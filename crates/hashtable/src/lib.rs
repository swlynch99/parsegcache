//! This module contains a concurrent hashmap which allows multiple readers
//! while only allowing a single writer.
//!
//! # Restrictions
//! - [`HashTable::get`] may be called from any thread at any time.
//! - [`HashTable::insert`] and [`HashTable::erase`] may only be called by one
//!   thread at a time and calls to them must be synchronized (e.g. by a mutex).
//! - Entries in the hashtable can be represented as a single pointer and are
//!   copiable. In effect, this means that they must be some sort of reference.
//!
//! Note that due to implementation details (specifically how the
//! [`sharded_slab`] crate is implemented) calling write methods from a large
//! number of distinct threads may result in panics.
//!
//! # Implementation
//! The hash table implementation here is a custom combination of folly's
//! F14HashMap design along with the existing hashtable used in segcache.
//!
//! The top level of the hash table is a fixed-size lookup table. Each slot is
//! the start of a linked-list of buckets, containing only the pointer to the
//! first bucket in the chain. By default, each slot points to nothing and slots
//! are filled in as needed.
//!
//! Next, we have the [`Bucket`]s. These contain two main things:
//! 1. A 32-bit portion of the hash value called a tag, and,
//! 2. The pointers to the `T` values stored in the hash table.
//!
//! Currently, this is laid out so that we have
//! - the tags for all values
//! - the pointer to the next value in the chain
//! - and the actual pointer values for the map
//!
//! Similar to folly's F14HashMap, each bucket stores 14 values. This is exactly
//! enough so that the size of the tags + the next pointer occupies a single
//! 64-byte cache line. Then, the remaining values take up the next two cache
//! lines.
//!
//! The goal of this is to avoid needing to touch the extra cache lines unless
//! we actually need to use or modify the corresponding value. There are two
//! main benefits to this approach:
//! - We cut the number of cache lines we need to read by up to 2/3rds in most
//!   cases. Since reads for a hash table are mostly random accesses, this can
//!   help significantly.
//! - Since buckets are cache-aligned, we minimize false-sharing induced by
//!   writes to the hash table.
//!
//! ## Concurrency Notes
//! Most of the edge cases when writing a concurrent hash table come from two
//! areas
//! - Concurrent non-atomic modifications to internal data structures, and,
//! - Safe deletion/removal of values within the hash table
//!
//! Here we address those by mainly ignoring them:
//! - we only have a single writer thread which means we never have multiple
//!   modifications ongoing at the same time. This solves most of the difficult
//!   edge cases.
//! - we punt completely on deallocation of the underlying values.
//!
//! Nevertheless, we still have a few cases we need to consider:
//! 1. Concurrent `get` and `insert` calls.
//! 2. Concurrent `get` and `erase` calls.
//!
//! To avoid externally-visible inconsistent state we use the following rules:
//! - A value must _always_ be either a null pointer or a valid pointer to a `T`
//!   instance.
//! - When reading a value, null pointers are treated the same as a missing
//!   value, even if the tag may match.
//!
//! Then, the process for reading a KV pair from a slot looks like this
//! 1. Check to see if the tag matches that for our search key, if not, move on.
//! 2. Load the value pointer, if null, move on.
//! 3. Finally, check that the actual key is equivalent to the one stored with
//!    the value.
//! 4. If all of these succeed, then we've found our value and can return it.
//!
//! For writing a new value, the process is
//! 1. Write the value pointer.
//! 2. Then, update the tag from EMPTY_TAG to the key tag. If updating the slot
//!    for a key that has already been inserted then this step can be ignored.
//!
//! For deleting a value, the process is
//! 1. Set the tag to EMPTY_TAG.
//! 2. Set the value to null.
//!
//! This one isn't used currently but moving a value can be done by first
//! inserting it into a later slot in the list and then clearing the earlier
//! slot. Note that this only works if moving to a _later_ slot. Since `get`
//! operates by iterating in a forward direction, moving a value to an earlier
//! slot could result in a reader spuriously missing a key that is actually
//! present in the hash table.
//!
//! All of these procedures only work because there is only a single writer
//! thread. If there were multiple then it would be possible for other writers
//! to mess things up between the writes to the tag and value which would result
//! in an inconsistent table state.
//!
//! # Future Work
//! - It should be possible to remove values from the hash table as long as
//!   entries are only moved to a later slot in the chain. When doing an erase
//!   we should move the earliest entry down into the newly filled slot and free
//!   the first bucket in the chain if empty.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::AtomicU64;

use sharded_slab::pool::Ref;
use sharded_slab::{Clear, Pool};
use thiserror::Error;

use crate::atomics::*;

mod entry;

pub use crate::entry::{Entry, RawEntry};

#[cfg(feature = "loom")]
mod atomics {
  pub(crate) use loom::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
  pub(crate) use loom::sync::Arc;
}

#[cfg(not(feature = "loom"))]
mod atomics {
  pub(crate) use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
  pub(crate) use std::sync::Arc;
}

const CACHE_LINE: usize = 64;
const BUCKET_ELEMS: usize =
  (CACHE_LINE - std::mem::size_of::<usize>()) / std::mem::size_of::<u32>();
const INVALID_BUCKET_IDX: usize = usize::MAX;
const EMPTY_TAG: u32 = 0;

#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommandError {
  #[error("no backing memory capacity available to insert into hashtable")]
  NoCapacity,

  #[error("the cache writer thread has shut down")]
  NoWriter,
}

#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
#[error("no backing memory capacity available to insert into the hashtable")]
pub struct CapacityError;

impl From<CapacityError> for CommandError {
  fn from(_: CapacityError) -> Self {
    Self::NoCapacity
  }
}

pub struct Writer<T, S = RandomState>
where
  T: RawEntry,
{
  reader: Reader<T, S>,
}

impl<T, S> Writer<T, S>
where
  T: RawEntry,
  S: BuildHasher + Default,
{
  pub fn with_capacity(capacity: usize) -> Self {
    Self::new(HashTable::with_capacity(capacity))
  }
}

impl<T, S> Writer<T, S>
where
  T: RawEntry,
  S: BuildHasher,
{
  fn new(table: HashTable<T, S>) -> Self {
    Self {
      reader: Reader {
        table: Arc::new(table),
      },
    }
  }

  pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
    Self::new(HashTable::with_capacity_and_hasher(capacity, hasher))
  }

  /// Insert a value into the hashtable.
  ///
  /// If there is a value with the same key in the hashtable then it will be
  /// replaced and returned. If the table has no capacity then an error will be
  /// returned.
  pub fn insert(&mut self, data: T) -> Result<Option<T>, CapacityError> {
    unsafe { self.reader.table.insert(data) }
  }

  /// Erase an existing value from the hashtable by its key.
  ///
  /// Returns the value contained in the hash table, or `None` if no value was
  /// present.
  pub fn erase(&mut self, key: &T::Key) -> Option<T> {
    unsafe { self.reader.table.erase(key) }
  }

  /// Erase an entry from the hash table only if it matches the exact entry
  /// here.
  ///
  /// This will erase an entry in the table if it has the same pointer
  /// representation as `entry`. Returns true if an entry was erased.
  pub fn erase_by_entry(&mut self, entry: T) -> bool {
    unsafe { self.reader.table.erase_by_entry(entry) }
  }

  /// Create a new reader.
  pub fn reader(&self) -> Reader<T, S> {
    self.reader.clone()
  }
}

impl<T, S> Deref for Writer<T, S>
where
  T: RawEntry,
{
  type Target = Reader<T, S>;

  fn deref(&self) -> &Self::Target {
    &self.reader
  }
}

pub struct Reader<T, S = RandomState>
where
  T: RawEntry,
{
  table: Arc<HashTable<T, S>>,
}

impl<T, S> Reader<T, S>
where
  T: RawEntry,
  S: BuildHasher,
{
  /// Read a value from the hashtable, if present.
  pub fn get(&self, key: &T::Key) -> Option<T> {
    self.table.get(key)
  }

  /// Update the existing value in the hash table only if it contains `old`.
  ///
  /// This method is equivalent to a compare-and-swap internally. If you want an
  /// unconditional exchange operation you will need to use [`Writer::insert`].
  pub fn update(&self, old: &T, new: T) -> bool {
    self.table.update(old, new)
  }
}

impl<T, S> Clone for Reader<T, S>
where
  T: RawEntry,
{
  fn clone(&self) -> Self {
    Self {
      table: self.table.clone(),
    }
  }
}

struct HashTable<T, S = RandomState>
where
  T: RawEntry,
{
  buckets: Vec<AtomicUsize>,
  storage: Pool<Bucket<T::Target>>,

  hasher: S,

  _phantom: PhantomData<T>,
}

unsafe impl<T, S> Send for HashTable<T, S>
where
  T: RawEntry + Send,
  S: Send,
{
}

unsafe impl<T, S> Sync for HashTable<T, S>
where
  T: RawEntry + Sync,
  S: Sync,
{
}

impl<T, S> HashTable<T, S>
where
  T: RawEntry,
  S: BuildHasher + Default,
{
  pub fn with_capacity(capacity: usize) -> Self {
    Self::with_capacity_and_hasher(capacity, Default::default())
  }
}

impl<T, S> HashTable<T, S>
where
  T: RawEntry,
  S: BuildHasher,
{
  pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
    let mut buckets = Vec::with_capacity(capacity.next_power_of_two());
    buckets.resize_with(capacity, || AtomicUsize::new(INVALID_BUCKET_IDX));

    Self {
      buckets,
      storage: Pool::new(),
      hasher,
      _phantom: PhantomData,
    }
  }

  fn hash(&self, key: &T::Key) -> u64 {
    let mut hasher = self.hasher.build_hasher();
    key.hash(&mut hasher);
    let mut hash = hasher.finish();

    // TODO: Some sort of finalizer here to get better mixing?

    // We need the top 32 bits of hash to be nonzero since zero is used as a
    // sentinel value for a slot containing no value.
    if hash >> 32 == 0 {
      hash += 1 << 32;
    }

    hash
  }

  fn hash_tag(&self, key: &T::Key) -> (u32, usize) {
    let hash = self.hash(key);
    let tag = (hash >> 32) as u32;
    let index = (hash as usize) & (self.buckets.len() - 1);

    (tag, index)
  }

  /// Read a value from the hashtable, if present.
  pub fn get(&self, key: &T::Key) -> Option<T> {
    let (tag, index) = self.hash_tag(key);

    let mut iter = BucketIter::new(self, index);
    while let Some((ktag, value)) = iter.next() {
      if tag != ktag.load(Ordering::SeqCst) {
        continue;
      }

      // # Note
      // This may not be the key that corresponds to ktag due to concurrent
      // modifications by the writer thread. However, in that case the keys will not
      // be equal so nothing bad happens anyway.
      let elem = value.load(Ordering::SeqCst);
      let data = match unsafe { T::from_ptr(elem) } {
        Some(data) => data,
        None => continue,
      };

      if data.key() == key {
        return Some(data);
      }
    }

    None
  }

  /// Update the existing value in the hashtable only if it contains `old`.
  ///
  /// This method is equivalent to a compare-and-swap internally.
  pub fn update(&self, old: &T, new: T) -> bool {
    debug_assert!(
      old.key() == new.key(),
      "update passed values with different keys"
    );

    let (tag, index) = self.hash_tag(new.key());

    let mut iter = BucketIter::new(self, index);
    while let Some((ktag, value)) = iter.next() {
      if tag != ktag.load(Ordering::SeqCst) {
        continue;
      }

      match value.compare_exchange(
        old.as_ptr() as _,
        new.as_ptr() as *mut _,
        Ordering::SeqCst,
        Ordering::SeqCst,
      ) {
        Ok(_) => return true,
        Err(_) => continue,
      }
    }

    return false;
  }

  /// Insert a value into the hashtable, replacing any existing value with the
  /// same key.
  ///
  /// # Safety
  /// It is only valid to call this function from the writer thread.
  pub unsafe fn insert(&self, data: T) -> Result<Option<T>, CapacityError> {
    let hash = self.hash(data.key());
    let tag = (hash >> 32) as u32;
    let index = (hash as usize) & (self.buckets.len() - 1);

    // First, check to see if a value for the key already exists in the hashtable.
    // If so, then replace it. We need to do this instead of simply inserting the
    // key wherever so that concurrent gets never miss the value.
    let mut iter = BucketIter::new(self, index);
    while let Some((ktag, value)) = iter.next() {
      if ktag.load(Ordering::SeqCst) != tag {
        continue;
      }

      let old = unsafe { T::from_ptr_unchecked(value.load(Ordering::SeqCst)) };
      if old.key() != data.key() {
        continue;
      }

      value.store(data.as_ptr() as _, Ordering::SeqCst);
      return Ok(Some(old));
    }

    // Now that we have verified that key is not in the table we can just insert it
    // in the first empty slot.
    //
    // We're the only thread writing things so we know that none of the values will
    // be modified under our feet.
    iter = BucketIter::new_extend(self, index)?;
    loop {
      let (ktag, value) = iter.next_extend()?;

      if ktag.load(Ordering::SeqCst) != EMPTY_TAG {
        continue;
      }

      value.store(data.as_ptr() as _, Ordering::SeqCst);
      ktag.store(tag, Ordering::SeqCst);
      break;
    }

    Ok(None)
  }

  /// Erase a value from the hashtable. Returns the value removed, if any.
  ///
  /// # Safety
  /// It is only valid to call this function from the writer thread.
  pub unsafe fn erase(&self, key: &T::Key) -> Option<T> {
    let hash = self.hash(key);
    let tag = (hash >> 32) as u32;
    let index = (hash as usize) & (self.buckets.len() - 1);

    let mut iter = BucketIter::new(self, index);
    while let Some((ktag, value)) = iter.next() {
      if tag != ktag.load(Ordering::SeqCst) {
        continue;
      }

      let elem = value.load(Ordering::SeqCst);
      let data = match unsafe { T::from_ptr(elem) } {
        Some(data) => data,
        None => continue,
      };

      if data.key() != key {
        continue;
      }

      ktag.clear(Ordering::SeqCst);
      value.store(std::ptr::null_mut(), Ordering::SeqCst);

      return Some(data);
    }

    None
  }

  /// Erase an entry from the hashtable only if it matches the exact entry here.
  ///
  /// # Safety
  /// It is only valid to call this function from the writer thread.
  pub unsafe fn erase_by_entry(&self, entry: T) -> bool {
    let (tag, index) = self.hash_tag(entry.key());

    let mut iter = BucketIter::new(self, index);
    while let Some((ktag, value)) = iter.next() {
      if tag != ktag.load(Ordering::SeqCst) {
        continue;
      }

      let elem = value.load(Ordering::SeqCst);
      let data = match unsafe { T::from_ptr(elem) } {
        Some(data) => data,
        None => continue,
      };

      if data.as_ptr() != entry.as_ptr() {
        continue;
      }

      ktag.clear(Ordering::SeqCst);
      value.store(std::ptr::null_mut(), Ordering::SeqCst);

      return true;
    }

    return false;
  }
}

#[repr(C, align(64))]
struct Bucket<T> {
  next: AtomicUsize,
  keys: AtomicU32Array<{ BUCKET_ELEMS / 2 }>,
  /// # Validity Invariant
  /// Each element will always either be null or point to a valid DataRef
  /// instance.
  values: [AtomicPtr<T>; BUCKET_ELEMS],
}

impl<T> Bucket<T> {
  pub fn new() -> Self {
    Self {
      keys: AtomicU32Array::new(),
      next: AtomicUsize::new(usize::MAX),
      values: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())),
    }
  }
}

impl<T> Default for Bucket<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T> Clear for Bucket<T> {
  fn clear(&mut self) {
    *self = Self::new();
  }
}

struct BucketIter<'tbl, T, S>
where
  T: RawEntry,
{
  table: &'tbl HashTable<T, S>,
  bucket: Option<Ref<'tbl, Bucket<T::Target>>>,
  index: usize,
}

impl<'tbl, T, S> BucketIter<'tbl, T, S>
where
  T: RawEntry,
{
  pub fn new(table: &'tbl HashTable<T, S>, index: usize) -> Self {
    let bucket = table.buckets[index].load(Ordering::SeqCst);

    Self {
      table,
      bucket: table.storage.get(bucket),
      index: 0,
    }
  }

  pub fn next(&mut self) -> Option<(AtomicU32Proxy, &AtomicPtr<T::Target>)> {
    loop {
      let bucket = self.bucket.as_ref()?;

      if self.index >= BUCKET_ELEMS {
        let bucket_idx = bucket.next.load(Ordering::SeqCst);

        self.index = 0;
        self.bucket = self.table.storage.get(bucket_idx);
      } else {
        let vals = (bucket.keys.get(self.index), &bucket.values[self.index]);
        self.index += 1;

        // Extend the lifetime of vals to avoid a borrow checker error.
        //
        // # Safety
        // The error here is due to a limitation in the borrow checker. It would be
        // valid to return the references except rustc sees the modification to
        // self.bucket in the branch above and errors, despite control flow never being
        // able to reach this branch.
        return Some(unsafe { std::mem::transmute(vals) });
      }
    }
  }

  /// Creates a new iterator, allocating a new bucket if necessary.
  ///
  /// This function should only be called from the write thread.
  pub fn new_extend(table: &'tbl HashTable<T, S>, index: usize) -> Result<Self, CapacityError> {
    let bucket_var = &table.buckets[index];
    let bucket_idx = bucket_var.load(Ordering::SeqCst);

    let bucket = match bucket_idx {
      INVALID_BUCKET_IDX => match table.storage.create() {
        Some(bucket) => {
          // No other threads are running this method concurrently so this works.
          //
          // We could use a store here but I've used a swap so that we can check
          // invariants.
          let prev = bucket_var.swap(bucket.key(), Ordering::SeqCst);
          assert_eq!(prev, INVALID_BUCKET_IDX);

          bucket.downgrade()
        }
        None => return Err(CapacityError),
      },
      index => match table.storage.get(index) {
        Some(bucket) => bucket,
        None => panic!("got invalid bucket index {}", index),
      },
    };

    Ok(Self {
      table,
      bucket: Some(bucket),
      index: 0,
    })
  }

  /// Step to the next key-value slot, allocating a new bucket if necessary.
  ///
  /// This function should only be called from the write thread.
  pub fn next_extend(&mut self) -> Result<(AtomicU32Proxy, &AtomicPtr<T::Target>), CapacityError> {
    loop {
      let bucket = self
        .bucket
        .as_ref()
        .expect("next_extend called with missing bucket");

      if self.index >= BUCKET_ELEMS {
        let next = match bucket.next.load(Ordering::SeqCst) {
          INVALID_BUCKET_IDX => match self.table.storage.create() {
            Some(next) => {
              bucket.next.store(next.key(), Ordering::SeqCst);
              next.downgrade()
            }
            None => return Err(CapacityError),
          },
          index => match self.table.storage.get(index) {
            Some(bucket) => bucket,
            None => panic!("bucket had invalid next index {}", index),
          },
        };

        self.index = 0;
        self.bucket = Some(next);
      } else {
        let vals = (bucket.keys.get(self.index), &bucket.values[self.index]);
        self.index += 1;

        // Extend the lifetime of vals to avoid a borrow checker error.
        //
        // # Safety
        // The error here is due to a limitation in the borrow checker. It would be
        // valid to return the references except rustc sees the modification to
        // self.bucket in the branch above and errors, despite control flow never being
        // able to reach this branch.
        return Ok(unsafe { std::mem::transmute(vals) });
      }
    }
  }

  #[allow(dead_code)]
  pub fn bucket(&self) -> Option<&Bucket<T::Target>> {
    self.bucket.as_deref()
  }

  #[allow(dead_code)]
  pub fn bucket_idx(&self) -> Option<usize> {
    self.bucket.as_ref().map(|bucket| bucket.key())
  }
}

struct AtomicU32Array<const HN: usize> {
  data: [AtomicU64; HN],
}

impl<const HN: usize> AtomicU32Array<HN> {
  const N: usize = HN * 2;

  const fn new() -> Self {
    const EMPTY_VAL: AtomicU64 = AtomicU64::new((EMPTY_TAG as u64) | ((EMPTY_TAG as u64) << 32));

    Self {
      data: [EMPTY_VAL; HN],
    }
  }

  fn get(&self, index: usize) -> AtomicU32Proxy {
    assert!(
      index < Self::N,
      "index was out of bounds: {index} >= {}",
      Self::N
    );

    let (index, half) = (index / 2, index % 2 == 1);

    AtomicU32Proxy {
      value: &self.data[index],
      half,
    }
  }
}

struct AtomicU32Proxy<'a> {
  value: &'a AtomicU64,
  half: bool,
}

impl AtomicU32Proxy<'_> {
  fn mask(&self) -> u64 {
    match self.half {
      true => (u32::MAX as u64) << 32,
      false => u32::MAX as u64,
    }
  }

  fn shift(&self) -> u64 {
    match self.half {
      true => 32,
      false => 0,
    }
  }

  fn load(&self, ordering: Ordering) -> u32 {
    (self.value.load(ordering) >> self.shift()) as u32
  }

  fn clear(&self, ordering: Ordering) {
    self.value.fetch_and(!self.mask(), ordering);
  }

  fn store(&self, value: u32, ordering: Ordering) {
    debug_assert_eq!(self.load(Ordering::Relaxed), 0);
    self
      .value
      .fetch_or((value as u64) << self.shift(), ordering);
  }
}

#[test]
fn test_bucket_field_offsets() {
  assert_eq!(memoffset::offset_of!(Bucket<u64>, values), CACHE_LINE);
  assert_eq!(memoffset::offset_of!(Bucket<*const ()>, values), CACHE_LINE);
}
