//! This module contains a concurrent hashmap which allows multiple readers
//! while only allowing a single writer.
//!
//! # Restrictions
//! - [`HashTable::get`] may be called from any thread at any time.
//! - [`HashTable::insert`] and [`HashTable::erase`] may only be called by one
//!   thread at a time and calls to them must be synchronized (e.g. by a mutex).
//! - The user of the hashtable is responsible for ensuring that [`DataRef`]
//!   instances within the hashtable are not released while they are contained
//!   within the hashtable or being used by any thread.
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
//! 2. The pointers to the [`DataRef`] values stored in the hash table.
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
//! - A value must _always_ be either a null pointer or a valid pointer to a
//!   [`DataRef`] instance.
//! - When reading a value, null pointers are treated the same as a missing
//!   value, even if the tag may match.
//!
//! Then, the process for reading a KV pair from a slot looks like this
//! 1. Check to see if the tag matches that for our search key, if not, move on.
//! 2. Load the value pointer, if null, move on.
//! 3. Finally, check that the actual key is equivalent to the one stored with
//!    the value.
//! 4. If all of these succeed, then we've found our value and can return the
//!    [`DataRef`].
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

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

use sharded_slab::pool::Ref;
use sharded_slab::{Clear, Pool};
use thiserror::Error;

use crate::DataRef;

const BUCKET_ELEMS: usize = (64 - std::mem::size_of::<usize>()) / std::mem::size_of::<u32>();
const INVALID_BUCKET_IDX: usize = usize::MAX;
const EMPTY_TAG: u32 = 0;

#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum InsertError {
  #[error("no backing memory capacity available to insert into hashtable")]
  NoCapacity,
}

pub(crate) struct HashTable<'seg, S = RandomState> {
  buckets: Vec<AtomicUsize>,
  storage: Pool<Bucket>,
  hasher: S,

  _phantom: PhantomData<DataRef<'seg>>,
}

#[repr(C, align(64))]
struct Bucket {
  keys: [AtomicU32; BUCKET_ELEMS],
  next: AtomicUsize,
  /// # Validity Invariant
  /// Each element will always either be null or point to a valid DataRef
  /// instance.
  values: [AtomicPtr<u8>; BUCKET_ELEMS],
}

impl<'seg, S> HashTable<'seg, S>
where
  S: BuildHasher + Default,
{
  pub(crate) fn with_capacity(capacity: usize) -> Self {
    Self::with_capacity_and_hasher(capacity, Default::default())
  }
}

impl<'seg, S> HashTable<'seg, S>
where
  S: BuildHasher,
{
  pub(crate) fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
    let mut buckets = Vec::with_capacity(capacity.next_power_of_two());
    buckets.resize_with(capacity, || AtomicUsize::new(0));

    Self {
      buckets,
      storage: Pool::new(),
      hasher,
      _phantom: PhantomData,
    }
  }

  fn hash(&self, key: &[u8]) -> u64 {
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

  fn hash_tag(&self, key: &[u8]) -> (u32, usize) {
    let hash = self.hash(key);
    let tag = (hash >> 32) as u32;
    let index = (hash as usize) & (self.buckets.len() - 1);

    (tag, index)
  }

  /// Read a value from the hashtable, if present.
  pub(crate) fn get(&self, key: &[u8]) -> Option<DataRef<'seg>> {
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
      let data = match unsafe { DataRef::new(elem) } {
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
  pub(crate) fn update(&self, old: DataRef<'seg>, new: DataRef<'seg>) -> bool {
    debug_assert_eq!(
      old.key(),
      new.key(),
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
        new.as_ptr() as _,
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
  pub(crate) unsafe fn insert(
    &self,
    data: DataRef<'seg>,
  ) -> Result<Option<DataRef<'seg>>, InsertError> {
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

      let old = unsafe { DataRef::from_ptr(value.load(Ordering::SeqCst)) };
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
  pub(crate) unsafe fn erase(&self, key: &[u8]) -> Option<DataRef<'seg>> {
    let hash = self.hash(key);
    let tag = (hash >> 32) as u32;
    let index = (hash as usize) & (self.buckets.len() - 1);

    let mut iter = BucketIter::new(self, index);
    while let Some((ktag, value)) = iter.next() {
      if tag != ktag.load(Ordering::SeqCst) {
        continue;
      }

      let elem = value.load(Ordering::SeqCst);
      let data = match unsafe { DataRef::new(elem) } {
        Some(data) => data,
        None => continue,
      };

      if data.key() != key {
        continue;
      }

      ktag.store(EMPTY_TAG, Ordering::SeqCst);
      value.store(std::ptr::null_mut(), Ordering::SeqCst);

      return Some(data);
    }

    None
  }
}

impl Bucket {
  pub(crate) const fn new() -> Self {
    const KEY_INIT: AtomicU32 = AtomicU32::new(EMPTY_TAG);
    const VAL_INIT: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());

    Self {
      keys: [KEY_INIT; BUCKET_ELEMS],
      next: AtomicUsize::new(usize::MAX),
      values: [VAL_INIT; BUCKET_ELEMS],
    }
  }
}

impl Default for Bucket {
  fn default() -> Self {
    Self::new()
  }
}

impl Clear for Bucket {
  fn clear(&mut self) {
    *self = Self::new();
  }
}

struct BucketIter<'tbl, 'seg, S> {
  table: &'tbl HashTable<'seg, S>,
  bucket: Option<Ref<'tbl, Bucket>>,
  index: usize,
}

impl<'tbl, 'seg, S> BucketIter<'tbl, 'seg, S> {
  pub(crate) fn new(table: &'tbl HashTable<'seg, S>, index: usize) -> Self {
    let bucket = table.buckets[index].load(Ordering::SeqCst);

    Self {
      table,
      bucket: table.storage.get(bucket),
      index: 0,
    }
  }

  pub(crate) fn next(&mut self) -> Option<(&AtomicU32, &AtomicPtr<u8>)> {
    loop {
      let bucket = self.bucket.as_ref()?;

      if self.index >= BUCKET_ELEMS {
        let bucket_idx = bucket.next.load(Ordering::SeqCst);

        self.index = 0;
        self.bucket = self.table.storage.get(bucket_idx);
      } else {
        let vals = (&bucket.keys[self.index], &bucket.values[self.index]);
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
  pub(crate) fn new_extend(
    table: &'tbl HashTable<'seg, S>,
    index: usize,
  ) -> Result<Self, InsertError> {
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
        None => return Err(InsertError::NoCapacity),
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
  pub(crate) fn next_extend(&mut self) -> Result<(&AtomicU32, &AtomicPtr<u8>), InsertError> {
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
            None => return Err(InsertError::NoCapacity),
          },
          index => match self.table.storage.get(index) {
            Some(bucket) => bucket,
            None => panic!("bucket had invalid next index {}", index),
          },
        };

        self.index = 0;
        self.bucket = Some(next);
      } else {
        let vals = (&bucket.keys[self.index], &bucket.values[self.index]);
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

  pub(crate) fn bucket(&self) -> Option<&Bucket> {
    self.bucket.as_deref()
  }

  pub(crate) fn bucket_idx(&self) -> Option<usize> {
    self.bucket.as_ref().map(|bucket| bucket.key())
  }
}

#[test]
fn test_bucket_field_offsets() {
  assert_eq!(memoffset::offset_of!(Bucket, values), 64);
}
