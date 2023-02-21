use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use parsegcache_hashtable::{Entry, RawEntry};

use crate::util::ByteWriter;

pub(crate) struct Segment<'seg> {
  data: SegmentData<'seg>,
}

impl<'seg> Segment<'seg> {
  pub(crate) fn new(data: &'seg mut [u8]) -> Self {
    assert!(data.len() >= std::mem::size_of::<SegmentHeader>());

    let mut data = SegmentData::new(data);
    unsafe {
      std::ptr::write(data.as_mut_ptr() as _, SegmentHeader::default());
    }

    Self { data }
  }

  /// Insert a new value into this segment. On success, returns a [`DataRef`]
  /// for the key-value pair.
  pub(crate) fn insert(&mut self, key: &[u8], value: &[u8]) -> Option<DataRef<'seg>> {
    let header = EntryHeader {
      klen: key.len(),
      vlen: value.len(),
    };

    if header.step_len() < self.remaining() {
      return None;
    }

    let seghdr = self.header();

    unsafe {
      let tail = seghdr.tail.load(Ordering::Relaxed);
      // We are only writing to a range that is not being written to by anyone else so
      // this is safe.
      let ptr = self.data.as_ptr().wrapping_add(tail) as *mut _;

      let mut writer = ByteWriter::new(ptr);
      writer.write(header);
      writer.write_bytes(key);
      writer.write_bytes(value);

      // The release store will synchronize with future acquire stores of tail.
      seghdr
        .tail
        .store(tail + header.step_len(), Ordering::Release);

      Some(DataRef::from_ptr(ptr))
    }
  }

  /// Undo the previous insertion.
  ///
  /// # Safety
  /// The [`DataRef`] passed in must have been the one returned by the prevous
  /// [`insert`](Self::insert) call to have been made on this segment.
  ///
  /// The [`DataRef`] instance must also not be accessed after calling this
  /// method.
  #[allow(dead_code)]
  pub(crate) unsafe fn uninsert(&mut self, data: DataRef<'seg>) {
    let seghdr = self.header();
    let step_len = data.step_len();

    let tail = seghdr.tail.fetch_sub(step_len, Ordering::AcqRel) - step_len;
    let tail_ptr = self.data.as_ptr().wrapping_add(tail);

    assert_eq!(
      tail_ptr,
      data.as_ptr(),
      "data was not at the end of this segment"
    );
  }

  pub(crate) fn iter(&self) -> EntryIter<'_, 'seg> {
    EntryIter::new(self)
  }

  pub(crate) fn take_next(&mut self) -> Option<Segment<'seg>> {
    unsafe { &mut *self.header().next.get() }.take()
  }

  /// Extend this segment with another one.
  ///
  /// # Panics
  /// Panics if this segment already has a next segment.
  pub(crate) fn extend<'a>(&'a mut self, next: Segment<'seg>) -> &'a mut Segment<'seg> {
    let next_ref = unsafe { &mut *self.header().next.get() };
    assert!(next_ref.is_none());

    *next_ref = Some(next);
    next_ref.as_mut().unwrap()
  }

  /// Reset this segment back to a blank state.
  ///
  /// Returns the next segment in the list, if there is one.
  pub(crate) fn reset(&mut self, expiry: SystemTime) -> Option<Segment<'seg>> {
    unsafe { self.header().reset(expiry) }
  }

  fn header(&self) -> &'seg SegmentHeader<'seg> {
    unsafe { &*(self.data.as_ptr() as *const _) }
  }

  fn remaining(&self) -> usize {
    self.data.len() - self.header().tail.load(Ordering::Relaxed)
  }
}

unsafe impl Send for Segment<'_> {}
unsafe impl Sync for Segment<'_> {}

/// Metadata fields for the segment.
///
/// Values in here should only be written to by the writer thread but may be
/// read by any thread.
pub(crate) struct SegmentHeader<'seg> {
  /// Offset at which the next entry will be inserted.
  ///
  /// Always read this field using acquire loads to ensure that the inserted
  /// data is visible.
  tail: AtomicUsize,

  /// Time at which this segment expires.
  expiry: AtomicU64,

  /// Next segment in the index.
  next: UnsafeCell<Option<Segment<'seg>>>,
}

impl<'seg> SegmentHeader<'seg> {
  unsafe fn reset(&self, expiry: SystemTime) -> Option<Segment<'seg>> {
    let ts = expiry
      .duration_since(SystemTime::UNIX_EPOCH)
      .unwrap()
      .as_millis()
      .try_into()
      .expect("Timestamp as milliseconds was larger than a u64");

    self
      .tail
      .store(std::mem::size_of::<Self>(), Ordering::SeqCst);
    self.expiry.store(ts, Ordering::SeqCst);
    std::mem::replace(&mut *self.next.get(), None)
  }

  pub(crate) fn expiry(&self) -> SystemTime {
    let expiry = self.expiry.load(Ordering::SeqCst);
    let expiry = Duration::from_millis(expiry);
    SystemTime::UNIX_EPOCH + expiry
  }
}

impl Default for SegmentHeader<'_> {
  fn default() -> Self {
    Self {
      tail: AtomicUsize::new(0),
      expiry: AtomicU64::new(0),
      next: UnsafeCell::new(None),
    }
  }
}

pub(crate) struct SegmentData<'seg> {
  ptr: NonNull<u8>,
  len: usize,
  _phantom: PhantomData<&'seg [u8]>,
}

impl<'seg> SegmentData<'seg> {
  pub(crate) fn new(data: &'seg mut [u8]) -> Self {
    assert_eq!(
      data
        .as_ptr()
        .align_offset(std::mem::align_of::<SegmentHeader>()),
      0
    );

    Self {
      ptr: unsafe { NonNull::new_unchecked(data.as_mut_ptr()) },
      len: data.len(),
      _phantom: PhantomData,
    }
  }

  pub(crate) fn as_ptr(&self) -> *const u8 {
    self.ptr.as_ptr() as *const u8
  }

  pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
    self.ptr.as_ptr()
  }

  pub(crate) fn len(&self) -> usize {
    self.len
  }
}

unsafe impl Send for SegmentData<'_> {}

#[derive(Copy, Clone, Debug)]
pub(crate) struct EntryHeader {
  klen: usize,
  vlen: usize,
}

impl EntryHeader {
  pub(crate) fn klen(&self) -> usize {
    self.klen
  }

  pub(crate) fn vlen(&self) -> usize {
    self.vlen
  }

  pub(crate) fn step_len(&self) -> usize {
    round_up(
      std::mem::size_of_val(self) + self.klen() + self.vlen(),
      std::mem::align_of_val(self),
    )
  }
}

/// A reference to a stored key-value pair in the cache.
///
/// # Validity Invariant
/// The implementation here assumes that the pointer contained within this
/// DataRef points to a valid cache entry. That is, in order,
/// - an EntryHeader
/// - the bytes for the key
/// - the bytes for the value
#[derive(Copy, Clone)]
pub struct DataRef<'seg> {
  ptr: NonNull<u8>,
  _phantom: PhantomData<&'seg [u8]>,
}

impl<'seg> DataRef<'seg> {
  /// Create a DataRef from an existing pointer.
  ///
  /// # Safety
  /// The pointer must point to a valid cache entry in memory.
  pub(crate) unsafe fn from_ptr(ptr: *const u8) -> Self {
    debug_assert!(!ptr.is_null());
    debug_assert!(is_aligned_to(ptr, std::mem::align_of::<EntryHeader>()));

    Self {
      ptr: NonNull::new_unchecked(ptr as _),
      _phantom: PhantomData,
    }
  }

  pub(crate) fn as_ptr(&self) -> *const u8 {
    self.ptr.as_ptr() as _
  }

  pub(crate) fn header(&self) -> &EntryHeader {
    unsafe { &*(self.as_ptr() as *const EntryHeader) }
  }

  fn data_ptr(&self) -> *const u8 {
    unsafe { self.as_ptr().add(std::mem::size_of::<EntryHeader>()) }
  }

  pub(crate) fn key(&self) -> &[u8] {
    let header = self.header();
    let ptr = self.data_ptr();
    unsafe { std::slice::from_raw_parts(ptr, header.klen()) }
  }

  pub(crate) fn value(&self) -> &[u8] {
    let header = self.header();
    let ptr = unsafe { self.data_ptr().add(header.klen()) };
    unsafe { std::slice::from_raw_parts(ptr, header.vlen()) }
  }

  pub(crate) fn step_len(&self) -> usize {
    self.header().step_len()
  }
}

impl<'seg> Entry for DataRef<'seg> {
  type Key = [u8];

  fn key(&self) -> &Self::Key {
    self.key()
  }
}

impl<'seg> RawEntry for DataRef<'seg> {
  type Target = u8;

  fn as_ptr(&self) -> *const Self::Target {
    self.as_ptr()
  }

  unsafe fn from_ptr_unchecked(ptr: *const Self::Target) -> Self {
    Self::from_ptr(ptr)
  }
}

unsafe impl Send for DataRef<'_> {}
unsafe impl Sync for DataRef<'_> {}

pub(crate) struct EntryIter<'a, 'seg> {
  ptr: *const u8,
  seg: &'a Segment<'seg>,
}

impl<'a, 'seg> EntryIter<'a, 'seg> {
  fn new(seg: &'a Segment<'seg>) -> Self {
    let ptr = seg
      .data
      .as_ptr()
      .wrapping_add(std::mem::size_of::<SegmentHeader>());

    Self { ptr, seg }
  }
}

impl<'a, 'seg> Iterator for EntryIter<'a, 'seg> {
  type Item = DataRef<'seg>;

  fn next(&mut self) -> Option<Self::Item> {
    let tail = self.seg.header().tail.load(Ordering::Acquire);
    let end = self.seg.data.as_ptr().wrapping_add(tail);
    if self.ptr >= end {
      return None;
    }

    let data = unsafe { DataRef::from_ptr(self.ptr) };
    self.ptr = self.ptr.wrapping_add(data.header().step_len());
    Some(data)
  }
}

fn round_up(value: usize, round: usize) -> usize {
  debug_assert!(round.is_power_of_two());

  (value + (round - 1)) & (round - 1)
}

// Copied from std::ptr::is_aligned_to
#[allow(dead_code)]
fn is_aligned_to<T>(ptr: *const T, align: usize) -> bool {
  ptr.cast::<()>().align_offset(align) == 0
}
