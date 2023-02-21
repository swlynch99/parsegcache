use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::util::{ByteWriter, WriterCell, WriterToken};

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
  ///
  /// # Safety
  /// This method may only be called from the writer thread.
  pub(crate) unsafe fn insert(&self, key: &[u8], value: &[u8]) -> Option<DataRef<'seg>> {
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

  pub(crate) fn iter(&self) -> SegmentIter<'_, 'seg> {
    SegmentIter::new(self)
  }

  pub(crate) fn is_tail(&self, token: &WriterToken) -> bool {
    self.header().next.get(token).is_none()
  }

  /// Extend this segment with another one.
  ///
  /// # Panics
  /// Panics if this segment already has a next segment.
  pub(crate) fn extend(&self, next: Segment<'seg>, token: &mut WriterToken) {
    let prev = std::mem::replace(self.header().next.get_mut(token), Some(next));
    assert!(prev.is_none());
  }

  /// Reset this segment back to a blank state.
  /// 
  /// Returns the next segment in the list, if there is one.
  pub(crate) fn reset(&self, token: &mut WriterToken) -> Option<Segment<'seg>> {
    self.header().reset(token)
  }

  fn header(&self) -> &'seg SegmentHeader {
    unsafe { &*(self.data.as_ptr() as *const _) }
  }

  fn remaining(&self) -> usize {
    self.data.len() - self.header().tail.load(Ordering::Relaxed)
  }
}

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
  next: WriterCell<Option<Segment<'seg>>>,
}

impl<'seg> SegmentHeader<'seg> {
  fn reset(&self, token: &mut WriterToken) -> Option<Segment<'seg>> {
    self
      .tail
      .store(std::mem::size_of::<Self>(), Ordering::SeqCst);
    self.expiry.store(0, Ordering::SeqCst);
    std::mem::replace(self.next.get_mut(token), None)
  }
}

impl Default for SegmentHeader<'_> {
  fn default() -> Self {
    Self {
      tail: AtomicUsize::new(0),
      expiry: AtomicU64::new(0),
      next: WriterCell::new(None),
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
pub(crate) struct DataRef<'seg> {
  ptr: NonNull<u8>,
  _phantom: PhantomData<&'seg [u8]>,
}

impl<'seg> DataRef<'seg> {
  pub(crate) unsafe fn new(ptr: *const u8) -> Option<Self> {
    if ptr.is_null() {
      None
    } else {
      Some(Self::from_ptr(ptr))
    }
  }

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
}

pub(crate) struct SegmentIter<'a, 'seg> {
  ptr: *const u8,
  seg: &'a Segment<'seg>,
}

impl<'a, 'seg> SegmentIter<'a, 'seg> {
  fn new(seg: &'a Segment<'seg>) -> Self {
    let ptr = seg
      .data
      .as_ptr()
      .wrapping_add(std::mem::size_of::<SegmentHeader>());

    Self { ptr, seg }
  }
}

impl<'a, 'seg> Iterator for SegmentIter<'a, 'seg> {
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
