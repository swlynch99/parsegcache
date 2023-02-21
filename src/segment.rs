use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::util::ByteWriter;

pub(crate) struct Segment<'seg> {
  data: SegmentData<'seg>,
  tail: usize,
}

impl<'seg> Segment<'seg> {
  pub(crate) fn insert(&mut self, key: &[u8], value: &[u8]) -> Option<DataRef<'seg>> {
    let header = EntryHeader {
      klen: key.len(),
      vlen: value.len(),
    };

    if header.step_len() < self.remaining() {
      return None;
    }

    unsafe {
      let ptr = self.write_ptr();
      self.tail += header.step_len();

      let mut writer = ByteWriter::new(ptr);
      writer.write(header);
      writer.write_bytes(key);
      writer.write_bytes(value);

      Some(DataRef::from_ptr(ptr))
    }
  }

  pub(crate) fn iter(&self) -> SegmentIter<'_, 'seg> {
    SegmentIter::new(self)
  }

  fn write_ptr(&mut self) -> *mut u8 {
    unsafe { self.data.as_mut_ptr().add(self.tail) }
  }

  fn remaining(&self) -> usize {
    self.data.len() - self.tail
  }
}

pub(crate) struct SegmentData<'seg> {
  ptr: NonNull<u8>,
  len: usize,
  _phantom: PhantomData<&'seg [u8]>,
}

impl<'seg> SegmentData<'seg> {
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
  end: *const u8,

  _phantom: PhantomData<&'a Segment<'seg>>,
}

impl<'a, 'seg> SegmentIter<'a, 'seg> {
  fn new(segment: &'a Segment<'seg>) -> Self {
    let ptr = segment.data.as_ptr();

    Self {
      ptr,
      end: ptr.wrapping_add(segment.tail),
      _phantom: PhantomData,
    }
  }
}

impl<'a, 'seg> Iterator for SegmentIter<'a, 'seg> {
  type Item = DataRef<'seg>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.ptr >= self.end {
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
