//! Module for various utilities that are not part of the crate API.

use std::cell::UnsafeCell;

/// Struct to make writing a sequence of values out to raw memory easier.
pub(crate) struct ByteWriter(*mut u8);

impl ByteWriter {
  pub unsafe fn new(ptr: *mut u8) -> Self {
    debug_assert!(!ptr.is_null());

    Self(ptr)
  }

  pub unsafe fn write<T>(&mut self, value: T) {
    debug_assert!(self.0.align_offset(std::mem::align_of::<T>()) == 0);

    std::ptr::write(self.0 as _, value);
    self.0 = self.0.add(std::mem::size_of::<T>());
  }

  pub unsafe fn write_bytes(&mut self, bytes: &[u8]) {
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.0, bytes.len());
    self.0 = self.0.add(bytes.len());
  }
}

pub(crate) struct WriterToken(());

impl WriterToken {
  /// Create a new writer token.
  /// 
  /// # Safety
  /// There should only be one writer token available at a time.
  pub(crate) unsafe fn new() -> Self {
    Self(())
  }
}

/// Cell type that only allows access by the writer thread.
#[derive(Default)]
pub(crate) struct WriterCell<T> {
  cell: UnsafeCell<T>,
}

impl<T> WriterCell<T> {
  pub(crate) fn new(value: T) -> Self {
    Self {
      cell: UnsafeCell::new(value),
    }
  }

  pub(crate) fn get<'c: 'v, 't: 'v, 'v>(&'c self, _token: &'t WriterToken) -> &'v T {
    unsafe { &*self.cell.get() }
  }

  pub(crate) fn get_mut<'c: 'v, 't: 'v, 'v>(&'c self, _token: &'t mut WriterToken) -> &'v mut T {
    unsafe { &mut *self.cell.get() }
  }
}
