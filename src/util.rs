//! Module for various utilities that are not part of the crate API.

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
