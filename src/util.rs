//! Module for various utilities that are not part of the crate API.

use std::cell::Cell;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Sub};

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

pub(crate) struct IntCell<T>(Cell<T>);

impl<T> IntCell<T> {
  pub fn new(value: T) -> Self {
    Self(Cell::new(value))
  }

  pub fn get_mut(&mut self) -> &mut T {
    self.0.get_mut()
  }
}

impl<T: Copy> IntCell<T> {
  pub fn get(&self) -> T {
    self.0.get()
  }

  pub fn replace(&self, val: T) -> T {
    self.0.replace(val)
  }
}

macro_rules! decl_fetch_op {
  ($trait:ident, $tmethod:ident, $method:ident) => {
    impl<T: Copy + $trait<Output = T>> IntCell<T> {
      pub fn $method(&self, value: T) -> T {
        self.replace($trait::$tmethod(self.get(), value))
      }
    }
  };
}

decl_fetch_op!(Add, add, fetch_add);
decl_fetch_op!(Sub, sub, fetch_sub);
decl_fetch_op!(Mul, mul, fetch_mul);
decl_fetch_op!(Div, div, fetch_div);
decl_fetch_op!(BitAnd, bitand, fetch_and);
decl_fetch_op!(BitOr, bitor, fetch_or);
decl_fetch_op!(BitXor, bitxor, fetch_xor);
