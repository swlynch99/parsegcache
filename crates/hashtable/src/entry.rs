use std::hash::Hash;

pub trait Entry {
  type Key: ?Sized + Hash + PartialEq;

  fn key(&self) -> &Self::Key;
}

pub trait RawEntry: Entry + Copy + Sized {
  type Target;

  /// Convert this entry to a raw pointer.
  ///
  /// Note that the returned pointer should not be null.
  fn as_ptr(&self) -> *const Self::Target;

  /// Convert a pointer back into an entry, returning `None` if the pointer is
  /// null.
  ///
  /// # Safety
  /// `ptr` must either be null or correspond to an instance returned by
  /// `as_ptr`.
  unsafe fn from_ptr(ptr: *const Self::Target) -> Option<Self> {
    match ptr {
      p if p.is_null() => None,
      p => Some(Self::from_ptr_unchecked(p)),
    }
  }

  unsafe fn from_ptr_unchecked(ptr: *const Self::Target) -> Self;
}

impl<'a, T> Entry for &'a T
where
  T: Entry,
{
  type Key = T::Key;

  fn key(&self) -> &Self::Key {
    (**self).key()
  }
}

impl<'a, T> RawEntry for &'a T
where
  T: Entry,
{
  type Target = T;

  fn as_ptr(&self) -> *const Self::Target {
    *self as *const _
  }

  unsafe fn from_ptr_unchecked(ptr: *const Self::Target) -> Self {
    &*ptr
  }
}

impl<'a> Entry for &'a str {
  type Key = Self;

  fn key(&self) -> &Self::Key {
    self
  }
}

impl<'a> Entry for &'a [u8] {
  type Key = Self;

  fn key(&self) -> &Self::Key {
    self
  }
}

impl<K, V> Entry for (K, V)
where
  K: PartialEq + Hash,
{
  type Key = K;

  fn key(&self) -> &Self::Key {
    &self.0
  }
}
