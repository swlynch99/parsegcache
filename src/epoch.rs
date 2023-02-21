//! Traits and types for epoch based garbage collection.

use std::ops::Deref;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};

use sharded_slab::{Clear, OwnedEntry, Slab};

/// Trait for deleting items once the epoch collector has deemed them to be
/// ready for collection.
pub trait EpochDeleter<T> {
  /// Destroy the value.
  fn destroy(value: T);
}

/// An [`EpochDeleter`] which just drops the items.
pub struct DefaultDeleter;

impl<T> EpochDeleter<T> for DefaultDeleter {
  fn destroy(_: T) {}
}

pub(crate) struct EpochGC<T>(Arc<EpochData<T>>);

// impl<T> EpochGC<T> {
//   pub(crate) fn create_guard(&self) -> EpochGuard<T> {
//     let index = self.0.counters.insert(PaddedCounter(AtomicU64::new(0)))
//       .expect("Unable to allocate slot for guard epoch counter");

//   }
// }

struct EpochData<T> {
  value: T,
  current: AtomicU64,
  counters: Slab<PaddedCounter>,
}

pub(crate) struct EpochGuard<T> {
  data: Weak<EpochData<T>>,
  index: usize,
}

// impl EpochGuard {
// }

#[repr(align(64))]
#[derive(Default)]
struct PaddedCounter(AtomicU64);

impl Clear for PaddedCounter {
  fn clear(&mut self) {
    *self.0.get_mut() = 0;
  }
}

impl Deref for PaddedCounter {
  type Target = AtomicU64;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}
