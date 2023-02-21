use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use stable_vec::StableVec;

pub(crate) struct Collector<T> {
  shared: Arc<Shared>,
  cleanup: VecDeque<CleanupItem<T>>,
  last_min: u64,
}

impl<T> Collector<T> {
  pub fn new() -> Self {
    Self {
      shared: Arc::new(Shared {
        epoch: AtomicU64::new(1),
        entries: RwLock::new(StableVec::new()),
      }),
      cleanup: VecDeque::new(),
      last_min: 0,
    }
  }

  pub fn defer(&mut self, item: T) {
    let epoch = self.shared.epoch.load(Ordering::Acquire);
    self.cleanup.push_back(CleanupItem { epoch, item });
  }

  pub fn advance(&mut self) {
    // Handle counters use the low bit to indicate that the counter is unpinned so
    // we increment by 2 here.
    let prev = self.shared.epoch.fetch_add(2, Ordering::AcqRel);

    // Now, we increment all the dead counters to the previous epoch.
    let entries = self
      .shared
      .entries
      .read()
      .expect("entries lock was poisoned");
    for (_, entry) in entries.iter() {
      let mut epoch = entry.load(Ordering::Acquire);

      loop {
        if epoch & 1 == 0 {
          break;
        }

        match entry.compare_exchange(epoch, prev | 1, Ordering::AcqRel, Ordering::Acquire) {
          Ok(_) => break,
          Err(val) => epoch = val,
        }
      }
    }
  }

  pub fn collect(&mut self) -> Option<T> {
    let front = self.cleanup.front()?;
    if front.epoch < self.last_min {
      return self.cleanup.pop_front().map(|front| front.item);
    }

    self.last_min = self.find_safe_epoch();
    if front.epoch < self.last_min {
      return self.cleanup.pop_front().map(|front| front.item);
    }

    None
  }

  pub fn register(&mut self) -> Handle {
    let mut entries = self
      .shared
      .entries
      .write()
      .expect("entries lock was poisoned");

    let index = entries.push(PaddedCounter(AtomicU64::new(self.last_min | 1)));
    Handle {
      shared: Arc::clone(&self.shared),
      entry: index,
    }
  }

  fn find_safe_epoch(&self) -> u64 {
    let entries = self
      .shared
      .entries
      .read()
      .expect("entries lock was poisoned");

    entries
      .iter()
      .map(|(_, entry)| entry.load(Ordering::Acquire))
      .min()
      .unwrap_or_else(|| self.shared.epoch.load(Ordering::Acquire))
  }
}

pub(crate) struct Handle {
  shared: Arc<Shared>,
  entry: usize,
}

impl Handle {
  pub(crate) fn pin(&mut self) -> EpochGuard {
    EpochGuard::new(self)
  }
}

impl Drop for Handle {
  fn drop(&mut self) {
    let mut entries = self
      .shared
      .entries
      .write()
      .expect("entries lock was poisoned");

    entries.remove(self.entry);
  }
}

pub(crate) struct EpochGuard<'h> {
  handle: &'h Handle,
}

impl<'h> EpochGuard<'h> {
  fn new(handle: &'h Handle) -> Self {
    let entries = handle
      .shared
      .entries
      .read()
      .expect("read lock was poisoned");
    let entry = &entries[handle.entry];

    let current = handle.shared.epoch.load(Ordering::Acquire);
    let mut expected = entry.load(Ordering::Acquire);

    loop {
      match entry.compare_exchange(expected, current, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => break,
        Err(val) => expected = val,
      }
    }

    Self { handle }
  }
}

impl<'h> Drop for EpochGuard<'h> {
  fn drop(&mut self) {
    let entries = self
      .handle
      .shared
      .entries
      .read()
      .expect("read lock was poisoned");
    let entry = &entries[self.handle.entry];

    entry.fetch_or(1, Ordering::Release);
  }
}

struct Shared {
  epoch: AtomicU64,
  entries: RwLock<StableVec<PaddedCounter>>,
}

struct CleanupItem<T> {
  epoch: u64,
  item: T,
}

#[repr(align(64))]
struct PaddedCounter(AtomicU64);

impl Deref for PaddedCounter {
  type Target = AtomicU64;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}
