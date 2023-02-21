use std::time::SystemTime;

use parsegcache_hashtable::Reader;

use crate::epoch::{EpochRef, Handle};
use crate::segment::DataRef;
use crate::shared::Shared;
use crate::{CacheConfig, Entry, Mmap};

#[derive(Clone)]
pub(crate) struct CacheReader<'seg> {
  handle: Handle,
  hashtable: Reader<DataRef<'seg>>,
  shared: Shared<'seg>,
}

impl<'seg> CacheReader<'seg> {
  pub(crate) fn new(
    handle: Handle,
    hashtable: Reader<DataRef<'seg>>,
    shared: Shared<'seg>,
  ) -> Self {
    Self {
      handle,
      hashtable,
      shared,
    }
  }

  pub fn get(&self, key: &[u8]) -> Option<EpochRef<Entry<'seg>>> {
    let guard = self.handle.pin();
    let data = self.hashtable.get(key)?;
    let expiry = self.shared.expiry_for(data);

    if expiry < SystemTime::now() {
      return None;
    }

    Some(EpochRef::new(guard, Entry::new(data, expiry)))
  }

  pub fn config(&self) -> &CacheConfig {
    &self.shared.config
  }

  /// Convert this reader to borrow from a different reference to the same
  /// underlying `MmapRaw`.
  ///
  /// # Panics
  /// Panics if data is not a reference to the same `MmapRaw` that was used to
  /// construct this `Reader`.
  pub(crate) fn transmute_lifetime<'a>(self, data: &'a Mmap) -> CacheReader<'a> {
    // This is only valid if we're referring to the same instance of data with
    // different lifetimes.
    assert_eq!(self.shared.data as *const Mmap, data as *const Mmap);

    // SAFETY: data is a reference to the same MmapRaw instance as self.shared.data.
    //        'seg is only for data borrowed from self.shared.data so it is safe to
    //        change our borrow to instead use the new lifetime.
    unsafe { std::mem::transmute(self) }
  }
}
