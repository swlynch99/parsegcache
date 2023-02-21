
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use memmap2::MmapRaw;
use ouroboros::self_referencing;
use parsegcache_hashtable::CapacityError;

use crate::epoch::EpochRef;
use crate::reader::CacheReader;
use crate::segment::DataRef;
use crate::writer::CacheWriter;

pub mod epoch;
pub mod segment;
mod reader;
mod shared;
mod util;
mod writer;

#[derive(Clone, Debug)]
pub struct CacheConfig {
  pub hashtable_capacity: usize,
  pub segment_len: usize,

  pub timer_span: Duration,
  pub timer_bucket: Duration,
  pub timer_margin: Duration,

  /// The maximum number of segments that can be expired between the execution
  /// of successive commands on the write thread.
  ///
  /// This acts as a bound on of the maximum extra latency between the
  /// processing of any two consecutive write commands. A lower value will
  /// result in less work being done between write commands under heavy load and
  /// may slow down the rate at which entries are expired.
  ///
  /// Note that this does not mean that expiry will stall if no write commands
  /// come in.
  pub expire_per_iter: usize,
}

pub struct Entry<'seg> {
  pub(crate) data: DataRef<'seg>,
  pub(crate) expiry: SystemTime,
}

impl<'seg> Entry<'seg> {
  pub(crate) fn new(data: DataRef<'seg>, expiry: SystemTime) -> Self {
    Self { data, expiry }
  }

  pub fn key(&self) -> &[u8] {
    self.data.key()
  }

  pub fn value(&self) -> &[u8] {
    self.data.value()
  }

  pub fn expiry(&self) -> SystemTime {
    self.expiry
  }
}

#[self_referencing]
struct WriterDetail {
  data: Arc<MmapRaw>,

  #[borrows(data)]
  #[not_covariant]
  cache: CacheWriter<'this>,
}

pub struct Writer(WriterDetail);

impl Writer {
  /// Fetch an entry from the cache, if present.
  pub fn get(&self, key: &[u8]) -> Option<Entry> {
    self.0.with_cache(|cache| cache.get(key))
  }

  /// Set an entry within the cache.
  /// 
  /// Returns the previous entry within the cache, or an error if there was no
  /// capacity to insert the new entry into the cache.
  pub fn set(
    &mut self,
    key: &[u8],
    value: &[u8],
    expiry: SystemTime,
  ) -> Result<Option<Entry>, CapacityError> {
    self.0.with_cache_mut(|cache| cache.set(key, value, expiry))
  }

  /// Delete an entry from the cache.
  /// 
  /// Returns the deleted entry, if one was present within the cache.
  pub fn delete(&mut self, key: &[u8]) -> Option<Entry> {
    self.0.with_cache_mut(|cache| cache.delete(key))
  }

  /// Perform background cleanup tasks on the cache.
  /// 
  /// This includes
  /// - removing entries that have expired,
  /// - (in the future) evicting cache entries when free space is low, and,
  /// - garbage collecting expired segments that are no longer referenced by a
  ///   reader thread.
  /// 
  /// It is recommended to call expire frequently. Ideally it would be called
  /// after every command and periodically if no commands are being issued to
  /// the `Writer`. Under the default config, `expire` is a lightweight and
  /// time-bounded operation and so it should be fine to call it frequently.
  pub fn expire(&mut self) {
    self.0.with_cache_mut(|cache| cache.expire())
  }

  /// Create a new reader for this cache.
  pub fn reader(&mut self) -> Reader {
    let data = self.0.borrow_data().clone();

    self
      .0
      .with_cache_mut(move |cache| Reader::new(data, cache.reader()))
  }

  /// Get the config that was used to construct this cache.
  pub fn config(&self) -> &CacheConfig {
    self.0.with_cache(|cache| cache.config())
  }
}

#[self_referencing]
pub struct ReaderDetail {
  data: Arc<MmapRaw>,

  #[borrows(data)]
  #[not_covariant]
  cache: CacheReader<'this>,
}

pub struct Reader(ReaderDetail);

impl Reader {
  /// Construct a new `Reader` from its component parts.
  ///
  /// # Panics
  /// Panics if `data` does not point to the same [`MmapRaw`] instance that the
  /// [`CacheReader`] was constructed with.
  pub(crate) fn new(data: Arc<MmapRaw>, cache: CacheReader<'_>) -> Self {
    Self(ReaderDetail::new(data, move |data| {
      cache.transmute_lifetime(&data)
    }))
  }

  pub fn get(&self, key: &[u8]) -> Option<EpochRef<Entry>> {
    self.0.with_cache(|cache| cache.get(key))
  }

  pub fn config(&self) -> &CacheConfig {
    self.0.with_cache(|cache| cache.config())
  }
}

impl Clone for Reader {
  fn clone(&self) -> Self {
    let data = self.0.borrow_data().clone();
    Self(ReaderDetail::new(data, |data| {
      self
        .0
        .with_cache(|cache| cache.clone().transmute_lifetime(&data))
    }))
  }
}


#[test]
fn assert_send_sync() {
  fn assert_send<T: Send>() {}
  fn assert_sync<T: Sync>() {}

  assert_send::<Writer>();
  assert_sync::<Writer>();

  assert_send::<Reader>();
}
