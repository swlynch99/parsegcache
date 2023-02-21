use std::collections::VecDeque;
use std::time::SystemTime;

use parsegcache_hashtable::{CapacityError, Writer};
use parsegcache_wheel::TimerWheel;

use crate::epoch::Collector;
use crate::reader::CacheReader;
use crate::segment::{DataRef, Segment, SegmentHeader};
use crate::shared::Shared;
use crate::{CacheConfig, Entry};

const CACHE_LINE: usize = 64;

pub(crate) enum WriteError {
  ExpiryInPast,
  CapacityError,
}

pub(crate) struct CacheWriter<'seg> {
  wheel: TimerWheel<TimeBucket<'seg>>,
  freelist: CacheFreelist<'seg>,
  hashtable: Writer<DataRef<'seg>>,
  collectq: VecDeque<Segment<'seg>>,

  shared: Shared<'seg>,
}

impl<'seg> CacheWriter<'seg> {
  pub(crate) fn new(shared: Shared<'seg>) -> Self {
    let data = shared.data;
    let config = &shared.config;

    assert_eq!(data.len() % config.segment_len, 0);
    assert_eq!(config.segment_len % CACHE_LINE, 0);
    assert_eq!(data.as_ptr().align_offset(64), 0);

    let mut segments = Vec::with_capacity(data.len() / config.segment_len);
    for offset in (0..data.len()).step_by(config.segment_len) {
      let slice = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr().add(offset), config.segment_len)
      };

      segments.push(Segment::new(slice));
    }

    Self {
      wheel: TimerWheel::new(config.timer_span, config.timer_bucket, config.timer_margin),
      freelist: CacheFreelist::new(segments, Collector::new()),
      collectq: VecDeque::new(),
      hashtable: Writer::with_capacity(config.hashtable_capacity),
      shared,
    }
  }

  pub fn reader(&mut self) -> CacheReader<'seg> {
    CacheReader::new(
      self.freelist.collector.register(),
      self.hashtable.reader(),
      self.shared.clone(),
    )
  }

  pub fn set(
    &mut self,
    key: &[u8],
    value: &[u8],
    expiry: SystemTime,
  ) -> Result<Option<Entry<'seg>>, CapacityError> {
    let entry = match self.write(key, value, expiry) {
      Ok(entry) => entry,
      Err(WriteError::ExpiryInPast) => {
        self.delete(key);
        return Ok(None);
      }
      Err(WriteError::CapacityError) => return Err(CapacityError),
    };

    match self.hashtable.insert(entry.data) {
      Ok(Some(prev)) => {
        let expiry = self.shared.expiry_for(prev);
        self.on_erase(prev);

        if expiry < SystemTime::now() {
          return Ok(None);
        }

        Ok(Some(Entry::new(prev, expiry)))
      },
      Ok(None) => Ok(None),
      Err(e) => {
        self.on_erase(entry.data);
        return Err(e);
      }
    }
  }

  pub fn delete(&mut self, key: &[u8]) -> Option<Entry<'seg>> {
    let data = self.hashtable.erase(key)?;
    let expiry = self.shared.expiry_for(data);
    self.on_erase(data);

    if expiry < SystemTime::now() {
      return None;
    }

    Some(Entry::new(data, expiry))
  }

  pub fn expire(&mut self) {
    if let Some(bucket) = self.wheel.reclaim() {
      if let Some(segment) = bucket.segment {
        self.collectq.push_back(segment);
      }
    }

    let mut next = self.collectq.pop_front();
    for _ in 0..self.shared.config.expire_per_iter {
      let mut segment = match next.take() {
        Some(segment) => segment,
        None => break,
      };

      next = segment.take_next();

      for entry in segment.iter() {
        self.hashtable.erase_by_entry(entry);
      }

      self.freelist.defer(segment);
    }
  }

  pub fn get(&self, key: &[u8]) -> Option<Entry<'seg>> {
    let data = self.hashtable.get(key)?;
    let expiry = self.shared.expiry_for(data);

    if expiry < SystemTime::now() {
      return None;
    }

    Some(Entry::new(data, expiry))
  }

  pub fn config(&self) -> &CacheConfig {
    &self.shared.config
  }

  fn on_erase(&mut self, data: DataRef<'seg>) {
    let header = self.data_segment(data);
    if let Some(bucket) = self.wheel.get_mut(header.expiry()) {
      bucket.stored -= data.step_len();
    }
  }

  fn write(
    &mut self,
    key: &[u8],
    value: &[u8],
    expiry: SystemTime,
  ) -> Result<Entry<'seg>, WriteError> {
    let bucket = self.wheel.get_mut(expiry).ok_or(WriteError::ExpiryInPast)?;
    let segment = match &mut bucket.segment {
      Some(segment) => segment,
      segment => segment.get_or_insert(
        self
          .freelist
          .next_free(expiry)
          .ok_or(WriteError::CapacityError)?,
      ),
    };

    loop {
      if let Some(data) = segment.insert(key, value) {
        bucket.allocated += data.step_len();
        bucket.stored += data.step_len();

        return Ok(Entry::new(data, self.shared.expiry_for(data)));
      }

      // Prepend the segment to the freelist so we don't do an unbounded
      // traversal.
      let next = self
        .freelist
        .next_free(expiry)
        .ok_or(WriteError::CapacityError)?;
      let prev = std::mem::replace(segment, next);
      segment.extend(prev);
    }
  }

  fn data_segment(&self, data: DataRef<'seg>) -> &SegmentHeader {
    self.shared.data_segment(data)
  }
}

struct CacheFreelist<'seg> {
  freelist: Vec<Segment<'seg>>,
  collector: Collector<Segment<'seg>>,
}

impl<'seg> CacheFreelist<'seg> {
  pub fn new(segments: Vec<Segment<'seg>>, collector: Collector<Segment<'seg>>) -> Self {
    Self {
      freelist: segments,
      collector,
    }
  }

  pub fn defer(&mut self, segment: Segment<'seg>) {
    self.collector.defer(segment)
  }

  pub fn next_free(&mut self, expiry: SystemTime) -> Option<Segment<'seg>> {
    if let Some(segment) = self.collector.recycle() {
      self.freelist.push(segment);
    }

    let mut segment = self.freelist.pop()?;
    if let Some(next) = segment.reset(expiry) {
      self.freelist.push(next);
    }

    Some(segment)
  }

  pub fn collect(&mut self) -> bool {
    self
      .collector
      .recycle()
      .map(|segment| self.freelist.push(segment))
      .is_some()
  }
}

#[derive(Default)]
struct TimeBucket<'seg> {
  segment: Option<Segment<'seg>>,

  /// The number of bytes that have been allocated in this bucket
  allocated: usize,

  /// The number of bytes that are still actively being used in this bucket.
  stored: usize,
}
