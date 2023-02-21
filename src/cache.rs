use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use crossbeam::channel::{Receiver, RecvTimeoutError};
use memmap2::MmapRaw;

use crate::epoch::Collector;
use crate::hashtable::{CommandError, HashTable};
use crate::segment::{DataRef, Segment, SegmentHeader};
use crate::wheel::{TimerWheel, WheelElement};
use crate::{CacheCommand, CacheConfig, UpdateResponse};

struct CacheWriter<'seg> {
  timer: TimerWheel<TimeBucket<'seg>>,
  freelist: CacheFreelist<'seg>,
  hashtable: &'seg HashTable<'seg>,
  channel: Receiver<CacheCommand>,
  config: CacheConfig,
  shared: &'seg Shared,

  /// Stores [`Segment`]s that are in the process of being collected.
  collectq: VecDeque<Segment<'seg>>,
}

impl<'seg> CacheWriter<'seg> {
  pub(crate) fn new(
    shared: &'seg Shared,
    collector: Collector<Segment<'seg>>,
    channel: Receiver<CacheCommand>,
    config: CacheConfig,
  ) -> Self {
    let data = &shared.data;

    assert!(data.len() % config.segment_len == 0);
    assert!(data.as_ptr().align_offset(64) == 0);

    let mut segments = Vec::with_capacity(data.len() / config.segment_len);
    for offset in (0..data.len()).step_by(config.segment_len) {
      let segment = Segment::new(unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr().add(offset), config.segment_len)
      });
      segments.push(segment);
    }

    Self {
      freelist: CacheFreelist::new(segments, collector),
      timer: TimerWheel::new(config.timer_span, config.timer_bucket, config.timer_margin),
      hashtable: shared.hashtable(),
      collectq: VecDeque::new(),
      channel,
      config,
      shared,
    }
  }

  pub(crate) fn run(&mut self) {
    let handle = self.freelist.collector.register();
    let mut duration = Duration::default();

    'outer: loop {
      loop {
        let deadline = Instant::now() + duration;
        let message = match self.channel.recv_deadline(deadline) {
          Ok(message) => message,
          Err(RecvTimeoutError::Disconnected) => break 'outer,
          Err(RecvTimeoutError::Timeout) => break,
        };

        let _guard = handle.pin();

        match message {
          CacheCommand::Set {
            mut key,
            mut value,
            expiry,
            chan,
          } => {
            let result = self.set(&key, &value, expiry);
            key.clear();
            value.clear();

            let err = match result {
              Ok(data) => {
                key.extend_from_slice(data.key());
                value.extend_from_slice(data.value());

                None
              }
              Err(e) => Some(e),
            };

            let _ = chan.send(UpdateResponse {
              key,
              val: Some(value),
              err,
            });
          }
          CacheCommand::Delete { key, chan } => {
            let value = self.delete(&key).map(|data| data.value().to_vec());
            let _ = chan.send(UpdateResponse {
              key,
              val: value,
              err: None,
            });
          }
        }

        self.expire();

        if self.freelist.collect() || !self.collectq.is_empty() {
          duration = Duration::default();
        } else {
          duration = self.config.timer_bucket / 2;
        }

        self.freelist.collector.advance();
      }
    }
  }
}

impl<'seg> CacheWriter<'seg> {
  fn data_segment(&self, data: DataRef<'seg>) -> &SegmentHeader {
    let offset: usize = unsafe { data.as_ptr().offset_from(self.shared.data.as_ptr()) }
      .try_into()
      .expect("Data item was not within the data array");

    let segment_idx = offset / self.config.segment_len;
    let segment_offset = segment_idx * self.config.segment_len;

    assert!(segment_offset < self.shared.data.len());
    let segment = unsafe { self.shared.data.as_ptr().add(segment_offset) };

    unsafe { &*(segment as *const SegmentHeader) }
  }

  fn write(&mut self, key: &[u8], value: &[u8], expiry: SystemTime) -> Option<DataRef<'seg>> {
    let bucket = self.timer.access(expiry)?;
    let segment = match &mut bucket.segment {
      Some(segment) => segment,
      segment => {
        *segment = Some(self.freelist.next_free(expiry)?);
        segment.as_mut().unwrap()
      }
    };

    loop {
      if let Some(data) = segment.insert(key, value) {
        bucket.allocated += data.step_len();
        bucket.stored += data.step_len();

        break Some(data);
      }

      // Prepend the segment to the freelist to avoid needing to do an unbounded
      // traversal.
      let next = self.freelist.next_free(expiry)?;
      let prev = std::mem::replace(segment, next);
      segment.extend(prev);
    }
  }

  pub(self) fn set(
    &mut self,
    key: &[u8],
    value: &[u8],
    expiry: SystemTime,
  ) -> Result<DataRef<'seg>, CommandError> {
    let data = self
      .write(key, value, expiry)
      .ok_or(CommandError::NoCapacity)?;
    if let Some(prev) = unsafe { self.hashtable.insert(data)? } {
      let header = self.data_segment(prev);
      if let Some(bucket) = self.timer.access(header.expiry()) {
        bucket.stored -= prev.step_len();
      }
    }

    Ok(data)
  }

  pub(self) fn delete(&mut self, key: &[u8]) -> Option<DataRef<'seg>> {
    let data = unsafe { self.hashtable.erase(key)? };
    let header = self.data_segment(data);
    if let Some(bucket) = self.timer.access(header.expiry()) {
      bucket.stored -= data.header().step_len();
    }

    Some(data)
  }

  pub(self) fn expire(&mut self) {
    if let Some(bucket) = self.timer.reclaim() {
      if let Some(segment) = bucket.segment {
        self.collectq.push_back(segment);
      }
    }

    let mut next = self.collectq.pop_front();

    for _ in 0..self.config.expire_per_iter {
      let mut segment = match next.take() {
        Some(segment) => segment,
        None => break,
      };

      next = segment.take_next();

      for entry in segment.iter() {
        unsafe { self.hashtable.erase_by_entry(entry) };
      }

      self.freelist.collector.defer(segment);
    }
  }
}

pub(crate) struct Shared {
  hashtable: HashTable<'static>,

  /// Here for lifetime-management purposes only, should not be touched except
  /// by the writer thread.
  ///
  /// Needs to be last to ensure it outlives any data references
  data: MmapRaw,
}

impl Shared {
  pub fn new(data: MmapRaw, config: &CacheConfig) -> Self {
    Self {
      hashtable: HashTable::with_capacity(config.hashtable_capacity),
      data,
    }
  }

  pub(crate) fn hashtable<'a>(&'a self) -> &HashTable<'a> {
    &self.hashtable
  }
}

impl<'seg> WheelElement for Option<Segment<'seg>> {
  fn is_empty(&self) -> bool {
    self.is_none()
  }

  fn empty() -> Self {
    None
  }
}

struct CacheFreelist<'seg> {
  freelist: Vec<Segment<'seg>>,
  collector: Collector<Segment<'seg>>,
}

impl<'seg> CacheFreelist<'seg> {
  pub(self) fn new(segments: Vec<Segment<'seg>>, collector: Collector<Segment<'seg>>) -> Self {
    Self {
      freelist: segments,
      collector,
    }
  }

  pub(self) fn next_free(&mut self, expiry: SystemTime) -> Option<Segment<'seg>> {
    if let Some(segment) = self.collector.recycle() {
      self.freelist.push(segment);
    }

    let mut segment = self.freelist.pop()?;
    if let Some(next) = segment.reset(expiry) {
      self.freelist.push(next);
    }

    Some(segment)
  }

  pub(self) fn collect(&mut self) -> bool {
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
