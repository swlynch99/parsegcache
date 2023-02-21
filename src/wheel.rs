//! This module contains a timer wheel implementation.

use std::time::{Duration, SystemTime};

pub(crate) trait WheelElement {
  fn empty() -> Self;

  fn is_empty(&self) -> bool;
}

pub(crate) struct TimerWheel<T> {
  buckets: Vec<T>,
  head: usize,

  width: Duration,
  start: SystemTime,
  margin: Duration,
}

impl<T> TimerWheel<T>
where
  T: Default,
{
  /// Create a new timer wheel.
  ///
  /// # Parameters
  /// - `span` - the total time span that the timer wheel will cover.
  /// - `bucket` - the time width covered by the bucket.
  /// - `margin` - how early a bucket will be removed from the timer wheel.
  pub(crate) fn new(span: Duration, bucket: Duration, margin: Duration) -> Self {
    let now = SystemTime::now();
    let nbuckets = (span.as_nanos() / bucket.as_nanos()) as usize;
    let mut buckets = Vec::with_capacity(nbuckets);
    buckets.resize_with(nbuckets, T::default);

    Self {
      buckets,
      head: 0,
      width: bucket,
      start: now,
      margin,
    }
  }

  /// Reclaim an existing segment that is about to expire.
  ///
  /// Returns the index to the segment contained in the bucket, if there is one.
  pub(crate) fn reclaim(&mut self) -> Option<T> {
    let now = SystemTime::now();

    if now + self.margin <= self.start {
      return None;
    }

    let current = std::mem::replace(&mut self.buckets[self.head], T::default());
    self.head = (self.head + 1) % self.buckets.len();
    self.start += self.width;

    Some(current)
  }

  pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
    let (tail, head) = self.buckets.split_at(self.head);
    head.iter().chain(tail.iter())
  }

  pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
    let (tail, head) = self.buckets.split_at_mut(self.head);
    head.iter_mut().chain(tail.iter_mut())
  }

  pub(crate) fn access(&mut self, expiry: SystemTime) -> Option<&mut T> {
    let span = expiry.duration_since(self.start).ok()?;
    if span < self.margin {
      return None;
    }

    let bucket_idx = ((span.as_nanos() / self.width.as_nanos()) as usize).min(self.buckets.len());
    let bucket_idx = (bucket_idx + self.head) % self.buckets.len();
    let bucket = &mut self.buckets[bucket_idx];

    Some(bucket)
  }
}
