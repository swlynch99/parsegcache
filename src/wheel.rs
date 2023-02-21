//! This module contains a timer wheel implementation.

use std::time::{Duration, SystemTime};

pub(crate) struct TimerWheel {
  buckets: Vec<usize>,
  head: usize,

  width: Duration,
  start: SystemTime,
  margin: Duration,
}

impl TimerWheel {
  /// Create a new timer wheel.
  /// 
  /// # Parameters
  /// - `span` - the total time span that the timer wheel will cover.
  /// - `bucket` - the time width covered by the bucket.
  /// - `margin` - how early a bucket will be removed from the timer wheel.
  pub(crate) fn new(span: Duration, bucket: Duration, margin: Duration) -> Self {
    let now = SystemTime::now();
    let nbuckets = (span.as_nanos() / bucket.as_nanos()) as usize;
    let buckets = vec![usize::MAX; nbuckets];

    Self {
      buckets,
      head: 0,
      width: bucket,
      start: now,
      margin,
    }
  }

  /// Reclaim existing segments that are about to expire.
  ///
  /// Returns the index to the segment contained in the bucket, if there is one.
  pub(crate) fn reclaim(&mut self) -> Option<usize> {
    let now = SystemTime::now();

    if now + self.margin <= self.start {
      return None;
    }

    let current = std::mem::replace(&mut self.buckets[self.head], usize::MAX);
    self.head = (self.head + 1) % self.buckets.len();
    self.start += self.width;

    match current {
      usize::MAX => None,
      _ => Some(current),
    }
  }

  /// Erase the first hash bucket that contains a segment reference.
  pub(crate) fn erase(&mut self) -> Option<usize> {
    let (tail, head) = self.buckets.split_at_mut(self.head);

    head
      .iter_mut()
      .chain(tail.iter_mut())
      .filter(|bucket| **bucket != usize::MAX)
      .map(|bucket| std::mem::replace(bucket, usize::MAX))
      .next()
  }

  pub(crate) fn access<F>(&mut self, expiry: SystemTime, or_create: F) -> Option<usize>
  where
    F: FnOnce() -> Option<usize>,
  {
    let span = expiry.duration_since(self.start).ok()?;
    if span < self.margin {
      return None;
    }

    let bucket_idx = ((span.as_nanos() / self.width.as_nanos()) as usize)
      .min(self.buckets.len());
    let bucket_idx = (bucket_idx + self.head) % self.buckets.len();
    let bucket = &mut self.buckets[bucket_idx];

    if *bucket == usize::MAX {
      *bucket = or_create()?;
    }

    Some(*bucket)
  }
}
