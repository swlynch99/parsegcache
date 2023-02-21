use std::time::SystemTime;

use crate::segment::{DataRef, SegmentHeader};
use crate::{CacheConfig, Mmap};

#[derive(Clone)]
pub(crate) struct Shared<'seg> {
  pub config: CacheConfig,
  pub data: &'seg Mmap,
}

impl<'seg> Shared<'seg> {
  pub fn data_segment(&self, data: DataRef<'seg>) -> &SegmentHeader {
    let offset: usize = unsafe { data.as_ptr().offset_from(self.data.as_ptr()) }
      .try_into()
      .expect("Data item was not within the data array");

    let segment_idx = offset / self.config.segment_len;
    let segment_offset = segment_idx * self.config.segment_len;

    assert!(segment_offset < self.data.len());
    let segment = unsafe { self.data.as_ptr().add(segment_offset) };

    unsafe { &*(segment as *const SegmentHeader) }
  }

  pub fn expiry_for(&self, data: DataRef<'seg>) -> SystemTime {
    self.data_segment(data).expiry()
  }
}
