#![allow(dead_code)]

pub mod hashtable;
pub mod segment;
mod util;
pub mod wheel;
pub mod epoch;

use std::sync::Arc;

use crate::hashtable::HashTable;
use crate::segment::{DataRef, Segment};

pub struct CacheConfig {
  pub segment_len: usize,
}

struct Shared<'seg> {
  hashtable: HashTable<'seg>,
}

pub struct CacheWriter<'seg> {
  shared: Arc<Shared<'seg>>,
}

pub struct CacheReader<'seg> {
  shared: Arc<Shared<'seg>>,
}

struct CacheData<'seg> {
  data: *mut u8,
  data_len: usize,
  segment_len: usize,
  segments: Vec<Segment<'seg>>,
}

impl<'seg> CacheData<'seg> {
  pub(crate) fn owning_segment_id(&self, ptr: *const u8) -> Option<usize> {
    let data = self.data as *const u8;
    let offset = unsafe { ptr.offset_from(data) as usize };

    if offset >= self.data_len {
      return None;
    }

    Some(offset / self.segment_len)
  }
}
