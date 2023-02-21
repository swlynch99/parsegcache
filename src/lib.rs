#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, SystemTime};

use crossbeam::channel::Sender;
use hashtable::CommandError;

use crate::epoch::EpochRef;
use crate::segment::DataRef;

pub mod cache;
pub mod epoch;
pub mod hashtable;
pub mod segment;
// pub mod threads;
mod util;
pub mod wheel;

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

#[derive(Clone)]
pub struct CacheHandle {
  shared: Arc<cache::Shared>,
  channel: Sender<CacheCommand>,
  handle: epoch::Handle,
}

impl CacheHandle {
  /// Read a value from the cache
  pub fn get(&self, key: &[u8]) -> Option<EpochRef<DataRef>> {
    let guard = self.handle.pin();
    let data = self.shared.hashtable().get(key)?;

    // TODO: Check expiry time?

    Some(EpochRef::new(guard, data))
  }

  pub fn set(&self, key: &[u8], value: &[u8], expiry: SystemTime) -> UpdateFuture {
    let (tx, rx) = oneshot::channel();
    if let Err(e) = self.channel.send(CacheCommand::Set {
      key: key.to_vec(),
      value: value.to_vec(),
      expiry,
      chan: tx,
    }) {
      match e.into_inner() {
        CacheCommand::Set {
          key, value, chan, ..
        } => chan
          .send(UpdateResponse {
            key,
            val: Some(value),
            err: Some(CommandError::NoWriter),
          })
          .unwrap(),
        _ => unreachable!(),
      }
    }

    UpdateFuture { recv: rx }
  }

  pub fn delete(&self, key: &[u8]) -> UpdateFuture {
    let (tx, rx) = oneshot::channel();

    let cmd = CacheCommand::Delete {
      key: key.to_vec(),
      chan: tx,
    };
    if let Err(e) = self.channel.send(cmd) {
      match e.into_inner() {
        CacheCommand::Delete { key, chan } => chan
          .send(UpdateResponse {
            key,
            val: None,
            err: Some(CommandError::NoWriter),
          })
          .unwrap(),
        _ => unreachable!(),
      }
    }

    UpdateFuture { recv: rx }
  }
}

#[must_use]
pub struct UpdateFuture {
  recv: oneshot::Receiver<UpdateResponse>,
}

impl UpdateFuture {
  pub fn get(self) -> Result<(), CommandError> {
    match self.recv.recv() {
      Ok(UpdateResponse { err: Some(e), .. }) => Err(e),
      Ok(UpdateResponse { .. }) => Ok(()),
      Err(_) => Err(CommandError::NoWriter),
    }
  }
}

impl Future for UpdateFuture {
  type Output = Result<(), CommandError>;

  fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
    Poll::Ready(match Pin::new(&mut self.recv).poll(cx) {
      Poll::Pending => return Poll::Pending,
      Poll::Ready(Ok(resp)) => match resp.err {
        Some(err) => Err(err),
        None => Ok(()),
      },
      Poll::Ready(Err(_)) => Err(CommandError::NoWriter),
    })
  }
}

struct UpdateResponse {
  key: Vec<u8>,
  val: Option<Vec<u8>>,
  err: Option<CommandError>,
}

enum CacheCommand {
  Set {
    key: Vec<u8>,
    value: Vec<u8>,
    expiry: SystemTime,
    chan: oneshot::Sender<UpdateResponse>,
  },
  Delete {
    key: Vec<u8>,
    chan: oneshot::Sender<UpdateResponse>,
  },
}
