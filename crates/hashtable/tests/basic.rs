#![cfg(not(feature = "loom"))]

use std::collections::hash_map::RandomState;

use assert_matches::assert_matches;
use parsegcache_hashtable::Writer;

static A: (&str, &str) = ("test", "a");
static B: (&str, &str) = ("test", "b");

#[test]
fn insert() {
  let mut writer = Writer::with_capacity_and_hasher(1024, RandomState::new());

  writer.insert(&A).unwrap();

  assert_matches!(writer.get(&"test"), Some(&v) if v == A);
}

#[test]
fn overwrite() {
  let mut writer = Writer::with_capacity_and_hasher(1024, RandomState::new());

  writer.insert(&A).unwrap();
  writer.insert(&B).unwrap();

  assert_matches!(writer.get(&"test"), Some(&v) if v == B);
}

#[test]
fn erase_empty() {
  let mut writer: Writer<&(&str, &str), _> =
    Writer::with_capacity_and_hasher(1024, RandomState::new());

  assert_matches!(writer.erase(&"test"), None);
}

#[test]
fn erase_present() {
  let mut writer = Writer::with_capacity_and_hasher(1024, RandomState::new());

  writer.insert(&A).unwrap();

  assert_matches!(writer.erase(&"test"), Some(&v) if v == A);
}
// #[test]
// fn
