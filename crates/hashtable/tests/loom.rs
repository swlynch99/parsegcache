#![cfg(feature = "loom")]

use std::collections::hash_map::RandomState;

use loom::thread;
use parsegcache_hashtable::Writer;

static A: (&str, &str) = ("test", "a");
static B: (&str, &str) = ("test", "b");

#[test]
fn concurrent_read_insert() {
  loom::model(|| {
    let mut writer: Writer<_, RandomState> = Writer::with_capacity(1024);
    let reader = writer.reader();

    writer.insert(&A).unwrap();

    thread::spawn(move || {
      writer.insert(&B).unwrap();
    });

    match reader.get(&"test") {
      Some(&(key, value)) => {
        assert_eq!(key, "test");
        assert!(value == "a" || value == "b", "value == {:?}", value);
      }
      None => panic!(),
    }
  })
}

static ENTRIES: &[&str] = &[
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
];

#[test]
fn concurrent_bucket_expand() {
  loom::model(|| {
    let mut writer: Writer<&&str, RandomState> = Writer::with_capacity(1);
    let reader = writer.reader();

    for entry in ENTRIES.iter().skip(1) {
      writer.insert(entry).unwrap();
    }

    thread::spawn(move || {
      writer.insert(&ENTRIES[0]).unwrap();
    });

    match reader.get(&ENTRIES[0]) {
      Some(&val) => assert_eq!(val, "0"),
      None => (),
    }
  })
}

#[test]
fn concurrent_update() {
  loom::model(|| {
    let mut writer: Writer<_, RandomState> = Writer::with_capacity(32);
    let reader = writer.reader();

    writer.insert(&A).unwrap();

    thread::spawn(move || {
      assert!(writer.update(&A, &B));
    });

    match reader.get(&"test") {
      Some(&(_, val)) => assert!(val == "a" || val == "b", "val == {val}"),
      None => panic!("value not present in table"),
    }
  })
}
