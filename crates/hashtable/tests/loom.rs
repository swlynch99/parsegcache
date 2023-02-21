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


