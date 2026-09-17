//! The prefix index through its public API: a lookup finds the longest
//! usable prefix, entries hold their checkpoints, and making room parks
//! an entry to the host without losing it.

mod common;

use std::sync::Arc;

use kern_pool::{Checkpoint, Denied, Evicted, Found, Hit, Host, Kept, Parked, Prefix, Tier};

use common::{hybrid_pool4, pool4, Rand};

fn table() -> Prefix<Checkpoint, Parked> {
    Prefix::new(4)
}

fn toks(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| i * 7 + 3).collect()
}

fn host() -> Arc<Host> {
    Arc::new(Host::new(64, 4, 4, 8))
}

fn tier<R, P>(h: &Hit<R, P>) -> Tier {
    match h.found {
        Found::Resident(_) => Tier::Resident,
        Found::Parked(_) => Tier::Parked,
    }
}

fn resident<R, P>(h: &Hit<R, P>) -> Option<&R> {
    match &h.found {
        Found::Resident(r) => Some(r),
        Found::Parked(_) => None,
    }
}

fn parked<R, P>(h: &Hit<R, P>) -> Option<&P> {
    match &h.found {
        Found::Parked(p) => Some(p),
        Found::Resident(_) => None,
    }
}

/// (length, tier) of what `t` finds for `tokens`.
fn find(t: &mut Prefix, tokens: &[i64]) -> Option<(usize, Tier)> {
    t.lookup(tokens).map(|h| (h.len, tier(&h)))
}

/// Park the coldest resident entry into `h`, no copies run; a full host
/// hands the checkpoint back, as the runtime's `room` does.
fn park(t: &mut Prefix, h: &Arc<Host>) -> Option<Evicted> {
    t.evict(Some(|cp: Checkpoint| Ok::<_, Denied>(h.park(&cp).map(|(p, _)| p).map_err(|_| cp)))).unwrap()
}

/// Drop the coldest resident entry.
fn drop_coldest(t: &mut Prefix) -> Option<Evicted> {
    t.evict(None::<fn(Checkpoint) -> Result<Result<Parked, Checkpoint>, Denied>>).unwrap()
}

#[test]
fn one_entry_grows_a_page_at_a_time() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    t.insert(&toks(4), p.checkpoint(&mut l, 4).unwrap().0);
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!((t.entries(), find(&mut t, &toks(9))), (1, Some((8, Tier::Resident))));
    // Ten tokens end inside the third page: a second entry with a tail.
    t.insert(&toks(10), p.checkpoint(&mut l, 10).unwrap().0);
    assert_eq!(t.entries(), 2);
    // A 12-token prompt of the same tokens may use 11: the 10-token entry.
    let h = t.lookup(&toks(12)).unwrap();
    assert_eq!((h.len, tier(&h), resident(&h).map(|c| c.tokens())), (10, Tier::Resident, Some(10)));
    // A 10-token prompt may use 9: 8 whole pages, the 8-token entry itself.
    let h = t.lookup(&toks(10)).unwrap();
    assert_eq!((h.len, resident(&h).map(|c| c.tokens())), (8, Some(8)));
    // Diverging inside the third page: still 8.
    let mut d = toks(12);
    d[9] = -1;
    assert_eq!(t.lookup(&d).map(|h| h.len), Some(8));
    // Diverging inside the second page: the entry is usable at its first page.
    d[5] = -1;
    let h = t.lookup(&d).unwrap();
    assert_eq!((h.len, resident(&h).map(|c| c.tokens())), (4, Some(8)));
    d[0] = -1;
    assert_eq!(t.lookup(&d).map(|h| h.len), None);
    assert!(t.lookup(&toks(1)).is_none());
    assert!(t.lookup(&[]).is_none());
    // Growing past the tailed entry: page 3 replaces the 8-token entry, not the 10.
    t.insert(&toks(12), p.checkpoint(&mut l, 12).unwrap().0);
    assert_eq!(t.entries(), 2);
    assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
    assert_eq!(t.lookup(&toks(11)).map(|h| h.len), Some(10));
}

#[test]
fn same_tokens_share_one_entry() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(p.used(), 2);
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!((t.entries(), p.used()), (1, 2));
    // Same pages but different tokens is a different entry.
    let mut other = toks(8);
    other[7] = -1;
    t.insert(&other, p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(t.entries(), 2);
}

#[test]
fn the_coldest_goes_first_and_a_hit_touches_its_path() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    let a = toks(8);
    t.insert(&a, p.checkpoint(&mut l, 8).unwrap().0);
    let mut c = toks(8);
    c[6] = -1;
    t.insert(&c, p.checkpoint(&mut l, 8).unwrap().0);
    // A hit at 9 tokens of `a` touches a alone: c is the colder.
    assert_eq!(t.lookup(&toks(9)).map(|h| h.len), Some(8));
    // Six tokens of c are usable at page 1 through either entry; the
    // one found is touched, so a is the colder again.
    assert_eq!(t.lookup(&c[..7]).map(|h| h.len), Some(4));
    assert_eq!(t.lookup(&[&c[..], &[0]].concat()).map(|h| h.len), Some(8));
    // Dropping the coldest (a) frees only what no lease still holds.
    assert_eq!(drop_coldest(&mut t), Some(Evicted::Dropped { key: Arc::from(&a[..]), dropped: 0 }));
    assert_eq!((t.entries(), p.used(), find(&mut t, &toks(9))), (1, 3, Some((4, Tier::Resident))));
    assert_eq!(find(&mut t, &[&c[..], &[0]].concat()), Some((8, Tier::Resident)));
    assert_eq!(drop_coldest(&mut t), Some(Evicted::Dropped { key: Arc::from(&c[..]), dropped: 0 }));
    assert_eq!((t.entries(), p.used(), drop_coldest(&mut t)), (0, 3, None));
    drop(l);
    assert_eq!(p.used(), 0);
}

#[test]
fn restore_then_checkpoint_deeper() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    drop(l);
    let hit = t.lookup(&toks(16)).unwrap();
    let (mut l2, _) = p.restore(resident(&hit).unwrap(), hit.len, 16).unwrap();
    assert_eq!(l2.prefix(), 8);
    // The sequence's checkpoints grow the entry a page at a time.
    t.insert(&toks(12), p.checkpoint(&mut l2, 12).unwrap().0);
    t.insert(&toks(16), p.checkpoint(&mut l2, 16).unwrap().0);
    drop(l2);
    assert_eq!(t.lookup(&toks(17)).map(|h| h.len), Some(16));
    assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
    assert_eq!((t.entries(), p.used()), (1, 4));
}

#[test]
fn a_branch_is_its_own_entry() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    drop(l);
    // Two sequences continue from the entry; the first extends it, the second branches.
    let hit = t.lookup(&toks(13)).unwrap();
    let (mut x, _) = p.restore(resident(&hit).unwrap(), hit.len, 12).unwrap();
    let (mut y, _) = p.restore(resident(&hit).unwrap(), hit.len, 12).unwrap();
    let mut ty = toks(12);
    ty[9] = -1;
    t.insert(&toks(12), p.checkpoint(&mut x, 12).unwrap().0);
    t.insert(&ty, p.checkpoint(&mut y, 12).unwrap().0);
    assert_eq!(t.entries(), 2);
    assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
    assert_eq!(t.lookup(&[&ty[..], &[0]].concat()).map(|h| h.len), Some(12));
    // Both usable at 8 whole tokens; the tree's first branch wins so the choice is stable.
    let h = t.lookup(&toks(9)).unwrap();
    assert_eq!((h.len, resident(&h).map(|c| c.tokens())), (8, Some(12)));
}

#[test]
fn a_stateful_entry_is_usable_at_its_length_only() {
    let p = hybrid_pool4();
    let mut t = table();
    let l = p.lease(12).unwrap();
    t.insert(&toks(10), p.retire(l, 10));
    assert_eq!(find(&mut t, &toks(12)), Some((10, Tier::Resident)));
    assert!(t.lookup(&toks(10)).is_none());
    let mut d = toks(12);
    d[9] = -1;
    assert!(t.lookup(&d).is_none());
    // A longer retirement on the same path is a second entry, not a growth.
    let l = p.lease(16).unwrap();
    t.insert(&toks(12), p.retire(l, 12));
    assert_eq!(t.entries(), 2);
    assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
}

#[test]
fn parked_entries_are_found_after_resident_ones() {
    let p = pool4();
    let h = host();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    let key = toks(12);
    t.insert(&key, p.checkpoint(&mut l, 12).unwrap().0);
    drop(l);
    assert_eq!(p.used(), 3);
    assert_eq!(park(&mut t, &h), Some(Evicted::Parked { key: Arc::from(&key[..]), dropped: 0 }));
    assert_eq!((p.used(), t.count(Tier::Parked), t.count(Tier::Resident)), (0, 1, 0));
    let hit = t.lookup(&toks(13)).unwrap();
    assert_eq!((hit.len, tier(&hit), parked(&hit).map(|q| q.offsets())), (12, Tier::Parked, Some(vec![0, 4, 8])));
    assert_eq!(find(&mut t, &toks(6)), Some((4, Tier::Parked)));
    // Nothing resident to make room with; a hit outlives what the index does to the entry.
    assert_eq!(park(&mut t, &h), None);
    assert_eq!((t.entries(), h.used()), (1, 12));
    drop(hit);
    // Resident and parked: the resident one is found first.
    let mut l = p.lease(12).unwrap();
    t.insert(&key, p.checkpoint(&mut l, 12).unwrap().0);
    assert_eq!(park(&mut t, &h), Some(Evicted::Parked { key: Arc::from(&key[..]), dropped: 0 }));
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(find(&mut t, &toks(9)), Some((8, Tier::Resident)));
    assert_eq!(find(&mut t, &toks(13)), Some((12, Tier::Parked)));
    // The same tokens resident again: one entry in both tiers, found
    // resident, and one page past the 8-token entry, which it replaces.
    t.insert(&key, p.checkpoint(&mut l, 12).unwrap().0);
    assert_eq!((t.entries(), t.count(Tier::Resident), t.count(Tier::Parked)), (1, 1, 0));
    assert_eq!(find(&mut t, &toks(13)), Some((12, Tier::Resident)));
    assert_eq!(find(&mut t, &toks(9)), Some((8, Tier::Resident)));
    // Parking it again copies nothing and keeps the host copy.
    let mut copied = None;
    let evicted = t
        .evict(Some(|cp: Checkpoint| {
            copied = Some(cp.tokens());
            Ok::<_, ()>(Err(cp))
        }))
        .unwrap();
    assert_eq!(
        (evicted, copied, t.count(Tier::Parked), h.used()),
        (Some(Evicted::Parked { key: Arc::from(&key[..]), dropped: 0 }), None, 1, 12)
    );
    drop(l);
    assert_eq!(p.used(), 0);
}

#[test]
fn a_stateful_park_keeps_its_slot_and_its_length() {
    let p = hybrid_pool4();
    let h = host();
    let mut t = table();
    let l = p.lease(12).unwrap();
    t.insert(&toks(10), p.retire(l, 10));
    assert_eq!((p.used(), p.slots_used()), (3, 1));
    assert_eq!(park(&mut t, &h), Some(Evicted::Parked { key: Arc::from(&toks(10)[..]), dropped: 0 }));
    assert_eq!((p.used(), p.slots_used(), h.used()), (0, 0, 20));
    let hit = t.lookup(&toks(12)).unwrap();
    assert_eq!((hit.len, parked(&hit).map(|q| q.has_slot())), (10, Some(true)));
    assert!(t.lookup(&toks(9)).is_none());
}

#[test]
fn a_full_host_drops_its_coldest_and_a_failed_copy_drops_the_entry() {
    let p = pool4();
    let h = Arc::new(Host::new(4, 4, 4, 0));
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    let mut l2 = p.lease(4).unwrap();
    let (a, b) = (toks(4), toks(8));
    t.insert(&a, p.checkpoint(&mut l, 4).unwrap().0);
    let mut c = toks(4);
    c[0] = -1;
    t.insert(&c, p.checkpoint(&mut l2, 4).unwrap().0);
    // a (the coldest) fills the block.
    assert_eq!(park(&mut t, &h), Some(Evicted::Parked { key: Arc::from(&a[..]), dropped: 0 }));
    assert_eq!((h.used(), t.count(Tier::Parked)), (4, 1));
    // c fits once a is dropped from the host: one parked entry gone for it.
    assert_eq!(park(&mut t, &h), Some(Evicted::Parked { key: Arc::from(&c[..]), dropped: 1 }));
    assert_eq!((h.used(), t.count(Tier::Parked), t.entries()), (4, 1, 1));
    assert_eq!(find(&mut t, &[&c[..], &[0]].concat()), Some((4, Tier::Parked)));
    // b is two pages: the block never holds it, c is dropped for nothing, then b.
    t.insert(&b, p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(park(&mut t, &h), Some(Evicted::Dropped { key: Arc::from(&b[..]), dropped: 1 }));
    assert_eq!((h.used(), t.entries()), (0, 0));
    // The copy failed: the entry is gone and the error is the caller's.
    t.insert(&b, p.checkpoint(&mut l, 8).unwrap().0);
    let failed = t.evict(Some(|_: Checkpoint| Err::<Result<Parked, Checkpoint>, _>("copy failed")));
    assert_eq!((failed, t.entries()), (Err("copy failed"), 0));
}

/// What the brute-force model keeps of an entry.
#[derive(Clone)]
struct K(usize, bool);

impl Kept for K {
    fn tokens(&self) -> usize {
        self.0
    }

    fn has_slot(&self) -> bool {
        self.1
    }
}

/// An entry as the model sees it: its tokens, whether a state pins it,
/// and which tiers hold it.
struct Entry {
    tokens: Vec<i64>,
    slot: bool,
    resident: bool,
    host: bool,
}

impl Entry {
    fn tier(&self) -> Tier {
        if self.resident {
            Tier::Resident
        } else {
            Tier::Parked
        }
    }
}

/// Random inserts, lookups, parks and drops against a brute-force model
/// that keeps every entry's tokens: a lookup's length is the longest
/// usable prefix any entry holds, a resident entry before a parked one;
/// making room parks one resident entry or, when the host refuses,
/// drops every parked one and then it, and names the one it took; and
/// the counts agree. Which entry is the coldest is the index's business.
/// Run twice, the index makes the same decisions in the same order.
#[test]
fn lookup_matches_the_brute_force_model() {
    let trace = |seed: u64| {
        let mut t: Prefix<K, K> = Prefix::new(4);
        let mut model: Vec<Entry> = Vec::new();
        let mut rand = Rand::new(seed);
        let mut out: Vec<(usize, Tier)> = Vec::new();
        for _ in 0..3000 {
            match rand.next(6) {
                0 | 1 => {
                    // Up to 12 tokens out of a 2-symbol alphabet: prefixes collide often.
                    let n = 1 + rand.next(12);
                    let tokens: Vec<i64> = (0..n).map(|_| rand.next(2) as i64).collect();
                    // Whether a state is held is the sequence's, not the insert's.
                    let slot = tokens.iter().sum::<i64>() % 3 == 0;
                    t.insert(&tokens, K(n, slot));
                    match model.iter_mut().find(|e| e.tokens == tokens) {
                        Some(e) => e.resident = true,
                        None => {
                            model.push(Entry { tokens: tokens.clone(), slot, resident: true, host: false });
                            // One page past a stateless resident-only entry replaces it.
                            if !slot && n > 4 && n.is_multiple_of(4) {
                                model.retain(|e| !(e.tokens == tokens[..n - 4] && !e.slot && e.resident && !e.host));
                            }
                        }
                    }
                }
                2 | 3 => {
                    let n = 1 + rand.next(14);
                    let tokens: Vec<i64> = (0..n).map(|_| rand.next(2) as i64).collect();
                    let usable = &tokens[..n - 1];
                    let want = model
                        .iter()
                        .map(|e| {
                            // Usable: all of it when the prompt covers it, else its whole pages when stateless.
                            let common = e.tokens.iter().zip(usable).take_while(|(a, b)| a == b).count();
                            let len = if common == e.tokens.len() {
                                e.tokens.len()
                            } else if e.slot {
                                0
                            } else {
                                common / 4 * 4
                            };
                            (len, std::cmp::Reverse(e.tier()))
                        })
                        .filter(|(len, _)| *len > 0)
                        .max();
                    let got = t.lookup(&tokens).map(|h| (h.len, std::cmp::Reverse(tier(&h))));
                    assert_eq!(got, want, "prompt {tokens:?}");
                    if let Some((len, std::cmp::Reverse(tier))) = got {
                        out.push((len, tier));
                    }
                }
                4 | 5 => {
                    // A host that takes the park, one that refuses it, or no host at all.
                    let host = rand.next(3);
                    let got = match host {
                        2 => t.evict(None::<fn(K) -> Result<Result<K, K>, ()>>),
                        _ => t.evict(Some(|r: K| Ok::<_, ()>(if host == 1 { Err(r) } else { Ok(K(r.0, r.1)) }))),
                    }
                    .unwrap();
                    let parked_before = model.iter().filter(|e| !e.resident).count();
                    match &got {
                        None => assert!(model.iter().all(|e| !e.resident)),
                        Some(Evicted::Parked { key, dropped }) => {
                            let e = model.iter_mut().find(|e| e.tokens[..] == key[..]).expect("an entry");
                            assert!(e.resident && (host == 0 || e.host), "parked {key:?}");
                            assert_eq!(*dropped, 0);
                            e.resident = false;
                            e.host = true;
                        }
                        Some(Evicted::Dropped { key, dropped }) => {
                            let e = model.iter().find(|e| e.tokens[..] == key[..]).expect("an entry");
                            assert!(e.resident && !e.host || host == 2, "dropped {key:?}");
                            assert_eq!(*dropped, if host == 2 { 0 } else { parked_before });
                            model.retain(|e| e.tokens[..] != key[..] && (host == 2 || e.resident));
                        }
                    }
                    out.push((
                        got.map_or(0, |e| match e {
                            Evicted::Parked { key, .. } => key.len(),
                            Evicted::Dropped { key, .. } => key.len() + 100,
                        }),
                        Tier::Parked,
                    ));
                }
                _ => {}
            }
            let parked = model.iter().filter(|e| !e.resident).count();
            assert_eq!((t.entries(), t.count(Tier::Parked)), (model.len(), parked));
        }
        out
    };
    assert_eq!(trace(0x1234_5678_9ABC_DEF1), trace(0x1234_5678_9ABC_DEF1));
}

#[test]
fn a_lookup_that_uses_nothing_touches_nothing() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(4).unwrap();
    let mut l2 = p.lease(4).unwrap();
    let (a, mut b) = (toks(4), toks(4));
    b[0] = -1;
    t.insert(&b, p.checkpoint(&mut l2, 4).unwrap().0);
    t.insert(&a, p.checkpoint(&mut l, 4).unwrap().0);
    // Nothing shared, then one token of b: under a page, so no hit, and
    // an entry that served nothing is not made warm by it.
    assert_eq!(find(&mut t, &[999, 998]), None);
    assert_eq!(find(&mut t, &[b[0], 55, 56]), None);
    assert_eq!(drop_coldest(&mut t), Some(Evicted::Dropped { key: Arc::from(&b[..]), dropped: 0 }));
}
