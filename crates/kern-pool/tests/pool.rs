//! The pool through its public API: leases, checkpoints, restores, forks,
//! retirements and the remaps that move chunks between pages and slots.

mod common;

use std::sync::Arc;

use kern_manifest::types::{Dim, Manifest};
use std::collections::{BTreeMap, BTreeSet};

use kern_pool::{chunks_for, Checkpoint, Copies, Denied, Kind, Lease, Pool, Remap};

use common::{hybrid, hybrid4, hybrid_pool, land, paged4, pool, pool_of, two_paged, Rand};

#[test]
fn a_new_pool_maps_every_chunk_in_its_first_plan() {
    let p = pool();
    assert_eq!((p.total(), p.unit(), p.max_seq_tokens(), p.chunk()), (4, 16, 48, 8));
    assert_eq!(p.tables().collect::<Vec<_>>(), ["block_table", "draft_block_table"]);
    assert_eq!((p.has_slots(), p.slots()), (false, 0));
    assert_eq!(p.lease(1).unwrap().seq_slot(), None);
    // Two chunks over what four pages take are spare, not a torn page.
    let (p2, plan) = Pool::new(&two_paged(), 8, 18, 0, None).unwrap();
    assert_eq!((p2.total(), p2.pages_max(), plan.map.len(), plan.unmap.len()), (4, 4, 16, 0));
    // A capacity in tokens is a cap on the pages, whatever the chunks hold.
    let (p3, plan3) = Pool::new(&two_paged(), 8, 18, 0, Some(32)).unwrap();
    assert_eq!((p3.total(), p3.pages_max(), plan3.map.len()), (2, 2, 8));
    assert_eq!(plan.made, [(Kind::Page, 0), (Kind::Page, 1), (Kind::Page, 2), (Kind::Page, 3)]);
    let names: Vec<(&str, Kind, u64, usize)> =
        p2.pooled().iter().map(|a| (a.state.as_str(), a.kind, a.object, a.positions)).collect();
    assert_eq!(names, [("draft_kv", Kind::Page, 16, 8), ("kv", Kind::Page, 16, 8)]);
    // With slots: every chunk mapped, one access grant per object, nothing unmapped.
    let (p, plan) = Pool::new(&hybrid(), 8, 20, 4, None).unwrap();
    assert_eq!((plan.map.len(), plan.access.len(), plan.unmap.len(), plan.unmade.len()), (20, 8, 0, 0));
    assert_eq!(plan.made.len(), 8);
    assert_eq!((p.total(), p.slots(), p.pages_max(), p.slots_max()), (4, 4, 10, 6));
    let arenas: Vec<(&str, Kind, usize)> = p.pooled().iter().map(|a| (a.state.as_str(), a.kind, a.positions)).collect();
    assert_eq!(arenas, [("kv", Kind::Page, 20), ("gdn", Kind::Slot, 18)]);
}

#[test]
fn pool_rejects_a_manifest_it_cannot_lay_out() {
    let err = |m: &Manifest, chunks: u32, first_slots: usize| match Pool::new(m, 8, chunks, first_slots, None) {
        Ok(_) => panic!("laid out"),
        Err(e) => e.to_string(),
    };
    // A state is paged or per sequence, not both.
    let mut m = hybrid();
    m.states.get_mut("gdn").unwrap().bytes = 8;
    assert!(err(&m, 20, 4).contains("one layout"));
    // The line table's shape is [lines, seqs, w] with the state's line count.
    let mut m = hybrid();
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Const(4), Dim::Var("seqs".into())];
    assert!(err(&m, 20, 4).contains("state holds 3"));
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Var("seqs".into())];
    assert!(err(&m, 20, 4).contains("[lines, seqs, w]"));
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Const(3), Dim::Var("seqs".into()), Dim::Const(8)];
    let p = pool_of(&m, 8, 20, 4);
    let a = p.lease(16).unwrap();
    assert_eq!((a.seq_width("line_index").unwrap(), a.seq_lines("line_index").unwrap()), (8, 3));
    // The budget must hold the first slots.
    assert!(err(&hybrid(), 5, 4).contains("hold 1 sequence slots"));
}

#[test]
fn the_chunk_budget_rounds_every_state_on_its_own() {
    // kv: 32 tokens of 1 byte = 4 chunks; gdn: 4 slots of 24 bytes = 12 chunks.
    assert_eq!(chunks_for(&hybrid(), 32, 4, 8), 16);
    // Two per-sequence states of 40 bytes each hold one slot in 2 chunks of 32
    // apiece, not the 3 the summed 80 bytes suggest: each arena rounds alone.
    let mut m = hybrid();
    m.states.get_mut("gdn").unwrap().bytes_per_seq = 40;
    m.states.insert("gdn2".into(), m.states["gdn"].clone());
    assert_eq!(chunks_for(&m, 0, 1, 32), 4);
    assert!(Pool::new(&m, 32, 4, 1, None).is_ok());
    assert!(Pool::new(&m, 32, 3, 1, None).is_err());
}

#[test]
fn drop_returns_pages() {
    let p = pool();
    let a = p.lease(17).unwrap(); // 2 pages
    let b = p.lease(1).unwrap();
    assert_eq!((a.pages(), a.tokens(), b.pages(), p.used()), (2, 32, 1, 3));
    drop(a);
    assert_eq!(p.used(), 1);
    drop(b);
    assert_eq!(p.used(), 0);
    let all: Vec<Lease> = (0..4).map(|_| p.lease(1).unwrap()).collect();
    let mut ids: Vec<i32> = all.iter().flat_map(|l| l.page_ids().to_vec()).collect();
    ids.sort();
    assert_eq!((ids, p.used()), (vec![0, 1, 2, 3], 4));
    drop(all);
    assert_eq!(p.used(), 0);
}

#[test]
fn denials() {
    let p = pool();
    assert_eq!(p.lease(49).unwrap_err(), Denied::ExceedsRow { limit: 48 });
    let a = p.lease(48).unwrap();
    // One page free, no other kind to take chunks from: busy.
    assert_eq!(p.lease(17).unwrap_err(), Denied::Busy);
    let b = p.lease(16).unwrap();
    assert_eq!(p.lease(1).unwrap_err(), Denied::Busy);
    drop(a);
    assert!(p.lease(33).is_ok());
    drop(b);
    // Two pages of budget cap the row at two pages.
    let small = pool_of(&two_paged(), 8, 8, 0);
    assert_eq!((small.pages_max(), small.lease(48).unwrap_err()), (2, Denied::ExceedsRow { limit: 32 }));
    // A restore is denied the same way: by the row, or by the fresh pages it needs.
    let mut a = p.lease(16).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 16).unwrap();
    assert_eq!(p.restore(&cp, cp.tokens(), 49).unwrap_err(), Denied::ExceedsRow { limit: 48 });
    let _b = p.lease(48).unwrap();
    assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Busy);
    drop(a);
    assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Busy);
}

#[test]
fn hybrid_denials() {
    let p = hybrid_pool();
    let mut a = p.lease(32).unwrap();
    let _b = p.lease(16).unwrap();
    let _c = p.lease(16).unwrap();
    // Every page and slot held: only an eviction helps.
    assert_eq!(p.lease(16).unwrap_err(), Denied::Busy);
    assert!(p.take_pending().is_none());
    // A checkpoint copies the slot, so it needs one too; retiring never does.
    assert_eq!(p.checkpoint(&mut a, 16).unwrap_err(), Denied::Busy);
    let cp = p.retire(a, 16);
    assert!(cp.seq_slot().is_some());
    // Five chunks: slot 0 and one page, room for no other slot ever.
    let small = pool_of(&hybrid(), 8, 5, 1);
    assert_eq!((small.slots_max(), small.total()), (1, 1));
    assert_eq!(small.lease(16).unwrap_err(), Denied::ExceedsPool);
}

#[test]
fn slots_and_rows() {
    let p = pool();
    let l = p.lease(20).unwrap(); // 2 pages
    let [p0, p1] = l.page_ids()[..] else { panic!() };
    assert_eq!(l.slot(0), p0 as i64 * 16);
    assert_eq!(l.slot(15), p0 as i64 * 16 + 15);
    assert_eq!(l.slot(16), p1 as i64 * 16);
    assert_eq!(l.slots(15..17), [p0 as i64 * 16 + 15, p1 as i64 * 16]);
    let mut t = Vec::new();
    l.extend_row("block_table", &mut t).unwrap();
    assert_eq!(t, [p0, p1, p0]);
    let mut d = Vec::new();
    l.extend_row("draft_block_table", &mut d).unwrap();
    let mut want: Vec<i32> = (0..4).map(|k| p0 * 4 + k).chain((0..4).map(|k| p1 * 4 + k)).collect();
    want.resize(16, p0 * 4);
    assert_eq!(d, want);
    // Position 17's draft entry (row index 17/4) names the draft page its slot falls in.
    assert_eq!(d[17 / 4], (l.slot(17) / 4) as i32);
    assert!(l.extend_row("slot_mapping", &mut t).is_err());
    assert!(l.seq_line("block_table", 0).is_err());
}

#[test]
#[should_panic(expected = "past the lease")]
fn slot_past_lease() {
    let p = pool();
    p.lease(16).unwrap().slot(16);
}

#[test]
fn seq_slots_and_lines() {
    let m = hybrid();
    // seqs 2 + pad + null: the manifest's first slots are 0..4, three of them leasable.
    assert_eq!(m.seq_slots(), 4);
    let p = hybrid_pool();
    assert_eq!((p.has_slots(), p.slots(), p.seq_tables().collect::<Vec<_>>()), (true, 4, vec!["line_index"]));
    let a = p.lease(16).unwrap();
    let b = p.lease(16).unwrap();
    let c = p.lease(16).unwrap();
    let (sa, sb, sc) = (a.seq_slot().unwrap(), b.seq_slot().unwrap(), c.seq_slot().unwrap());
    let mut got = vec![sa, sb, sc];
    got.sort();
    assert_eq!((got, p.slots_used()), (vec![1, 2, 3], 3));
    // A fourth page is free but no slot is, and one free page's two
    // chunks are short of a slot's three: busy, not exhausted.
    assert_eq!(p.lease(16).unwrap_err(), Denied::Busy);
    // Line r of a slot: slot × 3 + r; the null line 0 belongs to no lease.
    assert_eq!(a.seq_line("line_index", 2).unwrap(), sa * 3 + 2);
    assert_eq!(a.seq_lines("line_index").unwrap(), 3);
    assert!(a.seq_line("line_index", 3).is_err());
    assert_ne!(a.seq_line("line_index", 0).unwrap(), 0);
    drop(b);
    let d = p.lease(16).unwrap();
    assert_eq!(d.seq_slot().unwrap(), sb);
}

#[test]
fn pages_come_from_free_slots() {
    let p = hybrid_pool();
    let a = p.lease(48).unwrap(); // pages 0..3, slot 1
    assert_eq!((p.used(), p.slots_used()), (3, 1));
    // Two pages wanted, one free: the highest free slot's three chunks make page 4.
    assert_eq!(p.lease(32).unwrap_err(), Denied::Remapping);
    let plan = p.take_pending().unwrap();
    assert_eq!(
        (plan.unmap.len(), plan.map.len(), &plan.made[..], &plan.unmade[..]),
        (3, 2, &[(Kind::Page, 4)][..], &[(Kind::Slot, 3)][..])
    );
    assert_eq!(plan.access, [(0, 8..10)]);
    // Until it lands, a caller sees neither the page arriving nor the slot leaving.
    assert_eq!(p.lease(32).unwrap_err(), Denied::Remapping);
    assert!(p.take_pending().is_none());
    assert_eq!((p.total(), p.slots()), (4, 3));
    p.complete(plan);
    let b = p.lease(32).unwrap();
    assert_eq!(
        (b.page_ids().to_vec(), b.seq_slot(), p.total(), p.slots(), p.slots_used()),
        (vec![3, 4], Some(2), 5, 3, 2)
    );
    drop((a, b));
    assert_eq!((p.used(), p.slots_used()), (0, 0));
}

#[test]
fn a_slot_comes_from_free_pages() {
    // 24 chunks: 4 slots and 6 pages.
    let p = pool_of(&hybrid(), 8, 24, 4);
    let held: Vec<Lease> = (0..3).map(|_| p.lease(16).unwrap()).collect();
    assert_eq!((p.total(), p.used(), p.slots_used()), (6, 3, 3));
    // A page is free but every slot is held: two free pages give slot 4.
    assert_eq!(p.lease(16).unwrap_err(), Denied::Remapping);
    let plan = p.take_pending().unwrap();
    assert_eq!(
        (plan.unmade.clone(), plan.made.clone()),
        (vec![(Kind::Page, 5), (Kind::Page, 4)], vec![(Kind::Slot, 4)])
    );
    assert_eq!((plan.unmap.len(), plan.map.len()), (4, 3));
    p.complete(plan);
    let d = p.lease(16).unwrap();
    assert_eq!((d.seq_slot(), p.total(), p.slots(), p.slots_used()), (Some(4), 4, 5, 4));
    drop(held);
    // Chunks stay where they were: pages 4 and 5 are gone, slots 1..3 free.
    assert_eq!((p.total(), p.slots(), p.used()), (4, 5, 1));
}

#[test]
fn checkpoint_shares_pages_and_outlives_the_lease() {
    let p = pool();
    let mut a = p.lease(40).unwrap(); // 3 pages
    let (cp, copies) = p.checkpoint(&mut a, 32).unwrap(); // the first 2
    assert_eq!((cp.tokens(), cp.page_ids().len(), cp.seq_slot(), copies), (32, 2, None, Copies::default()));
    assert_eq!((cp.page_ids(), a.pages(), p.used()), (a.page_ids()[..2].to_vec(), 3, 3));
    drop(a);
    // The checkpoint keeps its 2 pages; the lease's third came back.
    assert_eq!(p.used(), 2);
    assert_eq!(p.lease(32).unwrap().pages(), 2);
    drop(cp);
    assert_eq!(p.used(), 0);
    // Retiring keeps the pages up to `len` and returns the rest.
    let a = p.lease(48).unwrap();
    let cp = p.retire(a, 17);
    assert_eq!((cp.tokens(), cp.page_ids().len(), p.used()), (17, 2, 2));
}

#[test]
fn checkpoints_along_one_lease_share_one_chain() {
    let p = pool();
    let mut a = p.lease(48).unwrap();
    let c1 = p.checkpoint(&mut a, 16).unwrap().0;
    let c2 = p.checkpoint(&mut a, 32).unwrap().0;
    let c3 = p.checkpoint(&mut a, 48).unwrap().0;
    // A shallower checkpoint taken after a deeper one shares the whole
    // pages up the chain and copies the page it ends inside.
    let (c2b, copies) = p.checkpoint(&mut a, 20).unwrap();
    assert_eq!((&c2b.page_ids()[..1], c2b.page_ids().len(), copies.pages.len()), (&c1.page_ids()[..], 2, 1));
    assert_ne!(c2b.page_ids()[1], c2.page_ids()[1]);
    assert_eq!(p.used(), 4);
    drop(a);
    drop(c3);
    assert_eq!(p.used(), 3);
    drop(c2);
    assert_eq!(p.used(), 2); // c2b holds page 1's node and its own copy of page 2
    drop(c2b);
    assert_eq!(p.used(), 1);
    drop(c1);
    assert_eq!(p.used(), 0);
}

#[test]
fn restore_shares_whole_pages_and_copies_the_partial_one() {
    let p = pool();
    let mut a = p.lease(40).unwrap();
    let [a0, a1, _] = a.page_ids()[..] else { panic!() };
    // 20 tokens: page a0 whole, a1 holds positions 16..20, copied into a
    // page of the checkpoint's own so the lease can go on writing a1.
    let (cp, copies) = p.checkpoint(&mut a, 20).unwrap();
    let [c0, c1] = cp.page_ids()[..] else { panic!() };
    assert_eq!((c0, copies), (a0, Copies { pages: vec![(a1, c1)], slot: None }));
    assert_ne!(c1, a1);
    drop(a);
    let (b, copies) = p.restore(&cp, cp.tokens(), 40).unwrap();
    let [b0, b1, b2] = b.page_ids()[..] else { panic!() };
    // Shares a0, gets a fresh copy of c1 (c1 itself stays the checkpoint's), one more fresh page.
    assert_eq!((b0, b.prefix(), b.tokens()), (a0, 20, 48));
    assert_ne!(b1, c1);
    assert_eq!(copies, Copies { pages: vec![(c1, b1)], slot: None });
    assert_eq!(p.used(), 4);
    // Positions from 20 on are the lease's to write, into its own copy.
    assert_eq!(b.slot(20), b1 as i64 * 16 + 4);
    assert_eq!(b.slots(31..33), [b1 as i64 * 16 + 15, b2 as i64 * 16]);
    drop(cp);
    // c1 is only the checkpoint's: freed with it; a0 is still b's.
    assert_eq!(p.used(), 3);
    drop(b);
    assert_eq!(p.used(), 0);
    // At a page boundary nothing is copied: the whole pages are the checkpoint's own.
    let mut a = p.lease(32).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 32).unwrap();
    drop(a);
    let (b, copies) = p.restore(&cp, cp.tokens(), 33).unwrap();
    assert_eq!((b.prefix(), &b.page_ids()[..2], b.pages(), copies), (32, &cp.page_ids()[..], 3, Copies::default()));
    assert_eq!(p.used(), 3);
    drop((b, cp));
    // Restoring shallower than the checkpoint shares its whole pages up to there.
    let mut a = p.lease(40).unwrap();
    let [a0, a1, _] = a.page_ids()[..] else { panic!() };
    let (cp, _) = p.checkpoint(&mut a, 36).unwrap(); // 3 pages, the third a copy
    drop(a);
    // 32 tokens: a0 and a1 shared, one fresh page.
    let (c, copies) = p.restore(&cp, 32, 33).unwrap();
    assert_eq!((&c.page_ids()[..2], c.prefix(), c.pages(), copies), (&[a0, a1][..], 32, 3, Copies::default()));
    drop(c);
    // 16 tokens: page a0 shared, nothing copied.
    let (b, copies) = p.restore(&cp, 16, 20).unwrap();
    assert_eq!((b.page_ids()[0], b.prefix(), b.pages(), copies), (a0, 16, 2, Copies::default()));
    assert_ne!(b.page_ids()[1], a1);
}

#[test]
#[should_panic(expected = "restoring 4 tokens of a checkpoint of 10 (with a slot)")]
fn a_stateful_checkpoint_restores_at_its_length_only() {
    let p = pool_of(&hybrid(), 8, 26, 4);
    let mut a = p.lease(16).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 10).unwrap();
    let _ = p.restore(&cp, 4, 20);
}

#[test]
#[should_panic(expected = "inside a sealed page")]
fn a_lease_refuses_the_pages_it_sealed() {
    let p = pool();
    let mut a = p.lease(32).unwrap();
    let (_cp, _) = p.checkpoint(&mut a, 20).unwrap();
    // Page 0 went whole into the checkpoint's chain; page 1 was copied and
    // stays the lease's own.
    assert_eq!(a.slot(20), a.page_ids()[1] as i64 * 16 + 4);
    a.slot(15);
}

#[test]
#[should_panic(expected = "inside the shared prefix")]
fn restored_lease_refuses_its_prefix() {
    let p = pool();
    let mut a = p.lease(32).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 20).unwrap();
    drop(a);
    let (b, _) = p.restore(&cp, cp.tokens(), 40).unwrap();
    b.slot(19);
}

#[test]
fn fork_shares_whole_pages_and_copies_the_partial_one_and_the_slot() {
    let p = pool_of(&hybrid(), 8, 26, 4); // 4 slots, 7 pages
    let mut a = p.lease(40).unwrap(); // 3 pages
    let [a0, a1, _] = a.page_ids()[..] else { panic!() };
    let sa = a.seq_slot().unwrap();
    // The parent is 20 tokens in: page a0 whole, a1 holds 16..20.
    let (b, copies) = p.fork(&mut a, 20, 40).unwrap();
    let [b0, b1, _] = b.page_ids()[..] else { panic!() };
    assert_eq!((b0, b.prefix()), (a0, 20));
    assert_ne!(b1, a1);
    assert_eq!(copies, Copies { pages: vec![(a1, b1)], slot: Some((sa, b.seq_slot().unwrap())) });
    // Both keep writing their own copy of the page; a0 stays until both are gone.
    assert_eq!((a.slot(20), b.slot(20)), (a1 as i64 * 16 + 4, b1 as i64 * 16 + 4));
    assert_eq!((p.used(), p.slots_used()), (5, 2));
    drop(a);
    assert_eq!(p.used(), 3);
    drop(b);
    assert_eq!((p.used(), p.slots_used()), (0, 0));
    // At a page boundary nothing is copied; before the first boundary nothing is shared.
    let mut a = p.lease(32).unwrap();
    let (b, copies) = p.fork(&mut a, 16, 32).unwrap();
    assert_eq!((copies.pages.len(), b.page_ids()[0], b.prefix()), (0, a.page_ids()[0], 16));
    let (c, copies) = p.fork(&mut a, 3, 32).unwrap();
    assert_eq!((c.prefix(), copies.pages), (3, vec![(a.page_ids()[0], c.page_ids()[0])]));
    assert_eq!(p.fork(&mut a, 3, 32).unwrap_err(), Denied::Busy);
}

#[test]
fn hybrid_checkpoint_copies_the_slot_and_retire_moves_it() {
    // 28 chunks: 4 slots and 8 pages.
    let p = pool_of(&hybrid(), 8, 28, 4);
    let mut a = p.lease(16).unwrap();
    let (a0, sa) = (a.page_ids()[0], a.seq_slot().unwrap());
    let (cp, copies) = p.checkpoint(&mut a, 10).unwrap();
    let (c0, sc) = (cp.page_ids()[0], cp.seq_slot().unwrap());
    assert_ne!((c0, sc), (a0, sa));
    assert_eq!((copies, p.used(), p.slots_used()), (Copies { pages: vec![(a0, c0)], slot: Some((sa, sc)) }, 2, 2));
    // A slot each for a and cp: one left; a restore takes it and copies the state in.
    let (b, copies) = p.restore(&cp, cp.tokens(), 17).unwrap();
    let sb = b.seq_slot().unwrap();
    assert_eq!((copies.slot, b.prefix(), p.slots_used()), (Some((sc, sb)), 10, 3));
    // No slot left; of the four free pages the restore needs two, the
    // other two hold a slot's worth of chunks.
    assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Remapping);
    land(&p);
    let (c, copies) = p.restore(&cp, cp.tokens(), 17).unwrap();
    assert_eq!((copies.slot, p.slots(), p.slots_used(), p.total()), (Some((sc, 4)), 5, 4, 6));
    drop((b, c));
    // Retiring a moves its slot and its page to the checkpoint: no copy.
    let a2 = p.retire(a, 10);
    assert_eq!((a2.seq_slot(), a2.page_ids(), p.slots_used(), p.used()), (Some(sa), vec![a0], 2, 2));
    drop(cp);
    assert_eq!((p.slots_used(), p.used()), (1, 1));
    drop(a2);
    assert_eq!((p.slots_used(), p.used()), (0, 0));
}

#[test]
fn slot_only_leases_move_the_slot_alone() {
    let p = hybrid_pool();
    let mut l = p.lease_slot().unwrap();
    assert_eq!((l.pages(), l.tokens(), l.seq_slot().is_some()), (0, 0, true));
    assert_eq!((p.used(), p.slots_used()), (0, 1));
    // A checkpoint copies the slot and holds no page, at any length.
    let (cp, c) = p.checkpoint(&mut l, 5).unwrap();
    assert_eq!((cp.tokens(), cp.page_ids().len()), (5, 0));
    assert_eq!((c.pages.len(), c.slot.map(|(a, _)| a)), (0, l.seq_slot()));
    // Restoring is at its own length and gives a slot-only lease with that prefix.
    let (r, c) = p.restore(&cp, 5, 100).unwrap();
    assert_eq!((r.pages(), r.prefix(), c.pages.len(), c.slot.map(|(a, _)| a)), (0, 5, 0, cp.seq_slot()));
    assert_eq!(p.slots_used(), 3);
    drop(r);
    drop(cp);
    // A fork the same way.
    let (f, c) = p.fork(&mut l, 9, 100).unwrap();
    assert_eq!((f.pages(), f.prefix(), c.slot.map(|(a, _)| a)), (0, 9, l.seq_slot()));
    drop(f);
    // Retiring hands the slot over as it is.
    let slot = l.seq_slot();
    let cp = p.retire(l, 7);
    assert_eq!((cp.tokens(), cp.page_ids().len(), cp.seq_slot()), (7, 0, slot));
    drop(cp);
    assert_eq!((p.used(), p.slots_used()), (0, 0));
}

#[test]
fn a_long_chain_drops_without_recursion() {
    let m = Manifest::from_json(
        r#"{
        "schema_version": 5, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
        "states": {"kv": {"bytes_per_token": 1}},
        "buffers": {
            "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 200000], "domain": {"index_into": "kv", "stride": 1}}
        },
        "modules": {}, "ops": {}, "programs": {}
    }"#,
    )
    .unwrap();
    let p: Arc<Pool> = pool_of(&m, 1, 200_000, 0);
    let mut a = p.lease(200_000).unwrap();
    let cp = p.checkpoint(&mut a, 200_000).unwrap().0;
    drop(a);
    assert_eq!(p.used(), 200_000);
    drop(cp);
    assert_eq!(p.used(), 0);
}

/// Which chunk is mapped at each (arena, position), kept from the plans
/// the pool hands out: a chunk is free or at one position, a plan maps
/// only free chunks onto empty positions and unmaps only mapped ones,
/// and what it makes is whole once it lands.
#[derive(Default)]
struct Mapping {
    chunks: usize,
    at: BTreeMap<(usize, usize), u32>,
    free: BTreeSet<u32>,
    /// Objects that exist, by (kind, object).
    made: BTreeSet<(Kind, i32)>,
    landed: usize,
}

impl Mapping {
    fn new(chunks: u32) -> Mapping {
        Mapping {
            chunks: chunks as usize,
            at: BTreeMap::new(),
            free: (0..chunks).collect(),
            made: BTreeSet::new(),
            landed: 0,
        }
    }

    /// Positions `object` of `kind` covers in each arena of that kind.
    fn positions(p: &Pool, kind: Kind, object: usize) -> Vec<(usize, usize)> {
        let chunk = p.chunk();
        p.pooled()
            .iter()
            .enumerate()
            .filter(|(_, a)| a.kind == kind)
            .flat_map(|(a, ar)| {
                let lo = ar.object * object as u64;
                ((lo / chunk) as usize..((lo + ar.object).div_ceil(chunk)) as usize).map(move |q| (a, q))
            })
            .collect()
    }

    fn land(&mut self, p: &Pool, plan: &Remap) {
        for &(a, q) in &plan.unmap {
            let c = self.at.remove(&(a, q)).expect("unmapping a mapped position");
            assert!(self.free.insert(c), "chunk {c} freed twice");
        }
        for &(a, q, c) in &plan.map {
            assert!(self.free.remove(&c), "chunk {c} mapped while in use");
            assert!(self.at.insert((a, q), c).is_none(), "position ({a}, {q}) mapped twice");
        }
        for &(k, o) in &plan.unmade {
            assert!(self.made.remove(&(k, o)), "unmaking {k:?} {o} that did not exist");
        }
        for &(k, o) in &plan.made {
            assert!(self.made.insert((k, o)), "making {k:?} {o} twice");
            for pos in Mapping::positions(p, k, o as usize) {
                assert!(self.at.contains_key(&pos), "{k:?} {o} landed with position {pos:?} unmapped");
            }
        }
        assert_eq!(self.at.len() + self.free.len(), self.chunks, "a chunk is free or at one position");
        let count = |k: Kind| self.made.iter().filter(|(kk, _)| *kk == k).count();
        assert_eq!((p.total(), p.slots()), (count(Kind::Page), count(Kind::Slot)));
        self.landed += 1;
    }

    /// Every page and slot a live handle names exists and is whole.
    fn covers(&self, p: &Pool, pages: &BTreeSet<i32>, slots: &BTreeSet<i32>) {
        for &page in pages {
            assert!(self.made.contains(&(Kind::Page, page)), "page {page} held but not made");
            for pos in Mapping::positions(p, Kind::Page, page as usize) {
                assert!(self.at.contains_key(&pos), "page {page} held with {pos:?} unmapped");
            }
        }
        for &slot in slots {
            assert!(self.made.contains(&(Kind::Slot, slot)), "slot {slot} held but not made");
            for pos in Mapping::positions(p, Kind::Slot, slot as usize) {
                assert!(self.at.contains_key(&pos), "slot {slot} held with {pos:?} unmapped");
            }
        }
    }
}

/// The positions and slots of a device, each holding the stamp last
/// written there.
#[derive(Default)]
struct Device {
    positions: BTreeMap<i64, u32>,
    slots: BTreeMap<i32, u32>,
}

impl Device {
    fn copy(&mut self, unit: usize, c: &Copies) {
        for &(s, d) in &c.pages {
            for k in 0..unit as i64 {
                match self.positions.get(&(s as i64 * unit as i64 + k)).copied() {
                    Some(v) => self.positions.insert(d as i64 * unit as i64 + k, v),
                    None => self.positions.remove(&(d as i64 * unit as i64 + k)),
                };
            }
        }
        if let Some((s, d)) = c.slot {
            match self.slots.get(&s).copied() {
                Some(v) => self.slots.insert(d, v),
                None => self.slots.remove(&d),
            };
        }
    }

    fn at(&self, pages: &[i32], unit: usize, pos: usize) -> Option<u32> {
        self.positions.get(&(pages[pos / unit] as i64 * unit as i64 + (pos % unit) as i64)).copied()
    }
}

fn state_of(content: &[u32]) -> u32 {
    content.iter().fold(7, |s, &v| s.wrapping_mul(31).wrapping_add(v))
}

struct Seq {
    lease: Lease,
    content: Vec<u32>,
}

struct Held {
    cp: Checkpoint,
    content: Vec<u32>,
}

/// Write positions `content.len()..upto` of `s`, then its state.
fn fill(dev: &mut Device, s: &mut Seq, upto: usize, stamp: &mut u32) {
    for pos in s.content.len()..upto {
        *stamp += 1;
        dev.positions.insert(s.lease.slot(pos), *stamp);
        s.content.push(*stamp);
    }
    if let Some(slot) = s.lease.seq_slot() {
        dev.slots.insert(slot, state_of(&s.content));
    }
}

/// Every position and slot a live handle names reads what its writer
/// put there, is mapped whole, and the pool holds exactly what the
/// handles name.
fn check(dev: &Device, map: &Mapping, p: &Pool, seqs: &[Seq], cps: &[Held]) {
    let unit = p.unit() as usize;
    let (mut pages, mut slots) = (BTreeSet::new(), BTreeSet::new());
    for s in seqs {
        let ids = s.lease.page_ids();
        for (pos, &v) in s.content.iter().enumerate() {
            assert_eq!(dev.at(ids, unit, pos), Some(v), "lease position {pos}");
        }
        if let Some(slot) = s.lease.seq_slot() {
            assert_eq!(dev.slots.get(&slot), Some(&state_of(&s.content)));
            slots.insert(slot);
        }
        pages.extend(ids.iter().copied());
    }
    for c in cps {
        let ids = c.cp.page_ids();
        for (pos, &v) in c.content.iter().enumerate() {
            assert_eq!(dev.at(&ids, unit, pos), Some(v), "checkpoint position {pos}");
        }
        if let Some(slot) = c.cp.seq_slot() {
            assert_eq!(dev.slots.get(&slot), Some(&state_of(&c.content)));
            slots.insert(slot);
        }
        pages.extend(ids);
    }
    assert_eq!((p.used(), p.slots_used()), (pages.len(), slots.len()));
    map.covers(p, &pages, &slots);
}

/// Random leases, fills, checkpoints, restores, forks, retirements and
/// drops, every position written by its lease and read back through
/// every handle that names it: a page is written by the lease it
/// belongs to and nobody else, so what a checkpoint held stays what
/// it held however far its sequence runs on. (A half-page checkpoint
/// that shared the page its lease kept writing was the bug behind
/// this rewrite.) With a recurrent state a checkpoint is the sequence
/// as of now and restores at its own length only. The remaps planned
/// along the way land into a model of the chunks: a chunk is free or
/// at one position, and whatever a handle names is mapped whole.
fn handles_read_what_their_writer_wrote(m: &Manifest, chunk: u64, chunks: u32, first_slots: usize, seed: u64) {
    let (p, first) = Pool::new(m, chunk, chunks, first_slots, None).unwrap();
    let p = Arc::new(p);
    let mut map = Mapping::new(chunks);
    map.land(&p, &first);
    let land = |map: &mut Mapping| {
        let plan = p.take_pending().expect("a remap planned");
        p.complete(plan.clone());
        map.land(&p, &plan);
    };
    let unit = p.unit() as usize;
    let max = p.max_seq_tokens();
    let stateful = p.has_slots();
    let mut dev = Device::default();
    let mut seqs: Vec<Seq> = Vec::new();
    let mut cps: Vec<Held> = Vec::new();
    let mut rand = Rand::new(seed);
    let mut stamp = 0;
    // A checkpoint length: the whole sequence when a state pins it, else anywhere in it.
    let cut = |rand: &mut Rand, filled: usize| if stateful { filled } else { 1 + rand.next(filled) };
    for _ in 0..4000 {
        match rand.next(9) {
            0 | 1 => match p.lease(1 + rand.next(max)) {
                Ok(lease) => {
                    let mut s = Seq { lease, content: Vec::new() };
                    let upto = rand.next(s.lease.tokens() + 1);
                    fill(&mut dev, &mut s, upto, &mut stamp);
                    seqs.push(s);
                }
                Err(Denied::Remapping) => land(&mut map),
                Err(_) => {}
            },
            2 if !seqs.is_empty() => {
                let i = rand.next(seqs.len());
                let s = &mut seqs[i];
                let upto = s.content.len() + rand.next(s.lease.tokens() - s.content.len() + 1);
                fill(&mut dev, s, upto, &mut stamp);
            }
            3 if !seqs.is_empty() => {
                let i = rand.next(seqs.len());
                let s = &mut seqs[i];
                if s.content.is_empty() {
                    continue;
                }
                let len = cut(&mut rand, s.content.len());
                match p.checkpoint(&mut s.lease, len) {
                    Ok((cp, copies)) => {
                        dev.copy(unit, &copies);
                        cps.push(Held { cp, content: s.content[..len].to_vec() });
                    }
                    Err(Denied::Remapping) => land(&mut map),
                    Err(_) => {}
                }
            }
            4 if !cps.is_empty() => {
                let c = &cps[rand.next(cps.len())];
                let whole = (c.cp.tokens() - 1) / unit;
                let len = if c.cp.has_slot() || whole == 0 || rand.next(2) == 0 {
                    c.cp.tokens()
                } else {
                    (1 + rand.next(whole)) * unit
                };
                if len >= max {
                    continue;
                }
                match p.restore(&c.cp, len, len + 1 + rand.next(max - len)) {
                    Ok((lease, copies)) => {
                        dev.copy(unit, &copies);
                        let mut s = Seq { lease, content: c.content[..len].to_vec() };
                        let upto = len + rand.next(s.lease.tokens() - len + 1);
                        fill(&mut dev, &mut s, upto, &mut stamp);
                        seqs.push(s);
                    }
                    Err(Denied::Remapping) => land(&mut map),
                    Err(_) => {}
                }
            }
            5 if !seqs.is_empty() => {
                let i = rand.next(seqs.len());
                let filled = seqs[i].content.len();
                if filled == 0 || filled >= max {
                    continue;
                }
                let len = cut(&mut rand, filled);
                let tokens = len + 1 + rand.next(max - len);
                match p.fork(&mut seqs[i].lease, len, tokens) {
                    Ok((lease, copies)) => {
                        dev.copy(unit, &copies);
                        let mut s = Seq { lease, content: seqs[i].content[..len].to_vec() };
                        let upto = len + rand.next(s.lease.tokens() - len + 1);
                        fill(&mut dev, &mut s, upto, &mut stamp);
                        seqs.push(s);
                    }
                    Err(Denied::Remapping) => land(&mut map),
                    Err(_) => {}
                }
            }
            6 if !seqs.is_empty() => {
                let s = seqs.swap_remove(rand.next(seqs.len()));
                if s.content.is_empty() {
                    continue;
                }
                let len = cut(&mut rand, s.content.len());
                let content = s.content[..len].to_vec();
                cps.push(Held { cp: p.retire(s.lease, len), content });
            }
            7 if !seqs.is_empty() => {
                seqs.swap_remove(rand.next(seqs.len()));
            }
            8 if !cps.is_empty() => {
                cps.swap_remove(rand.next(cps.len()));
            }
            _ => {}
        }
        check(&dev, &map, &p, &seqs, &cps);
    }
    drop((seqs, cps));
    assert_eq!((p.used(), p.slots_used()), (0, 0));
    assert!(!stateful || map.landed > 1, "pages and slots never traded a chunk");
}

#[test]
fn paged_handles_read_what_their_writer_wrote() {
    // 24 pages of 4 tokens, rows of 8.
    handles_read_what_their_writer_wrote(&paged4(), 4, 24, 0, 0x9E37_79B9_7F4A_7C15);
}

#[test]
fn stateful_handles_read_what_their_writer_wrote() {
    // 3 slots and 6 pages over 18 chunks: pages and slots trade chunks.
    handles_read_what_their_writer_wrote(&hybrid4(), 4, 18, 3, 0x2545_F491_4F6C_DD1D);
}

#[test]
fn hybrid_handles_read_what_their_writer_wrote() {
    // 8-byte chunks, 26 of them: 3 slots of 24 bytes, 8 pages of 16, a
    // spare chunk, so a page or a slot boundary can fall inside a chunk.
    handles_read_what_their_writer_wrote(&hybrid(), 8, 26, 3, 0x1234_5678_9ABC_DEF1);
}

#[test]
fn access_spans_merge_touching_grants_per_arena() {
    let plan =
        Remap { access: vec![(1, 4..6), (0, 0..2), (0, 1..3), (0, 3..4), (0, 6..7), (1, 2..4)], ..Remap::default() };
    assert_eq!(plan.access_spans(), [(0, 0..4), (0, 6..7), (1, 2..6)]);
    assert_eq!(Remap::default().access_spans(), []);
}
