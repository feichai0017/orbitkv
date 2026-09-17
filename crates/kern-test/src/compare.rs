//! Comparing what two sides wrote. Byte slices in, numbers out; the
//! dtype says how to read them.

use std::ops::Range;

use kern_manifest::types::DType;
use kern_manifest::values;
use serde::Serialize;

fn is_float(dt: DType) -> bool {
    matches!(dt, DType::Bf16 | DType::F16 | DType::F32 | DType::Fp8E4m3)
}

/// One buffer on two sides, element by element.
#[derive(Serialize, Clone, Default, Debug, PartialEq)]
pub struct Cmp {
    pub n: usize,
    pub n_diff: usize,
    pub max_ulp: Option<u64>,
    pub max_abs: f64,
    pub nan_only_one_side: usize,
    /// Bit-different but value-equal: +0 vs -0.
    pub signed_zero: usize,
}

pub fn compare(dt: DType, a: &[u8], b: &[u8]) -> Cmp {
    let w = dt.bytes() as usize;
    let mut c = Cmp { n: a.len() / w, ..Default::default() };
    for (x, y) in a.chunks_exact(w).zip(b.chunks_exact(w)) {
        if x == y {
            continue;
        }
        c.n_diff += 1;
        let (fx, fy) = (values::to_f64(dt, x)[0], values::to_f64(dt, y)[0]);
        if fx.is_nan() != fy.is_nan() {
            c.nan_only_one_side += 1;
            continue;
        }
        if fx == fy {
            c.signed_zero += 1;
            continue;
        }
        c.max_abs = c.max_abs.max((fx - fy).abs());
        if is_float(dt) {
            if let Some(u) = values::ulp_distance(dt, x, y) {
                c.max_ulp = Some(c.max_ulp.unwrap_or(0).max(u));
            }
        }
    }
    c
}

impl Cmp {
    /// A comparison a device kernel counted: `max_ulp` only if any float
    /// pair was measurable.
    pub fn from_counts(
        n: usize,
        n_diff: usize,
        signed_zero: usize,
        nan_only_one_side: usize,
        measured: usize,
        max_ulp: u64,
        max_abs: f64,
    ) -> Cmp {
        Cmp { n, n_diff, max_ulp: (measured > 0).then_some(max_ulp), max_abs, nan_only_one_side, signed_zero }
    }
    /// Two comparisons of disjoint ranges of one buffer as one.
    pub fn merge(self, o: Cmp) -> Cmp {
        Cmp {
            n: self.n + o.n,
            n_diff: self.n_diff + o.n_diff,
            max_ulp: match (self.max_ulp, o.max_ulp) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            },
            max_abs: self.max_abs.max(o.max_abs),
            nan_only_one_side: self.nan_only_one_side + o.nan_only_one_side,
            signed_zero: self.signed_zero + o.signed_zero,
        }
    }
    pub fn identical(&self) -> bool {
        self.n_diff == 0
    }
    /// Every difference is a signed zero.
    pub fn value_identical(&self) -> bool {
        self.n_diff == self.signed_zero
    }
    /// Worst first: differing elements, then ulps.
    pub fn severity(&self) -> (usize, u64) {
        (self.n_diff - self.signed_zero, self.max_ulp.unwrap_or(u64::MAX))
    }
}

/// One buffer at one span: A's write-set compared element by element, and
/// the bytes the other side wrote outside it (a block A left alone that
/// the other side changed: not compared, since A has no reference there,
/// but not nothing either).
#[derive(Serialize, Clone, Default, Debug, PartialEq)]
pub struct BufCmp {
    pub cmp: Cmp,
    pub outside: usize,
}

impl BufCmp {
    pub fn identical(&self) -> bool {
        self.cmp.identical() && self.outside == 0
    }
    pub fn value_identical(&self) -> bool {
        self.cmp.value_identical() && self.outside == 0
    }
    /// Worst first: a write outside the reference set outranks any ulp.
    pub fn severity(&self) -> (usize, u64) {
        if self.outside > 0 {
            (usize::MAX, u64::MAX)
        } else {
            self.cmp.severity()
        }
    }
}

/// Bytes of `b` that no range of `a` covers; both sorted and disjoint.
pub fn outside(b: &[Range<usize>], a: &[Range<usize>]) -> usize {
    b.iter()
        .map(|r| {
            let covered: usize = a.iter().map(|x| r.end.min(x.end).saturating_sub(r.start.max(x.start))).sum();
            r.len() - covered
        })
        .sum()
}

/// `ranges` with gaps under `gap` bytes closed: a scattered write-set as
/// few pieces (the bytes between are kept too, and compared, which is
/// harmless: they are the same on both sides unless the other side wrote
/// them, which is then seen as a difference rather than counted outside).
pub fn coalesce(ranges: Vec<Range<usize>>, gap: usize) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for r in ranges {
        match out.last_mut() {
            Some(last) if r.start <= last.end + gap => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

/// How many of A's most likely tokens the report follows: enough to see
/// whether a drift sits at the head of the distribution or in its tail.
pub const TOP: usize = 20;

/// One row of a `logits*` buffer, A against B, measured on the
/// distribution, not the storage: KL(A‖B) says how much probability mass
/// moved, whatever dtype the row is kept in, and a flip within a small KL
/// is a tie that was going to break either way. Element-wise ulps are
/// meaningless here — a 1-ulp change of the hidden state moves every logit
/// by about the same absolute amount, which is thousands of ulps for a
/// logit near zero and decides nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct LogitStats {
    pub cmp: Cmp,
    pub argmax_a: usize,
    pub argmax_b: usize,
    /// A's top-1 − top-2: how far the argmax was from flipping on its own.
    pub margin_a: f64,
    /// KL(A‖B) in nats; infinite when a side holds a NaN.
    pub kl: f64,
    /// How many of A's [`TOP`] most likely tokens are among B's.
    pub top: usize,
    /// Where A's argmax ranks in B, 1-based.
    pub rank_in_b: usize,
}

impl LogitStats {
    /// From what a kernel measured: A's top two values, the raw KL sum
    /// and the counts. A NaN on one side is an unbounded move, not one to
    /// skip: a row B poisons must never read as "moved nothing".
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        cmp: Cmp,
        argmax_a: usize,
        argmax_b: usize,
        top1: f64,
        top2: f64,
        kl: f64,
        top: usize,
        rank_in_b: usize,
    ) -> Self {
        let kl = if cmp.nan_only_one_side == 0 && kl.is_finite() { kl.max(0.0) } else { f64::INFINITY };
        LogitStats { cmp, argmax_a, argmax_b, margin_a: top1 - top2, kl, top, rank_in_b }
    }
    pub fn flip(&self) -> bool {
        self.argmax_a != self.argmax_b
    }
}

/// A [`LogitStats`] with the workload position it was read at.
#[derive(Clone, Debug)]
pub struct LogitRow {
    pub label: String,
    pub stats: LogitStats,
}

/// The definition: one row on the host.
pub fn logit_stats(dt: DType, a: &[u8], b: &[u8]) -> LogitStats {
    let (va, vb) = (values::to_f64(dt, a), values::to_f64(dt, b));
    let cmp = compare(dt, a, b);
    let order = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[j].total_cmp(&v[i]).then(i.cmp(&j)));
        idx
    };
    let (oa, ob) = (order(&va), order(&vb));
    let (argmax_a, argmax_b) = (oa.first().copied().unwrap_or(0), ob.first().copied().unwrap_or(0));
    let top1 = va.get(argmax_a).copied().unwrap_or(f64::NEG_INFINITY);
    let top2 = oa.get(1).map_or(f64::NEG_INFINITY, |&i| va[i]);
    let k = TOP.min(va.len());
    let top = oa[..k].iter().filter(|i| ob[..k].contains(i)).count();
    let rank_in_b = ob.iter().position(|&i| i == argmax_a).map_or(0, |r| r + 1);
    let lse = |v: &[f64], m: f64| m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
    let mb = vb.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (la, lb) = (lse(&va, top1), lse(&vb, mb));
    let kl = va.iter().zip(&vb).map(|(x, y)| (x - la).exp() * ((x - la) - (y - lb))).sum::<f64>();
    LogitStats::from_parts(cmp, argmax_a, argmax_b, top1, top2, kl, top, rank_in_b)
}

/// The 64-byte blocks where `pre` and `post` differ, as merged byte
/// ranges (adjacent changed blocks are one range; the last block is
/// clipped to the length). The unit a state's write-set is found in: a
/// kernel sets one bit per block, the host reads bits, not bytes.
pub const BLOCK: usize = 64;

pub fn changed_blocks(pre: &[u8], post: &[u8]) -> Vec<Range<usize>> {
    let n = pre.len();
    let block = |i: usize| i * BLOCK..((i + 1) * BLOCK).min(n);
    let mut out: Vec<Range<usize>> = Vec::new();
    for r in (0..n.div_ceil(BLOCK)).map(block).filter(|r| pre[r.clone()] != post[r.clone()]) {
        match out.last_mut() {
            Some(last) if last.end == r.start => last.end = r.end,
            _ => out.push(r),
        }
    }
    out
}
