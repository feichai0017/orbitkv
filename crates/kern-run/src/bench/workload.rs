//! What to measure: a sweep of call shapes, and the scenarios a manifest
//! makes of them.
//!
//! A workload names shapes, never programs. `groups` sequences of `rows`
//! rows each, on top of a previous context: [`Protocol`] turns that into a
//! program, so a decode step, a prefill chunk and a speculative round are
//! the same kind of entry in the file and nothing here knows their names.
//! One shape can become more than one scenario — a manifest that has both
//! a one-row step and a rows-as-fed chunk takes a single row either way,
//! and which is faster is a fair question — so the program is part of a
//! scenario's id rather than something the file chooses.
//!
//! A sweep is a cross product, because a cross product is how the mix is
//! found: the same 2,048-row chunk is mostly matrix multiply over an empty
//! cache and almost entirely attention over a long one. A combination no
//! program accepts is dropped and named in the report rather than failing
//! the run — which shapes a manifest serves is part of what a sweep asks.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{ensure, Context as _, Result};
use kern_manifest::protocol::Rows;
use kern_manifest::Protocol;
use serde::{Deserialize, Serialize};

/// One previous-context length for every sequence of a shape, or one each.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
enum Context {
    Every(usize),
    Each(Vec<usize>),
}

/// One cross product of shapes. `groups` defaults to a single sequence.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Sweep {
    #[serde(default = "alone")]
    groups: Vec<usize>,
    rows: Vec<usize>,
    context: Vec<Context>,
}

fn alone() -> Vec<usize> {
    vec![1]
}

/// A workload file: how many samples every measurement keeps, the seed its
/// prose context is drawn with, and the sweeps to expand. No sweep at all
/// is a workload that asks only for the hardware anchors.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub samples: usize,
    pub seed: u64,
    #[serde(default)]
    sweep: Vec<Sweep>,
}

/// A call shape: one sequence per `context` entry, `rows` rows each.
#[derive(Serialize, Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    pub groups: usize,
    pub rows: usize,
    pub context: Vec<usize>,
}

impl Shape {
    fn new(rows: usize, context: Vec<usize>) -> Shape {
        Shape { groups: context.len(), rows, context }
    }

    /// The context: one number when every sequence is at the same depth,
    /// and every sequence's otherwise.
    pub fn kv(&self) -> String {
        match self.context.split_first() {
            Some((n, rest)) if rest.iter().all(|c| c == n) => n.to_string(),
            _ => self.context.iter().map(usize::to_string).collect::<Vec<_>>().join("+"),
        }
    }

    /// What the shape is called before a program is chosen. It carries the
    /// whole shape, so anything printed beside it can say something else.
    pub fn label(&self) -> String {
        format!("g{}-r{}-kv{}", self.groups, self.rows, self.kv())
    }

    /// Tokens of state this shape reaches, its rows included, rounded up to
    /// whole pages with one page of slack.
    fn tokens(&self, unit: usize) -> usize {
        self.context.iter().map(|n| (n + self.rows).div_ceil(unit) * unit).sum::<usize>() + unit
    }
}

impl Workload {
    pub fn read(path: &Path) -> Result<Workload> {
        let text = std::fs::read_to_string(path).with_context(|| format!("workload {}", path.display()))?;
        Workload::parse(&text).with_context(|| format!("workload {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Workload> {
        let w: Workload = toml::from_str(text)?;
        ensure!((12..=256).contains(&w.samples), "samples must be in 12..=256, got {}", w.samples);
        for s in &w.sweep {
            ensure!(!s.rows.is_empty() && !s.context.is_empty(), "a sweep needs rows and context");
            ensure!(s.groups.iter().all(|&g| g > 0), "groups must be positive");
            ensure!(s.rows.iter().all(|&r| r > 0), "rows must be positive");
            for c in &s.context {
                if let Context::Each(v) = c {
                    ensure!(
                        s.groups.contains(&v.len()),
                        "a context of {} lengths pairs with no group count in {:?}",
                        v.len(),
                        s.groups
                    );
                }
            }
        }
        Ok(w)
    }

    /// Every sweep's cross product, in file order, without repeats. A
    /// per-sequence context only pairs with the group count it has lengths
    /// for, and `parse` has checked that count is in the sweep.
    pub fn shapes(&self) -> Vec<Shape> {
        let mut seen = BTreeSet::new();
        self.sweep
            .iter()
            .flat_map(|s| {
                s.groups.iter().flat_map(move |&g| {
                    s.rows.iter().flat_map(move |&r| {
                        s.context.iter().filter_map(move |c| match c {
                            Context::Every(n) => Some(Shape::new(r, vec![*n; g])),
                            Context::Each(v) if v.len() == g => Some(Shape::new(r, v.clone())),
                            Context::Each(_) => None,
                        })
                    })
                })
            })
            .filter(|s| seen.insert(s.label()))
            .collect()
    }
}

/// A shape this manifest has no program for.
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct Dropped {
    pub shape: String,
    pub why: String,
}

/// A shape and one program that takes it.
#[derive(Serialize, Debug)]
pub struct Scenario {
    pub id: String,
    #[serde(flatten)]
    pub shape: Shape,
    pub program: String,
}

/// What a workload asks of a manifest once the manifest has answered: the
/// scenarios to run, the shapes it turned away, and the state the largest
/// of them needs. Built by [`Plan::check`]; nothing downstream asks again
/// whether a scenario has a program.
#[derive(Debug)]
pub struct Plan {
    pub scenarios: Vec<Scenario>,
    pub dropped: Vec<Dropped>,
    /// Sized for the largest scenario: one plan, one allocation, so a
    /// sweep's cheap scenarios carry its expensive one's state.
    pub tokens: u64,
    pub seqs: u64,
}

impl Plan {
    pub fn check(w: &Workload, p: &Protocol, unit: usize) -> Result<Plan> {
        let (mut scenarios, mut dropped) = (Vec::new(), Vec::new());
        for shape in w.shapes() {
            match forwards(p, &shape) {
                Err(why) => dropped.push(Dropped { shape: shape.label(), why }),
                Ok(taken) => scenarios.extend(taken.into_iter().map(|program| Scenario {
                    id: format!("{program}-{}", shape.label()),
                    shape: shape.clone(),
                    program,
                })),
            }
        }
        ensure!(
            !scenarios.is_empty() || dropped.is_empty(),
            "no shape in this workload has a program: {}",
            dropped.iter().map(|d| format!("{} ({})", d.shape, d.why)).collect::<Vec<_>>().join(", ")
        );
        Ok(Plan {
            tokens: scenarios.iter().map(|s| s.shape.tokens(unit)).max().unwrap_or(unit) as u64,
            seqs: scenarios.iter().map(|s| s.shape.groups).max().unwrap_or(1) as u64,
            scenarios,
            dropped,
        })
    }
}

/// Every program that takes a shape: the one declaring exactly these rows,
/// and the one that takes rows as fed, when both do. A shape over context
/// also needs the second kind to build its prefix, whatever runs the step.
fn forwards(p: &Protocol, s: &Shape) -> std::result::Result<Vec<String>, String> {
    if s.groups * s.rows > p.rows.max as usize {
        return Err(format!("{} rows exceeds the manifest's {}", s.groups * s.rows, p.rows.max));
    }
    if s.context.iter().any(|&n| n > 0) && p.chunk().is_none() {
        return Err("a prefix needs a rows-as-fed program and this manifest has none".into());
    }
    let taken: Vec<String> = [Rows::Const(s.rows as u64), Rows::Var]
        .into_iter()
        .filter_map(|rows| p.forward(s.groups as u64, rows).map(|f| f.name.clone()))
        .collect();
    match taken.is_empty() {
        true => Err(format!("no program takes {} sequences × {} rows", s.groups, s.rows)),
        false => Ok(taken),
    }
}
