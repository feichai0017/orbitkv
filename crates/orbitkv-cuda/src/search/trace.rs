//! Optional search evidence. No operation selection or cost policy lives here.
//!
//! Program records retain every operation and ordered input, so callers can
//! inspect any provider/shape without a model-specific list of expensive ops.
//! Records are flushed outside device measurements; tracing can consume part
//! of the cooperative wall-clock search budget.

use std::{
    collections::{BTreeMap, HashSet},
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    time::Duration,
};

use orbitkv_compiler::{
    graph::{CompileOptions, LLIRGraph, llir_program_identity},
    prelude::{DynMap, petgraph::Direction, petgraph::visit::EdgeRef},
    search::{BucketContext, Candidate, Outcome, PendingFinalist, SelectedProgram},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::host::HostOp;

const TRACE_SCHEMA_VERSION: u32 = 2;
const TRACE_ENVIRONMENT_VARIABLE: &str = "ORBITKV_SEARCH_TRACE";

pub(super) struct SearchTrace<W: Write = BufWriter<File>> {
    writer: W,
    programs: HashSet<String>,
}

impl SearchTrace {
    pub(super) fn configured(options: &CompileOptions) -> Option<Self> {
        let path = options.search_trace.clone().or_else(|| {
            std::env::var_os(TRACE_ENVIRONMENT_VARIABLE).map(std::path::PathBuf::from)
        })?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap_or_else(|error| {
                panic!("cannot create search trace {}: {error}", path.display())
            });
        let mut trace = Self::new(BufWriter::new(file));
        trace.write(json!({
            "event": "search_started",
            "schema_version": TRACE_SCHEMA_VERSION,
            "backend": "cuda",
            "program_identity": "selected-schedule LLIR identity; build-local, not a source digest",
            "limit": options.limit,
            "keep_best": options.keep_best,
            "initial_population": options.initial_population,
            "hotspot_candidates": options.hotspot_candidates,
            "sampling_order": "sorted snapshot class/node IDs; fresh saturation is not canonicalized",
            "trials": options.trials,
            "search_time_limit_ns": finite_budget(options.search_time_limit),
            "candidate_timeout_ns": options.candidate_timeout.and_then(finite_budget),
            "execution_timeout_ns": options.execution_timeout.and_then(finite_budget),
        }));
        Some(trace)
    }
}

fn finite_budget(duration: Duration) -> Option<u128> {
    (duration != Duration::MAX).then_some(duration.as_nanos())
}

fn dimensions(dims: &DynMap) -> BTreeMap<String, usize> {
    dims.iter()
        .map(|(name, &value)| (name.to_string(), value))
        .collect()
}

impl<W: Write> SearchTrace<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            programs: HashSet::new(),
        }
    }

    fn try_write(&mut self, record: &Value) -> io::Result<()> {
        serde_json::to_writer(&mut self.writer, record)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }

    fn write(&mut self, record: Value) {
        self.try_write(&record)
            .expect("failed to write requested search trace");
    }

    fn program(&mut self, llir: &LLIRGraph) -> String {
        let identity = llir_program_identity(llir);
        if self.programs.insert(identity.clone()) {
            let operations = llir.node_indices().map(|node| {
                let mut inputs = llir.edges_directed(node, Direction::Incoming)
                    .map(|edge| (edge.id().index(), edge.source().index())).collect::<Vec<_>>();
                inputs.sort_unstable_by_key(|(edge, _)| *edge);
                json!({
                    "node": node.index(),
                    "semantic": format!("{:?}", llir[node]),
                    "host_provider": llir[node].to_dialect::<dyn HostOp>().and_then(|op| op.stats_name()),
                    "kernel": llir[node].to_dialect::<dyn crate::kernel::KernelOp>().map(|op| op.kernel_name()),
                    "inputs": inputs.into_iter().map(|(_, source)| source).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>();
            self.write(json!({"event": "program", "program": identity, "operations": operations}));
        }
        identity
    }

    pub(super) fn bucket(&mut self, ctx: &BucketContext<'_>) {
        let graph = ctx.egraph();
        let classes = graph
            .eclasses
            .iter()
            .map(|(id, value)| (id.as_ref(), value))
            .collect::<BTreeMap<_, _>>();
        let nodes = graph
            .enodes
            .iter()
            .map(|(id, value)| (id.as_ref(), value))
            .collect::<BTreeMap<_, _>>();
        let owners = graph
            .node_to_class
            .iter()
            .map(|(id, class)| (id.as_ref(), class))
            .collect::<BTreeMap<_, _>>();
        let custom = ctx
            .space
            .custom_ops
            .iter()
            .map(|op| format!("{op:?}"))
            .collect::<Vec<_>>();
        let mut digest = SnapshotDigest(Sha256::new());
        serde_json::to_writer(
            &mut digest,
            &(&classes, &nodes, &owners, &graph.roots, &custom),
        )
        .expect("search snapshot must be serializable");
        self.write(json!({
            "event": "bucket_started", "bucket": ctx.index,
            "snapshot_format": 1, "snapshot_sha256": format!("{:x}", digest.0.finalize()),
            "snapshot_scope": "ordered serialized egraph and custom-op descriptors; excludes compiler binary, data, and device state",
            "classes": classes.len(), "nodes": nodes.len(),
            "dimensions": dimensions(&ctx.representative_dyn_map),
        }));
    }

    pub(super) fn direct(
        &mut self,
        candidate: &Candidate<Duration>,
        outcome: &Outcome<Duration>,
        timed_out: bool,
        ctx: &BucketContext<'_>,
        evaluation_wall_time: Duration,
    ) {
        let program = self.program(&candidate.llir);
        let (status, metric, detail) = match outcome {
            Outcome::Measured(metric, detail) => (
                if timed_out { "timed_out" } else { "measured" },
                Some(metric.as_nanos()),
                detail,
            ),
            Outcome::Rejected(reason) => ("rejected", None, reason),
            Outcome::Invalid(reason) => ("invalid", None, reason),
        };
        self.write(json!({
            "event": "direct", "program": program, "bucket": ctx.index,
            "candidate": format!("{:?}", candidate.id),
            "sampling": format!("{:?}", candidate.sampling),
            "targeted_choice": candidate.targeted_choice.as_ref().map(|choice| json!({
                "parent": format!("{:?}", choice.parent), "class": choice.class,
                "from": choice.from, "to": choice.to, "cost_seconds": choice.cost,
            })),
            "profile_regions": candidate.profile.iter().map(|region| json!({
                "nodes": region.nodes.iter().map(|node| node.index()).collect::<Vec<_>>(),
                "cost_seconds": region.cost,
            })).collect::<Vec<_>>(),
            "bucket_indices": dimensions(ctx.bucket_indices()),
            "dimensions": dimensions(&candidate.profile_dyn_map),
            "status": status, "device_duration_ns": metric, "detail": detail,
            "candidate_timed_out": timed_out,
            "evaluation_wall_duration_ns": evaluation_wall_time.as_nanos(),
            "early_stop_hint": candidate.early_stop.map(|(best, factor)| json!({
                "best_duration_ns": best.as_nanos(), "factor": factor,
            })),
        }));
    }

    pub(super) fn deployment(
        &mut self,
        pending: &PendingFinalist<Duration>,
        ctx: &BucketContext<'_>,
        direct_rank: usize,
        result: &Result<Duration, String>,
        evaluation_wall_time: Duration,
    ) {
        let program = self.program(&pending.llir);
        self.write(json!({
            "event": "deployment", "program": program, "bucket": ctx.index,
            "dimensions": dimensions(&pending.dyn_map), "direct_rank": direct_rank,
            "direct_duration_ns": pending.metric.as_nanos(),
            "cuda_graph_duration_ns": result.as_ref().ok().map(|duration| duration.as_nanos()),
            "status": if result.is_ok() { "measured" } else { "rejected" },
            "reason": result.as_ref().err(),
            "evaluation_wall_duration_ns": evaluation_wall_time.as_nanos(),
        }));
    }

    pub(super) fn extraction_rejected(
        &mut self,
        bucket: usize,
        direct_rank: usize,
        reason: Option<&str>,
    ) {
        self.write(json!({
            "event": "deployment_extraction_rejected", "bucket": bucket,
            "direct_rank": direct_rank, "reason": reason,
        }));
    }

    pub(super) fn validation(
        &mut self,
        pending: &PendingFinalist<Duration>,
        ctx: &BucketContext<'_>,
        result: &Result<(), String>,
    ) {
        let program = self.program(&pending.llir);
        self.write(json!({
            "event": "finalist_validation", "program": program, "bucket": ctx.index,
            "deployment_rank": pending.rank,
            "status": if result.is_ok() { "accepted" } else { "rejected" },
            "reason": result.as_ref().err(),
        }));
    }

    pub(super) fn aggregate_rejected(&mut self, llirs: &[&LLIRGraph], reason: &str) {
        let programs = llirs
            .iter()
            .map(|llir| self.program(llir))
            .collect::<Vec<_>>();
        self.write(json!({"event": "aggregate_rejected", "programs": programs, "reason": reason}));
    }

    pub(super) fn selected(&mut self, selected: &[SelectedProgram]) {
        for selection in selected {
            let program = self.program(&selection.llir);
            self.write(json!({
                "event": "selected", "program": program,
                "bucket_indices": dimensions(&selection.bucket_indices),
                "dimensions": dimensions(&selection.representative_dyn_map),
            }));
        }
        self.write(json!({"event": "search_completed"}));
    }
}

struct SnapshotDigest(Sha256);

impl Write for SnapshotDigest {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/search/trace/mod.rs"]
mod tests;
