use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aletheia_contracts::{
    CaptureMode, HardwareFingerprint, HardwareTarget, InclusiveRangeU32, ModelIdentity, NumericalEvidence,
    PerformanceEvidence, Phase, PhysicalPlan, ProviderStep, QualificationCertificate, QualificationStatus,
    ResilienceEvidence, ResourceEvidence, ResourceRequirements, SCHEMA_VERSION, ServiceLevelObjective, WorkloadDomain,
    WorkloadPoint,
};
use aletheia_control::{PlanRegistry, SelectionRequest};
use aletheia_executor::reference::ReferenceProvider;
use aletheia_executor::{ExecutionState, Executor, Value};
use clap::{Parser, Subcommand};
use sha2::{Digest as _, Sha256};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(name = "aletheia-rt", about = "AletheiaRT proof-carrying inference tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate the structure of a physical plan.
    ValidatePlan { plan: PathBuf },
    /// Validate a qualification certificate against its exact plan digest.
    ValidateCertificate { plan: PathBuf, certificate: PathBuf },
    /// Validate every bundle and fallback edge in a plan registry directory.
    ValidateRegistry { registry: PathBuf },
    /// Run the CPU-only M0 selection and execution demonstration.
    Demo,
}

fn main() -> std::process::ExitCode {
    match run(Cli::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::ValidatePlan { plan } => {
            let plan: PhysicalPlan = read_json(&plan)?;
            plan.validate()?;
            println!("{}", plan.digest()?);
        }
        Command::ValidateCertificate { plan, certificate } => {
            let plan: PhysicalPlan = read_json(&plan)?;
            let certificate: QualificationCertificate = read_json(&certificate)?;
            plan.validate()?;
            certificate.validate_for(&plan)?;
            println!("qualified {}", plan.id);
        }
        Command::ValidateRegistry { registry } => validate_registry(&registry)?,
        Command::Demo => demo()?,
    }
    Ok(())
}

fn validate_registry(path: &Path) -> Result<()> {
    let mut pending = BTreeMap::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let directory_name = entry.file_name().to_string_lossy().into_owned();
        let plan: PhysicalPlan = read_json(&entry.path().join("plan.json"))?;
        let certificate: QualificationCertificate = read_json(&entry.path().join("certificate.json"))?;
        if directory_name != plan.id {
            return Err(format!("bundle directory {directory_name:?} does not match plan ID {:?}", plan.id).into());
        }
        if pending.insert(plan.id.clone(), (plan, certificate)).is_some() {
            return Err(format!("duplicate plan ID {directory_name:?}").into());
        }
        let (plan, _) = pending.get(&directory_name).expect("just inserted plan");
        for step in &plan.steps {
            let artifact = entry.path().join("artifacts").join(&step.artifact_sha256);
            let metadata = fs::symlink_metadata(&artifact)?;
            if !metadata.file_type().is_file() {
                return Err(format!("artifact {} is not a regular file", artifact.display()).into());
            }
            let actual = hex::encode(Sha256::digest(fs::read(&artifact)?));
            if actual != step.artifact_sha256 {
                return Err(format!("artifact {} digest mismatch: got {actual}", artifact.display()).into());
            }
        }
    }
    if pending.is_empty() {
        return Err("registry contains no plan bundles".into());
    }

    let total = pending.len();
    let mut registry = PlanRegistry::new();
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .filter(|(_, (plan, _))| plan.fallback_plan_ids.iter().all(|fallback| registry.get(fallback).is_some()))
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        if ready.is_empty() {
            let unresolved = pending
                .values()
                .map(|(plan, _)| format!("{} -> {:?}", plan.id, plan.fallback_plan_ids))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("registry has missing or cyclic fallback dependencies: {unresolved}").into());
        }
        for id in ready {
            let (plan, certificate) = pending.remove(&id).expect("ready plan must remain pending");
            registry.register(plan, certificate)?;
        }
    }
    println!("validated {total} qualified plan bundle(s)");
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn demo() -> Result<()> {
    let safe = demo_plan("safe", "identity", Vec::new());
    let fast = demo_plan("fast", "identity", vec![safe.id.clone()]);
    let mut registry = PlanRegistry::new();
    registry.register(safe.clone(), demo_certificate(&safe, 100.0, 80)?)?;
    registry.register(fast.clone(), demo_certificate(&fast, 200.0, 60)?)?;

    let hardware = demo_hardware();
    let workload = WorkloadPoint { phase: Phase::Decode, batch: 1, rows_per_sequence: 1, context_tokens: 32 };
    let request = SelectionRequest {
        model: demo_model(),
        hardware,
        workload: workload.clone(),
        slo: ServiceLevelObjective {
            max_p99_latency_micros: 100,
            min_goodput_tokens_per_second: 1.0,
            max_device_memory_bytes: 1024,
            max_abs_error: 0.0,
            min_output_agreement: 1.0,
        },
        provider_preference: vec!["reference".into()],
        available_providers: BTreeSet::from(["reference".into()]),
    };
    let selection = registry.select(&request)?;
    let selected_plan = selection.primary.plan.id.clone();
    let mut executor = Executor::new();
    executor.register_provider(Arc::new(ReferenceProvider));
    let prepared = executor.prepare(selection)?;
    let mut state = ExecutionState::new(workload);
    state.values.insert("input".into(), Value::F32(vec![1.0, 2.0, 3.0]));
    let report = prepared.execute(&state)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "selected_plan": selected_plan,
            "executed_plan": report.plan_id,
            "failed_attempts": report.failed_attempts.len(),
            "output": format!("{:?}", report.state.values.get("output")),
        }))?
    );
    Ok(())
}

fn hash(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn demo_model() -> ModelIdentity {
    ModelIdentity { name: "control-plane-demo".into(), semantics_sha256: hash('a') }
}

fn demo_hardware() -> HardwareFingerprint {
    HardwareFingerprint {
        accelerator: "cpu".into(),
        architecture: "reference".into(),
        device_count: 1,
        device_memory_bytes: 1024,
        driver: "none".into(),
        runtime: "rust".into(),
    }
}

fn demo_domain() -> WorkloadDomain {
    WorkloadDomain {
        phases: vec![Phase::Decode],
        batch: InclusiveRangeU32 { min: 1, max: 8 },
        rows_per_sequence: InclusiveRangeU32 { min: 1, max: 1 },
        context_tokens: InclusiveRangeU32 { min: 0, max: 4096 },
    }
}

fn demo_plan(id: &str, operation: &str, fallback_plan_ids: Vec<String>) -> PhysicalPlan {
    let mut config = BTreeMap::new();
    if operation == "add_scalar" {
        config.insert("scalar".into(), "1.0".into());
    }
    PhysicalPlan {
        schema_version: SCHEMA_VERSION,
        id: id.into(),
        model: demo_model(),
        hardware: HardwareTarget {
            accelerator: "cpu".into(),
            architecture: "reference".into(),
            device_count: 1,
            min_device_memory_bytes: 1,
        },
        workload: demo_domain(),
        resources: ResourceRequirements {
            peak_device_memory_bytes: 128,
            workspace_bytes: 0,
            stable_addresses: false,
            capture: CaptureMode::Eager,
        },
        steps: vec![ProviderStep {
            id: "forward".into(),
            provider: "reference".into(),
            operation: operation.into(),
            artifact_sha256: hash('b'),
            inputs: vec!["input".into()],
            outputs: vec!["output".into()],
            config,
        }],
        fallback_plan_ids,
        labels: BTreeMap::from([("tier".into(), "demo".into())]),
    }
}

fn demo_certificate(plan: &PhysicalPlan, goodput: f64, p99: u64) -> Result<QualificationCertificate> {
    Ok(QualificationCertificate {
        schema_version: SCHEMA_VERSION,
        plan_id: plan.id.clone(),
        plan_sha256: plan.digest()?,
        status: QualificationStatus::Qualified,
        hardware: demo_hardware(),
        workload: demo_domain(),
        numerical: NumericalEvidence {
            oracle: "reference".into(),
            cases: 32,
            atol: 0.0,
            rtol: 0.0,
            max_abs_error: 0.0,
            min_output_agreement: 1.0,
        },
        performance: PerformanceEvidence {
            workload_sha256: hash('c'),
            samples: 32,
            p50_latency_micros: p99 / 2,
            p99_latency_micros: p99,
            goodput_tokens_per_second: goodput,
        },
        resources: ResourceEvidence { peak_device_memory_bytes: 64, peak_workspace_bytes: 0 },
        resilience: ResilienceEvidence { soak_seconds: 1, passed_faults: vec!["provider_failure".into()] },
        software: BTreeMap::from([("aletheia-rt".into(), env!("CARGO_PKG_VERSION").into())]),
    })
}
