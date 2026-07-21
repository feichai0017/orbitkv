//! CUDA RMSNorm correctness and latency benchmark.

use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use loom_cuda::runtime::{CudaEvent, DeviceBuffer};
use loom_cuda::{CudaBackend, CudaExecutorError};
use loom_kernels::{
    rms_norm_bf16_reference, rms_norm_f16_reference, rms_norm_f32_reference, ContractError, DType,
    RmsNormSpec,
};
use serde::Serialize;
use std::error::Error;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchDType {
    F32,
    F16,
    Bf16,
}

impl BenchDType {
    const fn contract(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::Bf16,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    const fn max_abs_error_gate(self) -> f32 {
        match self {
            Self::F32 => 5.0e-5,
            Self::F16 => 4.0e-3,
            Self::Bf16 => 4.0e-2,
        }
    }

    const fn vectorization(self, hidden_size: usize) -> &'static str {
        match self {
            Self::F32 => "scalar",
            Self::F16 | Self::Bf16 if hidden_size.is_multiple_of(2) => "pair",
            Self::F16 | Self::Bf16 => "scalar-fallback",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Validate and benchmark Loom Kernels CUDA RMSNorm")]
struct Args {
    #[arg(long = "bench", hide = true)]
    _cargo_bench: bool,
    #[arg(long, value_enum, default_value_t = BenchDType::F32)]
    dtype: BenchDType,
    #[arg(long, default_value_t = 8)]
    rows: usize,
    #[arg(long, default_value_t = 4096)]
    hidden_size: usize,
    #[arg(long, default_value_t = 1.0e-5)]
    epsilon: f32,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    #[arg(long, default_value_t = 20)]
    samples: usize,
}

#[derive(Debug, Serialize)]
struct LatencySummary {
    minimum_us: f64,
    median_us: f64,
    maximum_us: f64,
}

#[derive(Debug, Serialize)]
struct Measurements {
    latency: LatencySummary,
    max_abs_error: f32,
    max_rel_error: f32,
}

#[derive(Debug, Serialize)]
struct Report {
    backend: &'static str,
    operator: &'static str,
    dtype: &'static str,
    vectorization: &'static str,
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    warmup: usize,
    iterations_per_sample: usize,
    samples: usize,
    max_abs_error_gate: f32,
    latency: LatencySummary,
    max_abs_error: f32,
    max_rel_error: f32,
}

fn main() -> BenchResult<()> {
    let args = Args::parse();
    if args.iterations == 0 || args.samples == 0 {
        return Err("iterations and samples must be positive".into());
    }

    let measurements = match args.dtype {
        BenchDType::F32 => run_typed(
            &args,
            |value| value,
            |value| value,
            rms_norm_f32_reference,
            CudaBackend::rms_norm_f32,
        )?,
        BenchDType::F16 => run_typed(
            &args,
            f16::from_f32,
            f16::to_f32,
            rms_norm_f16_reference,
            CudaBackend::rms_norm_f16,
        )?,
        BenchDType::Bf16 => run_typed(
            &args,
            bf16::from_f32,
            bf16::to_f32,
            rms_norm_bf16_reference,
            CudaBackend::rms_norm_bf16,
        )?,
    };

    let report = Report {
        backend: "loom-cuda",
        operator: "rms_norm",
        dtype: args.dtype.label(),
        vectorization: args.dtype.vectorization(args.hidden_size),
        rows: args.rows,
        hidden_size: args.hidden_size,
        epsilon: args.epsilon,
        warmup: args.warmup,
        iterations_per_sample: args.iterations,
        samples: args.samples,
        max_abs_error_gate: args.dtype.max_abs_error_gate(),
        latency: measurements.latency,
        max_abs_error: measurements.max_abs_error,
        max_rel_error: measurements.max_rel_error,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_typed<T, FromF32, ToF32, Reference, Launch>(
    args: &Args,
    from_f32: FromF32,
    to_f32: ToF32,
    reference: Reference,
    launch: Launch,
) -> BenchResult<Measurements>
where
    T: Copy + Default,
    FromF32: Fn(f32) -> T,
    ToF32: Fn(T) -> f32,
    Reference: Fn(&[T], &[T], &mut [T], RmsNormSpec) -> Result<(), ContractError>,
    Launch: Fn(
        &CudaBackend,
        &DeviceBuffer<T>,
        &DeviceBuffer<T>,
        &mut DeviceBuffer<T>,
        RmsNormSpec,
    ) -> Result<(), CudaExecutorError>,
{
    let spec = RmsNormSpec::new(
        args.rows,
        args.hidden_size,
        args.epsilon,
        args.dtype.contract(),
    )?;
    let input = deterministic_input(spec.numel())
        .into_iter()
        .map(&from_f32)
        .collect::<Vec<_>>();
    let weight = deterministic_weight(spec.hidden_size())
        .into_iter()
        .map(&from_f32)
        .collect::<Vec<_>>();
    let mut expected = vec![T::default(); spec.numel()];
    reference(&input, &weight, &mut expected, spec)?;

    let backend = CudaBackend::new()?;
    let input_device = DeviceBuffer::from_slice(&input)?;
    let weight_device = DeviceBuffer::from_slice(&weight)?;
    let mut output_device = DeviceBuffer::uninitialized(spec.numel())?;

    for _ in 0..args.warmup {
        launch(
            &backend,
            &input_device,
            &weight_device,
            &mut output_device,
            spec,
        )?;
    }
    backend.stream().synchronize()?;

    let actual = output_device.copy_to_vec()?;
    let (max_abs_error, max_rel_error) = compare(&expected, &actual, to_f32);
    let max_abs_error_gate = args.dtype.max_abs_error_gate();
    if !max_abs_error.is_finite() || max_abs_error > max_abs_error_gate {
        return Err(format!(
            "CUDA {} RMSNorm correctness gate failed: max_abs_error={max_abs_error}, max_rel_error={max_rel_error}, gate={max_abs_error_gate}",
            args.dtype.label()
        )
        .into());
    }

    let mut samples = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let start = CudaEvent::new()?;
        let end = CudaEvent::new()?;
        start.record(backend.stream())?;
        for _ in 0..args.iterations {
            launch(
                &backend,
                &input_device,
                &weight_device,
                &mut output_device,
                spec,
            )?;
        }
        end.record(backend.stream())?;
        end.synchronize()?;
        samples.push(f64::from(start.elapsed_ms(&end)?) * 1_000.0 / args.iterations as f64);
    }
    samples.sort_by(f64::total_cmp);

    Ok(Measurements {
        latency: LatencySummary {
            minimum_us: samples[0],
            median_us: samples[samples.len() / 2],
            maximum_us: samples[samples.len() - 1],
        },
        max_abs_error,
        max_rel_error,
    })
}

fn deterministic_input(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index.wrapping_mul(17) % 101) as f32 - 50.0) / 25.0)
        .collect()
}

fn deterministic_weight(hidden_size: usize) -> Vec<f32> {
    (0..hidden_size)
        .map(|index| 0.5 + (index.wrapping_mul(13) % 37) as f32 / 37.0)
        .collect()
}

fn compare<T, ToF32>(expected: &[T], actual: &[T], to_f32: ToF32) -> (f32, f32)
where
    T: Copy,
    ToF32: Fn(T) -> f32,
{
    expected
        .iter()
        .zip(actual)
        .fold((0.0_f32, 0.0_f32), |(max_abs, max_rel), (&lhs, &rhs)| {
            let lhs = to_f32(lhs);
            let rhs = to_f32(rhs);
            let absolute = (lhs - rhs).abs();
            let relative = absolute / lhs.abs().max(1.0e-8);
            (max_abs.max(absolute), max_rel.max(relative))
        })
}
