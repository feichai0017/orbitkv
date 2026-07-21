//! CUDA SiLU-and-Mul correctness and latency benchmark.

use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use loom_cuda::runtime::{CudaEvent, DeviceBuffer};
use loom_cuda::{CudaBackend, CudaExecutorError};
use loom_kernels::{
    silu_and_mul_bf16_reference, silu_and_mul_f16_reference, silu_and_mul_f32_reference,
    ContractError, DType, SiluAndMulSpec,
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
}

#[derive(Debug, Parser)]
#[command(about = "Validate and benchmark CUDA SiLU-and-Mul")]
struct Args {
    #[arg(long = "bench", hide = true)]
    _cargo_bench: bool,
    #[arg(long, value_enum, default_value_t = BenchDType::Bf16)]
    dtype: BenchDType,
    #[arg(long, default_value_t = 8)]
    rows: usize,
    #[arg(long, default_value_t = 11008)]
    width: usize,
    #[arg(long, default_value_t = 100)]
    warmup: usize,
    #[arg(long, default_value_t = 1000)]
    iterations: usize,
    #[arg(long, default_value_t = 9)]
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
    input_layout: &'static str,
    dtype: &'static str,
    rows: usize,
    width: usize,
    warmup: usize,
    iterations_per_sample: usize,
    samples: usize,
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
            |value: &f32| *value,
            silu_and_mul_f32_reference,
            CudaBackend::silu_and_mul_f32,
        )?,
        BenchDType::F16 => run_typed(
            &args,
            f16::from_f32,
            |value: &f16| value.to_f32(),
            silu_and_mul_f16_reference,
            CudaBackend::silu_and_mul_f16,
        )?,
        BenchDType::Bf16 => run_typed(
            &args,
            bf16::from_f32,
            |value: &bf16| value.to_f32(),
            silu_and_mul_bf16_reference,
            CudaBackend::silu_and_mul_bf16,
        )?,
    };

    let report = Report {
        backend: "loom-cuda",
        operator: "silu_and_mul",
        input_layout: "split-half [rows, 2 * width]",
        dtype: args.dtype.label(),
        rows: args.rows,
        width: args.width,
        warmup: args.warmup,
        iterations_per_sample: args.iterations,
        samples: args.samples,
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
    ToF32: Fn(&T) -> f32,
    Reference: Fn(&[T], &mut [T], SiluAndMulSpec) -> Result<(), ContractError>,
    Launch: Fn(
        &CudaBackend,
        &DeviceBuffer<T>,
        &mut DeviceBuffer<T>,
        SiluAndMulSpec,
    ) -> Result<(), CudaExecutorError>,
{
    let spec = SiluAndMulSpec::new(args.rows, args.width, args.dtype.contract())?;
    let input = deterministic_input(spec.input_numel())
        .into_iter()
        .map(&from_f32)
        .collect::<Vec<_>>();
    let mut expected = vec![T::default(); spec.output_numel()];
    reference(&input, &mut expected, spec)?;

    let backend = CudaBackend::new()?;
    let input_device = DeviceBuffer::from_slice(&input)?;
    let mut output_device = DeviceBuffer::uninitialized(spec.output_numel())?;
    launch(&backend, &input_device, &mut output_device, spec)?;
    backend.stream().synchronize()?;
    let actual = output_device.copy_to_vec()?;
    let (max_abs_error, max_rel_error) = compare(&expected, &actual, &to_f32);
    let tolerance = match args.dtype {
        BenchDType::F32 => 2.0e-5,
        BenchDType::F16 => 2.0e-3,
        BenchDType::Bf16 => 2.0e-2,
    };
    if max_abs_error > tolerance {
        return Err(format!(
            "CUDA {} SiLU-and-Mul correctness gate failed: abs={max_abs_error}, rel={max_rel_error}",
            args.dtype.label()
        )
        .into());
    }

    for _ in 0..args.warmup {
        launch(&backend, &input_device, &mut output_device, spec)?;
    }
    backend.stream().synchronize()?;

    let mut samples = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let start = CudaEvent::new()?;
        let end = CudaEvent::new()?;
        start.record(backend.stream())?;
        for _ in 0..args.iterations {
            launch(&backend, &input_device, &mut output_device, spec)?;
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
        .map(|index| ((index.wrapping_mul(19) % 127) as f32 - 63.0) / 21.0)
        .collect()
}

fn compare<T, ToF32>(expected: &[T], actual: &[T], to_f32: &ToF32) -> (f32, f32)
where
    ToF32: Fn(&T) -> f32,
{
    expected
        .iter()
        .zip(actual)
        .fold((0.0_f32, 0.0_f32), |(max_abs, max_rel), (lhs, rhs)| {
            let lhs = to_f32(lhs);
            let rhs = to_f32(rhs);
            let absolute = (lhs - rhs).abs();
            let relative = absolute / lhs.abs().max(1.0e-8);
            (max_abs.max(absolute), max_rel.max(relative))
        })
}
