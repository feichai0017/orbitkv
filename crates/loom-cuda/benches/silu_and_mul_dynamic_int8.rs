//! CUDA SwiGLU plus dynamic per-token INT8 correctness and latency benchmark.

use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use loom_cuda::runtime::{CudaEvent, DeviceBuffer};
use loom_cuda::{CudaBackend, CudaExecutorError};
use loom_kernels::{
    silu_and_mul_dynamic_int8_bf16_reference, silu_and_mul_dynamic_int8_f16_reference,
    ContractError, DType, SiluAndMulDynamicInt8Spec,
};
use serde::Serialize;
use std::error::Error;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchDType {
    F16,
    Bf16,
}

impl BenchDType {
    const fn contract(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::Bf16 => DType::Bf16,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Validate and benchmark CUDA SiLU-and-Mul+dynamic INT8")]
struct Args {
    #[arg(long = "bench", hide = true)]
    _cargo_bench: bool,
    #[arg(long, value_enum, default_value_t = BenchDType::Bf16)]
    dtype: BenchDType,
    #[arg(long, default_value_t = 8)]
    rows: usize,
    #[arg(long, default_value_t = 4864)]
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
    output_mismatches: usize,
    maximum_int8_delta: i16,
    max_scale_abs_error: f32,
    max_scale_rel_error: f32,
}

#[derive(Debug, Serialize)]
struct Report {
    backend: &'static str,
    operator: &'static str,
    input_dtype: &'static str,
    output_dtype: &'static str,
    quantization: &'static str,
    input_layout: &'static str,
    scale_layout: &'static str,
    rows: usize,
    width: usize,
    warmup: usize,
    iterations_per_sample: usize,
    samples: usize,
    measurements: Measurements,
    claim_boundary: [&'static str; 2],
}

fn main() -> BenchResult<()> {
    let args = Args::parse();
    if args.iterations == 0 || args.samples == 0 {
        return Err("iterations and samples must be positive".into());
    }

    let measurements = match args.dtype {
        BenchDType::F16 => run_typed(
            &args,
            f16::from_f32,
            silu_and_mul_dynamic_int8_f16_reference,
            CudaBackend::silu_and_mul_dynamic_int8_f16,
        )?,
        BenchDType::Bf16 => run_typed(
            &args,
            bf16::from_f32,
            silu_and_mul_dynamic_int8_bf16_reference,
            CudaBackend::silu_and_mul_dynamic_int8_bf16,
        )?,
    };

    let report = Report {
        backend: "loom-cuda",
        operator: "silu_and_mul_dynamic_int8",
        input_dtype: args.dtype.label(),
        output_dtype: "int8",
        quantization: "symmetric-dynamic-per-token",
        input_layout: "split-half [rows, 2 * width]",
        scale_layout: "row-major [rows, 1]",
        rows: args.rows,
        width: args.width,
        warmup: args.warmup,
        iterations_per_sample: args.iterations,
        samples: args.samples,
        measurements,
        claim_boundary: [
            "This measures the Loom operator only, not an engine or GEMM.",
            "Use the Python vLLM boundary benchmark for native two-kernel A/B.",
        ],
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_typed<T, FromF32, Reference, Launch>(
    args: &Args,
    from_f32: FromF32,
    reference: Reference,
    launch: Launch,
) -> BenchResult<Measurements>
where
    T: Copy + Default,
    FromF32: Fn(f32) -> T,
    Reference:
        Fn(&[T], &mut [i8], &mut [f32], SiluAndMulDynamicInt8Spec) -> Result<(), ContractError>,
    Launch: Fn(
        &CudaBackend,
        &DeviceBuffer<T>,
        &mut DeviceBuffer<i8>,
        &mut DeviceBuffer<f32>,
        SiluAndMulDynamicInt8Spec,
    ) -> Result<(), CudaExecutorError>,
{
    let spec = SiluAndMulDynamicInt8Spec::new(args.rows, args.width, args.dtype.contract())?;
    let input = deterministic_input(spec.input_numel())
        .into_iter()
        .map(from_f32)
        .collect::<Vec<_>>();
    let mut expected_output = vec![0_i8; spec.output_numel()];
    let mut expected_scales = vec![0.0_f32; spec.scale_count()];
    reference(&input, &mut expected_output, &mut expected_scales, spec)?;

    let backend = CudaBackend::new()?;
    let input_device = DeviceBuffer::from_slice(&input)?;
    let mut output_device = DeviceBuffer::uninitialized(spec.output_numel())?;
    let mut scales_device = DeviceBuffer::uninitialized(spec.scale_count())?;
    launch(
        &backend,
        &input_device,
        &mut output_device,
        &mut scales_device,
        spec,
    )?;
    backend.stream().synchronize()?;

    let actual_output = output_device.copy_to_vec()?;
    let actual_scales = scales_device.copy_to_vec()?;
    let output_mismatches = expected_output
        .iter()
        .zip(&actual_output)
        .filter(|(expected, actual)| expected != actual)
        .count();
    let maximum_int8_delta = expected_output
        .iter()
        .zip(&actual_output)
        .map(|(&expected, &actual)| (i16::from(expected) - i16::from(actual)).abs())
        .max()
        .unwrap_or(0);
    let (max_scale_abs_error, max_scale_rel_error) =
        compare_scales(&expected_scales, &actual_scales);
    if maximum_int8_delta > 1 || max_scale_rel_error > 5.0e-4 {
        return Err(format!(
            "CUDA {} SiLU-and-Mul+INT8 correctness gate failed: mismatches={output_mismatches}, int8_delta={maximum_int8_delta}, scale_rel={max_scale_rel_error}",
            args.dtype.label()
        )
        .into());
    }

    for _ in 0..args.warmup {
        launch(
            &backend,
            &input_device,
            &mut output_device,
            &mut scales_device,
            spec,
        )?;
    }
    backend.stream().synchronize()?;

    let mut samples = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let start = CudaEvent::new()?;
        let end = CudaEvent::new()?;
        start.record(backend.stream())?;
        for _ in 0..args.iterations {
            launch(
                &backend,
                &input_device,
                &mut output_device,
                &mut scales_device,
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
        output_mismatches,
        maximum_int8_delta,
        max_scale_abs_error,
        max_scale_rel_error,
    })
}

fn deterministic_input(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index.wrapping_mul(19) % 127) as f32 - 63.0) / 21.0)
        .collect()
}

fn compare_scales(expected: &[f32], actual: &[f32]) -> (f32, f32) {
    expected
        .iter()
        .zip(actual)
        .fold((0.0_f32, 0.0_f32), |(max_abs, max_rel), (&lhs, &rhs)| {
            let absolute = (lhs - rhs).abs();
            let relative = absolute / lhs.abs().max(1.0e-12);
            (max_abs.max(absolute), max_rel.max(relative))
        })
}
