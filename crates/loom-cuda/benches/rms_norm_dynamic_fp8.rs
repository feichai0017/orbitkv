//! CUDA RMSNorm plus dynamic per-token FP8 correctness and latency benchmark.

use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use loom_cuda::runtime::{CudaEvent, DeviceBuffer};
use loom_cuda::{CudaBackend, CudaExecutorError};
use loom_kernels::{
    fp8_e4m3fn_to_f32, rms_norm_dynamic_fp8_bf16_reference, rms_norm_dynamic_fp8_f16_reference,
    rms_norm_dynamic_fp8_f32_reference, ContractError, DType, RmsNormDynamicFp8Spec,
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
#[command(about = "Validate and benchmark CUDA RMSNorm+dynamic FP8")]
struct Args {
    #[arg(long = "bench", hide = true)]
    _cargo_bench: bool,
    #[arg(long, value_enum, default_value_t = BenchDType::Bf16)]
    dtype: BenchDType,
    #[arg(long, default_value_t = 8)]
    rows: usize,
    #[arg(long, default_value_t = 4096)]
    hidden_size: usize,
    #[arg(long, default_value_t = 1.0e-5)]
    epsilon: f32,
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
    output_byte_mismatches: usize,
    max_scale_abs_error: f32,
    max_scale_rel_error: f32,
    max_dequantized_abs_error: f32,
}

#[derive(Debug, Serialize)]
struct Report {
    backend: &'static str,
    operator: &'static str,
    input_dtype: &'static str,
    output_dtype: &'static str,
    quantization: &'static str,
    scale_semantics: &'static str,
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    warmup: usize,
    iterations_per_sample: usize,
    samples: usize,
    latency: LatencySummary,
    output_byte_mismatches: usize,
    max_scale_abs_error: f32,
    max_scale_rel_error: f32,
    max_dequantized_abs_error: f32,
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
            rms_norm_dynamic_fp8_f32_reference,
            CudaBackend::rms_norm_dynamic_fp8_f32,
        )?,
        BenchDType::F16 => run_typed(
            &args,
            f16::from_f32,
            rms_norm_dynamic_fp8_f16_reference,
            CudaBackend::rms_norm_dynamic_fp8_f16,
        )?,
        BenchDType::Bf16 => run_typed(
            &args,
            bf16::from_f32,
            rms_norm_dynamic_fp8_bf16_reference,
            CudaBackend::rms_norm_dynamic_fp8_bf16,
        )?,
    };

    let report = Report {
        backend: "loom-cuda",
        operator: "rms_norm_dynamic_fp8",
        input_dtype: args.dtype.label(),
        output_dtype: "fp8_e4m3fn",
        quantization: "symmetric-dynamic-per-token",
        scale_semantics: "normalized_value ~= fp8(output) * scale; scale shape [rows,1]",
        rows: args.rows,
        hidden_size: args.hidden_size,
        epsilon: args.epsilon,
        warmup: args.warmup,
        iterations_per_sample: args.iterations,
        samples: args.samples,
        latency: measurements.latency,
        output_byte_mismatches: measurements.output_byte_mismatches,
        max_scale_abs_error: measurements.max_scale_abs_error,
        max_scale_rel_error: measurements.max_scale_rel_error,
        max_dequantized_abs_error: measurements.max_dequantized_abs_error,
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
        Fn(&[T], &[T], &mut [u8], &mut [f32], RmsNormDynamicFp8Spec) -> Result<(), ContractError>,
    Launch: Fn(
        &CudaBackend,
        &DeviceBuffer<T>,
        &DeviceBuffer<T>,
        &mut DeviceBuffer<u8>,
        &mut DeviceBuffer<f32>,
        RmsNormDynamicFp8Spec,
    ) -> Result<(), CudaExecutorError>,
{
    let spec = RmsNormDynamicFp8Spec::new(
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
    let mut expected_output = vec![0_u8; spec.numel()];
    let mut expected_scales = vec![0.0_f32; spec.scale_count()];
    reference(
        &input,
        &weight,
        &mut expected_output,
        &mut expected_scales,
        spec,
    )?;

    let backend = CudaBackend::new()?;
    let input_device = DeviceBuffer::from_slice(&input)?;
    let weight_device = DeviceBuffer::from_slice(&weight)?;
    let mut output_device = DeviceBuffer::uninitialized(spec.numel())?;
    let mut scales_device = DeviceBuffer::uninitialized(spec.scale_count())?;
    launch(
        &backend,
        &input_device,
        &weight_device,
        &mut output_device,
        &mut scales_device,
        spec,
    )?;
    backend.stream().synchronize()?;

    let actual_output = output_device.copy_to_vec()?;
    let actual_scales = scales_device.copy_to_vec()?;
    let output_byte_mismatches = expected_output
        .iter()
        .zip(&actual_output)
        .filter(|(expected, actual)| expected != actual)
        .count();
    let (max_scale_abs_error, max_scale_rel_error) =
        compare_scales(&expected_scales, &actual_scales);
    let max_dequantized_abs_error = compare_dequantized(
        &expected_output,
        &expected_scales,
        &actual_output,
        &actual_scales,
        spec.hidden_size(),
    );
    if max_scale_rel_error > 5.0e-3 || max_dequantized_abs_error > 5.0e-2 {
        return Err(format!(
            "CUDA {} RMSNorm+FP8 correctness gate failed: byte_mismatches={output_byte_mismatches}, scale_rel={max_scale_rel_error}, dequant_abs={max_dequantized_abs_error}",
            args.dtype.label()
        )
        .into());
    }

    for _ in 0..args.warmup {
        launch(
            &backend,
            &input_device,
            &weight_device,
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
                &weight_device,
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
        output_byte_mismatches,
        max_scale_abs_error,
        max_scale_rel_error,
        max_dequantized_abs_error,
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

fn compare_dequantized(
    expected_output: &[u8],
    expected_scales: &[f32],
    actual_output: &[u8],
    actual_scales: &[f32],
    hidden_size: usize,
) -> f32 {
    expected_output
        .chunks_exact(hidden_size)
        .zip(actual_output.chunks_exact(hidden_size))
        .zip(expected_scales.iter().zip(actual_scales))
        .fold(
            0.0_f32,
            |maximum, ((expected, actual), (&lhs_scale, &rhs_scale))| {
                expected
                    .iter()
                    .zip(actual)
                    .fold(maximum, |row_maximum, (&lhs, &rhs)| {
                        let lhs = fp8_e4m3fn_to_f32(lhs) * lhs_scale;
                        let rhs = fp8_e4m3fn_to_f32(rhs) * rhs_scale;
                        row_maximum.max((lhs - rhs).abs())
                    })
            },
        )
}
