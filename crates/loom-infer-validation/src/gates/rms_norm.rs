use cuda_core::{CudaContext, DeviceBuffer};
use half::{bf16, f16};
use loom_infer::{
    DType, RmsNormSpec, rms_norm_bf16_reference, rms_norm_f16_reference, rms_norm_f32_reference,
};
use loom_infer_cuda::command::{BindingElement, CommandQueue, CommandScope};
use loom_infer_cuda::rms_norm::{
    RmsNormArgs, RmsNormBf16Plan, RmsNormEnqueueError, RmsNormF16Plan, RmsNormKernelPath,
    RmsNormPlanError, RmsNormProvider,
};
use std::sync::Arc;

const F32_MAX_ABS_ERROR: f32 = 5.0e-5;
const F16_MAX_ABS_ERROR: f32 = 4.0e-3;
const BF16_MAX_ABS_ERROR: f32 = 4.0e-2;
const LOW_PRECISION_MAX_ULP_ERROR: u32 = 2;

fn fixture(rows: usize, hidden_size: usize) -> (Vec<f32>, Vec<f32>) {
    let input = (0..rows * hidden_size)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let weight = (0..hidden_size)
        .map(|index| 0.5 + (index * 13 % 37) as f32 / 37.0)
        .collect();
    (input, weight)
}

trait LowPrecisionPlan<T: LowPrecision> {
    fn kernel_path(&self) -> RmsNormKernelPath;

    fn enqueue(
        &self,
        scope: &mut CommandScope<'_>,
        args: RmsNormArgs<T>,
    ) -> Result<(), RmsNormEnqueueError>;
}

trait LowPrecision: BindingElement + Copy + std::fmt::Debug + 'static {
    type Plan: LowPrecisionPlan<Self>;

    const DTYPE: DType;
    const NAME: &'static str;
    const MAX_ABS_ERROR: f32;

    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
    fn bits(self) -> u16;
    fn plan(provider: &RmsNormProvider, spec: RmsNormSpec) -> Result<Self::Plan, RmsNormPlanError>;
    fn reference(
        input: &[Self],
        weight: &[Self],
        output: &mut [Self],
        spec: RmsNormSpec,
    ) -> Result<(), loom_infer::ContractError>;
}

impl LowPrecisionPlan<f16> for RmsNormF16Plan {
    fn kernel_path(&self) -> RmsNormKernelPath {
        self.kernel_path()
    }

    fn enqueue(
        &self,
        scope: &mut CommandScope<'_>,
        args: RmsNormArgs<f16>,
    ) -> Result<(), RmsNormEnqueueError> {
        self.enqueue_into(scope, args)
    }
}

impl LowPrecision for f16 {
    type Plan = RmsNormF16Plan;

    const DTYPE: DType = DType::F16;
    const NAME: &'static str = "f16";
    const MAX_ABS_ERROR: f32 = F16_MAX_ABS_ERROR;

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn bits(self) -> u16 {
        self.to_bits()
    }

    fn plan(provider: &RmsNormProvider, spec: RmsNormSpec) -> Result<Self::Plan, RmsNormPlanError> {
        provider.plan_f16(spec)
    }

    fn reference(
        input: &[Self],
        weight: &[Self],
        output: &mut [Self],
        spec: RmsNormSpec,
    ) -> Result<(), loom_infer::ContractError> {
        rms_norm_f16_reference(input, weight, output, spec)
    }
}

impl LowPrecisionPlan<bf16> for RmsNormBf16Plan {
    fn kernel_path(&self) -> RmsNormKernelPath {
        self.kernel_path()
    }

    fn enqueue(
        &self,
        scope: &mut CommandScope<'_>,
        args: RmsNormArgs<bf16>,
    ) -> Result<(), RmsNormEnqueueError> {
        self.enqueue_into(scope, args)
    }
}

impl LowPrecision for bf16 {
    type Plan = RmsNormBf16Plan;

    const DTYPE: DType = DType::Bf16;
    const NAME: &'static str = "bf16";
    const MAX_ABS_ERROR: f32 = BF16_MAX_ABS_ERROR;

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn bits(self) -> u16 {
        self.to_bits()
    }

    fn plan(provider: &RmsNormProvider, spec: RmsNormSpec) -> Result<Self::Plan, RmsNormPlanError> {
        provider.plan_bf16(spec)
    }

    fn reference(
        input: &[Self],
        weight: &[Self],
        output: &mut [Self],
        spec: RmsNormSpec,
    ) -> Result<(), loom_infer::ContractError> {
        rms_norm_bf16_reference(input, weight, output, spec)
    }
}

#[derive(Clone, Copy, Debug)]
struct ErrorStats {
    max_abs: f32,
    max_rel: f32,
    max_ulp: u32,
    max_abs_index: usize,
}

fn compare_low_precision<T: LowPrecision>(
    actual: &[T],
    expected: &[T],
) -> Result<ErrorStats, Box<dyn std::error::Error>> {
    let mut stats = ErrorStats {
        max_abs: 0.0,
        max_rel: 0.0,
        max_ulp: 0,
        max_abs_index: 0,
    };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual_f32 = actual.to_f32();
        let expected_f32 = expected.to_f32();
        if !actual_f32.is_finite() {
            return Err(format!("{} output at index {index} is not finite", T::NAME).into());
        }
        let absolute = (actual_f32 - expected_f32).abs();
        let relative = absolute / expected_f32.abs().max(f32::MIN_POSITIVE);
        let ulp = ordered_low_precision_bits(actual.bits())
            .abs_diff(ordered_low_precision_bits(expected.bits()));
        if absolute > stats.max_abs {
            stats.max_abs = absolute;
            stats.max_abs_index = index;
        }
        stats.max_rel = stats.max_rel.max(relative);
        stats.max_ulp = stats.max_ulp.max(ulp);
    }
    Ok(stats)
}

fn ordered_low_precision_bits(bits: u16) -> u32 {
    if bits & 0x8000 == 0 {
        0x8000 + u32::from(bits)
    } else {
        0x8000 - u32::from(bits & 0x7fff)
    }
}

fn check_low_precision_case<T: LowPrecision>(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
    rows: usize,
    hidden_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = queue.stream().clone();
    let spec = RmsNormSpec::new(rows, hidden_size, 1.0e-5, T::DTYPE)?;
    let plan = T::plan(provider, spec)?;
    let expected_path = if hidden_size.is_multiple_of(2) {
        RmsNormKernelPath::Packed2
    } else {
        RmsNormKernelPath::Scalar
    };
    if plan.kernel_path() != expected_path {
        return Err(format!(
            "{} plan selected {:?}, expected {expected_path:?}",
            T::NAME,
            plan.kernel_path()
        )
        .into());
    }

    let (input_f32, weight_f32) = fixture(rows, hidden_size);
    let input_host = input_f32.into_iter().map(T::from_f32).collect::<Vec<_>>();
    let weight_host = weight_f32.into_iter().map(T::from_f32).collect::<Vec<_>>();
    let mut expected = vec![T::from_f32(0.0); spec.numel()];
    T::reference(&input_host, &weight_host, &mut expected, spec)?;

    let input = Arc::new(DeviceBuffer::from_host(&stream, &input_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(&stream, &weight_host)?);
    let output = DeviceBuffer::<T>::zeroed(&stream, spec.numel())?;
    let mut bindings = queue.bindings(3)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue(
        &mut scope,
        RmsNormArgs::new(input_handle, weight_handle, output_handle.write()),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err(format!("{} single scope retained the wrong command count", T::NAME).into());
    }
    let mut bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;

    let actual = output.to_host_vec(&stream)?;
    let stats = compare_low_precision(&actual, &expected)?;
    println!(
        "rms_norm_{} rows={rows} hidden={hidden_size} path={:?} stream=non_default \
         finite=true max_abs_error={:.9e} max_rel_error={:.9e} max_ulp_error={} \
         max_error_index={}",
        T::NAME,
        plan.kernel_path(),
        stats.max_abs,
        stats.max_rel,
        stats.max_ulp,
        stats.max_abs_index,
    );
    if stats.max_abs > T::MAX_ABS_ERROR || stats.max_ulp > LOW_PRECISION_MAX_ULP_ERROR {
        return Err(format!(
            "{} RMSNorm exceeded error gate: abs={:.9e}/{:.9e}, ulp={}/{}",
            T::NAME,
            stats.max_abs,
            T::MAX_ABS_ERROR,
            stats.max_ulp,
            LOW_PRECISION_MAX_ULP_ERROR,
        )
        .into());
    }
    Ok(())
}

fn check_f32_case(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
    rows: usize,
    hidden_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = queue.stream().clone();
    let spec = RmsNormSpec::new(rows, hidden_size, 1.0e-5, DType::F32)?;
    let plan = provider.plan_f32(spec)?;
    let (input_host, weight_host) = fixture(rows, hidden_size);
    let mut expected = vec![0.0_f32; spec.numel()];
    rms_norm_f32_reference(&input_host, &weight_host, &mut expected, spec)?;

    let input = Arc::new(DeviceBuffer::from_host(&stream, &input_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(&stream, &weight_host)?);
    let output = DeviceBuffer::<f32>::zeroed(&stream, spec.numel())?;
    let mut bindings = queue.bindings(3)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        RmsNormArgs::new(input_handle, weight_handle, output_handle.write()),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("single RMSNorm scope did not retain exactly one launch".into());
    }
    let mut bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;

    let actual = output.to_host_vec(&stream)?;
    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    let mut max_error_index = 0_usize;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        if !actual.is_finite() {
            return Err(format!(
                "non-finite RMSNorm output at index {index} for ({rows}, {hidden_size})"
            )
            .into());
        }
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
        if absolute > max_abs_error {
            max_abs_error = absolute;
            max_error_index = index;
        }
        max_rel_error = max_rel_error.max(relative);
    }

    println!(
        "rms_norm_f32 rows={rows} hidden={hidden_size} stream=non_default finite=true \
         max_abs_error={max_abs_error:.9e} max_rel_error={max_rel_error:.9e} \
         max_error_index={max_error_index}"
    );
    if max_abs_error > F32_MAX_ABS_ERROR {
        return Err(format!(
            "RMSNorm error {max_abs_error:.9e} exceeds {F32_MAX_ABS_ERROR:.9e} \
             for ({rows}, {hidden_size}) at index {max_error_index}"
        )
        .into());
    }

    Ok(())
}

fn check_f32_rejected_input_length(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = queue.stream().clone();
    let spec = RmsNormSpec::new(1, 4, 1.0e-5, DType::F32)?;
    let plan = provider.plan_f32(spec)?;
    let input = Arc::new(DeviceBuffer::from_host(&stream, &[1.0_f32; 3])?);
    let weight = Arc::new(DeviceBuffer::from_host(&stream, &[1.0_f32; 4])?);
    let output = DeviceBuffer::<f32>::zeroed(&stream, 4)?;

    let mut bindings = queue.bindings(3)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let mut scope = queue.begin(bindings)?;
    if plan
        .enqueue_into(
            &mut scope,
            RmsNormArgs::new(input_handle, weight_handle, output_handle.write()),
        )
        .is_ok()
    {
        return Err("RMSNorm accepted 3 input elements, but the contract requires 4".into());
    }
    drop(scope);

    println!("rms_norm_f32 input_length expected=4 actual=3 rejected=true");
    Ok(())
}

fn check_f32_chained_scope(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = queue.stream().clone();
    let spec = RmsNormSpec::new(8, 4096, 1.0e-5, DType::F32)?;
    let plan = provider.plan_f32(spec)?;
    let (input_host, weight_one_host) = fixture(spec.rows(), spec.hidden_size());
    let weight_two_host = (0..spec.hidden_size())
        .map(|index| 0.75 + (index * 19 % 43) as f32 / 43.0)
        .collect::<Vec<_>>();
    let mut intermediate_expected = vec![0.0_f32; spec.numel()];
    let mut expected = vec![0.0_f32; spec.numel()];
    rms_norm_f32_reference(
        &input_host,
        &weight_one_host,
        &mut intermediate_expected,
        spec,
    )?;
    rms_norm_f32_reference(
        &intermediate_expected,
        &weight_two_host,
        &mut expected,
        spec,
    )?;

    let input = Arc::new(DeviceBuffer::from_host(&stream, &input_host)?);
    let weight_one = Arc::new(DeviceBuffer::from_host(&stream, &weight_one_host)?);
    let weight_two = Arc::new(DeviceBuffer::from_host(&stream, &weight_two_host)?);
    let intermediate = DeviceBuffer::<f32>::zeroed(&stream, spec.numel())?;
    let output = DeviceBuffer::<f32>::zeroed(&stream, spec.numel())?;

    let mut bindings = queue.bindings(5)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let weight_one_handle = bindings.bind_read(Arc::clone(&weight_one))?;
    let intermediate_handle = bindings.bind_read_write(intermediate)?;
    let weight_two_handle = bindings.bind_read(Arc::clone(&weight_two))?;
    let output_handle = bindings.bind_read_write(output)?;
    for _ in 0..2 {
        let mut scope = queue.begin(bindings)?;
        plan.enqueue_into(
            &mut scope,
            RmsNormArgs::new(input_handle, weight_one_handle, intermediate_handle.write()),
        )?;
        plan.enqueue_into(
            &mut scope,
            RmsNormArgs::new(
                intermediate_handle.read(),
                weight_two_handle,
                output_handle.write(),
            ),
        )?;
        let completion = scope.finish();
        if completion.submitted() != 2 {
            return Err("chained RMSNorm scope did not retain exactly two launches".into());
        }
        bindings = completion.wait()?;
    }
    let output = bindings.take_read_write(output_handle)?;

    let actual = output.to_host_vec(&stream)?;
    let max_abs_error = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if max_abs_error > F32_MAX_ABS_ERROR {
        return Err(format!(
            "chained RMSNorm error {max_abs_error:.9e} exceeds {F32_MAX_ABS_ERROR:.9e}"
        )
        .into());
    }

    println!(
        "rms_norm_f32 chained rows=8 hidden=4096 stream=non_default scopes=2 \
         commands_per_scope=2 completion_records_per_scope=1 intermediate_waits=0 \
         queue_reused=true bindings_reused=true max_abs_error={max_abs_error:.9e}"
    );
    Ok(())
}

fn check_f32_partial_scope_rejection(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = queue.stream().clone();
    let spec = RmsNormSpec::new(1, 4, 1.0e-5, DType::F32)?;
    let plan = provider.plan_f32(spec)?;
    let input_host = [1.0_f32, -2.0, 3.0, -4.0];
    let weight_host = [0.5_f32, 0.75, 1.0, 1.25];
    let mut expected = [0.0_f32; 4];
    rms_norm_f32_reference(&input_host, &weight_host, &mut expected, spec)?;

    let input = Arc::new(DeviceBuffer::from_host(&stream, &input_host)?);
    let short_input = Arc::new(DeviceBuffer::from_host(&stream, &[1.0_f32; 3])?);
    let weight = Arc::new(DeviceBuffer::from_host(&stream, &weight_host)?);
    let intermediate = DeviceBuffer::<f32>::zeroed(&stream, spec.numel())?;
    let output = DeviceBuffer::<f32>::zeroed(&stream, spec.numel())?;
    let mut bindings = queue.bindings(5)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let short_input_handle = bindings.bind_read(Arc::clone(&short_input))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let intermediate_handle = bindings.bind_read_write(intermediate)?;
    let output_handle = bindings.bind_read_write(output)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        RmsNormArgs::new(input_handle, weight_handle, intermediate_handle.write()),
    )?;
    if plan
        .enqueue_into(
            &mut scope,
            RmsNormArgs::new(short_input_handle, weight_handle, output_handle.write()),
        )
        .is_ok()
    {
        return Err("partial scope accepted a short second input".into());
    }
    let mut bindings = scope.finish().wait()?;
    let intermediate = bindings.take_read_write(intermediate_handle)?;

    let actual = intermediate.to_host_vec(&stream)?;
    let max_abs_error = actual
        .iter()
        .zip(expected)
        .map(|(&actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if max_abs_error > F32_MAX_ABS_ERROR {
        return Err(format!(
            "partial-scope first launch error {max_abs_error:.9e} exceeds {F32_MAX_ABS_ERROR:.9e}"
        )
        .into());
    }

    let drop_intermediate = DeviceBuffer::<f32>::zeroed(&stream, spec.numel())?;
    let drop_output = DeviceBuffer::<f32>::zeroed(&stream, spec.numel())?;
    let mut drop_bindings = queue.bindings(5)?;
    let drop_input_handle = drop_bindings.bind_read(Arc::clone(&input))?;
    let drop_short_input_handle = drop_bindings.bind_read(Arc::clone(&short_input))?;
    let drop_weight_handle = drop_bindings.bind_read(Arc::clone(&weight))?;
    let drop_intermediate_handle = drop_bindings.bind_read_write(drop_intermediate)?;
    let drop_output_handle = drop_bindings.bind_read_write(drop_output)?;
    let mut drop_scope = queue.begin(drop_bindings)?;
    plan.enqueue_into(
        &mut drop_scope,
        RmsNormArgs::new(
            drop_input_handle,
            drop_weight_handle,
            drop_intermediate_handle.write(),
        ),
    )?;
    if plan
        .enqueue_into(
            &mut drop_scope,
            RmsNormArgs::new(
                drop_short_input_handle,
                drop_weight_handle,
                drop_output_handle.write(),
            ),
        )
        .is_ok()
    {
        return Err("drop-guard scope accepted a short second input".into());
    }
    drop(drop_scope);

    println!(
        "rms_norm_f32 partial_scope submitted_before_error=1 second_launch_rejected=true \
         drop_guard_exercised=true first_launch_max_abs_error={max_abs_error:.9e}"
    );
    Ok(())
}

fn expect_low_precision_rejection<T: LowPrecision>(
    queue: &mut CommandQueue,
    plan: &T::Plan,
    input_len: usize,
    weight_len: usize,
    output_len: usize,
    operand: &'static str,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = queue.stream().clone();
    let input = Arc::new(DeviceBuffer::from_host(
        &stream,
        &vec![T::from_f32(1.0); input_len],
    )?);
    let weight = Arc::new(DeviceBuffer::from_host(
        &stream,
        &vec![T::from_f32(1.0); weight_len],
    )?);
    let output = DeviceBuffer::<T>::zeroed(&stream, output_len)?;
    let mut bindings = queue.bindings(3)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let output_handle = bindings.bind_read_write(output)?;
    let mut scope = queue.begin(bindings)?;
    if plan
        .enqueue(
            &mut scope,
            RmsNormArgs::new(input_handle, weight_handle, output_handle.write()),
        )
        .is_ok()
    {
        return Err(format!("{} RMSNorm accepted a short {operand} buffer", T::NAME).into());
    }
    drop(scope);
    Ok(())
}

fn check_low_precision_rejections<T: LowPrecision>(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = RmsNormSpec::new(1, 4, 1.0e-5, T::DTYPE)?;
    let plan = T::plan(provider, spec)?;

    expect_low_precision_rejection::<T>(queue, &plan, 3, 4, 4, "input")?;
    expect_low_precision_rejection::<T>(queue, &plan, 4, 3, 4, "weight")?;
    expect_low_precision_rejection::<T>(queue, &plan, 4, 4, 3, "output")?;

    println!(
        "rms_norm_{} short_buffers input_rejected=true weight_rejected=true \
         output_rejected=true",
        T::NAME
    );
    Ok(())
}

fn check_low_precision_chained_scope<T: LowPrecision>(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = queue.stream().clone();
    let spec = RmsNormSpec::new(8, 4096, 1.0e-5, T::DTYPE)?;
    let plan = T::plan(provider, spec)?;
    let (input_f32, weight_one_f32) = fixture(spec.rows(), spec.hidden_size());
    let input_host = input_f32.into_iter().map(T::from_f32).collect::<Vec<_>>();
    let weight_one_host = weight_one_f32
        .into_iter()
        .map(T::from_f32)
        .collect::<Vec<_>>();
    let weight_two_host = (0..spec.hidden_size())
        .map(|index| T::from_f32(0.75 + (index * 19 % 43) as f32 / 43.0))
        .collect::<Vec<_>>();
    let mut intermediate_expected = vec![T::from_f32(0.0); spec.numel()];
    let mut expected = vec![T::from_f32(0.0); spec.numel()];
    T::reference(
        &input_host,
        &weight_one_host,
        &mut intermediate_expected,
        spec,
    )?;
    T::reference(
        &intermediate_expected,
        &weight_two_host,
        &mut expected,
        spec,
    )?;

    let input = Arc::new(DeviceBuffer::from_host(&stream, &input_host)?);
    let weight_one = Arc::new(DeviceBuffer::from_host(&stream, &weight_one_host)?);
    let weight_two = Arc::new(DeviceBuffer::from_host(&stream, &weight_two_host)?);
    let intermediate = DeviceBuffer::<T>::zeroed(&stream, spec.numel())?;
    let output = DeviceBuffer::<T>::zeroed(&stream, spec.numel())?;
    let metadata = Arc::new(DeviceBuffer::from_host(&stream, &[1.0_f32])?);
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, 256)?;

    let mut bindings = queue.bindings(7)?;
    let input_handle = bindings.bind_read(Arc::clone(&input))?;
    let weight_one_handle = bindings.bind_read(Arc::clone(&weight_one))?;
    let intermediate_handle = bindings.bind_read_write(intermediate)?;
    let weight_two_handle = bindings.bind_read(Arc::clone(&weight_two))?;
    let output_handle = bindings.bind_read_write(output)?;
    let _metadata_handle = bindings.bind_read(Arc::clone(&metadata))?;
    let _workspace_handle = bindings.bind_read_write(workspace)?;
    for _ in 0..2 {
        let mut scope = queue.begin(bindings)?;
        plan.enqueue(
            &mut scope,
            RmsNormArgs::new(input_handle, weight_one_handle, intermediate_handle.write()),
        )?;
        plan.enqueue(
            &mut scope,
            RmsNormArgs::new(
                intermediate_handle.read(),
                weight_two_handle,
                output_handle.write(),
            ),
        )?;
        let completion = scope.finish();
        if completion.submitted() != 2 {
            return Err(
                format!("{} chained scope retained the wrong command count", T::NAME).into(),
            );
        }
        bindings = completion.wait()?;
    }
    let output = bindings.take_read_write(output_handle)?;

    let actual = output.to_host_vec(&stream)?;
    let stats = compare_low_precision(&actual, &expected)?;
    if stats.max_abs > T::MAX_ABS_ERROR || stats.max_ulp > LOW_PRECISION_MAX_ULP_ERROR {
        return Err(format!(
            "{} chained RMSNorm exceeded gate: abs={:.9e}, ulp={}",
            T::NAME,
            stats.max_abs,
            stats.max_ulp,
        )
        .into());
    }
    println!(
        "rms_norm_{} chained rows=8 hidden=4096 path={:?} scopes=2 \
         commands_per_scope=2 completion_records_per_scope=1 intermediate_waits=0 \
         queue_reused=true bindings_reused=true heterogeneous_bindings=f32+u8+{} \
         max_abs_error={:.9e} max_ulp_error={}",
        T::NAME,
        plan.kernel_path(),
        T::NAME,
        stats.max_abs,
        stats.max_ulp,
    );
    Ok(())
}

fn check_low_precision_signed_zero<T: LowPrecision>(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = queue.stream().clone();
    for hidden_size in [2, 3] {
        let spec = RmsNormSpec::new(1, hidden_size, 1.0e-5, T::DTYPE)?;
        let plan = T::plan(provider, spec)?;
        let input_host = (0..hidden_size)
            .map(|index| T::from_f32(if index.is_multiple_of(2) { -0.0 } else { 0.0 }))
            .collect::<Vec<_>>();
        let weight_host = vec![T::from_f32(1.0); hidden_size];
        let mut expected = vec![T::from_f32(0.0); hidden_size];
        T::reference(&input_host, &weight_host, &mut expected, spec)?;
        let input = Arc::new(DeviceBuffer::from_host(&stream, &input_host)?);
        let weight = Arc::new(DeviceBuffer::from_host(&stream, &weight_host)?);
        let output = DeviceBuffer::<T>::zeroed(&stream, hidden_size)?;
        let mut bindings = queue.bindings(3)?;
        let input_handle = bindings.bind_read(Arc::clone(&input))?;
        let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
        let output_handle = bindings.bind_read_write(output)?;
        let mut scope = queue.begin(bindings)?;
        plan.enqueue(
            &mut scope,
            RmsNormArgs::new(input_handle, weight_handle, output_handle.write()),
        )?;
        let mut bindings = scope.finish().wait()?;
        let output = bindings.take_read_write(output_handle)?;
        let actual = output.to_host_vec(&stream)?;
        let actual_bits = actual.iter().map(|value| value.bits()).collect::<Vec<_>>();
        let expected_bits = expected
            .iter()
            .map(|value| value.bits())
            .collect::<Vec<_>>();
        if actual_bits != expected_bits {
            return Err(format!(
                "{} {:?} signed-zero mismatch: actual={actual_bits:x?}, expected={expected_bits:x?}",
                T::NAME,
                plan.kernel_path(),
            )
            .into());
        }
        println!(
            "rms_norm_{} signed_zero path={:?} hidden={hidden_size} bit_exact=true",
            T::NAME,
            plan.kernel_path(),
        );
    }
    Ok(())
}

fn run_low_precision_suite<T: LowPrecision>(
    queue: &mut CommandQueue,
    provider: &RmsNormProvider,
) -> Result<(), Box<dyn std::error::Error>> {
    for (rows, hidden_size) in [
        (1, 1),
        (3, 127),
        (3, 4097),
        (1, 2),
        (32, 256),
        (8, 4096),
        (16, 8192),
        (1, 11008),
    ] {
        check_low_precision_case::<T>(queue, provider, rows, hidden_size)?;
    }
    check_low_precision_rejections::<T>(queue, provider)?;
    check_low_precision_chained_scope::<T>(queue, provider)?;
    check_low_precision_signed_zero::<T>(queue, provider)?;
    println!(
        "cuda-oxide {} RMSNorm passed 8 single-launch cases, 3 short-buffer \
         rejections, reusable chained scopes, and scalar/packed signed-zero checks",
        T::NAME
    );
    Ok(())
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let provider = RmsNormProvider::load(&context)?;
    let stream = context.new_stream()?;
    let mut queue = CommandQueue::new(stream, 2)?;

    for (rows, hidden_size) in [(1, 1), (3, 127), (8, 4096), (16, 8192)] {
        check_f32_case(&mut queue, &provider, rows, hidden_size)?;
    }
    check_f32_rejected_input_length(&mut queue, &provider)?;
    check_f32_chained_scope(&mut queue, &provider)?;
    check_f32_partial_scope_rejection(&mut queue, &provider)?;

    println!(
        "cuda-oxide F32 RMSNorm passed 4 single-launch cases, 2 rejection paths, and reusable chained scopes"
    );
    run_low_precision_suite::<f16>(&mut queue, &provider)?;
    run_low_precision_suite::<bf16>(&mut queue, &provider)?;
    println!(
        "cuda-oxide RMSNorm H20 gate passed F32, FP16, and BF16 on one checked execution lifecycle"
    );
    Ok(())
}
