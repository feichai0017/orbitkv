use crate::comparison::{Comparison, compare_bf16};
use crate::reporting::GateCase;
use crate::support::gemm_fixture::{
    CENSUS_SHAPES, MINIMUM_SHAPE, exact_activation_value, exact_weight_value,
};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use half::bf16;
use loom_infer::{Bf16DenseGemmSpec, bf16_dense_gemm_reference};
use loom_infer_cuda::command::{CommandError, CommandQueue};
use loom_infer_cuda::gemm::{
    Bf16DenseGemmAlgorithm, Bf16DenseGemmEnqueueError, Bf16DenseGemmOperands, Bf16DenseGemmPlan,
    Bf16DenseGemmPlanError, Bf16DenseGemmSelection, GemmPlanner, GemmProviderId,
    GemmProviderVersion,
};
use loom_infer_cuda::graph::GraphQueue;
use loom_infer_cuda::memory::{ReadDeviceRegion, ReadWriteDeviceRegion};
use std::error::Error;
use std::sync::Arc;

const TOLERANCE_SPEC: (usize, usize, usize) = (1, 1_536, 8_960);
const OUTPUT_GUARD_ELEMENTS: usize = 2;
const LOOM_TENSOR_ALIGNMENT_BYTES: u64 = 4;
const LOOM_WORKSPACE_ALIGNMENT_BYTES: u64 = 1;
const GENERAL_MAX_ABS_ERROR: f32 = 3.125e-2;
const GENERAL_MAX_REL_ERROR: f32 = 1.5625e-2;

fn fixture(spec: Bf16DenseGemmSpec) -> (Vec<bf16>, Vec<bf16>) {
    // Every product is an integer multiple of 2^-14. The largest possible
    // partial sum has fewer than 24 significant integer bits for the covered
    // census K range, so the F32 result is independent of reduction order.
    let activation = (0..spec.k())
        .map(|column| bf16::from_f32(exact_activation_value(column)))
        .collect();
    let mut weight = Vec::with_capacity(spec.weight_numel());
    for row in 0..spec.n() {
        weight.extend((0..spec.k()).map(|column| bf16::from_f32(exact_weight_value(row, column))));
    }
    (activation, weight)
}

fn cancellation_fixture(spec: Bf16DenseGemmSpec) -> (Vec<bf16>, Vec<bf16>) {
    let activation = (0..spec.k())
        .map(|column| {
            let signed = ((column * 17 + 11) % 101) as f32 - 50.0;
            bf16::from_f32(signed / 37.0)
        })
        .collect();
    let weight = (0..spec.weight_numel())
        .map(|index| {
            let row = index / spec.k();
            let column = index % spec.k();
            let signed = ((row * 29 + column * 13 + 7) % 127) as f32 - 63.0;
            let sign = if (column / 5).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            bf16::from_f32(sign * signed / 53.0)
        })
        .collect();
    (activation, weight)
}

fn cancellation_ratio(activation: &[bf16], weight: &[bf16], spec: Bf16DenseGemmSpec) -> f32 {
    (0..usize::min(spec.n(), 16)).fold(0.0_f32, |largest, row| {
        let weight_row = &weight[row * spec.k()..(row + 1) * spec.k()];
        let (sum, absolute_sum) = activation.iter().zip(weight_row).fold(
            (0.0_f32, 0.0_f32),
            |(sum, absolute_sum), (&activation, &weight)| {
                let product = activation.to_f32() * weight.to_f32();
                (sum + product, absolute_sum + product.abs())
            },
        );
        largest.max(absolute_sum / sum.abs().max(f32::MIN_POSITIVE))
    })
}

fn reference(
    activation: &[bf16],
    weight: &[bf16],
    spec: Bf16DenseGemmSpec,
) -> Result<Vec<bf16>, Box<dyn Error>> {
    let mut expected = vec![bf16::ZERO; spec.output_numel()];
    bf16_dense_gemm_reference(activation, weight, &mut expected, spec)?;
    require_transpose_sensitive(activation, weight, &expected, spec)?;
    Ok(expected)
}

fn require_transpose_sensitive(
    activation: &[bf16],
    weight: &[bf16],
    expected: &[bf16],
    spec: Bf16DenseGemmSpec,
) -> Result<(), Box<dyn Error>> {
    let sample_columns = usize::min(spec.n(), 16);
    let differs_from_untransposed = (0..sample_columns).any(|output_column| {
        let accumulator = (0..spec.k()).fold(0.0_f32, |sum, reduction| {
            activation[reduction]
                .to_f32()
                .mul_add(weight[reduction * spec.n() + output_column].to_f32(), sum)
        });
        bf16::from_f32(accumulator).to_bits() != expected[output_column].to_bits()
    });
    if !differs_from_untransposed {
        return Err("GEMV fixture does not distinguish row-major W[N,K]^T from W[K,N]".into());
    }
    Ok(())
}

fn guarded_output(output_numel: usize) -> Vec<bf16> {
    let mut output = vec![bf16::from_bits(0x7fc1); output_numel + 2 * OUTPUT_GUARD_ELEMENTS];
    output[0] = bf16::from_bits(0x3f80);
    output[1] = bf16::from_bits(0xc020);
    let suffix = output_numel + OUTPUT_GUARD_ELEMENTS;
    output[suffix] = bf16::from_bits(0x40a0);
    output[suffix + 1] = bf16::from_bits(0xc0e0);
    output
}

fn require_bits_equal(
    actual: &[bf16],
    expected: &[bf16],
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(format!("{label} changed").into());
    }
    Ok(())
}

fn check_guarded_output(
    label: &str,
    actual: &[bf16],
    expected: &[bf16],
) -> Result<Comparison, Box<dyn Error>> {
    let expected_len = expected.len() + 2 * OUTPUT_GUARD_ELEMENTS;
    if actual.len() != expected_len {
        return Err(format!(
            "{label} allocation length changed: expected {expected_len}, got {}",
            actual.len()
        )
        .into());
    }
    let sentinel = guarded_output(expected.len());
    require_bits_equal(
        &actual[..OUTPUT_GUARD_ELEMENTS],
        &sentinel[..OUTPUT_GUARD_ELEMENTS],
        &format!("{label} prefix sentinel"),
    )?;
    let suffix = expected.len() + OUTPUT_GUARD_ELEMENTS;
    require_bits_equal(
        &actual[suffix..],
        &sentinel[suffix..],
        &format!("{label} suffix sentinel"),
    )?;
    let comparison = compare_bf16(&actual[OUTPUT_GUARD_ELEMENTS..suffix], expected, label)?;
    if comparison.bit_mismatches != 0 {
        return Err(format!(
            "{label} exceeded the bit-exact dyadic-fixture gate: mismatches={} max_abs={:.9e}",
            comparison.bit_mismatches, comparison.max_abs
        )
        .into());
    }
    Ok(comparison)
}

fn check_guarded_output_with_tolerance(
    label: &str,
    actual: &[bf16],
    expected: &[bf16],
) -> Result<(Comparison, f32), Box<dyn Error>> {
    let expected_len = expected.len() + 2 * OUTPUT_GUARD_ELEMENTS;
    if actual.len() != expected_len {
        return Err(format!(
            "{label} allocation length changed: expected {expected_len}, got {}",
            actual.len()
        )
        .into());
    }
    let sentinel = guarded_output(expected.len());
    require_bits_equal(
        &actual[..OUTPUT_GUARD_ELEMENTS],
        &sentinel[..OUTPUT_GUARD_ELEMENTS],
        &format!("{label} prefix sentinel"),
    )?;
    let suffix = expected.len() + OUTPUT_GUARD_ELEMENTS;
    require_bits_equal(
        &actual[suffix..],
        &sentinel[suffix..],
        &format!("{label} suffix sentinel"),
    )?;
    let actual = &actual[OUTPUT_GUARD_ELEMENTS..suffix];
    let comparison = compare_bf16(actual, expected, label)?;
    let mut max_rel = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let absolute = (actual.to_f32() - expected.to_f32()).abs();
        let relative = absolute / expected.to_f32().abs().max(GENERAL_MAX_ABS_ERROR);
        max_rel = max_rel.max(relative);
        let limit = GENERAL_MAX_ABS_ERROR + GENERAL_MAX_REL_ERROR * expected.to_f32().abs();
        if absolute > limit {
            return Err(format!(
                "{label} exceeded its mixed tolerance at index {index}: \
                 actual={} expected={} abs={absolute:.9e} limit={limit:.9e}",
                actual.to_f32(),
                expected.to_f32(),
            )
            .into());
        }
    }
    Ok((comparison, max_rel))
}

fn make_output_region(
    stream: &Arc<CudaStream>,
    output_numel: usize,
) -> Result<ReadWriteDeviceRegion<bf16>, Box<dyn Error>> {
    let allocation = DeviceBuffer::from_host(stream, &guarded_output(output_numel))?;
    Ok(ReadWriteDeviceRegion::from_buffer_range(
        allocation,
        OUTPUT_GUARD_ELEMENTS..OUTPUT_GUARD_ELEMENTS + output_numel,
    )?)
}

fn check_plan_metadata(
    planner: &GemmPlanner,
    plan: &Bf16DenseGemmPlan,
    expected_spec: Bf16DenseGemmSpec,
) -> Result<(), Box<dyn Error>> {
    let info = plan.plan_info();
    if plan.spec() != expected_spec
        || info.provider() != GemmProviderId::Loom
        || info.algorithm() != Bf16DenseGemmAlgorithm::LoomSm90SimtGemvM1N16K64
        || info.workspace_required_bytes() != 0
        || plan.workspace_required_bytes() != 0
        || info.tensor_alignment_bytes() != LOOM_TENSOR_ALIGNMENT_BYTES
        || plan.tensor_alignment_bytes() != LOOM_TENSOR_ALIGNMENT_BYTES
        || info.workspace_alignment_bytes() != LOOM_WORKSPACE_ALIGNMENT_BYTES
        || plan.workspace_alignment_bytes() != LOOM_WORKSPACE_ALIGNMENT_BYTES
        || plan.estimated_waves_count().is_some()
        || planner.workspace_limit_bytes(Bf16DenseGemmSelection::Loom) != 0
    {
        return Err("Loom GEMV plan metadata does not match the frozen algorithm contract".into());
    }
    match planner.provider_version(GemmProviderId::Loom) {
        GemmProviderVersion::Loom(version) if !version.is_empty() => Ok(()),
        other => Err(format!("Loom GEMV reported an invalid provider version: {other:?}").into()),
    }
}

fn check_eager_reuse(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
    activation: &Arc<DeviceBuffer<bf16>>,
    weight: &Arc<DeviceBuffer<bf16>>,
    expected: &[bf16],
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let spec = plan.spec();
    let first_output = make_output_region(&stream, spec.output_numel())?;
    let second_output = make_output_region(&stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, 0)?;
    let mut bindings = queue.bindings(5)?;
    let activation_handle = bindings.bind_read(Arc::clone(activation))?;
    let weight_handle = bindings.bind_read(Arc::clone(weight))?;
    let first_output_handle = bindings.bind_read_write_region(first_output)?;
    let second_output_handle = bindings.bind_read_write_region(second_output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    for output in [first_output_handle.write(), second_output_handle.write()] {
        let mut scope = queue.begin(bindings)?;
        plan.enqueue_into(
            &mut scope,
            Bf16DenseGemmOperands::new(
                activation_handle,
                weight_handle,
                output,
                workspace_handle.write(),
            ),
        )?;
        let completion = scope.finish();
        if completion.submitted() != 1 {
            return Err("Loom GEMV eager completion covered the wrong command count".into());
        }
        bindings = completion.wait()?;
    }

    let first_output = bindings
        .take_read_write_region(first_output_handle)?
        .into_buffer()
        .map_err(|_| "eager output unexpectedly used external storage")?;
    let second_output = bindings
        .take_read_write_region(second_output_handle)?
        .into_buffer()
        .map_err(|_| "eager output unexpectedly used external storage")?;
    let workspace = bindings.take_read_write(workspace_handle)?;
    drop(bindings);
    if !workspace.is_empty() {
        return Err("Loom GEMV plan used a nonempty caller workspace".into());
    }
    let first_actual = first_output.to_host_vec(&stream)?;
    let second_actual = second_output.to_host_vec(&stream)?;
    let first_comparison = check_guarded_output("first eager Loom GEMV", &first_actual, expected)?;
    let second_comparison =
        check_guarded_output("second eager Loom GEMV", &second_actual, expected)?;
    println!(
        "{} m={} n={} k={} commands_per_scope=1 scopes=2 plan_reused=true \
         workspace_required=0 output_sentinels=preserved fixture=dyadic_exact \
         tolerance=bit_exact \
         first_digest={:016x} second_digest={:016x}",
        GateCase::new("loom_sm90_simt_gemv_h20", "eager_shape"),
        spec.m(),
        spec.n(),
        spec.k(),
        first_comparison.digest,
        second_comparison.digest,
    );
    Ok(())
}

fn check_graph_replay(
    context: &Arc<CudaContext>,
    plan: Bf16DenseGemmPlan,
    activation: Arc<DeviceBuffer<bf16>>,
    weight: Arc<DeviceBuffer<bf16>>,
    expected: &[bf16],
) -> Result<(), Box<dyn Error>> {
    let spec = plan.spec();
    if !expected.iter().any(|value| value.to_bits() != 0) {
        return Err("Loom GEMV graph fixture has an all-zero expected output".into());
    }
    let upload_stream = context.new_stream()?;
    let zero_activation_host = vec![bf16::ZERO; spec.a_numel()];
    let zero_activation = Arc::new(DeviceBuffer::<bf16>::from_host(
        &upload_stream,
        &zero_activation_host,
    )?);
    let zero_output = make_output_region(&upload_stream, spec.output_numel())?;
    let expected_output = make_output_region(&upload_stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(&upload_stream, 0)?;
    let graph_queue = GraphQueue::new(context, 3)?;
    let mut bindings = graph_queue.bindings(6)?;
    let zero_activation_handle = bindings.bind_read(Arc::clone(&zero_activation))?;
    let activation_handle = bindings.bind_read(Arc::clone(&activation))?;
    let weight_handle = bindings.bind_read(Arc::clone(&weight))?;
    let zero_output_handle = bindings.bind_read_write_region(zero_output)?;
    let expected_output_handle = bindings.bind_read_write_region(expected_output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;

    let captured = graph_queue.capture(bindings, |scope| {
        plan.enqueue_into(
            scope,
            Bf16DenseGemmOperands::new(
                zero_activation_handle,
                weight_handle,
                zero_output_handle.write(),
                workspace_handle.write(),
            ),
        )?;
        plan.enqueue_into(
            scope,
            Bf16DenseGemmOperands::new(
                zero_activation_handle,
                weight_handle,
                expected_output_handle.write(),
                workspace_handle.write(),
            ),
        )?;
        plan.enqueue_into(
            scope,
            Bf16DenseGemmOperands::new(
                activation_handle,
                weight_handle,
                expected_output_handle.write(),
                workspace_handle.write(),
            ),
        )
    })?;
    if captured.commands() != 3 {
        return Err("Loom GEMV graph captured the wrong command count".into());
    }
    drop(plan);
    drop(zero_activation);
    drop(activation);
    drop(weight);

    let mut exec = captured.instantiate()?;
    for expected_launch in 1..=2 {
        let mut completion = exec.launch()?;
        if completion.launch_index() != expected_launch {
            return Err("Loom GEMV graph completion reported the wrong replay index".into());
        }
        let _ = completion.is_complete()?;
        if expected_launch == 1 {
            completion.wait()?;
        } else {
            drop(completion);
        }
    }
    if exec.launches() != 2 || exec.commands() != 3 {
        return Err("Loom GEMV graph accounting changed across replay".into());
    }
    let mut bindings = exec.into_bindings()?;
    let zero_output = bindings
        .take_read_write_region(zero_output_handle)?
        .into_buffer()
        .map_err(|_| "graph zero output unexpectedly used external storage")?;
    let expected_output = bindings
        .take_read_write_region(expected_output_handle)?
        .into_buffer()
        .map_err(|_| "graph expected output unexpectedly used external storage")?;
    let workspace = bindings.take_read_write(workspace_handle)?;
    drop(bindings);
    if !workspace.is_empty() {
        return Err("Loom GEMV graph used a nonempty caller workspace".into());
    }
    let zero_actual = zero_output.to_host_vec(&upload_stream)?;
    let expected_actual = expected_output.to_host_vec(&upload_stream)?;
    let zero_expected = vec![bf16::ZERO; spec.output_numel()];
    let zero_comparison =
        check_guarded_output("replayed Loom GEMV zero node", &zero_actual, &zero_expected)?;
    let expected_comparison = check_guarded_output(
        "replayed Loom GEMV expected node",
        &expected_actual,
        expected,
    )?;
    println!(
        "{} m={} n={} k={} commands=3 replays=2 fixed_bindings=true \
         replay_nodes=independent_zero_then_expected_poison_then_expected \
         independent_zero_node_observable=true expected_output_poisoned_each_replay=true \
         initial_outputs=poisoned zero_output=verified expected_output_nonzero=true \
         owner_drop_before_replay=true leases_retained=true workspace_required=0 \
         output_sentinels=preserved fixture=dyadic_exact tolerance=bit_exact \
         zero_digest={:016x} expected_digest={:016x}",
        GateCase::new("loom_sm90_simt_gemv_h20", "graph_shape"),
        spec.m(),
        spec.n(),
        spec.k(),
        zero_comparison.digest,
        expected_comparison.digest,
    );
    Ok(())
}

fn check_census_shape(
    context: &Arc<CudaContext>,
    queue: &mut CommandQueue,
    planner: &GemmPlanner,
    dimensions: (usize, usize, usize),
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16DenseGemmSpec::new(dimensions.0, dimensions.1, dimensions.2)?;
    let plan = planner.plan_bf16_dense(spec, Bf16DenseGemmSelection::Loom)?;
    check_plan_metadata(planner, &plan, spec)?;
    let (activation_host, weight_host) = fixture(spec);
    let expected = reference(&activation_host, &weight_host, spec)?;
    let stream = queue.stream().clone();
    let activation = Arc::new(DeviceBuffer::from_host(&stream, &activation_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(&stream, &weight_host)?);
    check_eager_reuse(queue, &plan, &activation, &weight, &expected)?;
    check_graph_replay(context, plan, activation, weight, &expected)?;
    Ok(())
}

fn check_general_numerics(
    queue: &mut CommandQueue,
    planner: &GemmPlanner,
) -> Result<(), Box<dyn Error>> {
    let spec = Bf16DenseGemmSpec::new(TOLERANCE_SPEC.0, TOLERANCE_SPEC.1, TOLERANCE_SPEC.2)?;
    let plan = planner.plan_bf16_dense(spec, Bf16DenseGemmSelection::Loom)?;
    let (activation_host, weight_host) = cancellation_fixture(spec);
    let cancellation_ratio = cancellation_ratio(&activation_host, &weight_host, spec);
    if cancellation_ratio < 8.0 {
        return Err(format!(
            "general numerical fixture is not cancellation-sensitive: ratio={cancellation_ratio:.6}"
        )
        .into());
    }
    let expected = reference(&activation_host, &weight_host, spec)?;
    let stream = queue.stream().clone();
    let activation = Arc::new(DeviceBuffer::from_host(&stream, &activation_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(&stream, &weight_host)?);
    let output = make_output_region(&stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, 0)?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(activation)?;
    let weight_handle = bindings.bind_read(weight)?;
    let output_handle = bindings.bind_read_write_region(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(
        &mut scope,
        Bf16DenseGemmOperands::new(
            activation_handle,
            weight_handle,
            output_handle.write(),
            workspace_handle.write(),
        ),
    )?;
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("general-numerics GEMV completion covered the wrong command count".into());
    }
    let mut bindings = completion.wait()?;
    let output = bindings
        .take_read_write_region(output_handle)?
        .into_buffer()
        .map_err(|_| "general-numerics output unexpectedly used external storage")?;
    drop(bindings);
    let actual = output.to_host_vec(&stream)?;
    let (comparison, max_rel) =
        check_guarded_output_with_tolerance("general-numerics Loom GEMV", &actual, &expected)?;
    println!(
        "{} m={} n={} k={} fixture=cancellation_sensitive oracle=cpu_f32_sequential \
         tolerance=abs_plus_rel max_abs_limit={:.9e} max_rel_limit={:.9e} \
         max_abs={:.9e} max_rel={:.9e} bit_mismatches={} cancellation_ratio={:.6} \
         output_sentinels=preserved digest={:016x}",
        GateCase::new("loom_sm90_simt_gemv_h20", "general_numerics"),
        spec.m(),
        spec.n(),
        spec.k(),
        GENERAL_MAX_ABS_ERROR,
        GENERAL_MAX_REL_ERROR,
        comparison.max_abs,
        max_rel,
        comparison.bit_mismatches,
        cancellation_ratio,
        comparison.digest,
    );
    Ok(())
}

fn check_shape_rejections(planner: &GemmPlanner) -> Result<(), Box<dyn Error>> {
    let cases = [
        ("m", Bf16DenseGemmSpec::new(2, 16, 64)?),
        ("n", Bf16DenseGemmSpec::new(1, 15, 64)?),
        ("k", Bf16DenseGemmSpec::new(1, 16, 63)?),
    ];
    for (name, spec) in cases {
        let result = planner.plan_bf16_dense(spec, Bf16DenseGemmSelection::Loom);
        let matched = match (name, result) {
            ("m", Err(Bf16DenseGemmPlanError::LoomMNotOne { m: 2 }))
            | ("n", Err(Bf16DenseGemmPlanError::LoomNNotMultipleOf16 { n: 15 }))
            | ("k", Err(Bf16DenseGemmPlanError::LoomKNotMultipleOf64 { k: 63 })) => true,
            (_, _) => false,
        };
        if !matched {
            return Err(format!("Loom GEMV returned the wrong {name} admission result").into());
        }
    }
    println!(
        "{} m_not_one=rejected n_not_multiple_of_16=rejected \
         k_not_multiple_of_64=rejected before_submission=true fallback=false \
         alternate_layout=unrepresentable_by_fixed_spec alternate_transpose=unrepresentable_by_fixed_spec",
        GateCase::new("loom_sm90_simt_gemv_h20", "shape_contract"),
    );
    Ok(())
}

fn expect_length_rejection(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
    expected_operand: &'static str,
    activation_len: usize,
    weight_len: usize,
    output_len: usize,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let poison = vec![bf16::from_bits(0x7fc1); output_len];
    let activation = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, activation_len)?);
    let weight = Arc::new(DeviceBuffer::<bf16>::zeroed(&stream, weight_len)?);
    let output = DeviceBuffer::from_host(&stream, &poison)?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, 0)?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(activation)?;
    let weight_handle = bindings.bind_read(weight)?;
    let output_handle = bindings.bind_read_write(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let mut scope = queue.begin(bindings)?;
    let result = plan.enqueue_into(
        &mut scope,
        Bf16DenseGemmOperands::new(
            activation_handle,
            weight_handle,
            output_handle.write(),
            workspace_handle.write(),
        ),
    );
    match result {
        Err(Bf16DenseGemmEnqueueError::LengthMismatch { operand, .. })
            if operand == expected_operand => {}
        other => {
            return Err(format!(
                "invalid-length {expected_operand} buffer returned the wrong result: {other:?}"
            )
            .into());
        }
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err(
            format!("invalid-length {expected_operand} buffer submitted device work").into(),
        );
    }
    let mut bindings = completion.wait()?;
    let output = bindings.take_read_write(output_handle)?;
    drop(bindings);
    require_bits_equal(
        &output.to_host_vec(&stream)?,
        &poison,
        "rejected length output sentinel",
    )?;
    Ok(())
}

fn check_buffer_lengths(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    let spec = plan.spec();
    expect_length_rejection(
        queue,
        plan,
        "A",
        spec.a_numel() - 1,
        spec.weight_numel(),
        spec.output_numel(),
    )?;
    expect_length_rejection(
        queue,
        plan,
        "W",
        spec.a_numel(),
        spec.weight_numel() - 1,
        spec.output_numel(),
    )?;
    expect_length_rejection(
        queue,
        plan,
        "D",
        spec.a_numel(),
        spec.weight_numel(),
        spec.output_numel() - 1,
    )?;
    expect_length_rejection(
        queue,
        plan,
        "A",
        spec.a_numel() + 1,
        spec.weight_numel(),
        spec.output_numel(),
    )?;
    expect_length_rejection(
        queue,
        plan,
        "W",
        spec.a_numel(),
        spec.weight_numel() + 1,
        spec.output_numel(),
    )?;
    expect_length_rejection(
        queue,
        plan,
        "D",
        spec.a_numel(),
        spec.weight_numel(),
        spec.output_numel() + 1,
    )?;
    println!(
        "{} a_short=rejected w_short=rejected d_short=rejected \
         a_long=rejected w_long=rejected d_long=rejected \
         exact_regions_required=true before_submission=true sentinels_preserved=true",
        GateCase::new("loom_sm90_simt_gemv_h20", "buffer_lengths"),
    );
    Ok(())
}

fn expect_alignment_rejection(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
    expected_operand: &'static str,
) -> Result<(), Box<dyn Error>> {
    let stream = queue.stream().clone();
    let spec = plan.spec();
    let activation_offset = usize::from(expected_operand == "A");
    let weight_offset = usize::from(expected_operand == "W");
    let output_offset = usize::from(expected_operand == "D");
    let activation = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.a_numel() + activation_offset,
    )?);
    let weight = Arc::new(DeviceBuffer::<bf16>::zeroed(
        &stream,
        spec.weight_numel() + weight_offset,
    )?);
    let poison = vec![bf16::from_bits(0x7fc1); spec.output_numel() + output_offset];
    let output = DeviceBuffer::from_host(&stream, &poison)?;
    let activation_region = ReadDeviceRegion::from_buffer_range(
        activation,
        activation_offset..activation_offset + spec.a_numel(),
    )?;
    let weight_region = ReadDeviceRegion::from_buffer_range(
        weight,
        weight_offset..weight_offset + spec.weight_numel(),
    )?;
    let output_region = ReadWriteDeviceRegion::from_buffer_range(
        output,
        output_offset..output_offset + spec.output_numel(),
    )?;
    let workspace = DeviceBuffer::<u8>::zeroed(&stream, 0)?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read_region(activation_region)?;
    let weight_handle = bindings.bind_read_region(weight_region)?;
    let output_handle = bindings.bind_read_write_region(output_region)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let mut scope = queue.begin(bindings)?;
    let result = plan.enqueue_into(
        &mut scope,
        Bf16DenseGemmOperands::new(
            activation_handle,
            weight_handle,
            output_handle.write(),
            workspace_handle.write(),
        ),
    );
    match result {
        Err(Bf16DenseGemmEnqueueError::MisalignedBuffer {
            operand,
            alignment: LOOM_TENSOR_ALIGNMENT_BYTES,
            ..
        }) if operand == expected_operand => {}
        other => {
            return Err(format!(
                "misaligned {expected_operand} returned the wrong result: {other:?}"
            )
            .into());
        }
    }
    let completion = scope.finish();
    if completion.submitted() != 0 {
        return Err(format!("misaligned {expected_operand} submitted device work").into());
    }
    let mut bindings = completion.wait()?;
    let output = bindings
        .take_read_write_region(output_handle)?
        .into_buffer()
        .map_err(|_| "alignment output unexpectedly used external storage")?;
    drop(bindings);
    require_bits_equal(
        &output.to_host_vec(&stream)?,
        &poison,
        "rejected alignment output sentinel",
    )?;
    Ok(())
}

fn check_alignment(
    queue: &mut CommandQueue,
    plan: &Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    for operand in ["A", "W", "D"] {
        expect_alignment_rejection(queue, plan, operand)?;
    }
    println!(
        "{} a=rejected w=rejected d=rejected required_alignment=4 \
         before_submission=true sentinels_preserved=true",
        GateCase::new("loom_sm90_simt_gemv_h20", "alignment"),
    );
    Ok(())
}

fn check_command_capacity(
    stream: &Arc<CudaStream>,
    plan: &Bf16DenseGemmPlan,
) -> Result<(), Box<dyn Error>> {
    let spec = plan.spec();
    let (activation_host, weight_host) = fixture(spec);
    let expected = reference(&activation_host, &weight_host, spec)?;
    let activation = Arc::new(DeviceBuffer::from_host(stream, &activation_host)?);
    let weight = Arc::new(DeviceBuffer::from_host(stream, &weight_host)?);
    let output = make_output_region(stream, spec.output_numel())?;
    let workspace = DeviceBuffer::<u8>::zeroed(stream, 0)?;
    let mut queue = CommandQueue::new(Arc::clone(stream), 1, 1)?;
    let mut bindings = queue.bindings(4)?;
    let activation_handle = bindings.bind_read(activation)?;
    let weight_handle = bindings.bind_read(weight)?;
    let output_handle = bindings.bind_read_write_region(output)?;
    let workspace_handle = bindings.bind_read_write(workspace)?;
    let operands = Bf16DenseGemmOperands::new(
        activation_handle,
        weight_handle,
        output_handle.write(),
        workspace_handle.write(),
    );
    let mut scope = queue.begin(bindings)?;
    plan.enqueue_into(&mut scope, operands)?;
    match plan.enqueue_into(&mut scope, operands) {
        Err(Bf16DenseGemmEnqueueError::Command(CommandError::CommandCapacityExceeded {
            capacity: 1,
        })) => {}
        other => return Err(format!("second Loom GEMV command returned {other:?}").into()),
    }
    let completion = scope.finish();
    if completion.submitted() != 1 {
        return Err("command-capacity rejection changed the submitted command count".into());
    }
    let mut bindings = completion.wait()?;
    let output = bindings
        .take_read_write_region(output_handle)?
        .into_buffer()
        .map_err(|_| "capacity output unexpectedly used external storage")?;
    drop(bindings);
    let actual = output.to_host_vec(stream)?;
    let comparison = check_guarded_output("capacity-admitted Loom GEMV", &actual, &expected)?;
    println!(
        "{} capacity=1 first_submitted=true second_rejected_before_ffi=true \
         submitted=1 output_sentinels=preserved digest={:016x}",
        GateCase::new("loom_sm90_simt_gemv_h20", "command_capacity"),
        comparison.digest,
    );
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let planner = GemmPlanner::load(&context)?;
    let stream = context.new_stream()?;
    let mut queue = CommandQueue::new(Arc::clone(&stream), 1, 1)?;

    check_shape_rejections(&planner)?;
    let minimum_spec = Bf16DenseGemmSpec::new(MINIMUM_SHAPE.0, MINIMUM_SHAPE.1, MINIMUM_SHAPE.2)?;
    let minimum_plan = planner.plan_bf16_dense(minimum_spec, Bf16DenseGemmSelection::Loom)?;
    check_plan_metadata(&planner, &minimum_plan, minimum_spec)?;
    let GemmProviderVersion::Loom(provider_version) =
        planner.provider_version(GemmProviderId::Loom)
    else {
        return Err("Loom GEMV provider reported the wrong version identity".into());
    };
    println!(
        "{} provider=Loom provider_version={} \
         algorithm=LoomSm90SimtGemvM1N16K64 workspace_required=0 \
         tensor_alignment=4 workspace_alignment=1 estimated_waves=none \
         device=NVIDIA_H20 compute_capability=9.0 artifact_target=sm_90a \
         dtype=bf16_by_type post_ops=unrepresentable_by_fixed_spec",
        GateCase::new("loom_sm90_simt_gemv_h20", "plan"),
        provider_version,
    );
    check_buffer_lengths(&mut queue, &minimum_plan)?;
    check_alignment(&mut queue, &minimum_plan)?;
    check_command_capacity(&stream, &minimum_plan)?;
    drop(minimum_plan);
    check_general_numerics(&mut queue, &planner)?;

    for dimensions in CENSUS_SHAPES {
        check_census_shape(&context, &mut queue, &planner, dimensions)?;
    }
    println!(
        "gate=loom_sm90_simt_gemv_h20 suite=all status=pass \
         census_shapes=5 eager_scopes_per_shape=2 graph_commands_per_shape=3 \
         graph_replays_per_shape=2 \
         layout_contract=fixed_row_major_transposed_weight \
         exact_fixture_tolerance=bit_exact general_fixture_tolerance=abs_plus_rel"
    );
    Ok(())
}
