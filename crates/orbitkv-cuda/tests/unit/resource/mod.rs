use orbitkv_compiler::prelude::Symbol;

use super::*;

fn test_device(total_memory_bytes: usize) -> CudaDeviceResourceLimits {
    CudaDeviceResourceLimits {
        max_candidate_memory_bytes: total_memory_bytes,
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [i32::MAX as usize, 65_535, 65_535],
        max_shared_memory_per_block: 48 * 1024,
        max_kernel_parameter_bytes: 32_764,
    }
}

#[test]
fn extended_kernel_parameter_abi_requires_arch_nvrtc_and_driver_support() {
    let extended = EXTENDED_MAX_KERNEL_PARAMETER_BYTES;
    let legacy = LEGACY_MAX_KERNEL_PARAMETER_BYTES;

    assert_eq!(
        kernel_parameter_abi_limit(7, Some(12_010), Some(12_010)),
        extended
    );
    assert_eq!(
        kernel_parameter_abi_limit(6, Some(13_030), Some(13_030)),
        legacy,
        "pre-Volta hardware retains the 4 KiB ABI"
    );
    assert_eq!(
        kernel_parameter_abi_limit(9, Some(12_000), Some(13_030)),
        legacy,
        "pre-12.1 NVRTC cannot emit the extended ABI"
    );
    assert_eq!(
        kernel_parameter_abi_limit(9, Some(13_030), Some(12_000)),
        legacy,
        "a pre-R530 driver cannot launch the extended ABI"
    );
    assert_eq!(
        kernel_parameter_abi_limit(9, None, Some(13_030)),
        legacy,
        "an unavailable NVRTC version query must fall back conservatively"
    );
    assert_eq!(
        kernel_parameter_abi_limit(9, Some(13_030), None),
        legacy,
        "an unavailable driver version query must fall back conservatively"
    );
}

fn plan(intermediate_bytes: usize) -> CandidateResourcePlan {
    CandidateResourcePlan {
        intermediate_lower_bound_bytes: intermediate_bytes,
        ..Default::default()
    }
}

#[test]
fn memory_filter_rejects_impossible_but_not_merely_large_candidates() {
    let gib = 1024usize.pow(3);
    let device = test_device(80 * gib);

    assert!(
        validate_resource_plan(
            &plan(79 * gib),
            CandidateResourceCaps::default(),
            Some(device)
        )
        .is_ok(),
        "a large plan that fits is still a searchable candidate"
    );
    assert!(matches!(
        validate_resource_plan(
            &plan(524 * gib),
            CandidateResourceCaps::default(),
            Some(device)
        ),
        Err(ResourceViolation::CandidateDeviceMemory { .. })
    ));
    assert!(matches!(
        validate_resource_plan(
            &plan(70 * gib),
            CandidateResourceCaps {
                max_intermediate_bytes: Some(64 * gib),
                ..Default::default()
            },
            Some(device)
        ),
        Err(ResourceViolation::IntermediateMemory { .. })
    ));
}

#[test]
fn configured_intermediate_cap_and_total_device_memory_are_separate() {
    let mut candidate = plan(64);
    candidate.host_persistent_bytes = 128;
    candidate.host_transient_peak_bytes = 32;
    candidate
        .shared_device_allocations
        .push(SharedDeviceMemoryAllocation {
            key: "shared-test-workspace",
            bytes: 16,
        });

    assert!(
        validate_resource_plan(
            &candidate,
            CandidateResourceCaps {
                max_intermediate_bytes: Some(64),
                max_kernel_source_bytes: None,
            },
            Some(test_device(240)),
        )
        .is_ok(),
        "the configured intermediate cap applies only to the 64-byte arena"
    );
    assert!(matches!(
        validate_resource_plan(
            &candidate,
            CandidateResourceCaps {
                max_intermediate_bytes: Some(63),
                max_kernel_source_bytes: None,
            },
            Some(test_device(240)),
        ),
        Err(ResourceViolation::IntermediateMemory {
            required: 64,
            limit: 63,
        })
    ));
    assert!(matches!(
        validate_resource_plan(
            &candidate,
            CandidateResourceCaps {
                max_intermediate_bytes: Some(64),
                max_kernel_source_bytes: None,
            },
            Some(test_device(239)),
        ),
        Err(ResourceViolation::CandidateDeviceMemory {
            required: 240,
            limit: 239,
        })
    ));
}

#[test]
fn exact_arena_plan_cannot_hide_a_larger_static_lower_bound() {
    let mut candidate = plan(256);
    candidate.planned_intermediate_bytes = Some(128);

    assert_eq!(candidate.required_intermediate_bytes(), 256);
    assert!(matches!(
        validate_resource_plan(
            &candidate,
            CandidateResourceCaps {
                max_intermediate_bytes: Some(192),
                ..Default::default()
            },
            None,
        ),
        Err(ResourceViolation::IntermediateMemory {
            required: 256,
            limit: 192,
        })
    ));
}

#[test]
fn source_size_uses_a_configurable_compile_budget() {
    let mut candidate = plan(0);
    candidate.kernels.push(KernelResourcePlan {
        name: "large_but_legal",
        source_bytes: Some(2_000_000),
        parameter_bytes: 16,
        grid: [1, 1, 1],
        block: [1, 1, 1],
        dynamic_shared_memory_bytes: 0,
        static_shared_memory_bytes: 0,
        function_max_threads_per_block: None,
    });

    assert!(matches!(
        validate_resource_plan(&candidate, CandidateResourceCaps::default(), None),
        Err(ResourceViolation::KernelSource { .. })
    ));
    assert!(
        validate_resource_plan(
            &candidate,
            CandidateResourceCaps {
                max_kernel_source_bytes: None,
                ..Default::default()
            },
            None
        )
        .is_ok()
    );
    assert!(matches!(
        validate_resource_plan(
            &candidate,
            CandidateResourceCaps {
                max_kernel_source_bytes: Some(1_000_000),
                ..Default::default()
            },
            None
        ),
        Err(ResourceViolation::KernelSource { .. })
    ));
}

#[test]
fn launch_validation_checks_hard_device_resources() {
    let device = test_device(1024usize.pow(3));
    let mut candidate = plan(0);
    candidate.kernels.push(KernelResourcePlan {
        name: "too_many_threads",
        source_bytes: None,
        parameter_bytes: 16,
        grid: [1, 1, 1],
        block: [1024, 2, 1],
        dynamic_shared_memory_bytes: 0,
        static_shared_memory_bytes: 0,
        function_max_threads_per_block: None,
    });
    assert!(matches!(
        validate_resource_plan(&candidate, CandidateResourceCaps::default(), Some(device)),
        Err(ResourceViolation::ThreadsPerBlock { .. })
    ));

    candidate.kernels[0].block = [1, 1, 1];
    candidate.kernels[0].dynamic_shared_memory_bytes = 49 * 1024;
    assert!(matches!(
        validate_resource_plan(&candidate, CandidateResourceCaps::default(), Some(device)),
        Err(ResourceViolation::SharedMemory { .. })
    ));

    candidate.kernels[0].dynamic_shared_memory_bytes = 0;
    candidate.kernels[0].grid = [0, 1, 1];
    assert!(matches!(
        validate_resource_plan(&candidate, CandidateResourceCaps::default(), Some(device)),
        Err(ResourceViolation::ZeroLaunchDimension { kind: "grid", .. })
    ));
}

#[test]
fn parameter_accounting_matches_current_custom_build_param_contracts() {
    use crate::kernel::{hlir::KernelScatter, other_ops::KernelScatterNoCopy};

    // KernelScatter::build_params emits (out, dest, indexes, src [, dyn]).
    let copying = KernelScatter::default();
    assert_eq!(kernel_parameter_bytes(&copying, 3, false).unwrap(), 4 * 8);
    assert_eq!(kernel_parameter_bytes(&copying, 3, true).unwrap(), 5 * 8);

    // KernelScatterNoCopy aliases dest and its override emits
    // (dest, indexes, src [, dyn]) without a separate output pointer.
    let in_place = KernelScatterNoCopy::default();
    assert_eq!(kernel_parameter_bytes(&in_place, 3, false).unwrap(), 3 * 8);
    assert_eq!(kernel_parameter_bytes(&in_place, 3, true).unwrap(), 4 * 8);
}

#[derive(Debug)]
struct TestKernel {
    bytes: Expression,
    aliases_input: bool,
    mutates_aliased_input: bool,
}

impl KernelOp for TestKernel {
    fn compile(
        &self,
        _: &std::sync::Arc<CudaStream>,
        _: &mut FxHashMap<
            String,
            (
                std::sync::Arc<cudarc::driver::CudaModule>,
                cudarc::driver::CudaFunction,
            ),
        >,
    ) -> (
        cudarc::driver::CudaFunction,
        std::sync::Arc<cudarc::driver::CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, cudarc::driver::CudaSlice<u8>>,
    ) {
        unreachable!("static resource planning must not compile test kernels")
    }

    fn output_size(&self) -> Expression {
        self.bytes
    }

    fn output_bytes(&self) -> Expression {
        self.bytes
    }

    fn output_aliases_input(&self) -> Option<usize> {
        self.aliases_input.then_some(0)
    }

    fn mutates_aliased_input(&self) -> bool {
        self.mutates_aliased_input
    }

    fn kernel_name(&self) -> &'static str {
        "TestKernel"
    }
}

#[test]
fn static_llir_plan_computes_a_simultaneous_buffer_lower_bound() {
    use orbitkv_compiler::{hlir::Input, op::LLIROp};

    let mut llir = LLIRGraph::default();
    let input = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
    let first = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
        bytes: Expression::from('s') * 4,
        aliases_input: false,
        mutates_aliased_input: false,
    })));
    let second = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
        bytes: 128.into(),
        aliases_input: false,
        mutates_aliased_input: false,
    })));
    llir.add_edge(input, first, ());
    llir.add_edge(first, second, ());

    let dyn_map = FxHashMap::from_iter([(Symbol::from('s'), 16)]);
    let plan = plan_static_llir_resources(&llir, &dyn_map).unwrap();

    // The first output is 64 B and must coexist with the second's 128 B
    // output while the second kernel executes.
    assert_eq!(plan.intermediate_lower_bound_bytes, 192);
    assert!(plan.planned_intermediate_bytes.is_none());
}

#[test]
fn mutating_alias_allows_a_prior_ordered_read() {
    use orbitkv_compiler::{hlir::Input, op::LLIROp};

    let mut llir = LLIRGraph::default();
    let dest = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
    let prior_read = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
        bytes: 16.into(),
        aliases_input: false,
        mutates_aliased_input: false,
    })));
    let mutation = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
        bytes: 16.into(),
        aliases_input: true,
        mutates_aliased_input: true,
    })));
    llir.add_edge(dest, prior_read, ());
    llir.add_edge(dest, mutation, ());
    llir.add_edge(prior_read, mutation, ());

    assert!(plan_static_llir_resources(&llir, &FxHashMap::default()).is_ok());
}

#[test]
fn mutating_alias_rejects_an_unordered_competing_read() {
    use orbitkv_compiler::{hlir::Input, op::LLIROp};

    let mut llir = LLIRGraph::default();
    let dest = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
    let competing_read = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
        bytes: 16.into(),
        aliases_input: false,
        mutates_aliased_input: false,
    })));
    let mutation = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
        bytes: 16.into(),
        aliases_input: true,
        mutates_aliased_input: true,
    })));
    llir.add_edge(dest, competing_read, ());
    llir.add_edge(dest, mutation, ());

    assert!(matches!(
        plan_static_llir_resources(&llir, &FxHashMap::default()),
        Err(ResourceViolation::AliasingHazard { .. })
    ));
}

#[test]
fn mutating_alias_batches_check_mutations_after_the_first_word() {
    use orbitkv_compiler::{hlir::Input, op::LLIROp};

    let mut llir = LLIRGraph::default();
    let mut version = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
    let mut before_last = version;
    for _ in 0..65 {
        before_last = version;
        let mutation = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
            bytes: 16.into(),
            aliases_input: true,
            mutates_aliased_input: true,
        })));
        llir.add_edge(version, mutation, ());
        version = mutation;
    }

    plan_static_llir_resources(&llir, &FxHashMap::default())
        .expect("a dependency-ordered mutation chain is valid");

    let competing_read = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(TestKernel {
        bytes: 16.into(),
        aliases_input: false,
        mutates_aliased_input: false,
    })));
    llir.add_edge(before_last, competing_read, ());
    assert!(matches!(
        plan_static_llir_resources(&llir, &FxHashMap::default()),
        Err(ResourceViolation::AliasingHazard { .. })
    ));
}

#[test]
fn static_llir_plan_accounts_for_search_grown_fused_source_and_parameters() {
    use crate::kernel::fusion::{
        elementwise::CudaUnaryElementwise,
        markers::{FusionEnd, FusionStart},
    };
    use orbitkv_compiler::{
        dtype::DType,
        hlir::{Input, Output},
        op::LLIROp,
    };

    let shape = vec![128.into()];
    let strides = vec![1.into()];
    let mut llir = LLIRGraph::default();
    let input = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
    let start = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionStart {
        shape: shape.clone(),
        strides: strides.clone(),
        dtype: DType::F32,
    })));
    let unary = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
        CudaUnaryElementwise {
            op: "Sin".to_string(),
            shape: shape.clone(),
            in_strides: strides.clone(),
            out_strides: strides.clone(),
            dtype: DType::F32,
        },
    )));
    let end = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionEnd {
        shape,
        strides,
        dtype: DType::F32,
    })));
    let output = llir.add_node(LLIROp::new::<Output>(Box::new(Output::default())));
    llir.add_edge(input, start, ());
    llir.add_edge(start, unary, ());
    llir.add_edge(unary, end, ());
    llir.add_edge(end, output, ());

    let plan = plan_static_llir_resources(&llir, &FxHashMap::default()).unwrap();
    assert_eq!(plan.kernels.len(), 1);
    let kernel = &plan.kernels[0];
    assert_eq!(kernel.name, "FusedRegion");
    assert!(kernel.source_bytes.is_some_and(|bytes| bytes > 0));
    assert_eq!(kernel.parameter_bytes, 2 * 8); // output + one external input
    assert_eq!(kernel.grid, [1, 1, 1]);
    assert_eq!(kernel.block, [128, 1, 1]);
}

#[test]
fn fused_diamond_intermediates_are_registers_not_static_buffers() {
    use crate::kernel::fusion::{
        elementwise::{CudaBinaryElementwise, CudaUnaryElementwise},
        markers::{FusionEnd, FusionStart},
    };
    use orbitkv_compiler::{
        dtype::DType,
        hlir::{Input, Output},
        op::LLIROp,
    };

    let shape = vec![128.into()];
    let strides = vec![1.into()];
    let mut llir = LLIRGraph::default();
    let lhs = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
    let rhs = llir.add_node(LLIROp::new::<Input>(Box::new(Input::default())));
    let lhs_start = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionStart {
        shape: shape.clone(),
        strides: strides.clone(),
        dtype: DType::F32,
    })));
    let rhs_start = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionStart {
        shape: shape.clone(),
        strides: strides.clone(),
        dtype: DType::F32,
    })));
    let lhs_unary = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
        CudaUnaryElementwise {
            op: "Sin".to_string(),
            shape: shape.clone(),
            in_strides: strides.clone(),
            out_strides: strides.clone(),
            dtype: DType::F32,
        },
    )));
    let rhs_unary = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
        CudaUnaryElementwise {
            op: "Sqrt".to_string(),
            shape: shape.clone(),
            in_strides: strides.clone(),
            out_strides: strides.clone(),
            dtype: DType::F32,
        },
    )));
    let binary = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(
        CudaBinaryElementwise {
            op: "Add".to_string(),
            out_shape: shape.clone(),
            a_stride: strides.clone(),
            b_stride: strides.clone(),
            out_stride: strides.clone(),
            dtype: DType::F32,
        },
    )));
    let end = llir.add_node(LLIROp::new::<dyn KernelOp>(Box::new(FusionEnd {
        shape,
        strides,
        dtype: DType::F32,
    })));
    let output = llir.add_node(LLIROp::new::<Output>(Box::new(Output::default())));

    llir.add_edge(lhs, lhs_start, ());
    llir.add_edge(rhs, rhs_start, ());
    llir.add_edge(lhs_start, lhs_unary, ());
    llir.add_edge(rhs_start, rhs_unary, ());
    llir.add_edge(lhs_unary, binary, ());
    llir.add_edge(rhs_unary, binary, ());
    llir.add_edge(binary, end, ());
    llir.add_edge(end, output, ());

    let plan = plan_static_llir_resources(&llir, &FxHashMap::default()).unwrap();

    // Only the FusionEnd output is materialized. The two unary branches
    // and the binary root are locals in the generated fused kernel.
    assert_eq!(plan.intermediate_lower_bound_bytes, 128 * 4);
}
