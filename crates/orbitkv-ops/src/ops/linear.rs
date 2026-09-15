//! Block-scaled FP8 linear semantics, independent of implementation libraries.
use orbitkv_compiler::{
    dtype::DType,
    op::CustomOp,
    prelude::{Expression, GraphTensor},
};

/// Numerical tile of the current activation and weight quantization contract.
/// Changing this changes the operation, not just a kernel tuning parameter.
pub const FP8_SCALE_BLOCK: usize = 128;
/// Largest finite E4M3 value used by dynamic activation quantization.
pub const FP8_MAX_FINITE: f32 = 448.0;
/// Rounded F32 reciprocal used before multiplying the block maximum. Keeping
/// the order explicit preserves FP8 midpoint decisions across implementations.
pub const FP8_INVERSE_MAX_FINITE: f32 = 1.0 / FP8_MAX_FINITE;
/// Clamp each activation block's absolute maximum before deriving its scale.
pub const QUANTIZATION_AMAX_FLOOR: f32 = 1.0e-4;

pub const BLOCK_SCALED_LINEAR_DECLARATIONS: &str =
    "(relation block-scaled-linear-op (i64 Expression Expression Expression i64 i64))";

/// Compile-time contract for `BF16[M,K] x FP8[N,K]^T -> BF16[M,N]`.
#[derive(Clone, Copy, Debug)]
pub struct BlockScaledLinearSpec {
    pub rows: Expression,
    pub output_features: usize,
    pub input_features: usize,
    pub weight_block_rows: usize,
    pub weight_block_columns: usize,
}

/// Insert the provider-neutral block-FP8 linear operation.
///
/// Activations are dynamically quantized independently for every `(row, 128-K)`
/// tile. Checkpoint weights use one inverse scale for every `(128-N, 128-K)`
/// tile. F32 operations use nearest-even rounding at each boundary: the scale is
/// `max(amax, QUANTIZATION_AMAX_FLOOR) * FP8_INVERSE_MAX_FINITE`, and each value
/// is multiplied by the rounded F32 reciprocal of that scale before conversion
/// to E4M3 (also nearest-even). Division by the maximum or by the scale is not
/// an interchangeable expression at quantization midpoints. Products accumulate
/// in F32 and are rounded once to BF16.
pub fn block_scaled_linear(
    input: GraphTensor,
    weight: GraphTensor,
    weight_scale: GraphTensor,
    spec: BlockScaledLinearSpec,
) -> GraphTensor {
    assert!(spec.output_features > 0 && spec.input_features > 0);
    assert!(spec.weight_block_rows > 0 && spec.weight_block_columns > 0);
    assert_eq!(input.dims(), [spec.rows, spec.input_features.into()]);
    assert_eq!(
        weight.dims(),
        [
            Expression::from(spec.output_features),
            Expression::from(spec.input_features),
        ]
    );
    assert_eq!(
        weight_scale.dims(),
        [
            Expression::from(spec.output_features.div_ceil(spec.weight_block_rows)),
            Expression::from(spec.input_features.div_ceil(spec.weight_block_columns)),
        ]
    );
    assert_eq!(input.dtype, DType::Bf16);
    assert_eq!(weight.dtype, DType::F8E4M3);
    assert_eq!(weight_scale.dtype, DType::F32);
    assert_eq!(spec.weight_block_rows, FP8_SCALE_BLOCK);
    assert_eq!(spec.weight_block_columns, FP8_SCALE_BLOCK);
    assert!(spec.input_features.is_multiple_of(FP8_SCALE_BLOCK));
    assert!(
        [weight, weight_scale]
            .iter()
            .all(|tensor| tensor.graph_ref == input.graph_ref),
        "block-scaled linear inputs must belong to one graph"
    );
    let graph = input.graph();
    graph.custom_op(
        BlockScaledLinear { spec },
        vec![input, weight, weight_scale],
        (spec.rows, spec.output_features),
        DType::Bf16,
    )
}

#[derive(Debug)]
struct BlockScaledLinear {
    spec: BlockScaledLinearSpec,
}

impl CustomOp for BlockScaledLinear {
    fn compiler_declarations(&self) -> &'static str {
        BLOCK_SCALED_LINEAR_DECLARATIONS
    }

    fn compiler_facts(&self, custom_op_id: usize) -> String {
        format!(
            "(block-scaled-linear-op {custom_op_id} {} (MNum {}) (MNum {}) {} {})",
            self.spec.rows.to_egglog(),
            self.spec.output_features,
            self.spec.input_features,
            self.spec.weight_block_rows,
            self.spec.weight_block_columns,
        )
    }
}
