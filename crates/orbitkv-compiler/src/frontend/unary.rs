use itertools::Itertools;

use crate::prelude::*;
use std::ops::{Add, Mul, Neg};

/// Scatter `arange(ax_size)` into rank positions to convert per-element ranks
/// into sort indices. Handles multi-dim by computing flat scatter offsets.
fn scatter_ranks_to_sort_indices(
    ranks: GraphTensor,
    dims: Vec<Expression>,
    axis: usize,
    g: &mut Graph,
) -> GraphTensor {
    let ax_size = dims[axis];
    let ndim = dims.len();

    // Values: [0, 1, ..., ax_size-1] along axis, expanded to full shape
    let mut values = g.arange(ax_size);
    let mut zeros = g.iota(Expression::from(0usize), ax_size);
    for (i, &dim) in dims.iter().enumerate() {
        if i != axis {
            values = values.expand_dim(i, dim);
            zeros = zeros.expand_dim(i, dim);
        }
    }

    if ndim == 1 {
        return values.scatter(ranks, zeros);
    }

    // Multi-dim: ranks are per-axis (0..ax_size) but scatter uses flat indices.
    // Compute: adjusted = base_offset + ranks * axis_stride
    let mut strides = vec![Expression::from(1usize); ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = (strides[d + 1] * dims[d + 1]).simplify();
    }
    let axis_stride = strides[axis];
    let ranks_scaled = ranks * axis_stride;

    let mut base_offset: Option<GraphTensor> = None;
    for d in 0..ndim {
        if d == axis {
            continue;
        }
        let expr = Expression::from('z') * strides[d];
        let mut component = g.iota(expr, dims[d]);
        for (i, &dim) in dims.iter().enumerate() {
            if i != d {
                component = component.expand_dim(i, dim);
            }
        }
        base_offset = Some(match base_offset {
            None => component,
            Some(acc) => acc + component,
        });
    }

    let adjusted = match base_offset {
        None => ranks_scaled,
        Some(base) => base + ranks_scaled,
    };

    values.scatter(adjusted, zeros)
}

impl Neg for GraphTensor {
    type Output = GraphTensor;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}

impl GraphTensor {
    /// Base 2 log
    pub fn log2(self) -> GraphTensor {
        let new_id = self.graph().add_op(
            crate::hlir::Log2 {
                input_shape: self.shape,
                ..Default::default()
            },
            &[self.id],
        );
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor {
        let new_id = self.graph().add_op(
            crate::hlir::Exp2 {
                input_shape: self.shape,
                ..Default::default()
            },
            &[self.id],
        );
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor {
        (self * (1.0 / f32::ln(2.))).exp2()
    }

    /// Natural log
    pub fn log(self) -> GraphTensor {
        self.log2() * f32::ln(2.)
    }

    /// Take the reciprocal of each element
    pub fn reciprocal(self) -> GraphTensor {
        let new_id = self.graph().add_op(
            crate::hlir::Recip {
                input_shape: self.shape,
                ..Default::default()
            },
            &[self.id],
        );
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor {
        let new_id = self.graph().add_op(
            crate::hlir::Sin {
                input_shape: self.shape,
                ..Default::default()
            },
            &[self.id],
        );
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// The cos(x) function
    pub fn cos(self) -> GraphTensor {
        ((std::f32::consts::PI / 2.) - self).sin()
    }

    /// Square every element in the tensor
    pub fn square(self) -> GraphTensor {
        self * self
    }

    /// The square root function
    pub fn sqrt(self) -> GraphTensor {
        let new_id = self.graph().add_op(
            crate::hlir::Sqrt {
                input_shape: self.shape,
                ..Default::default()
            },
            &[self.id],
        );
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Scale so std is 1.0
    pub fn std_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        (self * self)
            .mean(axes.to_axes())
            .add(epsilon)
            .sqrt()
            .reciprocal()
            .expand_to_shape_on_axes(self.shape, axes)
            .mul(self)
    }

    /// Center so mean is 0.0
    pub fn mean_norm(self, axes: impl ToAxes) -> GraphTensor {
        self - self
            .mean(axes.to_axes())
            .expand_to_shape_on_axes(self.shape, axes)
    }

    /// Applies a layer norm along an axis
    pub fn layer_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        self.mean_norm(axes.to_axes()).std_norm(axes, epsilon)
    }

    /// Normalize the tensor along `axes` using an Lp norm.
    pub fn normalize(self, p: f32, axes: impl ToAxes, epsilon: f32) -> GraphTensor {
        let norm = self.abs().pow(p).sum(axes.to_axes()).pow(1.0 / p);
        self / norm
            .maximum_f32(epsilon)
            .expand_to_shape_on_axes(self.shape, axes)
    }

    /// Applies a softmax function along an axis
    pub fn softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self
            - self
                .max(axes.to_axes())
                .expand_to_shape_on_axes(self.shape, axes.to_axes());
        let exp = m.exp();
        exp / exp
            .sum(axes.to_axes())
            .expand_to_shape_on_axes(self.shape, axes)
    }

    /// Applies a log softmax function along an axis
    pub fn log_softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self
            - self
                .max(axes.to_axes())
                .expand_to_shape_on_axes(self.shape, axes.to_axes());
        m - m
            .exp()
            .sum(axes.to_axes())
            .log()
            .expand_to_shape_on_axes(m.shape, axes)
    }

    /// Gets the indices of the maximum elements along an axis.
    ///
    /// For finite inputs, equal maxima resolve to the highest index along the
    /// reduced axis. Backends must preserve this tie policy when fusing argmax.
    pub fn argmax(self, axis: usize) -> GraphTensor {
        // Get one-hot along last dimension
        let x_equal = self
            .eq(self.max(axis).expand_dim(axis, self.dims()[axis]))
            .cast(DType::Int);
        // Create index arange for last dimension
        let r = self.graph().arange(self.dims()[axis]);
        let axes = (0..self.shape.len()).filter(|i| *i != axis).collect_vec();
        // Multiply one-hot by expanded index arange
        (x_equal * r.expand_to_shape_on_axes(self.shape, axes)).max(axis)
    }

    /// Gets the indices of the minimum elements along an axis.
    ///
    /// For finite inputs, equal minima resolve to the highest index along the
    /// reduced axis, matching [`Self::argmax`].
    pub fn argmin(self, axis: usize) -> GraphTensor {
        (-self).argmax(axis)
    }

    /// Compute the sample variance along axes
    pub fn var(self, axes: impl ToAxes) -> GraphTensor {
        self.var_options(axes, 1)
    }

    /// Compute the sample variance along an axes with options
    pub fn var_options(self, axes: impl ToAxes, correction: usize) -> GraphTensor {
        let axes = axes.to_axes();
        let n = axes
            .to_axes()
            .into_iter()
            .map(|i| self.dims()[i])
            .product::<Expression>();
        let mean = self
            .mean(axes.to_axes())
            .expand_to_shape_on_axes(self.shape, axes.to_axes());
        let centered = self - mean;
        (centered * centered).sum(axes) / (n - correction)
    }

    /// Compute the sample standard deviation along axes
    pub fn std(self, axes: impl ToAxes) -> GraphTensor {
        self.std_options(axes, 1)
    }

    /// Compute the standard deviation along axes with options
    pub fn std_options(self, axes: impl ToAxes, correction: usize) -> GraphTensor {
        self.var_options(axes, correction).sqrt()
    }

    /// Take the absolute value
    pub fn abs(self) -> GraphTensor {
        match self.dtype {
            DType::U4 | DType::U8 | DType::U16 => self,
            DType::I4 | DType::I8 | DType::I16 | DType::Int | DType::I64 => {
                let zero = self
                    .graph()
                    .constant_float(0.0)
                    .cast(self.dtype)
                    .expand_rhs(self.shape);
                let negative = self.lt(zero).cast(self.dtype);
                self * (1.0 - negative * 2.0)
            }
            _ => self.relu() + (-self).relu(),
        }
    }

    /// Get the sign of each element, '1' for positive and '-1' for negative
    pub fn sign(self) -> GraphTensor {
        self / (self.abs() + 1e-10)
    }

    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor {
        self.maximum_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        (1. + (-self).exp()).reciprocal()
    }

    /// The swish (aka silu) activation function
    pub fn swish(self) -> GraphTensor {
        self * self.sigmoid()
    }

    /// The silu (aka swish) activation function
    pub fn silu(self) -> GraphTensor {
        self.swish()
    }

    /// The tanh activation function
    pub fn tanh(self) -> GraphTensor {
        (self * 2.0).sigmoid() * 2.0 - 1.0
    }

    /// The leaky relu activation function
    pub fn leaky_relu(self, neg_slope: f32) -> GraphTensor {
        self.relu() - (self * -neg_slope).relu()
    }

    /// The Gaussian Error Linear Unit activation function: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    ///
    /// This is the exact (erf-based) form, matching PyTorch's default `aten.gelu`
    /// (`approximate="none"`). `erf` is composed from existing primitives via the
    /// Abramowitz & Stegun 7.1.26 rational+exp approximation (max abs error ~1.5e-7),
    /// so this is a single, deterministic elementwise composition with no
    /// special-cased kernel. For the cheaper tanh approximation, see
    /// [`gelu_fast_tanh_approximation`](Self::gelu_fast_tanh_approximation).
    #[allow(clippy::excessive_precision)]
    pub fn gelu(self) -> GraphTensor {
        // erf(u), u = x / sqrt(2), via A&S 7.1.26:
        //   t = 1 / (1 + p*|u|)
        //   erf(u) = sign(u) * (1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) * exp(-u^2))
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        const P: f32 = 0.3275911;
        const A1: f32 = 0.254829592;
        const A2: f32 = -0.284496736;
        const A3: f32 = 1.421413741;
        const A4: f32 = -1.453152027;
        const A5: f32 = 1.061405429;

        let u = INV_SQRT2 * self;
        let t = (1. + P * u.abs()).reciprocal();
        // Horner form of the degree-5 polynomial in t.
        let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
        let erf = u.sign() * (1. - poly * (-(u * u)).exp());
        0.5 * self * (1. + erf)
    }

    /// The tanh approximation of GELU (PyTorch's `approximate="tanh"`):
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    ///
    /// Cheaper than the exact [`gelu`](Self::gelu) (fewer ops, ~3e-4 max error) and
    /// what cuBLASLt's GELU epilogue / the GLUMoE Gemma-GELU recognition match.
    #[allow(clippy::excessive_precision)]
    pub fn gelu_fast_tanh_approximation(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9fc4465557831b614b56dd645eebc940ca0fa1bb/tinygrad/tensor.py#L1162C26-L1162C104
        let scaled = 1.5957691216 * self * (1. + 0.044715 * self * self);
        self * scaled.sigmoid()
    }

    /// Compute the sorted indexes of this tensor along a certian axis
    pub fn argsort(self, axis: usize, descending: bool) -> GraphTensor {
        // Compare all elements with all other elements by making an axis
        let ax_size = self.dims()[axis];
        let a = self.expand_dim(axis + 1, ax_size) + 0.0;
        let b = self.expand_dim(axis, ax_size) + 1e-9;
        let cmp = if descending { a.gt(b) } else { a.lt(b) };
        // ind[j] = rank of element j (how many elements are smaller/larger)
        let ranks = (cmp.cast(DType::F32) + 0.0).sum(axis).cast(DType::Int);
        // Scatter original indices into rank positions to get sort indices
        scatter_ranks_to_sort_indices(ranks, self.dims(), axis, self.graph())
    }

    /// Stable argsort: like `argsort`, but breaks ties by original index
    /// (lower index first). Guarantees unique ranks even when values are equal.
    pub fn stable_argsort(self, axis: usize, descending: bool) -> GraphTensor {
        let ax_size = self.dims()[axis];
        let dims = self.dims();

        // Expanded shape: original dims with ax_size inserted at axis
        let mut exp_dims = dims.clone();
        exp_dims.insert(axis, ax_size);

        // Pairwise value tensors (* 1.0 forces materialization to avoid stride issues)
        let a_val = self.expand_dim(axis + 1, ax_size) * 1.0;
        let b_val = self.expand_dim(axis, ax_size) * 1.0;

        // Index tensors for tiebreaking
        let mut iota_a = self.graph().arange(ax_size).cast(DType::F32);
        for (i, dim) in exp_dims.iter().take(axis).enumerate() {
            iota_a = iota_a.expand_dim(i, *dim);
        }
        iota_a = iota_a.expand_dim(axis + 1, ax_size);
        for (i, dim) in exp_dims.iter().enumerate().skip(axis + 2) {
            iota_a = iota_a.expand_dim(i, *dim);
        }
        let mut iota_b = self.graph().arange(ax_size).cast(DType::F32);
        for (i, dim) in exp_dims.iter().take(axis + 1).enumerate() {
            iota_b = iota_b.expand_dim(i, *dim);
        }
        for (i, dim) in exp_dims.iter().enumerate().skip(axis + 2) {
            iota_b = iota_b.expand_dim(i, *dim);
        }

        // Lexicographic comparison with stable tiebreaking (lower index first):
        // ascending:  rank[j] = count of i where (x[i] < x[j]) || (x[i]==x[j] && i < j)
        // descending: rank[j] = count of i where (x[i] > x[j]) || (x[i]==x[j] && i < j)
        let primary = if descending {
            a_val.gt(b_val)
        } else {
            a_val.lt(b_val)
        };
        let idx_cmp = iota_a.lt(iota_b);
        let not_lt = 1.0 - a_val.lt(b_val).cast(DType::F32);
        let not_gt = 1.0 - a_val.gt(b_val).cast(DType::F32);
        let val_eq = not_lt * not_gt;
        let cmp = primary.cast(DType::F32) + val_eq * idx_cmp.cast(DType::F32);

        // Scatter original indices into rank positions to get sort indices
        let ranks = cmp.sum(axis).cast(DType::Int);
        scatter_ranks_to_sort_indices(ranks, dims, axis, self.graph())
    }

    /// Sort the tensor along a certian axis
    pub fn sort(self, axis: usize, descending: bool) -> GraphTensor {
        self.gather(self.argsort(axis, descending))
    }

    /// Sort and retrieve top-k **indexes**
    pub fn topk_indexes(self, k: usize, axis: usize) -> GraphTensor {
        // Stable sort is required: with exact value ties (e.g. softmax rows
        // where several entries underflow to exactly 0.0), the unstable
        // argsort produces duplicate ranks and the rank→index scatter emits
        // garbage — nondeterministically across search candidates, since tie
        // outcomes depend on each candidate's float details (FTZ, fusion).
        self.stable_argsort(axis, true).slice_along(..k, axis)
    }

    /// Sort and retrieve top-k **values** (largest first)
    pub fn topk_values(self, k: usize, axis: usize) -> GraphTensor {
        let top_k_idx = self.topk_indexes(k, axis);
        self.gather_elements(top_k_idx, axis)
    }

    /// Apply a cumulative reduction operation along dimensions
    ///
    /// See `cumsum` or `cummax` for usage examples.
    pub fn cumop(
        mut self,
        axes: impl ToAxes,
        op: impl Fn(GraphTensor, usize) -> GraphTensor,
        pad_elem: f32,
    ) -> Self {
        let n_dims = self.shape.len();
        for axis in axes.to_axes() {
            // Pad out length
            let mut kernel = vec![1.into(); n_dims];
            let mut padding = vec![(Expression::from(0), Expression::from(0)); n_dims];
            let orig_length = self.dims()[axis];
            padding[axis] = (orig_length - 1, 0.into());
            kernel[axis] = orig_length;
            self = self.pad(padding, pad_elem);
            // Unfold
            self = self.unfold(kernel, vec![1; n_dims], vec![1; n_dims]);
            // Remove non-cumulative dimensions
            for i in (0..n_dims).rev() {
                if i != axis {
                    self = self.squeeze(n_dims + i);
                }
            }
            // apply operation along cumulative dimensions
            self = op(self, n_dims);
        }
        self
    }

    /// Apply a cumulative sum along dimensions
    pub fn cumsum(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.sum(axes), 0.)
    }

    /// Apply a cumulative max along dimensions
    pub fn cummax(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.max(axes), f32::MIN)
    }

    /// Apply a cumulative product along dimensions
    pub fn cumprod(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.prod(axes), 1.)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/frontend/unary/mod.rs"]
pub(super) mod tests;
