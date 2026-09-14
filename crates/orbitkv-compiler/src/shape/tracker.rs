use std::fmt::Display;

use itertools::Itertools;
use tinyvec::ArrayVec;

use crate::prelude::*;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub struct ShapeTracker {
    pub dims: ArrayVec<[Expression; 10]>,
    pub strides: ArrayVec<[Expression; 10]>,
    /// Bits per element in memory storage. Controls byte-size computation.
    /// Defaults to 32 (F32). Set from dtype.bits() at tensor creation.
    pub element_stride_bits: usize,
}

impl Display for ShapeTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) : ({})",
            self.dims.iter().map(|e| format!("{e}")).join(", "),
            self.strides.iter().map(|e| format!("{e}")).join(", ")
        )
    }
}

impl ShapeTracker {
    /// Make a new row-major shape tracker. Defaults element_stride_bits to 32 (F32).
    pub fn new(dims: impl ToShape) -> ShapeTracker {
        let mut s = Self {
            dims: Default::default(),
            strides: Default::default(),
            element_stride_bits: 32,
        };
        let mut stride = expr('z');
        for d in dims.to_shape().into_iter().rev() {
            // A size may not depend on the runtime loop index, though a
            // stride must.
            assert!(
                !d.uses_reserved_index(),
                "dimension size {d:?} uses the reserved runtime loop index; \
                 it is an index, not a dimension"
            );
            s.dims.insert(0, d);
            s.strides.insert(0, stride);
            stride *= d;
        }
        s
    }

    /// Make a new row-major shape tracker with explicit element stride in bits.
    pub fn new_with_element_bits(dims: impl ToShape, element_bits: usize) -> ShapeTracker {
        let mut s = Self::new(dims);
        s.element_stride_bits = element_bits;
        s
    }

    /// Set element stride bits. Chainable builder.
    pub fn with_element_bits(mut self, bits: usize) -> Self {
        self.element_stride_bits = bits;
        self
    }

    /// Make a new shape tracker with fake dimensions
    pub fn fake(dims: impl ToShape) -> Self {
        let mut s = Self {
            dims: Default::default(),
            strides: Default::default(),
            element_stride_bits: 32,
        };
        for d in dims.to_shape().into_iter() {
            s.dims.push(d);
            s.strides.push(0.into());
        }
        s
    }

    /// Make a new shape tracker with custom strides
    pub fn new_strided(dims: impl ToShape, strides: impl ToShape) -> Self {
        let dims = dims.to_shape();
        let strides = strides.to_shape();
        assert_eq!(
            dims.len(),
            strides.len(),
            "Dimensions and strides need to be the same size!"
        );
        let mut s = Self {
            dims: Default::default(),
            strides: Default::default(),
            element_stride_bits: 32,
        };
        for (dim, stride) in dims.into_iter().zip(strides) {
            s.dims.push(dim);
            s.strides.push(stride);
        }
        s
    }

    /// Add dim along a certian axis
    pub fn add_dim(
        &mut self,
        axis: usize,
        dim: impl Into<Expression>,
        stride: impl Into<Expression>,
    ) {
        self.dims.insert(axis, dim.into());
        self.strides.insert(axis, stride.into());
    }

    /// Add fake dim along a certian axis
    pub fn expand_dim(&mut self, axis: usize, dim: impl Into<Expression>) {
        self.add_dim(axis, dim, 0);
    }

    /// Expand this shape to a new shape following PyTorch semantics
    pub fn expand(&mut self, new_shape: impl ToShape) {
        let new_shape = new_shape.to_shape();
        assert!(
            new_shape.len() >= self.len(),
            "Cannot expand from {} dims to {} dims",
            self.len(),
            new_shape.len()
        );

        while self.len() < new_shape.len() {
            self.expand_dim(0, 1);
        }

        for (axis, ((size, dim), stride)) in new_shape
            .into_iter()
            .zip(&mut self.dims)
            .zip(&mut self.strides)
            .enumerate()
        {
            if *dim == size {
                continue;
            }
            if dim.to_usize() == Some(1) {
                *dim = size;
                *stride = 0.into();
            } else {
                let (dim_simplified, size_simplified) = (dim.simplify(), size.simplify());
                if dim_simplified == size_simplified {
                    *dim = size;
                } else {
                    panic!(
                        "Cannot expand dim {axis} from {dim} to {size} \
                         (simplified: {dim_simplified} vs {size_simplified})",
                    );
                }
            }
        }
    }

    /// Tile the tensor along each existing dimension without materializing new storage.
    pub fn repeat(&mut self, repeats: impl ToShape) {
        let repeats = repeats.to_shape();
        assert_eq!(
            repeats.len(),
            self.len(),
            "Repeat shape ({}) doesn't match tensor dimensions ({})",
            repeats.len(),
            self.len()
        );

        for ((dim, stride), repeat) in self
            .dims
            .iter_mut()
            .zip(self.strides.iter_mut())
            .zip(repeats)
        {
            // r == 1 leaves the axis untouched; skip the mod-wrap so untiled
            // axes keep clean stride expressions.
            if repeat == Expression::from(1) {
                continue;
            }
            let original_dim = *dim;
            *dim = (*dim * repeat).simplify();
            *stride = stride.substitute('z', expr('z') % original_dim).simplify();
        }
    }

    /// Remove a dimension
    pub fn remove_dim(&mut self, axis: usize) -> Expression {
        self.strides.remove(axis);
        self.dims.remove(axis)
    }

    /// Permute the dimensions
    pub fn permute(&mut self, axes: impl ToAxes) {
        let axes = axes.to_axes();
        assert!(
            axes.len() == self.len(),
            "Permute axes ({}) doesn't match shape axes ({})",
            axes.len(),
            self.len()
        );
        self.dims = axes.iter().map(|i| self.dims[*i]).collect();
        self.strides = axes.iter().map(|i| self.strides[*i]).collect();
    }

    /// Create an expression to translate logical indexes into physical indexes, without expression simplification
    pub fn index_expression_no_simplify(&self) -> Expression {
        if self.is_contiguous() {
            return 'z'.into();
        }
        let mut ind_expr = 0.into(); // The final index expression
        let mut current_elem_size = expr(1); // Keep track of the size of each element of the current dim (last dim elem size: 1)

        // Loop through all dims in reverse order
        for (d, s) in self.dims.iter().zip(&self.strides).rev() {
            // Don't include fake dimensions in the index expression
            if *s == 0 {
                current_elem_size *= d;
                continue;
            }
            let mut dim_ind = expr('z');
            // Remove other dim components
            dim_ind /= current_elem_size;
            // Get position in current dim
            dim_ind %= d;
            // Add to index expression (substitute z in stride with the dimension index)
            ind_expr += s.substitute('z', dim_ind);
            // Keep track of element size for next dimension
            current_elem_size *= d;
        }
        ind_expr
    }

    /// Create an expression to translate logical indexes into physical indexes
    pub fn index_expression(&self) -> Expression {
        self.index_expression_no_simplify().simplify()
    }

    /// If this expression evaluates to 0, the logical index is invalid. Otherwise it is valid. No simplification
    pub fn valid_expression_no_simplify(&self) -> Expression {
        true.into()
    }

    /// If this expression evaluates to 0, the logical index is invalid. Otherwise it is valid
    pub fn valid_expression(&self) -> Expression {
        self.valid_expression_no_simplify().simplify()
    }

    /// Check if contiguous (no permutes or fake dimensions)
    pub fn is_contiguous(&self) -> bool {
        self.dims
            .iter()
            .rev()
            .scan(expr('z'), |acc, d| {
                let r = *acc;
                *acc *= d;
                Some(r)
            })
            .zip(self.strides.iter().rev())
            .all(|(a, b)| a == *b)
    }

    /// The number of elements in this tensor, including padding and mask
    pub fn n_elements(&self) -> Expression {
        self.dims.into_iter().product::<Expression>().max(1)
    }

    /// The number of elements in this tensor, not including pads and mask
    pub fn n_physical_elements(&self) -> Expression {
        self.dims
            .into_iter()
            .zip(&self.strides)
            .filter(|(_, s)| **s != 0)
            .map(|(s, _)| s)
            .product::<Expression>()
            .max(1)
    }

    /// The number of physical elements this view can address: max linear
    /// offset + 1. Differs from `n_physical_elements` for sliced views, where
    /// the addressed span exceeds the count of viewed elements (e.g. a
    /// (3,4)-slice of (3,16) views 12 elements but addresses offsets 0..36).
    /// Stride expressions are nondecreasing in z, so each axis peaks at
    /// `dim - 1`.
    pub fn physical_span(&self) -> Expression {
        self.dims
            .into_iter()
            .zip(&self.strides)
            .map(|(d, s)| s.substitute('z', d - 1))
            .sum::<Expression>()
            .max(0)
            .simplify()
            + 1
    }

    /// The number of dimensions
    pub fn len(&self) -> usize {
        self.dims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn all_axes(&self) -> Vec<usize> {
        (0..self.len()).collect()
    }

    pub fn last_axis(&self) -> usize {
        self.len() - 1
    }

    /// Required bytes to store this tensor's physical elements. Rounds up to nearest byte.
    pub fn required_total_bytes(&self) -> Expression {
        (self.n_physical_elements() * self.element_stride_bits).ceil_div(8)
    }

    /// Create a contiguous version, preserving element_stride_bits
    pub fn contiguous(self) -> Self {
        Self::new_with_element_bits(
            self.dims
                .into_iter()
                .map(|i| i.simplify())
                .collect::<Vec<_>>(),
            self.element_stride_bits,
        )
    }

    /// Realize the true shape and convert it to usizes. All dyn dims must be replaced already
    pub fn shape_usize(&self) -> Vec<usize> {
        self.dims.iter().map(|e| e.to_usize().unwrap()).collect()
    }

    /// Given a dyn dim map, resolve global dyn dims into known dims
    pub fn resolve_dyn_dims(&mut self, dyn_dim_map: &DynMap) {
        for d in self.dims.iter_mut().chain(&mut self.strides) {
            *d = d.resolve_vars(dyn_dim_map);
        }
    }

    /// Merge two dimensions together.
    ///
    /// The merged dimension is computed as `outer_stride * inner_stride`
    /// The merged stride is computed as: `outer_stride(z / inner_dim) + inner_stride(z % inner_dim)`
    pub fn merge_dims(&mut self, axis1: usize, axis2: usize) {
        assert!(axis1 < axis2, "axis1 must be less than axis2");
        // Move axis2 to axis1+1 if not already adjacent
        if axis2 != axis1 + 1 {
            let dim = self.dims.remove(axis2);
            let stride = self.strides.remove(axis2);
            self.dims.insert(axis1 + 1, dim);
            self.strides.insert(axis1 + 1, stride);
        }
        let inner_dim = self.dims[axis1 + 1];
        let outer_stride = self.strides[axis1];
        let inner_stride = self.strides[axis1 + 1];
        // When outer_stride == inner_stride * inner_dim, the dims are contiguous
        // and the merged stride is just inner_stride (avoids complex z/N, z%N expressions
        // that the e-graph simplifier can't always reduce).
        let merged_stride = if (inner_stride * inner_dim)
            .simplify()
            .egglog_equal(outer_stride.simplify())
        {
            inner_stride
        } else {
            let z = expr('z');
            (outer_stride.substitute('z', z / inner_dim)
                + inner_stride.substitute('z', z % inner_dim))
            .simplify()
        };
        self.dims[axis1] = self.dims[axis1] * self.dims[axis1 + 1];
        self.strides[axis1] = merged_stride;
        self.dims.remove(axis1 + 1);
        self.strides.remove(axis1 + 1);
    }

    /// Flatten all dimensions into a single dimension by iteratively merging.
    pub fn flatten(&mut self) {
        while self.dims.len() > 1 {
            self.merge_dims(0, 1);
        }
    }

    /// Split a dim into 2 dims, new dim is placed directly after original dim
    pub fn split_dims(&mut self, axis: usize, new_dim_size: impl Into<Expression>) {
        let new_dim_size = new_dim_size.into();
        assert!(
            new_dim_size.as_num().is_none_or(|n| n > 0),
            "split_dims inner dimension must be positive, got {new_dim_size}"
        );
        let old_dim = self.dims[axis];
        let outer_dim = (old_dim / new_dim_size).simplify();
        assert!(
            (outer_dim * new_dim_size)
                .simplify()
                .egglog_equal(old_dim.simplify()),
            "split_dims requires the old dimension ({old_dim}) to be exactly divisible by the inner dimension ({new_dim_size})"
        );

        let old_stride = self.strides[axis];
        let zero = old_stride.substitute('z', 0).simplify();
        let outer_stride = (old_stride.substitute('z', expr('z') * new_dim_size) - zero).simplify();
        let inner_stride = old_stride;

        assert!(
            split_stride_is_separable(old_stride, old_dim, outer_dim, new_dim_size),
            "split_dims cannot represent stride {old_stride} as independent outer/inner strides for inner dimension {new_dim_size}"
        );

        self.dims.insert(axis + 1, new_dim_size);
        self.strides.insert(axis + 1, inner_stride);
        self.dims[axis] = outer_dim;
        self.strides[axis] = outer_stride;
    }
}

fn split_stride_is_separable(
    old_stride: Expression,
    _old_dim: Expression,
    outer_dim: Expression,
    inner_dim: Expression,
) -> bool {
    if let (Some(outer), Some(inner)) = (outer_dim.as_num(), inner_dim.as_num()) {
        if outer < 0 || inner <= 0 {
            return false;
        }
        if split_stride_is_separable_concrete(old_stride, outer as usize, inner as usize) {
            return true;
        }
    }

    if split_stride_symbolic_base_identity(old_stride, inner_dim) {
        return true;
    }

    if let Some(inner) = inner_dim.as_num()
        && inner > 0
    {
        return split_stride_symbolic_inner_points(old_stride, inner as usize);
    }

    false
}

fn split_stride_is_separable_concrete(
    old_stride: Expression,
    outer_dim: usize,
    inner_dim: usize,
) -> bool {
    let Some(zero) = eval_stride_at(old_stride, 0) else {
        return false;
    };

    for outer in 0..outer_dim {
        let Some(outer_base) = eval_stride_at(old_stride, outer * inner_dim) else {
            return false;
        };
        for inner in 0..inner_dim {
            let Some(old) = eval_stride_at(old_stride, outer * inner_dim + inner) else {
                return false;
            };
            let Some(inner_base) = eval_stride_at(old_stride, inner) else {
                return false;
            };
            if old != outer_base - zero + inner_base {
                return false;
            }
        }
    }
    true
}

fn split_stride_symbolic_base_identity(old_stride: Expression, inner_dim: Expression) -> bool {
    let zero = old_stride.substitute('z', 0).simplify();
    let outer_var = fresh_split_var(&[inner_dim, old_stride], &[]);
    let inner_var = fresh_split_var(&[inner_dim, old_stride], &[outer_var]);
    let outer = expr(outer_var);
    let inner = expr(inner_var);
    let old_split_stride = old_stride
        .substitute('z', outer * inner_dim + inner)
        .simplify();
    let new_split_stride = ((old_stride.substitute('z', outer * inner_dim) - zero)
        + old_stride.substitute('z', inner))
    .simplify();
    old_split_stride.egglog_equal(new_split_stride)
}

fn split_stride_symbolic_inner_points(old_stride: Expression, inner_dim: usize) -> bool {
    let zero = old_stride.substitute('z', 0).simplify();
    let outer = expr('z');
    let inner_dim_expr = expr(inner_dim);
    for inner in 0..inner_dim {
        let old_split_stride = old_stride
            .substitute('z', outer * inner_dim_expr + inner)
            .simplify();
        let new_split_stride = ((old_stride.substitute('z', outer * inner_dim_expr) - zero)
            + old_stride.substitute('z', inner))
        .simplify();
        if old_split_stride != new_split_stride && !old_split_stride.egglog_equal(new_split_stride)
        {
            return false;
        }
    }
    true
}

fn eval_stride_at(stride: Expression, z: usize) -> Option<i64> {
    let mut stack = Vec::new();
    for term in stride.terms.read().iter() {
        match *term {
            Term::Num(n) => stack.push(n),
            Term::Var(v) if v.is_reserved() => stack.push(z as i64),
            Term::Var(_) => return None,
            _ => {
                let a = stack.pop()?;
                let b = stack.pop()?;
                stack.push(term.as_op()?(a, b)?);
            }
        }
    }
    stack.pop()
}

/// A scratch dimension for the split_dims proof, distinct from `excluded` and
/// from anything `expressions` already mentions.
fn fresh_split_var(expressions: &[Expression], excluded: &[Symbol]) -> Symbol {
    (0..)
        .map(|n| Symbol::new(&format!("split{n}")))
        .find(|candidate| {
            !excluded.contains(candidate)
                && expressions
                    .iter()
                    .all(|expr| !expr.to_symbols().contains(candidate))
        })
        .expect("usize is not exhaustible")
}

#[cfg(test)]
#[path = "../../tests/unit/shape/tracker/mod.rs"]
mod tests;
