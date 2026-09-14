use crate::hlir::*;
use crate::prelude::*;

impl Graph {
    /// A scalar expression constant
    pub fn constant(&mut self, i: impl Into<Expression>) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(Iota(i.into(), 1.into()), &[]),
            ShapeTracker::new(()),
            self,
            DType::Int,
        )
    }

    /// An exact scalar I64 constant built from 16-bit limbs.
    ///
    /// `Iota` stores values as the default 32-bit integer dtype, so emitting a
    /// large literal there and casting afterward has already truncated it.
    /// Horner assembly performs every multiply and add after promotion to I64
    /// and covers the complete signed 64-bit range without a new HLIR op.
    pub fn constant_i64(&mut self, value: i64) -> GraphTensor {
        let base = self.constant(1i64 << 16).cast(DType::I64);
        let mut result = self.constant(value >> 48).cast(DType::I64);
        for shift in [32, 16, 0] {
            let limb = self.constant((value >> shift) & 0xffff).cast(DType::I64);
            result = result * base + limb;
        }
        result
    }

    /// A scalar float constant
    pub fn constant_float(&mut self, i: f32) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(Constant(i), &[]),
            ShapeTracker::new(()),
            self,
            DType::F32,
        )
    }

    /// A scalar F64 constant. The value remains F64 through HLIR instead of
    /// being narrowed to F32 and widened again by a cast.
    pub fn constant_float64(&mut self, i: f64) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(ConstantF64(i), &[]),
            ShapeTracker::new(()),
            self,
            DType::F64,
        )
    }

    /// Iota expression
    pub fn iota(&mut self, i: impl Into<Expression>, shape: impl ToShape) -> GraphTensor {
        let sh = shape.to_shape();
        GraphTensor::from_id(
            self.add_op(
                Iota(
                    i.into().simplify(),
                    sh.iter().copied().product::<Expression>().simplify(),
                ),
                &[],
            ),
            ShapeTracker::new(sh),
            self,
            DType::Int,
        )
    }

    /// ARange from 0 to N
    pub fn arange(&mut self, to: impl Into<Expression>) -> GraphTensor {
        self.iota('z', to)
    }

    /// ARange from beginning to end
    pub fn arange_options(
        &mut self,
        start: impl Into<Expression>,
        end: impl Into<Expression>,
        step: impl Into<Expression>,
    ) -> GraphTensor {
        let (start, end, step) = (start.into(), end.into(), step.into());
        self.iota((Expression::from('z') * step) + start, (end - start) / step)
    }

    /// Lower left-hand triangle of 1s. Currently required to be square
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.tril
    pub fn tril(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).cast(DType::F32).expand_dim(0, size);
        let vertical = self.arange(size).cast(DType::F32).expand_dim(1, size);
        (horizontal - (diagonal as f32 + 1.)).lt(vertical)
    }

    /// Upper right-hand triangle of 1s
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.triu
    pub fn triu(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).cast(DType::F32).expand_dim(0, size);
        let vertical = self.arange(size).cast(DType::F32).expand_dim(1, size);
        (horizontal - (diagonal as f32 - 1.)).gt(vertical)
    }

    /// Stack tensors along a new dimension
    pub fn stack(&mut self, tensors: &[GraphTensor], axis: usize) -> GraphTensor {
        assert!(!tensors.is_empty(), "Cannot stack empty tensor list");
        let first = tensors[0].unsqueeze(axis);
        tensors[1..]
            .iter()
            .fold(first, |acc, t| acc.concat_along(t.unsqueeze(axis), axis))
    }
}

impl GraphTensor {
    pub fn cast(self, dtype: DType) -> GraphTensor {
        if self.dtype == dtype {
            return self;
        }
        // Cast converts the addressed span of the underlying buffer and the
        // view passes through unchanged; sliced views address beyond
        // n_physical_elements, so size by span.
        let id = self
            .graph()
            .add_op(Cast(self.shape.physical_span(), dtype), &[self.id]);
        let mut shape = self.shape;
        shape.element_stride_bits = dtype.bits();
        GraphTensor::from_id(id, shape, self.graph_ref, dtype)
    }

    /// Sets this tensor's dtype without doing a cast
    pub fn as_dtype(mut self, dtype: DType) -> GraphTensor {
        self.dtype = dtype;
        self.shape.element_stride_bits = dtype.bits();
        if let Some(gmem) = self.graph().try_get_op_mut::<Input>(self.id) {
            gmem.dtype = dtype;
        }
        if let Some((_, d)) = self.graph().input_meta.get_mut(&self.id) {
            *d = dtype;
        }
        self
    }
}

#[cfg(test)]
#[path = "../../tests/unit/frontend/other/mod.rs"]
mod tests;
