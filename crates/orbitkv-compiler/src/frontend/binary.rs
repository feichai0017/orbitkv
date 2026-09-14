use crate::hlir::*;
use crate::prelude::*;
use std::ops::AddAssign;
use std::ops::DivAssign;
use std::ops::MulAssign;
use std::ops::RemAssign;
use std::ops::SubAssign;
use std::ops::{Add, Div, Mul, Rem, Sub};

impl Add for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: GraphTensor) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to add tensors.");
        assert_eq!(
            self.dtype, rhs.dtype,
            "Dtypes must match to add tensors. Got {:?} and {:?}",
            self.dtype, rhs.dtype
        );
        let new_id = self.graph().add_op(
            crate::hlir::Add {
                input_shapes: vec![self.shape, rhs.shape],
                ..Default::default()
            },
            &[self.id, rhs.id],
        );
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }
}

impl Add<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn add(self, rhs: GraphTensor) -> Self::Output {
        rhs + self
    }
}

impl<T> AddAssign<T> for GraphTensor
where
    GraphTensor: Add<T, Output = GraphTensor>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl Sub for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        self + -rhs
    }
}

impl Sub<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        self + -rhs
    }
}

impl<T> SubAssign<T> for GraphTensor
where
    GraphTensor: Sub<T, Output = GraphTensor>,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl Mul for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: GraphTensor) -> Self::Output {
        assert_eq!(
            self.dims(),
            rhs.dims(),
            "Dims must match to multiply tensors."
        );
        assert_eq!(
            self.dtype, rhs.dtype,
            "Dtypes must match to multiply tensors. Got {:?} and {:?}",
            self.dtype, rhs.dtype
        );
        let new_id = self.graph().add_op(
            crate::hlir::Mul {
                input_shapes: vec![self.shape, rhs.shape],
                ..Default::default()
            },
            &[self.id, rhs.id],
        );
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }
}

impl Mul<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn mul(self, rhs: GraphTensor) -> Self::Output {
        rhs * self
    }
}

impl<T> MulAssign<T> for GraphTensor
where
    GraphTensor: Mul<T, Output = GraphTensor>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        self * rhs.reciprocal()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        self * rhs.reciprocal()
    }
}

impl<T> DivAssign<T> for GraphTensor
where
    GraphTensor: Div<T, Output = GraphTensor>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl Rem<GraphTensor> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: GraphTensor) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to mod tensors.");
        assert_eq!(
            self.dtype, rhs.dtype,
            "Dtypes must match to mod tensors. Got {:?} and {:?}",
            self.dtype, rhs.dtype
        );
        let new_id = self.graph().add_op(
            Mod {
                input_shapes: vec![self.shape, rhs.shape],
                ..Default::default()
            },
            &[self.id, rhs.id],
        );
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }
}

impl<T> RemAssign<T> for GraphTensor
where
    GraphTensor: Rem<T, Output = GraphTensor>,
{
    fn rem_assign(&mut self, rhs: T) {
        *self = *self % rhs;
    }
}

impl Add<f32> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: f32) -> Self::Output {
        self + self
            .graph()
            .constant_float(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Add<S> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: S) -> Self::Output {
        self + self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl Sub<f32> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self
            .graph()
            .constant_float(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Sub<S> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: S) -> Self::Output {
        self - self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl Mul<f32> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self * self
            .graph()
            .constant_float(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Mul<S> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: S) -> Self::Output {
        self * self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: f32) -> Self::Output {
        self * self
            .graph()
            .constant_float(rhs.recip())
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Div<S> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: S) -> Self::Output {
        self / self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl Rem<f32> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self
            .graph()
            .constant_float(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Rem<S> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: S) -> Self::Output {
        self % self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

// Comparisons, all redurn bools (based on https://github.com/tinygrad/tinygrad/blob/3e0c2d256fe9f4f5f85cd3e4d8733a51d7b4a984/tinygrad/tensor.py#L653)
impl GraphTensor {
    /// Less than comparison
    pub fn lt(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to lt tensors.");
        assert_eq!(
            self.dtype, rhs.dtype,
            "Dtypes must match to compare tensors. Got {:?} and {:?}",
            self.dtype, rhs.dtype
        );
        let new_id = self.graph().add_op(
            LessThan {
                input_shapes: vec![self.shape, rhs.shape],
                ..Default::default()
            },
            &[self.id, rhs.id],
        );
        // Comparison operations always output Bool
        GraphTensor::from_id(
            new_id,
            self.shape
                .contiguous()
                .with_element_bits(DType::Bool.bits()),
            self.graph_ref,
            DType::Bool,
        )
    }

    /// Greater than comparison
    pub fn gt(self, rhs: GraphTensor) -> GraphTensor {
        rhs.lt(self)
    }

    /// Less than or equal
    pub fn le(self, rhs: GraphTensor) -> GraphTensor {
        (-self.gt(rhs).cast(DType::F32) + 1.0).cast(DType::Bool)
    }

    /// Greater than or equal
    pub fn ge(self, rhs: GraphTensor) -> GraphTensor {
        (-self.lt(rhs).cast(DType::F32) + 1.0).cast(DType::Bool)
    }

    /// Not equal
    pub fn ne(self, rhs: GraphTensor) -> GraphTensor {
        (self.lt(rhs).cast(DType::F32) + self.gt(rhs).cast(DType::F32)).cast(DType::Bool)
    }

    /// Equal
    pub fn eq(self, rhs: GraphTensor) -> GraphTensor {
        // Keep the inequality indicator numeric until the final cast. Calling
        // `ne` here would create a Bool -> F32 round trip, forcing backends
        // without Bool storage (currently Metal) to materialize an otherwise
        // internal boolean buffer.
        let not_equal = self.lt(rhs).cast(DType::F32) + self.gt(rhs).cast(DType::F32);
        (-not_equal + 1.0).cast(DType::Bool)
    }

    /// Raise the tensor to a power
    pub fn pow<T>(self, e: T) -> GraphTensor
    where
        Self: Mul<T, Output = Self>,
    {
        // Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
        self.abs().log().mul(e).exp()
    }

    // Clipping ops (minimum, maximum, clip)

    /// Take the elementwise maximum of two tensors
    pub fn maximum(self, rhs: GraphTensor) -> GraphTensor {
        (self.lt(rhs).cast(self.dtype) * rhs) + (rhs.le(self).cast(self.dtype) * self)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn maximum_f32(self, rhs: f32) -> GraphTensor {
        // `constant_float` always emits F32; cast it to `self.dtype` so the
        // downstream `lt`/`le` comparisons inside `maximum` don't panic when
        // `self` is Int (e.g. `aten.clamp` on Int top-k indices coming out
        // of an MoE router). For Int self the cast floors the bound, which
        // matches PyTorch's `clamp(int_tensor, min=<float>)` semantics.
        self.maximum(
            self.graph()
                .constant_float(rhs)
                .cast(self.dtype)
                .expand_rhs(self.shape),
        )
    }

    /// Take the elementwise minimum of two tensors
    pub fn minimum(self, rhs: GraphTensor) -> GraphTensor {
        -(-self).maximum(-rhs)
    }

    /// Take the elementwise minimum of a tensor and a float
    pub fn minimum_f32(self, rhs: f32) -> GraphTensor {
        -(-self).maximum_f32(-rhs)
    }

    /// Clip (clamp) a tensor into the range [`min`, `max`]
    pub fn clip(self, min: f32, max: f32) -> GraphTensor {
        self.maximum_f32(min).minimum_f32(max)
    }

    /// Return a tensor of elements selected from either self or other, depending on condition. Condition should be a boolean tensor
    pub fn cond(self, cond: GraphTensor, other: GraphTensor) -> GraphTensor {
        assert_eq!(
            self.dtype, other.dtype,
            "self and other need to be the same dtype!"
        );
        (cond.cast(self.dtype) * self) + ((1.0 - cond.cast(DType::F32)).cast(other.dtype) * other)
    }
}

pub trait F32Pow {
    fn pow(self, e: GraphTensor) -> GraphTensor;
}

impl F32Pow for f32 {
    fn pow(self, e: GraphTensor) -> GraphTensor {
        e.mul(self.abs().ln()).exp()
    }
}

// #[cfg(test)]
#[cfg(test)]
#[path = "../../tests/unit/frontend/binary/mod.rs"]
pub(super) mod tests;
