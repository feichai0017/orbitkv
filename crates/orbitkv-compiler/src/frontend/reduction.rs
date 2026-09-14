use crate::hlir::*;
use crate::prelude::*;

impl GraphTensor {
    /// Reduce a dimension of the tensor by summing all elements along that axis.
    pub fn sum(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Sum reduce each dimension
        let mut axes = axes.to_axes();
        for dim in 0..axes.len() {
            id = self.graph().add_op(
                SumReduce {
                    dim: axes[dim],
                    input_shape: shape,
                    ..Default::default()
                },
                &[id],
            );
            shape.remove_dim(axes[dim]);
            shape = shape.contiguous();
            let axis = axes[dim];
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        GraphTensor::from_id(id, shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Reduce a dimension of the tensor by taking the maximum of all elements along that axis.
    pub fn max(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Max reduce each dimension
        let mut axes = axes.to_axes();
        for dim in 0..axes.len() {
            id = self.graph().add_op(
                MaxReduce {
                    dim: axes[dim],
                    input_shape: shape,
                    ..Default::default()
                },
                &[id],
            );
            shape.remove_dim(axes[dim]);
            shape = shape.contiguous();
            let axis = axes[dim];
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        GraphTensor::from_id(id, shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Reduce a dimension of the tensor by taking the minimum of all elements along that axis.
    pub fn min(self, axes: impl ToAxes) -> GraphTensor {
        -(-self).max(axes)
    }

    /// Reduce a dimension of the tensor by taking the mean of all elements along that axis.
    pub fn mean(self, axes: impl ToAxes) -> GraphTensor {
        let reduced_elements = axes
            .to_axes()
            .into_iter()
            .map(|i| self.dims()[i])
            .product::<Expression>();
        self.sum(axes) / reduced_elements
    }

    /// Reduce a dimension of the tensor by multiplying all elements along that axis.
    pub fn prod(self, axes: impl ToAxes) -> GraphTensor {
        self.log().sum(axes).exp()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/frontend/reduction/mod.rs"]
mod tests;
