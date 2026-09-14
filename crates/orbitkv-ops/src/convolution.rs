use orbitkv_compiler::prelude::*;

/// Generic N-dimensional convolution layer implemented with the GraphTensor `unfold` helper.
///
/// The layer expects inputs shaped like `[batch..., channels, spatial...]` where the number of
/// spatial dimensions is greater than zero. The kernel configuration controls how many spatial
/// axes are convolved (N) and must be shorter than the input rank (K): `K > N` is asserted.
pub struct ConvND {
    pub weight: GraphTensor, // (ch_out, ch_in * kernel_product)
    pub bias: Option<GraphTensor>,
    kernel: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    ch_in: usize,
    ch_out: usize,
}

impl ConvND {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: impl AsRef<[usize]>,
        stride: impl AsRef<[usize]>,
        dilation: impl AsRef<[usize]>,
        padding: impl AsRef<[usize]>,
        bias: bool,
        cx: &mut Graph,
    ) -> Self {
        let kernel = kernel.as_ref().to_vec();
        let stride = stride.as_ref().to_vec();
        let dilation = dilation.as_ref().to_vec();
        let padding = padding.as_ref().to_vec();
        assert!(
            !kernel.is_empty(),
            "ConvND requires at least one spatial dimension in the kernel",
        );
        let k = kernel.len();
        assert_eq!(
            stride.len(),
            k,
            "Stride dimensions ({}) must match kernel dimensions ({k})",
            stride.len()
        );
        assert_eq!(
            dilation.len(),
            k,
            "Dilation dimensions ({}) must match kernel dimensions ({k})",
            dilation.len()
        );
        assert_eq!(
            padding.len(),
            k,
            "Padding dimensions ({}) must match kernel dimensions ({k})",
            padding.len()
        );

        let kernel_product: usize = kernel.iter().product();

        Self {
            weight: cx
                .named_tensor("ConvWeight", (ch_out, ch_in * kernel_product))
                .persist(),
            bias: if bias {
                Some(cx.named_tensor("ConvBias", ch_out).persist())
            } else {
                None
            },
            kernel,
            stride,
            dilation,
            padding,
            ch_in,
            ch_out,
        }
    }

    /// Apply convolution to an input shaped `[batch..., channels, spatial...]`.
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let input_dims = input.dims();
        let rank = input_dims.len();
        let spatial = self.kernel.len();

        assert!(
            rank > spatial,
            "ConvND expects input rank ({rank}) to be greater than kernel dims ({spatial})",
        );

        let batch_len = rank - spatial - 1;
        assert_eq!(
            input_dims[batch_len],
            Expression::from(self.ch_in),
            "Input channel dimension ({}) must match ch_in ({})",
            input_dims[batch_len],
            self.ch_in
        );
        assert_eq!(
            self.weight.dims()[0],
            Expression::from(self.ch_out),
            "Weight output channels ({}) must match ch_out ({})",
            self.weight.dims()[0],
            self.ch_out
        );

        // Pad only the spatial dimensions.
        let mut padding = vec![(Expression::from(0), Expression::from(0)); rank];
        for (i, pad) in self.padding.iter().enumerate() {
            let axis = batch_len + 1 + i;
            padding[axis] = (Expression::from(*pad), Expression::from(*pad));
        }
        let padded = input.pad(padding, 0.0);

        // Build unfold parameters with ones for non-spatial axes.
        let mut kernel_shape = vec![1; rank];
        let mut stride_shape = vec![1; rank];
        let mut dilation_shape = vec![1; rank];
        for i in 0..spatial {
            let axis = batch_len + 1 + i;
            kernel_shape[axis] = self.kernel[i];
            stride_shape[axis] = self.stride[i];
            dilation_shape[axis] = self.dilation[i];
        }

        // unfold yields [window..., kernel...] — windows already in front.
        let unfolded = padded.unfold(kernel_shape, stride_shape, dilation_shape);
        let unfolded_dims = unfolded.dims();

        // Capture output spatial dimensions from the unfolded view.
        let output_dims: Vec<Expression> =
            unfolded_dims[batch_len + 1..batch_len + 1 + spatial].to_vec();

        // Reorder to [batch..., out..., channels, kernel_spatial..., kernel_batch..., kernel_channel].
        let mut order2 = Vec::with_capacity(2 * rank);
        // window batch dims
        order2.extend(0..batch_len);
        // window spatial dims (outputs)
        order2.extend(batch_len + 1..batch_len + 1 + spatial);
        // window channel dim
        order2.push(batch_len);
        // kernel spatial dims
        order2.extend(rank + batch_len + 1..rank + batch_len + 1 + spatial);
        // kernel batch dims and kernel channel dim (to be merged away)
        order2.extend(rank..rank + batch_len + 1);
        let mut patches = unfolded.permute(order2);

        // Drop kernel axes for batch + channel by merging them into the previous dimension.
        for _ in 0..=batch_len {
            let last = patches.dims().len();
            patches = patches.merge_dims(last - 2, last - 1);
        }

        // Flatten channel and kernel spatial dimensions together.
        for _ in 0..spatial {
            let channel_axis = batch_len + spatial;
            patches = patches.merge_dims(channel_axis, channel_axis + 1);
        }

        // Collapse batch dimensions into one and output dimensions into one for matmul.
        for _ in 1..batch_len {
            patches = patches.merge_dims(0, 1);
        }
        for _ in 1..spatial {
            patches = patches.merge_dims(1, 2);
        }

        let mut out = patches.matmul(self.weight.permute((1, 0)));

        // Restore batch and spatial dimensions. The collapse loops merged
        // k dims into 1, so restore splits k-1 times: splitting by every dim
        // including the outermost would leave a spurious leading 1-dim.
        let batch_dims = self.input_batch_dims(&input_dims, batch_len);
        for dim in batch_dims.iter().skip(1).rev() {
            out = out.split_dims(0, *dim);
        }
        for dim in output_dims.iter().skip(1).rev() {
            out = out.split_dims(batch_len, *dim);
        }

        // Move channel dimension ahead of the spatial axes: [batch..., ch_out, spatial...]
        let mut final_order: Vec<usize> = (0..batch_len).collect();
        final_order.push(batch_len + spatial);
        final_order.extend(batch_len..batch_len + spatial);
        out = out.permute(final_order);

        if let Some(_b) = self.bias {
            todo!()
            // out += b.expand(out.shape);
        }

        out
    }

    fn input_batch_dims(&self, input_dims: &[Expression], batch_len: usize) -> Vec<Expression> {
        input_dims[..batch_len].to_vec()
    }

    pub fn infer_output_shape(&self, input: &[usize]) -> Vec<usize> {
        let rank = input.len();
        let spatial = self.kernel.len();

        assert!(rank > spatial, "expected input rank > spatial dims");
        let batch_len = rank - spatial - 1;
        assert_eq!(
            input[batch_len], self.ch_in,
            "input channel dimension does not match ch_in",
        );

        let batch_prefix = &input[..batch_len];
        let spatial_dims = &input[batch_len + 1..];
        let out_spatial: Vec<usize> = spatial_dims
            .iter()
            .zip(
                self.kernel
                    .iter()
                    .zip(self.stride.iter())
                    .zip(self.dilation.iter())
                    .zip(self.padding.iter()),
            )
            .map(|(dim, (((k, s), d), p))| (dim + 2 * p - d * (k - 1) - 1) / s + 1)
            .collect();

        let mut shape = batch_prefix.to_vec();
        shape.push(self.ch_out);
        shape.extend(out_spatial);
        shape
    }
}

#[cfg(test)]
#[path = "../tests/unit/convolution/mod.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/convolution/forward_tests.rs"]
mod forward_tests;
