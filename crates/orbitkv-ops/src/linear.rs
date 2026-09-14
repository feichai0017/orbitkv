use orbitkv_compiler::prelude::*;

/// A simple linear layer
pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
    permute: bool,
}

impl Linear {
    pub fn new(inp: usize, out: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (inp, out)).persist(),
            bias: if bias {
                Some(cx.named_tensor("Bias", out).persist())
            } else {
                None
            },
            permute: false,
        }
    }

    pub fn new_permuted(inp: usize, out: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (out, inp)).persist(),
            bias: if bias {
                Some(cx.named_tensor("Bias", out).persist())
            } else {
                None
            },
            permute: true,
        }
    }
}

impl Linear {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let output = input.matmul(if self.permute {
            self.weight.permute((1, 0))
        } else {
            self.weight
        });
        if let Some(bias) = self.bias {
            output + bias.expand_lhs(&output.dims()[..output.dims().len() - 1])
        } else {
            output
        }
    }
}
