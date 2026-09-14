use orbitkv_compiler::prelude::*;

/// A simple layer norm with an optional weight and bias
#[derive(Default)]
pub struct LayerNorm {
    pub weight: Option<GraphTensor>,
    pub bias: Option<GraphTensor>,
    mean_norm: bool,
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(
        dim: usize,
        weight: Option<&str>,
        bias: Option<&str>,
        mean_norm: bool,
        epsilon: f32,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: weight.map(|w| cx.named_tensor(w, dim).persist()),
            bias: bias.map(|b| cx.named_tensor(b, dim).persist()),
            mean_norm,
            epsilon,
        }
    }
}

impl LayerNorm {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        if self.mean_norm {
            input = input.mean_norm(input.shape.last_axis());
        }
        input = input.std_norm(input.shape.last_axis(), self.epsilon);
        if let Some(w) = self.weight {
            input *= w.expand_lhs(&input.dims()[..input.dims().len() - 1]);
        }
        if let Some(b) = self.bias {
            input += b.expand_lhs(&input.dims()[..input.dims().len() - 1]);
        }
        input
    }
}
