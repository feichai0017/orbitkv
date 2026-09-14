use orbitkv_compiler::prelude::*;

/// Rectified Linear Unit activation function
#[derive(Default)]
pub struct ReLU;

impl ReLU {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.relu()
    }
}

/// Gaussian Error Linear Unit activation function.
///
/// Uses the exact erf form (`gelu`), matching PyTorch's `nn.GELU()` default
/// (`approximate="none"`). For the cheaper tanh approximation, call
/// `GraphTensor::gelu_fast_tanh_approximation()`.
#[derive(Default)]
pub struct GeLU;

impl GeLU {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.gelu()
    }
}

/// Sigmoid activation function
#[derive(Default)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.sigmoid()
    }
}

/// Swish activation function
#[derive(Default)]
pub struct Swish;

impl Swish {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.swish()
    }
}

/// Tanh activation function
#[derive(Default)]
pub struct Tanh;

impl Tanh {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.tanh()
    }
}
