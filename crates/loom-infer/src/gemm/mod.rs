//! Dense BF16 GEMM contract and CPU reference implementation.

use crate::ContractError;
use crate::error::require_len;
use half::bf16;

/// Contract for `D[M, N] = A[M, K] * W[N, K]^T`.
///
/// `A`, `W`, and `D` are contiguous row-major BF16 tensors. Each dot product
/// accumulates in F32, and the completed result is rounded once to BF16.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bf16DenseGemmSpec {
    m: usize,
    n: usize,
    k: usize,
}

impl Bf16DenseGemmSpec {
    /// Creates a fixed-shape contiguous BF16 GEMM contract.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self, ContractError> {
        if m == 0 || n == 0 || k == 0 {
            return Err(ContractError::ZeroDimension);
        }

        m.checked_mul(k)
            .ok_or(ContractError::ElementCountOverflow)?;
        n.checked_mul(k)
            .ok_or(ContractError::ElementCountOverflow)?;
        m.checked_mul(n)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self { m, n, k })
    }

    /// Returns the number of rows in `A` and `D`.
    pub const fn m(self) -> usize {
        self.m
    }

    /// Returns the number of rows in `W` and columns in `D`.
    pub const fn n(self) -> usize {
        self.n
    }

    /// Returns the shared reduction dimension.
    pub const fn k(self) -> usize {
        self.k
    }

    /// Returns the exact number of BF16 elements required by `A`.
    pub const fn a_numel(self) -> usize {
        self.m * self.k
    }

    /// Returns the exact number of BF16 elements required by `W`.
    pub const fn weight_numel(self) -> usize {
        self.n * self.k
    }

    /// Returns the exact number of BF16 elements required by `D`.
    pub const fn output_numel(self) -> usize {
        self.m * self.n
    }
}

/// Computes the fixed contiguous BF16 GEMM contract on the CPU.
///
/// The reference visits the reduction dimension in ascending order and uses
/// one F32 fused multiply-add per element. This fixes the oracle's reduction
/// order; a parallel provider may require a declared numerical tolerance while
/// still using F32 accumulation.
pub fn bf16_dense_gemm_reference(
    a: &[bf16],
    weight: &[bf16],
    output: &mut [bf16],
    spec: Bf16DenseGemmSpec,
) -> Result<(), ContractError> {
    require_len("A", a.len(), spec.a_numel())?;
    require_len("W", weight.len(), spec.weight_numel())?;
    require_len("D", output.len(), spec.output_numel())?;

    for (a_row, output_row) in a
        .chunks_exact(spec.k())
        .zip(output.chunks_exact_mut(spec.n()))
    {
        for (column, destination) in output_row.iter_mut().enumerate() {
            let weight_row = &weight[column * spec.k()..(column + 1) * spec.k()];
            let accumulator =
                a_row
                    .iter()
                    .zip(weight_row)
                    .fold(0.0_f32, |sum, (&a_value, &weight_value)| {
                        a_value.to_f32().mul_add(weight_value.to_f32(), sum)
                    });
            *destination = bf16::from_f32(accumulator);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
