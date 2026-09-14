use orbitkv_compiler::prelude::*;

/// A layer of E experts and a router
pub struct MoE {
    pub expert_weights: GraphTensor, // [E, in, out]
    pub router: GraphTensor,         // [in, E]
    pub k: usize,
}

impl MoE {
    pub fn forward(&self, activations: GraphTensor) -> GraphTensor {
        let n = activations.dims().len();
        let e_dim = *self.router.dims().last().unwrap();
        let (_, in_size, out_size) = self.expert_weights.dims3();
        let io = in_size * out_size;
        let k_expr = Expression::from(self.k);

        // 1. Routing probabilities: [batch.., E]
        let routing_weights = activations.matmul(self.router).softmax(n - 1);

        // 2. Top-k expert indices: [batch.., k] (Int)
        let top_k_indices = routing_weights.topk_indexes(self.k, n - 1);

        // 3. Gather top-k routing values: [batch.., k]
        //    flat_idx = batch_row * E + expert_idx
        //    iota(z / k * E) gives batch_row * E at each position in [batch.., k]
        let row_offsets = activations
            .graph()
            .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
        let routing_flat_idx =
            (row_offsets.cast(DType::F32) + top_k_indices.cast(DType::F32)).cast(DType::Int);
        let top_k_values = routing_weights.gather(routing_flat_idx); // [batch.., k]

        // 4. Gather expert weight matrices: [batch.., k, in, out]
        //    flat_idx[.., ki, i, o] = expert_idx[.., ki] * in*out + i * out + o
        let base = (top_k_indices * io).cast(DType::F32); // [batch.., k]
        let within = activations
            .graph()
            .iota(Expression::from('z'), (in_size, out_size))
            .cast(DType::F32); // [in, out] values 0..in*out-1

        // Expand base to [batch.., k, in, out]
        let n_base = base.dims().len();
        let exp_base = base
            .expand_dim(n_base, in_size)
            .expand_dim(n_base + 1, out_size);

        // Expand within to [batch.., k, in, out]
        let mut exp_within = within;
        for (i, dim) in base.dims().iter().enumerate() {
            exp_within = exp_within.expand_dim(i, *dim);
        }

        let expert_flat_idx = (exp_base + exp_within).cast(DType::Int);
        let gathered = self.expert_weights.gather(expert_flat_idx); // [batch.., k, in, out]

        // 5. Batched matmul: [batch.., k, 1, in] @ [batch.., k, in, out] → [batch.., k, out]
        let expanded_act = activations
            .expand_dim(n - 1, self.k) // [batch.., k, in]
            .unsqueeze(n); // [batch.., k, 1, in]
        let expert_out = expanded_act.matmul(gathered).squeeze(n); // [batch.., k, out]

        // 6. Weighted sum over experts: [batch.., k, out] * [batch.., k, 1] → sum(k) → [batch.., out]
        let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len()); // [batch.., k, 1]
        weights_exp.shape.expand(expert_out.dims());
        (expert_out * weights_exp).sum(n - 1)
    }
}

#[cfg(test)]
#[path = "../tests/unit/moe/mod.rs"]
mod tests;
