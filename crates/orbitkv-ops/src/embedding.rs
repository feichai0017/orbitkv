// use orbitkv_compiler::{prelude::*, tests::random_vec};

// pub struct Embedding {
//     permute: bool,
//     pub weight: GraphTensor, // n embeddings x embedding dim
//     embedding_dim: usize,
// }

// impl Embedding {
//     pub fn new(n_embeddings: usize, embedding_dim: usize, cx: &mut Graph) -> Self {
//         Self {
//             weight: cx.named_tensor("Embedding Weight", (n_embeddings, embedding_dim)),
//             permute: false,
//             embedding_dim,
//         }
//     }

//     pub fn new_permuted(n_embeddings: usize, embedding_dim: usize, cx: &mut Graph) -> Self {
//         Self {
//             weight: cx.named_tensor("Embedding Weight", (embedding_dim, n_embeddings)),
//             permute: true,
//             embedding_dim,
//         }
//     }

//     pub fn initialize(self) -> Self {
//         self.weight.set(random_vec(
//             self.weight.shape.n_elements().to_usize().unwrap(),
//         ));
//         self
//     }
// }

// impl SerializeModule for Embedding {
//     fn serialize(&self, s: &mut orbitkv_compiler::module::Serializer) {
//         s.tensor("weight", self.weight);
//     }
// }

// impl Module<GraphTensor> for Embedding {
//     type Output = GraphTensor;

//     fn forward(&self, input: GraphTensor) -> Self::Output {
//         // Flatten batches
//         let batch_size = input.shape.n_elements();
//         let inp = input.reshape(batch_size);
//         // Gather
//         let out = if self.permute {
//             self.weight.permute((1, 0)).gather(inp)
//         } else {
//             self.weight.gather(inp)
//         };
//         // Unflatten
//         let mut new_shape = input.dims();
//         new_shape.push(self.embedding_dim.into());
//         out.reshape(new_shape)
//     }
// }

// impl Embedding {
//     // Reverse from embedding to token distribution
//     pub fn reverse(&self, input: GraphTensor) -> GraphTensor {
//         if self.permute {
//             input.matmul(self.weight)
//         } else {
//             input.matmul(self.weight.permute((1, 0)))
//         }
//     }
// }
