# ADR 0002: Name the platform AletheiaRT

Status: accepted.

The product name is **AletheiaRT** and the command-line executable is
`aletheia-rt`. Aletheia is the Greek idea of truth as disclosure or unconcealment;
RT identifies the system as a runtime and control plane.
The name describes the product contract: an optimization must reveal its model
semantics, hardware and workload scope, numerical envelope, raw measurements,
resource use, exact artifacts, and fallback before it can be deployed.

This is not a correctness prover in the theorem-proving sense. The
"proof-carrying" phrase means machine-checkable deployment evidence and
explicit contracts. It does not imply formal verification of arbitrary CUDA
kernels.

The Aletheia namespace applies to the Rust crates, Python integrations, cache
and registry layout; AletheiaRT is the product and CLI name. The current filesystem checkout remains
`/workspace/orbitkv` until the repository and remote are renamed as a separate
external operation.
