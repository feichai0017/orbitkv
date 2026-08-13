# TileLang artifact manifest and launch ABI

**State:** the schema, pure-Rust validation, byte verification, and exact
registry selection are current. CUDA module loading and launch remain planned
under roadmap phase E1.

Oxide compiles TileLang outside the production process. The serving runtime
accepts only immutable cubin or PTX bytes accompanied by a versioned manifest.
The manifest is a compatibility contract, not build-system metadata.

The current Rust implementation is
`crates/oxide-infer-cuda/src/artifact.rs`. It has no CUDA dependency, so a
release builder and CI can reject malformed artifacts before a GPU or driver
is present.

## Artifact identity

One registry key is the exact tuple:

```text
(contract name, contract version, algorithm, CUDA architecture)
```

Properties such as dtype, layout, and mask mode and all declared dimensions
must also match the selected manifest. A request with a missing, additional,
or incompatible field fails. Registry selection never searches for a nearby
shape, another architecture, or another provider.

The schema and launch ABI are versioned separately:

- `schema_version` changes when manifest interpretation changes;
- `launch_abi_version` changes when the Rust-to-kernel calling convention
  changes;
- `contract.version` changes when operator behavior changes;
- `algorithm` changes when an implementation is independently qualified.

Only provider `oxide_tile` is accepted by the target artifact path.

## Version 1 example

```json
{
  "schema_version": 1,
  "provider": "oxide_tile",
  "contract": {
    "name": "attention.paged_prefill",
    "version": 1
  },
  "algorithm": "gqa4_bf16",
  "artifact": {
    "file_name": "paged_prefill_sm90a.cubin",
    "format": "cubin",
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "size_bytes": 123456
  },
  "toolchain": {
    "tilelang_version": "0.1.13",
    "source_revision": "8001cc4ccf6149382d2019654a19f59c1d4d0482",
    "cuda_toolkit_version": "13.1"
  },
  "target": {
    "architecture": "sm_90a",
    "minimum_driver": {
      "major": 590,
      "minor": 48,
      "patch": 1
    }
  },
  "launch_abi_version": 1,
  "entry_points": [
    {
      "symbol": "oxide_paged_prefill",
      "parameters": [
        {
          "name": "query",
          "parameter_type": {
            "kind": "device_pointer",
            "element": "bf16",
            "access": "read",
            "alignment": 16
          }
        },
        {
          "name": "query_tokens",
          "parameter_type": {
            "kind": "scalar",
            "scalar": "u32"
          }
        }
      ],
      "allowed_aliases": [],
      "launch": {
        "grid_dimensions": [96, 16, 1],
        "block_dimensions": [128, 1, 1],
        "cluster_dimensions": null,
        "dynamic_shared_memory_bytes": 49152
      },
      "workspace": {
        "bytes": 0,
        "alignment": 256
      }
    }
  ],
  "numerics": {
    "accumulation": "f32",
    "output": "bf16",
    "determinism": "required"
  },
  "qualification": {
    "correctness": "results/paged_prefill-correctness.json",
    "sanitizer": "results/paged_prefill-sanitizer.json",
    "graph": "results/paged_prefill-graph.json",
    "benchmark": "results/paged_prefill-benchmark.json"
  },
  "properties": {
    "dtype": "bf16",
    "layout": "paged",
    "mask": "causal"
  },
  "dimensions": {
    "head_dim": {
      "min": 128,
      "max": 128,
      "multiple_of": 8
    },
    "query_tokens": {
      "min": 1,
      "max": 4096,
      "multiple_of": 1
    }
  }
}
```

Unknown JSON fields are rejected. File names must be a single local component
with an extension matching the declared format. SHA-256 is canonical lowercase
hexadecimal. Entry-point and parameter names are unique, alignments are
non-zero powers of two, and dimension ranges are internally consistent.

## Runtime admission order

The checked path is:

```text
strict JSON decode
  -> schema, provider, ABI, and structural validation
  -> exact device architecture and minimum-driver validation
  -> byte length and SHA-256 verification
  -> verified bytes owned by the registry
  -> exact property and dimension selection
  -> planned CUDA module and function loading
  -> checked operands and CommandScope launch
```

`VerifiedTileArtifact` owns the bytes that were hashed. It does not return a
validated path that could later be replaced with different file contents.
CUDA loading will consume those owned bytes.

## Deliberately excluded from version 1

Version 1 does not perform runtime JIT, auto-tuning, network retrieval,
filesystem discovery, provider fallback, or best-effort forward compatibility.
It also does not claim that a structurally valid artifact is numerically
correct or fast. Those facts require separate immutable correctness, sanitizer,
Graph, and benchmark records before release promotion.

The next E1 slice connects one manifest-verified BF16 paged-prefill cubin to
the existing memory, stream, command, and completion lifecycle.
