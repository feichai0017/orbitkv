# Roadmap

## M0: renamed, reproducible baseline

- import the complete PegaFlow 0.24.5 source snapshot;
- rename crates, modules, packages, binaries, protocol namespaces, metrics,
  configuration keys, scripts, tests, and documentation to OrbitKV;
- retain upstream provenance and Apache-2.0 obligations;
- restore the existing OrbitKV website and logo;
- pass host-side Rust, Python, and website checks.

## M1: SGLang HiCache backend

- implement the dynamic `HiCacheStorage` contract;
- support batch prefix existence, get, and put for MHA/MLA and named auxiliary
  pools;
- isolate keys by model, parallel rank, layout, dtype, and pool;
- make storage failure fail open for serving while exposing explicit metrics;
- validate cold miss, warm hit, partial prefix, restart, and cancellation.

## M2: native transfer path

- remove Python byte copies from the steady state;
- register SGLang host pages with the Rust engine;
- batch page descriptors and overlap layer-wise copy with attention;
- reuse SSD and RDMA tiers through one completion contract.

## M3: proof-carrying lifetime plans

- describe `may_read(query, key)` for full, sliding, sink-local, and recurrent
  state;
- compile retirement and placement plans;
- enforce the semantic-frontier plus execution-frontier reuse rule;
- report Retention Amplification alongside TTFT, TPOT, throughput, and traffic.

## M4: adaptive physical planning

- learn next-touch and transfer costs from SGLang traces;
- choose retention, prefetch, compression, tier, and replica placement jointly;
- treat ring layouts and migration thresholds as derived plans;
- canary plan changes and roll back on correctness or SLO regressions.
