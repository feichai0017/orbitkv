# OrbitKV Mooncake Sys

`orbitkv-mooncake-sys` is the native artifact and raw ABI boundary for Mooncake
TENT. It does not build or load the legacy Transfer Engine backend.

The crate has three responsibilities:

1. build the pinned stable Mooncake release from `third-party/mooncake`;
2. build the `tent_shared` target and stage `libtent_shared.so`,
   `libmooncake_common.so`, and `libasio.so` in
   a relocatable runtime directory;
3. load and expose the C ABI required by `orbitkv-transfer`.

It deliberately contains no OrbitKV cache identity, lease, retry, timeout, or
transfer-plan policy. Those belong to `orbitkv-transfer` and `orbitkv-core`.

## Upstream Rust bindings

Mooncake's older Rust wrapper targets the compatibility Transfer Engine API and
is not published on crates.io in the pinned release. OrbitKV binds the stable
TENT C ABI dynamically so cancellation, explicit memory options, notification
and rail-load semantics remain available without linking C++ into every binary.

OrbitKV therefore keeps this small sys crate for reproducible native builds and
relocatable shared-library packaging. The higher-level API remains isolated in
`orbitkv-transfer`, so the raw bindings can be replaced by the official crate
later without changing cache or framework code.

## Native source

- release: `v0.3.13.post1`
- commit: `719735896c86b56fabec6cf3e825fb2ea640597a`

To use compatible prebuilt libraries instead of invoking CMake, set
`ORBITKV_MOONCAKE_LIB_DIR` to a directory containing the three TENT runtime
shared objects. A directory containing only `libtransfer_engine.so` is rejected.
