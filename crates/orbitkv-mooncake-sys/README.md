# OrbitKV Mooncake Sys

`orbitkv-mooncake-sys` is the native artifact and raw ABI boundary for the
Mooncake Transfer Engine. It is not an alternative transfer backend.

The crate has three responsibilities:

1. build the pinned stable Mooncake release from `third-party/mooncake`;
2. stage `libtransfer_engine.so`, `libmooncake_common.so`, and `libasio.so` in
   a relocatable runtime directory;
3. load and expose the C ABI required by `orbitkv-transfer`.

It deliberately contains no OrbitKV cache identity, lease, retry, timeout, or
transfer-plan policy. Those belong to `orbitkv-transfer` and `orbitkv-core`.

## Upstream Rust bindings

Mooncake ships `mooncake-transfer-engine/rust/transfer_engine_rust`, but the
official crate is not published on crates.io in the pinned release. Its build
script expects a separately built native Mooncake tree and links the native
libraries into the final Rust artifact. It does not build or package the shared
libraries required by the OrbitKV Python wheel.

OrbitKV therefore keeps this small sys crate for reproducible native builds and
relocatable shared-library packaging. The higher-level API remains isolated in
`orbitkv-transfer`, so the raw bindings can be replaced by the official crate
later without changing cache or framework code.

## Native source

- release: `v0.3.13.post1`
- commit: `719735896c86b56fabec6cf3e825fb2ea640597a`

To use compatible prebuilt libraries instead of invoking CMake, set
`ORBITKV_MOONCAKE_LIB_DIR` to a directory containing the three shared objects.
