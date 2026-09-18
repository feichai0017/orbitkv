# SGLang patch stack

## 0001-hopper-only-kernel-build.patch

Applies to SGLang `095ec6c997bfdd25d3864cb0ce77a6562a934b96`. It adds
the build-only `ALETHEIA_HOPPER_ONLY` option. When enabled, the source build:

- keeps SM90/SM90a code and the runtime's existing `sm90/common_ops` loader;
- disables below-SM90, SM100 and SM120 gencodes;
- does not build or package the duplicate `common_ops_sm100` library;
- keeps Hopper FA3 and FlashMLA sources.

No runtime Python, scheduler, allocator, or kernel semantics are changed. This
profile exists because SGLang's default AOT wheel is multi-architecture even
when `TORCH_CUDA_ARCH_LIST=9.0`, while AletheiaRT qualifies artifacts for an
explicit hardware domain. Remove the patch when upstream exposes an equivalent
build profile.
