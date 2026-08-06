SHELL := /bin/bash
.DEFAULT_GOAL := help

CUDA_CRATE := crates/loom-infer-cuda
VALIDATION_CRATE := crates/loom-infer-validation
CUDA_ARCH ?= sm_90
PACKAGE_FLAGS ?= --allow-dirty
LOOM_SOURCE_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)
MISE := $(shell command -v mise 2>/dev/null)
ifneq ($(MISE),)
RUN := $(MISE) exec --
endif
ifneq ($(LOOM_CARGO_HOME),)
CARGO_ENV := CARGO_HOME=$(LOOM_CARGO_HOME)
endif
CARGO := $(CARGO_ENV) $(RUN) cargo
NPM := $(RUN) npm --prefix website

.PHONY: help check check-rust check-website install-website cuda-doctor \
	cuda-check cuda-test h20-rms-norm h20-gemm h20-attention h20-paged-attention \
	h20-ragged-prefill h20 bench-loom bench-paged-loom bench-split-k

help:
	@printf '%s\n' \
		'make check          Run CPU-only Rust and website gates' \
		'make cuda-doctor    Check the pinned cuda-oxide environment' \
		'make cuda-check     Run CUDA-feature Clippy' \
		'make cuda-test      Run release tests through cuda-oxide' \
		'make h20            Run all H20 correctness programs sequentially' \
		'make h20-paged-attention  Run paged batch-decode H20 correctness' \
		'make h20-ragged-prefill  Run ragged prefill H20 correctness' \
		'make bench-loom     Run the Loom side of the matched H20 benchmark' \
		'make bench-paged-loom  Run Loom matched paged-decode cases only' \
		'make bench-split-k  Sweep Loom split-K choices on H20'

check: check-rust check-website

check-rust:
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	$(CARGO) test --workspace --all-targets
	$(CARGO) check --workspace --release
	$(CARGO) package -p loom-infer $(PACKAGE_FLAGS)

install-website:
	NODE_ENV=development $(NPM) ci --include=dev

check-website:
	@test -d website/node_modules || { echo 'website dependencies missing; run make install-website'; exit 1; }
	NODE_ENV=development $(NPM) run check
	NODE_ENV=development $(NPM) run build
	NODE_ENV=development $(NPM) audit

cuda-doctor:
	cd $(CUDA_CRATE) && $(CARGO) +nightly-2026-04-03 oxide doctor

cuda-check:
	$(CARGO) +nightly-2026-04-03 clippy --workspace --all-targets --features cuda -- -D warnings

cuda-test:
	cd $(VALIDATION_CRATE) && CARGO_BUILD_JOBS=1 $(CARGO) +nightly-2026-04-03 oxide test --arch $(CUDA_ARCH) -- --workspace --features cuda --release

h20-rms-norm:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run rms_norm_h20 --bin rms_norm_h20 --features cuda --arch $(CUDA_ARCH)

h20-gemm:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run bf16_gemm_h20 --bin bf16_gemm_h20 --features cuda --arch $(CUDA_ARCH)

h20-attention:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run single_decode_h20 --bin single_decode_h20 --features cuda --arch $(CUDA_ARCH)

h20-paged-attention:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run paged_batch_decode_h20 --bin paged_batch_decode_h20 --features cuda --arch $(CUDA_ARCH)

h20-ragged-prefill:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run ragged_prefill_h20 --bin ragged_prefill_h20 --features cuda --arch $(CUDA_ARCH)

h20: h20-rms-norm h20-gemm h20-attention h20-paged-attention h20-ragged-prefill

bench-loom:
	@set -o pipefail; cd $(VALIDATION_CRATE) && LOOM_SOURCE_COMMIT="$(LOOM_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run loom_matched_bench_h20 --bin loom_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-paged-loom:
	@set -o pipefail; cd $(VALIDATION_CRATE) && LOOM_BENCH_OPERATORS=paged_batch_decode LOOM_SOURCE_COMMIT="$(LOOM_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run loom_matched_bench_h20 --bin loom_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-split-k:
	@set -o pipefail; cd $(VALIDATION_CRATE) && LOOM_SOURCE_COMMIT="$(LOOM_SOURCE_COMMIT)" LOOM_SOURCE_STATE=working_tree $(CARGO) +nightly-2026-04-03 oxide run split_k_sweep_h20 --bin split_k_sweep_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'
