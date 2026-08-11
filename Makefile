SHELL := /bin/bash
.DEFAULT_GOAL := help

CUDA_CRATE := crates/oxide-infer-cuda
VALIDATION_CRATE := crates/oxide-infer-lab
CUDA_ARCH ?= sm_90
PACKAGE_FLAGS ?= --allow-dirty
OXIDE_SOURCE_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)
MISE := $(shell command -v mise 2>/dev/null)
ifneq ($(MISE),)
RUN := $(MISE) exec --
endif
ifneq ($(OXIDE_CARGO_HOME),)
CARGO_ENV := CARGO_HOME=$(OXIDE_CARGO_HOME)
endif
CARGO := $(CARGO_ENV) $(RUN) cargo
NPM := $(RUN) npm --prefix website

.PHONY: help check check-rust check-tools check-website install-website cuda-doctor \
	cuda-check cuda-test h20-rms-norm h20-gemm h20-oxide-gemm h20-attention h20-paged-attention \
	h20-ragged-prefill h20-paged-prefill h20-rope h20-engine-interop h20 \
	bench-oxide bench-paged-oxide bench-ragged-oxide \
	bench-paged-prefill-oxide bench-paged-prefill-graph-oxide bench-ragged-graph-oxide \
	bench-rope-oxide bench-rope-append-oxide \
	bench-rope-append-tokens-oxide bench-rope-append-tokens-graph-oxide bench-split-k

help:
	@printf '%s\n' \
		'make check          Run CPU-only Rust, evidence-tool, and website gates' \
		'make check-tools    Check Python evidence tools' \
		'make cuda-doctor    Check the pinned cuda-oxide environment' \
		'make cuda-check     Run CUDA-feature Clippy' \
		'make cuda-test      Run release tests through cuda-oxide' \
		'make h20            Run all H20 correctness programs sequentially' \
		'make h20-oxide-gemm  Validate the experimental Oxide SM90a M=1 GEMV' \
		'make h20-engine-interop  Validate single and paged external-stream interop' \
		'make h20-paged-attention  Run paged batch-decode H20 correctness' \
		'make h20-ragged-prefill  Run ragged prefill H20 correctness' \
		'make h20-paged-prefill  Run paged prefill H20 correctness' \
		'make h20-rope       Run standard RoPE H20 correctness' \
		'make bench-oxide     Run the Oxide side of the matched H20 benchmark' \
		'make bench-paged-oxide  Run Oxide matched paged-decode cases only' \
		'make bench-ragged-oxide  Run Oxide matched ragged-prefill cases only' \
		'make bench-paged-prefill-oxide  Run Oxide matched paged-prefill cases only' \
		'make bench-paged-prefill-graph-oxide  Run Oxide paged-prefill Graph replay benchmark' \
		'make bench-ragged-graph-oxide  Run Oxide ragged Graph replay benchmark' \
		'make bench-rope-oxide  Run Oxide matched RoPE case only' \
		'make bench-rope-append-oxide  Run Oxide fused RoPE paged append case' \
		'make bench-rope-append-tokens-oxide  Run Oxide explicit multi-token RoPE append case' \
		'make bench-rope-append-tokens-graph-oxide  Run Oxide multi-token RoPE append Graph case' \
		'make bench-split-k  Sweep Oxide split-K choices on H20'

check: check-rust check-tools check-website

check-rust:
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	$(CARGO) test --workspace --all-targets
	$(CARGO) check --workspace --release
	$(CARGO) package -p oxide-infer $(PACKAGE_FLAGS)

check-tools:
	python3 -m py_compile tools/flashinfer/*.py tools/gemm/*.py
	python3 -m unittest discover -s tools/flashinfer -p 'test_*.py'
	python3 -m unittest discover -s tools/gemm -p 'test_*.py'

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

h20-oxide-gemm:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run oxide_sm90_simt_gemv_h20 --bin oxide_sm90_simt_gemv_h20 --features cuda --arch sm_90a

h20-attention:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run single_decode_h20 --bin single_decode_h20 --features cuda --arch $(CUDA_ARCH)

h20-paged-attention:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run paged_batch_decode_h20 --bin paged_batch_decode_h20 --features cuda --arch $(CUDA_ARCH)

h20-ragged-prefill:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run ragged_prefill_h20 --bin ragged_prefill_h20 --features cuda --arch $(CUDA_ARCH)

h20-paged-prefill:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run paged_prefill_h20 --bin paged_prefill_h20 --features cuda --arch $(CUDA_ARCH)

h20-rope:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run rope_h20 --bin rope_h20 --features cuda --arch $(CUDA_ARCH)

h20-engine-interop:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run engine_interop_h20 --bin engine_interop_h20 --features cuda --arch $(CUDA_ARCH)

h20: h20-rms-norm h20-gemm h20-oxide-gemm h20-attention h20-paged-attention h20-ragged-prefill h20-paged-prefill h20-rope h20-engine-interop

bench-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-paged-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=paged_batch_decode OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-ragged-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=ragged_prefill OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-paged-prefill-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=paged_prefill OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-paged-prefill-graph-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run paged_prefill_graph_bench_h20 --bin paged_prefill_graph_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-ragged-graph-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run ragged_graph_bench_h20 --bin ragged_graph_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-rope-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=rope OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-rope-append-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=rope_paged_kv_append OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-rope-append-tokens-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=rope_paged_kv_append_tokens OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-rope-append-tokens-graph-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run rope_append_graph_bench_h20 --bin rope_append_graph_bench_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'

bench-split-k:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" OXIDE_SOURCE_STATE=working_tree $(CARGO) +nightly-2026-04-03 oxide run split_k_sweep_h20 --bin split_k_sweep_h20 --features cuda --arch $(CUDA_ARCH) | sed -n '/^{/p'
