SHELL := /bin/bash
.DEFAULT_GOAL := help

CUDA_CRATE := crates/oxide-infer-cuda
VALIDATION_CRATE := crates/oxide-infer-lab
CUDA_ARCH ?= sm_90
override H20_ARCH := sm_90a
H20_RUNNER ?=
H20_RUNNER_SHA256 ?=
H20_RUNNERS := rms_norm_h20 bf16_gemm_h20 oxide_sm90_simt_gemv_h20 \
	single_decode_h20 paged_batch_decode_h20 ragged_prefill_h20 \
	paged_prefill_h20 rope_h20 engine_interop_h20
override H20_TARGET_DIR := $(CURDIR)/target
override H20_BINARY = $(H20_TARGET_DIR)/release/$(H20_RUNNER)
override COMPUTE_SANITIZER := compute-sanitizer
override SHA256SUM := sha256sum
PACKAGE_FLAGS ?= --allow-dirty
OXIDE_SOURCE_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)
USE_MISE ?= 0
ifeq ($(USE_MISE),1)
MISE := $(shell command -v mise 2>/dev/null)
ifeq ($(MISE),)
$(error USE_MISE=1 requires mise on PATH)
endif
RUN := $(MISE) exec --
else ifneq ($(USE_MISE),0)
$(error USE_MISE must be 0 or 1)
endif
ifneq ($(OXIDE_CARGO_HOME),)
CARGO_ENV := CARGO_HOME=$(OXIDE_CARGO_HOME)
endif
CARGO := $(CARGO_ENV) $(RUN) cargo
NPM := $(RUN) npm --prefix website

.PHONY: help check check-rust check-tools check-website install-website cuda-doctor \
	cuda-check cuda-test h20-rms-norm h20-gemm h20-oxide-gemm h20-attention h20-paged-attention \
	h20-ragged-prefill h20-paged-prefill h20-rope h20-engine-interop h20 \
	h20-runner-preflight h20-build-runner h20-sanitize-runner \
	bench-oxide bench-paged-oxide bench-ragged-oxide \
	bench-sm90-gemv-oxide bench-sm90-gemv-cublaslt \
	bench-paged-prefill-oxide bench-paged-prefill-graph-oxide bench-ragged-graph-oxide \
	bench-rope-oxide bench-rope-append-oxide \
	bench-rope-append-tokens-oxide bench-rope-append-tokens-graph-oxide bench-split-k

help:
	@printf '%s\n' \
		'make check          Run CPU-only Rust, evidence-tool, and website gates' \
		'make check-tools    Check Python evidence tools' \
		'make cuda-doctor    Check the pinned cuda-oxide environment' \
		'make cuda-check     Run CUDA-feature Clippy' \
		'make cuda-test      Run generic release tests; CUDA_ARCH selects the target' \
		'make h20            Run all H20 correctness programs for fixed sm_90a' \
		'make h20-build-runner H20_RUNNER=<name>  Build and hash one permanent runner' \
		'make h20-sanitize-runner H20_RUNNER=<name> H20_RUNNER_SHA256=<sha256>  Sanitize the exact built binary' \
		'make h20-oxide-gemm  Validate the experimental Oxide SM90a M=1 GEMV' \
		'make h20-engine-interop  Validate single and paged external-stream interop' \
		'make h20-paged-attention  Run paged batch-decode H20 correctness' \
		'make h20-ragged-prefill  Run ragged prefill H20 correctness' \
		'make h20-paged-prefill  Run paged prefill H20 correctness' \
		'make h20-rope       Run standard RoPE H20 correctness' \
		'make bench-oxide     Run the Oxide side of the matched H20 benchmark' \
		'make bench-sm90-gemv-oxide  Run the five-shape native M=1 GEMV benchmark' \
		'make bench-sm90-gemv-cublaslt  Run its matched cuBLASLt baseline' \
		'make bench-paged-oxide  Run Oxide matched paged-decode cases only' \
		'make bench-ragged-oxide  Run Oxide matched ragged-prefill cases only' \
		'make bench-paged-prefill-oxide  Run Oxide matched paged-prefill cases only' \
		'make bench-paged-prefill-graph-oxide  Run Oxide paged-prefill Graph replay benchmark' \
		'make bench-ragged-graph-oxide  Run Oxide ragged Graph replay benchmark' \
		'make bench-rope-oxide  Run Oxide matched RoPE case only' \
		'make bench-rope-append-oxide  Run Oxide fused RoPE paged append case' \
		'make bench-rope-append-tokens-oxide  Run Oxide explicit multi-token RoPE append case' \
		'make bench-rope-append-tokens-graph-oxide  Run Oxide multi-token RoPE append Graph case' \
		'make bench-split-k  Sweep Oxide split-K choices on H20' \
		'USE_MISE=1 make <target>  Run a target through the trusted mise environment'

check: check-rust check-tools check-website

check-rust:
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	$(CARGO) test --workspace --all-targets
	$(CARGO) check --workspace --release
	$(CARGO) package -p oxide-infer $(PACKAGE_FLAGS)

check-tools:
	python3 -m py_compile tools/flashinfer/*.py tools/gemm/*.py tools/tilelang/*.py
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
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run rms_norm_h20 --bin rms_norm_h20 --features cuda --arch $(H20_ARCH)

h20-gemm:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run bf16_gemm_h20 --bin bf16_gemm_h20 --features cuda --arch $(H20_ARCH)

h20-oxide-gemm:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run oxide_sm90_simt_gemv_h20 --bin oxide_sm90_simt_gemv_h20 --features cuda --arch $(H20_ARCH)

h20-attention:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run single_decode_h20 --bin single_decode_h20 --features cuda --arch $(H20_ARCH)

h20-paged-attention:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run paged_batch_decode_h20 --bin paged_batch_decode_h20 --features cuda --arch $(H20_ARCH)

h20-ragged-prefill:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run ragged_prefill_h20 --bin ragged_prefill_h20 --features cuda --arch $(H20_ARCH)

h20-paged-prefill:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run paged_prefill_h20 --bin paged_prefill_h20 --features cuda --arch $(H20_ARCH)

h20-rope:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run rope_h20 --bin rope_h20 --features cuda --arch $(H20_ARCH)

h20-engine-interop:
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide run engine_interop_h20 --bin engine_interop_h20 --features cuda --arch $(H20_ARCH)

h20: h20-rms-norm h20-gemm h20-oxide-gemm h20-attention h20-paged-attention h20-ragged-prefill h20-paged-prefill h20-rope h20-engine-interop

h20-runner-preflight:
	@case " $(H20_RUNNERS) " in \
		*" $(H20_RUNNER) "*) ;; \
		*) printf 'H20_RUNNER must be one of: %s\n' "$(H20_RUNNERS)" >&2; exit 2 ;; \
	esac

h20-build-runner: h20-runner-preflight
	cd $(VALIDATION_CRATE) && $(CARGO) +nightly-2026-04-03 oxide build --lineinfo --arch $(H20_ARCH) --cargo-target-dir "$(H20_TARGET_DIR)" -- --package oxide-infer-lab --bin $(H20_RUNNER) --features cuda --release
	@set -eu; \
		command -v "$(SHA256SUM)" >/dev/null || { echo 'sha256sum is required' >&2; exit 2; }; \
		artifact_hash="$$($(SHA256SUM) "$(H20_BINARY)" | awk '{print $$1}')"; \
		printf 'runner_binary=%s sha256=%s arch=%s\n' "$(H20_BINARY)" "$$artifact_hash" "$(H20_ARCH)"

h20-sanitize-runner: h20-runner-preflight
	@set -eu; \
		expected_hash="$(H20_RUNNER_SHA256)"; \
		[[ "$$expected_hash" =~ ^[0-9a-f]{64}$$ ]] || { echo 'H20_RUNNER_SHA256 must be the 64-character lowercase SHA-256 printed by h20-build-runner' >&2; exit 2; }; \
		command -v "$(COMPUTE_SANITIZER)" >/dev/null || { echo 'compute-sanitizer is required' >&2; exit 2; }; \
		command -v "$(SHA256SUM)" >/dev/null || { echo 'sha256sum is required' >&2; exit 2; }; \
		binary="$(H20_BINARY)"; \
		test -x "$$binary" || { printf 'missing runner binary: %s; run make h20-build-runner H20_RUNNER=%s first\n' "$$binary" "$(H20_RUNNER)" >&2; exit 2; }; \
		read_hash() { "$(SHA256SUM)" "$$binary" | awk '{print $$1}'; }; \
		artifact_hash="$$(read_hash)"; \
		test "$$artifact_hash" = "$$expected_hash" || { printf 'runner hash does not match build output: expected=%s actual=%s\n' "$$expected_hash" "$$artifact_hash" >&2; exit 2; }; \
		printf 'sanitizer_binary=%s sha256=%s arch=%s\n' "$$binary" "$$artifact_hash" "$(H20_ARCH)"; \
		for tool in memcheck racecheck synccheck initcheck; do \
			current_hash="$$(read_hash)"; \
			test "$$current_hash" = "$$artifact_hash" || { printf 'runner hash changed before %s: expected=%s actual=%s\n' "$$tool" "$$artifact_hash" "$$current_hash" >&2; exit 2; }; \
			if test "$$tool" = memcheck; then \
				"$(COMPUTE_SANITIZER)" --tool "$$tool" --leak-check full --error-exitcode 99 "$$binary"; \
			else \
				"$(COMPUTE_SANITIZER)" --tool "$$tool" --error-exitcode 99 "$$binary"; \
			fi; \
			current_hash="$$(read_hash)"; \
			test "$$current_hash" = "$$artifact_hash" || { printf 'runner hash changed after %s: expected=%s actual=%s\n' "$$tool" "$$artifact_hash" "$$current_hash" >&2; exit 2; }; \
		done; \
		printf 'sanitizer_status=pass binary=%s sha256=%s tools=memcheck,racecheck,synccheck,initcheck\n' "$$binary" "$$artifact_hash"

bench-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-sm90-gemv-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=gemv_m1 OXIDE_BENCH_GEMV_PROVIDER=oxide OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-sm90-gemv-cublaslt:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=gemv_m1 OXIDE_BENCH_GEMV_PROVIDER=cublaslt OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-paged-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=paged_batch_decode OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-ragged-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=ragged_prefill OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-paged-prefill-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=paged_prefill OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-paged-prefill-graph-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run paged_prefill_graph_bench_h20 --bin paged_prefill_graph_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-ragged-graph-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run ragged_graph_bench_h20 --bin ragged_graph_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-rope-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=rope OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-rope-append-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=rope_paged_kv_append OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-rope-append-tokens-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_BENCH_OPERATORS=rope_paged_kv_append_tokens OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run oxide_matched_bench_h20 --bin oxide_matched_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-rope-append-tokens-graph-oxide:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" $(CARGO) +nightly-2026-04-03 oxide run rope_append_graph_bench_h20 --bin rope_append_graph_bench_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'

bench-split-k:
	@set -o pipefail; cd $(VALIDATION_CRATE) && OXIDE_SOURCE_COMMIT="$(OXIDE_SOURCE_COMMIT)" OXIDE_SOURCE_STATE=working_tree $(CARGO) +nightly-2026-04-03 oxide run split_k_sweep_h20 --bin split_k_sweep_h20 --features cuda --arch $(H20_ARCH) | sed -n '/^{/p'
