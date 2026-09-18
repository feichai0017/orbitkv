.PHONY: source-init source-verify doctor test stage-reference qualify-rmsnorm smoke-sglang-kernel \
        bootstrap-sglang-dry-run bootstrap-autodeploy-dry-run bootstrap-tensorrt-source-dry-run bootstrap-kernels-dry-run \
        bootstrap-sglang bootstrap-autodeploy bootstrap-tensorrt-source bootstrap-kernels

source-init:
	git submodule update --init --recursive

source-verify:
	python3 tools/source_workspace.py verify

doctor:
	python3 tools/source_workspace.py doctor

test:
	cargo build --locked -p aletheia-cli
	cargo test --locked --workspace --all-targets
	python3 -m unittest discover -s integrations/sglang/tests -v
	python3 -m unittest discover -s integrations/autodeploy/tests -v
	python3 -m unittest discover -s integrations/providers/tests -v
	python3 -m unittest discover -s tools/tests -v

stage-reference:
	python3 tools/stage_bundle.py \
		--plan examples/staging/reference-plan.json \
		--evidence examples/staging/reference-evidence.json \
		--trace examples/staging/reference-trace.jsonl \
		--registry .aletheia/registry \
		--aletheia-bin target/debug/aletheia-rt

qualify-rmsnorm:
	env FLASHINFER_WORKSPACE_BASE=$(CURDIR)/.aletheia/cache/flashinfer \
		TORCH_EXTENSIONS_DIR=$(CURDIR)/.aletheia/cache/torch \
		DG_JIT_CACHE_DIR=$(CURDIR)/.aletheia/cache/deepgemm \
		CUDA_HOME=/usr/local/cuda-13.1 \
		PATH=$(CURDIR)/.venv/kernels/bin:/usr/local/cuda-13.1/bin:$(PATH) \
		.venv/kernels/bin/aletheia-qualify-rmsnorm \
		--out .aletheia/candidates/flashinfer-rmsnorm-sm90-b1-h4096-c32

smoke-sglang-kernel:
	env CUDA_HOME=/usr/local/cuda-13.1 \
		PATH=$(CURDIR)/.venv/sglang/bin:/usr/local/cuda-13.1/bin:$(PATH) \
		.venv/sglang/bin/python tools/smoke_sglang_kernel.py

bootstrap-sglang-dry-run:
	python3 tools/source_workspace.py bootstrap --stack sglang

bootstrap-autodeploy-dry-run:
	python3 tools/source_workspace.py bootstrap --stack autodeploy

bootstrap-tensorrt-source-dry-run:
	python3 tools/source_workspace.py bootstrap --stack tensorrt-source

bootstrap-kernels-dry-run:
	python3 tools/source_workspace.py bootstrap --stack kernels

bootstrap-sglang:
	python3 tools/source_workspace.py bootstrap --stack sglang --execute

bootstrap-autodeploy:
	python3 tools/source_workspace.py bootstrap --stack autodeploy --execute

bootstrap-tensorrt-source:
	python3 tools/source_workspace.py bootstrap --stack tensorrt-source --execute

bootstrap-kernels:
	python3 tools/source_workspace.py bootstrap --stack kernels --execute
