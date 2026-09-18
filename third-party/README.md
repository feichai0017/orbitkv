# Reading and updating upstream source

`sources.lock.toml` is the human-readable source lock layered on top of Git
submodule pins. A fresh clone becomes source-complete with:

```bash
git submodule update --init --recursive
python3 tools/source_workspace.py verify
```

The top-level submodules are:

| Path | Role | Pinned ref |
| --- | --- | --- |
| `sglang` | production serving and scheduling | `095ec6c997...` |
| `tensorrt-llm` | AutoDeploy compiler and NVIDIA baseline | `v1.2.0` |
| `flashinfer` | attention/GEMM/MoE candidates and autotuner | `v0.6.17` |
| `deepgemm` | SGLang-compatible FP8/grouped-GEMM backend | `v0.1.5.post3` |

FlashInfer and DeepGEMM retain their own nested submodules, so their exact
CUTLASS/CCCL/fmt sources are visible as well.

For exploratory learning, work on a branch inside a submodule. Product changes
must either be upstreamed or preserved as a reviewable fork commit/patch with a
removal gate. Updating a source requires changing the Gitlink and source lock in
the same review.
