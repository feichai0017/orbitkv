# OrbitKV ABI8 SGLang Full and Full+SWA Prefix qualification

Status: ABI8 SGLang Full/Full+SWA Prefix correctness qualified; performance pending.

Source commit: `6f62a23b9abaa9bf12e9b060389259fa9185e70f`

Verified pairs: 12 across 3 epoch(s).

Cases: Qwen2.5-7B Full/FA3 B1+B4; GPT-OSS-20B Full+SWA/FA3 B1+B4.

Excluded: token relocation, MLA, fixed-state, overlap scheduling, CUDA Graphs, speculation, distributed execution, and performance qualification.

The `qualification/source` directory is a source closure containing the qualification runner, benchmark, checkpoint helper, packaging metadata, reviewed SGLang patch/prepare helper, and complete `orbitkv_sglang` Python package. Verification is offline and does not require models, SGLang worktrees, the original repository, or an installed editable package.

Establish trust first from a trusted OrbitKV checkout with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/verify_manifests.py /path/to/seal/manifest.json
```

That command verifies the Git source closure, every artifact, the ELF symbol table, and all record semantics without executing bundled code or loading the bundled library. After it succeeds, this optional portable consistency check may be run from the seal root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 qualification/source/qualify_abi8_h20.py verify-seal .
```

Do not use the bundled command as the initial authenticity check.
