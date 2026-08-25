# OrbitKV ABI8 H20 token-relocation qualification

Status: scoped Full-attention token-relocation correctness and lifecycle qualified; performance pending. `hardware_attested=false` and `performance_go=false`. No capacity, memory-saving, or general speedup claim is made.

Verify first from a trusted OrbitKV checkout with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/verify_manifests.py /path/to/seal/manifest.json
```

After that succeeds, the bundled portable consistency check is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -S qualification/source/integrations/sglang/qualify_token_relocation_h20.py verify-seal .
```
