"""Give Qwen manifest dimensions semantic names after ABI normalization.

This is deliberately context-aware: equal numbers are not necessarily the same
constant (HIDDEN_SIZE vs block-table capacity, or Q width vs GDN V width).
Unclassified kernel tuning parameters stay literal. No kernel or capture is
needed to reapply this presentation pass.
"""
import copy


def name_constants(manifest):
    m = copy.deepcopy(manifest)
    if "constants" in m:
        return m
    hybrid = m["model"].startswith("qwen3.8")
    constants = {}

    def ref(name, value):
        name = name.upper()
        if isinstance(value, str):
            return value
        assert name not in constants or constants[name] == value, name
        constants[name] = value
        return name

    hidden, vocab, ffn = (5120, 248320, 17408) if hybrid else (2560, 151936, 9728)
    common = {hidden: "hidden_size", vocab: "vocab_size", ffn: "intermediate_size", 2 * ffn: "gate_up_dim"}
    attention = {14336: "qkv_dim", 6144: "q_dim", 1024: "kv_dim", 256: "head_dim"} if hybrid else {
        6144: "qkv_dim", 4096: "q_dim", 1024: "kv_dim", 128: "head_dim"}
    draft = {6144: "draft_qkv_dim", 4096: "draft_q_dim", 1024: "draft_kv_dim", 128: "draft_head_dim"} if hybrid else attention
    gdn = {16384: "gdn_qkvz_dim", 10240: "gdn_conv_dim", 2048: "gdn_qk_dim", 6144: "gdn_v_dim",
           96: "gdn_ba_dim", 48: "gdn_num_v_heads", 128: "gdn_head_dim", 64: "fla_chunk_size"}
    gdn_buffers = {"qkvz", "ba", "conv_out", "gdn_q", "gdn_k", "gdn_v", "Ai", "w", "u", "v_new",
                   "h", "core_attn_out", "z_c", "g", "beta", "g_cum", "A", "h0"}
    for name, b in m["buffers"].items():
        is_draft = name.startswith(("draft.", "d.", "d_"))
        dims = common | (draft if is_draft else attention)
        if hybrid and (name in gdn_buffers or ".linear_attn." in name):
            dims = common | gdn
        if name.startswith("draft.selector."):
            dims = common | {256: "selector_rank"}
        elif "markov" in name:
            dims = common | {256: "markov_rank"}
        elif name == "draft.fused_kv.weight":
            dims = common | {10240: "draft_fused_kv_dim"}
        elif "kernel_projection" in name:
            dims = common | {1280: "draft_conv_proj_dim"}
        # These buffers have capacity axes, not feature axes.
        if name in ("block_table", "draft_block_table"):
            dims = {b["shape"][-1]: "draft_max_blocks_per_seq" if name.startswith("draft") else "max_blocks_per_seq"}
        elif name == "cu_seqlens_q":
            dims = {b["shape"][0]: "seq_offsets_capacity"}
        elif "rope." in name:
            dims = {b["shape"][0]: "max_position_embeddings", b["shape"][1]:
                    "rotary_half_dim" if name in ("rope.cos", "rope.sin") else
                    "draft_head_dim" if hybrid and is_draft else "head_dim"}
        elif name in ("cos_g", "sin_g"):
            dims = {32: "rotary_half_dim"}
        elif name in ("draft_tokens", "verify_tokens"):
            dims = {b["shape"][-1]: "draft_tokens_per_round" if name == "draft_tokens" else "verify_tokens_per_round"}
        elif name.endswith("kv_scales"):
            dims = {b["shape"][0]: "draft_kv_scale_count" if is_draft else "kv_scale_count"}
        elif name in ("gdn.line_index", "line_adv"):
            dims = {48: "gdn_num_layers", 8: "verify_tokens_per_round"}
        elif name.startswith("fla.chunk_"):
            dims = {manifest["buffers"]["h"]["shape"][0]: "max_fla_chunks"}
        elif name in ("conv.batch_ptr", "conv.token_chunk_offset", "gdn.has_initial"):
            dims = {}  # capture/kernel metadata capacity, not a head dimension
        if hybrid and name in ("cand_ids", "cand_vals", "hidden_r", "succ_g", "pred_g", "pred_anchor"):
            dims = {1024: "max_verify_tokens", 16: "selector_candidates", 256: "selector_rank", 16384: "max_candidate_rows"}
        elif hybrid and name in ("k_save", "v_save", "a_save", "b_save"):
            dims = {1024: "max_verify_tokens", 2048: "gdn_qk_dim", 6144: "gdn_v_dim"}
        elif hybrid and name in ("a_c", "b_c"):
            dims = {48: "gdn_num_v_heads"}
        elif hybrid and name == "kv_flat":
            dims = {10240: "draft_fused_kv_dim"}
        elif hybrid and name == "d_coef":
            dims = {1280: "draft_conv_proj_dim"}
        b["shape"] = [ref(dims[d], d) if isinstance(d, int) and d in dims else d for d in b["shape"]]
        if hybrid and name in ("k_save", "v_save", "a_save", "b_save"):
            b["shape"][0] = ref("gdn_num_layers", 48)
            if name in ("a_save", "b_save"):
                b["shape"][2] = ref("gdn_num_v_heads", 48)
        if hybrid and name == "h":
            b["shape"][0] = ref("max_fla_chunks", manifest["buffers"][name]["shape"][0])
        if hybrid and name == "logits_blk":
            b["shape"][0] = ref("max_verify_tokens", manifest["buffers"][name]["shape"][0])
        if "stride" in b.get("domain", {}) and name in ("block_table", "draft_block_table"):
            b["domain"]["stride"] = ref("draft_kv_block_size" if name.startswith("draft") else "kv_block_size", b["domain"]["stride"])

    for name, v in m["vars"].items():
        if name in ("tokens", "seqs"):
            v["max"] = ref("max_" + name, v["max"])
    for name, s in m.get("states", {}).items():
        for field in ("bytes_per_token", "bytes_per_seq", "bytes_fixed"):
            if field in s:
                s[field] = ref(f"{name}_{field}", s[field])

    def scalars(value, dims, op):
        if isinstance(value, list):
            for v in value:
                scalars(v, dims, op)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k in ("i32", "i64") and isinstance(v, int) and v in dims:
                    value[k] = ref(dims[v], v)
                elif k == "f32" and isinstance(v, (int, float)):
                    # An epsilon is orders of magnitude below any attention
                    # scale (1/sqrt(head_dim)), whichever op carries it.
                    if 0 < v < 0.001:
                        value[k] = ref("rms_norm_eps", v)
                    elif op.startswith("attn") and 0 < v < 1:
                        value[k] = ref("draft_attention_scale" if op == "attn_draft" else "attention_scale", v)
                elif k == "expr" and isinstance(v, dict):
                    for expr in ("mul", "ceil_div"):
                        if expr in v and v[expr][1] in dims:
                            v[expr][1] = ref(dims[v[expr][1]], v[expr][1])
                else:
                    scalars(v, dims, op)

    for name, op in m["ops"].items():
        # Only classify dimensions in interfaces whose semantics are known.
        dims = {}
        if name in ("embedding", "silu_mul", "argmax", "argmax_row", "last_row", "copy_rows") or "norm" in name:
            dims = common.copy()
        if "qhead" in name or "khead" in name or name in ("rotary_embedding", "mrope", "reshape_and_cache") or name.startswith("attn"):
            dims |= attention
            dims |= {24 if hybrid else 32: "num_heads", 4 if hybrid else 8: "num_kv_heads"}
        if name.startswith("attn"):
            table = manifest["buffers"]["block_table"]["shape"][-1]
            dims[table] = "max_blocks_per_seq"
        if hybrid and (name.startswith("d_") or name == "attn_draft"):
            dims = common | draft | {32: "draft_num_heads", 8: "draft_num_kv_heads"}
            if name == "attn_draft":
                dims[16384] = "draft_max_blocks_per_seq"
        if name in ("gated_norm", "gated_norm_decode"):
            dims = gdn
        scalars(op, dims, name)

    for program in m["programs"].values():
        if "batch" in program:
            batch = program["batch"]
            if batch["groups"] == manifest["vars"]["seqs"]["max"]:
                batch["groups"] = ref("max_seqs", batch["groups"])
            if isinstance(batch["rows"], int) and batch["rows"] > 1:
                batch["rows"] = ref("verify_tokens_per_round", batch["rows"])
        for call in program["calls"]:
            args = call["args"]
            if call["op"] in ("gemm", "gemm_acc"):
                # N and K are precisely the two weight axes, even for GDN
                # and draft projections whose dimensions happen to coincide.
                weight = m["buffers"][args[1]["buf"]]["shape"]
                for a, dim in zip(args[4:6], weight):
                    if isinstance(dim, str) and "i32" in a:
                        assert constants[dim] == a["i32"]
                        a["i32"] = dim
            if not hybrid:
                for arg in args:
                    if arg.get("buf") == "qkv" and arg.get("offset") in (8192, 10240):
                        offset = arg["offset"]
                        arg["offset"] = ref("qkv_k_offset_bytes" if offset == 8192 else "qkv_v_offset_bytes", offset)
            if call["op"] in ("rms_norm_qhead", "rms_norm_khead"):
                scalars(args, {32: "num_heads", 8: "num_kv_heads"}, call["op"])
    return {"schema_version": m.pop("schema_version"), "model": m.pop("model"), "constants": constants, **m}
