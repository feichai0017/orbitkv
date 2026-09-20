"""vLLM cache-group layout and resumable hybrid checkpoint boundaries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheGroupLayout:
    """Stable vLLM cache-group order shared by scheduler and worker.

    `storage_group_ids` maps each connector cache group onto the engine's
    hybrid storage groups: every attention-like group shares storage group 0
    (prefix cadence, raw hash keys), while each recurrent group gets its own
    id starting at 1 (membership semantics, group-encoded keys).
    """

    layer_names: tuple[tuple[str, ...], ...]
    hash_group_index: int
    has_recurrent_state: bool
    recurrent_group_indices: frozenset[int]
    recurrent_layer_names: frozenset[str]
    storage_group_ids: tuple[int, ...] = (0,)

    @classmethod
    def from_config(cls, kv_cache_config) -> "CacheGroupLayout":
        groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
        if not groups:
            return cls(
                layer_names=((),),
                hash_group_index=0,
                has_recurrent_state=False,
                recurrent_group_indices=frozenset(),
                recurrent_layer_names=frozenset(),
            )

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MambaSpec,
            MLAAttentionSpec,
            UniformTypeKVCacheSpecs,
        )

        specs = tuple(group.kv_cache_spec for group in groups)
        if len(specs) == 1:
            spec = specs[0]
            is_uniform_mla = (
                type(spec) is UniformTypeKVCacheSpecs
                and bool(spec.kv_cache_specs)
                and all(
                    type(layer_spec) is MLAAttentionSpec
                    for layer_spec in spec.kv_cache_specs.values()
                )
            )
            if type(spec) not in (FullAttentionSpec, MLAAttentionSpec) and not is_uniform_mla:
                raise RuntimeError(
                    "OrbitKV supports a single cache group only for FullAttention, MLA, "
                    "or uniformly grouped MLA layers"
                )
        else:
            if any(not isinstance(spec, (FullAttentionSpec, MambaSpec)) for spec in specs):
                raise RuntimeError("OrbitKV HMA supports only FullAttention and Mamba cache groups")

            has_full_attention = any(isinstance(spec, FullAttentionSpec) for spec in specs)
            has_mamba = any(isinstance(spec, MambaSpec) for spec in specs)
            if not has_full_attention:
                raise RuntimeError(
                    "OrbitKV requires a dense FullAttention cache group for block hashes"
                )
            if not has_mamba:
                raise RuntimeError("OrbitKV HMA requires both FullAttention and Mamba cache groups")
            if any(
                isinstance(spec, MambaSpec) and spec.mamba_cache_mode != "align" for spec in specs
            ):
                raise RuntimeError("OrbitKV HMA requires mamba_cache_mode='align'")

        block_sizes = {group.kv_cache_spec.block_size for group in groups}
        if len(groups) > 1 and len(block_sizes) != 1:
            raise RuntimeError(
                "OrbitKV HMA requires cache groups with identical logical block sizes"
            )

        hash_group_index = (
            0
            if len(groups) == 1
            else next(
                (
                    index
                    for index, group in enumerate(groups)
                    if isinstance(group.kv_cache_spec, FullAttentionSpec)
                ),
                None,
            )
        )
        if hash_group_index is None:
            raise RuntimeError(
                "OrbitKV requires a dense FullAttention cache group for block hashes"
            )

        recurrent_group_indices = frozenset(
            index
            for index, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        )
        # Attention-like groups all share storage group 0 (they advance in
        # per-block prefix cadence); each recurrent group gets a dense id
        # from 1. Engine keys are raw only for group 0, so this keeps every
        # existing single-group cache layout bit-identical.
        storage_group_ids = tuple(
            0
            if index not in recurrent_group_indices
            else 1 + sum(1 for other in recurrent_group_indices if other < index)
            for index in range(len(groups))
        )

        return cls(
            layer_names=tuple(tuple(group.layer_names) for group in groups),
            hash_group_index=hash_group_index,
            has_recurrent_state=any(isinstance(group.kv_cache_spec, MambaSpec) for group in groups),
            recurrent_group_indices=recurrent_group_indices,
            recurrent_layer_names=frozenset(
                layer_name
                for group in groups
                if isinstance(group.kv_cache_spec, MambaSpec)
                for layer_name in group.layer_names
            ),
            storage_group_ids=storage_group_ids,
        )

    @property
    def group_count(self) -> int:
        return len(self.layer_names)

    def layer_to_group(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for group_index, names in enumerate(self.layer_names):
            for name in names:
                if name in result:
                    raise RuntimeError(f"KV cache layer belongs to multiple groups: {name}")
                result[name] = group_index
        return result

    def storage_group_of(self, group_index: int) -> int:
        """Engine storage group id for a connector cache group index."""
        return self.storage_group_ids[group_index]


def reconcile_hybrid_hit(
    attention_hit_blocks: int,
    recurrent_hits: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[int, int | None, frozenset[int]]:
    """Combine per-group query results into one hybrid hit.

    ``attention_hit_blocks`` is the (already shard-minimized) attention prefix
    length in blocks. ``recurrent_hits[g][s]`` lists the query positions whose
    checkpoint block is cached in recurrent group ``g`` on shard ``s``.

    HMA needs the whole prefix resumable: every recurrent group must hold a
    checkpoint state inside the attention prefix (attention KV alone cannot
    skip mamba's sequential prefill), and that state must exist on every TP
    shard. Returns ``(hit_blocks, checkpoint, usable)`` where ``hit_blocks``
    is ``checkpoint + 1`` — the checkpoint covers tokens through the end of
    its own block — and ``usable`` is every legal boundary position (for
    re-derivation when the token budget later shrinks the hit). ``(0, None,
    frozenset())`` means no usable boundary: recompute from scratch.
    """
    if attention_hit_blocks <= 0 or not recurrent_hits:
        return 0, None, frozenset()
    # A checkpoint position is usable only inside the attention prefix AND
    # present in every recurrent group on every shard.
    usable: set[int] | None = None
    for group_hits in recurrent_hits:
        for shard_hits in group_hits:
            in_prefix = {p for p in shard_hits if p < attention_hit_blocks}
            usable = in_prefix if usable is None else usable & in_prefix
            if not usable:
                return 0, None, frozenset()
    if not usable:
        return 0, None, frozenset()
    checkpoint = max(usable)
    return checkpoint + 1, checkpoint, frozenset(usable)
