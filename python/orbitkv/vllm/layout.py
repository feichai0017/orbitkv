"""Compile vLLM cache groups into storage and shared recovery requirements."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheGroupLayout:
    """Stable vLLM cache-group order shared by scheduler and worker.

    `storage_group_ids` maps each connector cache group onto the engine's
    storage groups: full-attention layers share group 0; each sliding-window
    or recurrent group has an independent key and recovery requirement.
    """

    layer_names: tuple[tuple[str, ...], ...]
    hash_group_index: int
    has_recurrent_state: bool
    recurrent_group_indices: frozenset[int]
    recurrent_layer_names: frozenset[str]
    storage_group_ids: tuple[int, ...] = (0,)
    recovery_groups: tuple[tuple[int, str, int], ...] = ((0, "attention", 0),)
    window_group_indices: frozenset[int] = frozenset()

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
            SlidingWindowSpec,
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
            if any(
                not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec, MambaSpec))
                for spec in specs
            ):
                raise RuntimeError(
                    "OrbitKV supports FullAttention, SlidingWindow and aligned Mamba groups"
                )

            has_full_attention = any(isinstance(spec, FullAttentionSpec) for spec in specs)
            has_mamba = any(isinstance(spec, MambaSpec) for spec in specs)
            if not has_full_attention:
                raise RuntimeError(
                    "OrbitKV requires a dense FullAttention cache group for block hashes"
                )
            if not has_mamba and not any(isinstance(spec, SlidingWindowSpec) for spec in specs):
                raise RuntimeError("OrbitKV hybrid layouts require window or recurrent state")
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
        window_group_indices = frozenset(
            index for index, spec in enumerate(specs) if isinstance(spec, SlidingWindowSpec)
        )
        if any(specs[index].sliding_window <= 1 for index in window_group_indices):
            raise RuntimeError("OrbitKV window recovery requires at least one past token")
        auxiliary = recurrent_group_indices | window_group_indices
        storage_group_ids = tuple(
            0 if index not in auxiliary else 1 + sum(1 for other in auxiliary if other < index)
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
            window_group_indices=window_group_indices,
            storage_group_ids=storage_group_ids,
            recovery_groups=(
                (
                    0,
                    "mla"
                    if all(
                        isinstance(spec, (MLAAttentionSpec, UniformTypeKVCacheSpecs))
                        for index, spec in enumerate(specs)
                        if index not in auxiliary
                    )
                    else "attention",
                    0,
                ),
                *(
                    (
                        storage_group_ids[index],
                        "window" if index in window_group_indices else "recurrent",
                        specs[index].sliding_window - 1 if index in window_group_indices else 0,
                    )
                    for index in sorted(auxiliary)
                ),
            ),
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
