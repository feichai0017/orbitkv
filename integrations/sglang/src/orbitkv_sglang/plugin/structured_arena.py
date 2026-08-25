"""Structured SGLang MHA arenas and pure external-append mapping.

SGLang stores one logical KV class in separate K/V tensors for every layer.
The engine-neutral runtime contract deliberately treats a token record as one
opaque byte row, so this module describes how that row is split across the
SGLang-owned tensors without giving the adapter allocation authority.

The module has no eager Torch or SGLang dependency.  Builders inspect the
already-validated pool objects structurally, which also keeps their geometry
and lowering tests host-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

from orbitkv_runtime import (
    ArenaRegistration,
    BackendPageAddress,
    BackendTokenAddress,
    DataPlaneOperation,
    ExternalTokenWrite,
    OperationContext,
    PageLease,
    RequestLease,
    StepLease,
)


_MISSING = object()
_COMPONENT_NAMES = ("key", "value")


def _integer(name: str, value: Any, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeError(f"{name} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise RuntimeError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class SglangArenaComponent:
    """One layer-local K or V tensor in a structured class arena.

    ``token_byte_offset`` is an offset in the class's conceptual opaque token
    record.  It does not imply that the independently allocated tensors are
    physically contiguous.
    """

    global_layer_id: int
    local_layer_id: int
    name: str
    tensor: object
    token_byte_offset: int
    token_byte_count: int

    def __post_init__(self) -> None:
        _integer("global_layer_id", self.global_layer_id)
        _integer("local_layer_id", self.local_layer_id)
        if self.name not in _COMPONENT_NAMES:
            raise ValueError("structured MHA component must be 'key' or 'value'")
        if self.tensor is None:
            raise ValueError("structured arena component tensor must not be None")
        _integer("token_byte_offset", self.token_byte_offset)
        _integer("token_byte_count", self.token_byte_count, positive=True)


@dataclass(frozen=True, slots=True)
class SglangStructuredArena:
    """One neutral arena registration backed by structured SGLang tensors."""

    registration: ArenaRegistration
    retention: str
    storage: str
    storage_token_bias: int
    components: tuple[SglangArenaComponent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.registration, ArenaRegistration):
            raise TypeError("registration must be an orbitkv_runtime ArenaRegistration")
        if self.retention not in ("full", "sliding"):
            raise ValueError("structured arena retention is unsupported")
        if self.storage != "token_kv":
            raise ValueError("structured arena supports only token_kv storage")
        if self.storage_token_bias != self.registration.page_tokens:
            raise ValueError("SGLang storage bias must be exactly one dummy page")
        if not isinstance(self.components, tuple) or not self.components:
            raise TypeError("components must be a nonempty tuple")
        if any(not isinstance(item, SglangArenaComponent) for item in self.components):
            raise TypeError("components must contain SglangArenaComponent values")

        offset = 0
        expected_local = 0
        index = 0
        seen_tensors: set[int] = set()
        while index < len(self.components):
            if index + 1 >= len(self.components):
                raise ValueError("every structured MHA layer requires key and value")
            key, value = self.components[index : index + 2]
            if (key.name, value.name) != _COMPONENT_NAMES:
                raise ValueError("structured components are not in layer/key/value order")
            if (key.local_layer_id, value.local_layer_id) != (
                expected_local,
                expected_local,
            ) or key.global_layer_id != value.global_layer_id:
                raise ValueError("structured component layer identity is not canonical")
            for component in (key, value):
                if component.token_byte_offset != offset:
                    raise ValueError("structured component byte offsets are not contiguous")
                offset += component.token_byte_count
                tensor_identity = id(component.tensor)
                if tensor_identity in seen_tensors:
                    raise ValueError("structured arena aliases a component tensor")
                seen_tensors.add(tensor_identity)
            expected_local += 1
            index += 2
        if offset != self.registration.token_bytes:
            raise ValueError("structured component bytes differ from arena token width")

    @property
    def class_id(self) -> int:
        return self.registration.class_id

    @property
    def page_tokens(self) -> int:
        return self.registration.page_tokens

    @property
    def token_bytes(self) -> int:
        return self.registration.token_bytes

    @property
    def managed_token_count(self) -> int:
        return self.registration.page_count * self.registration.page_tokens


@dataclass(frozen=True, slots=True)
class ExternalAppendManifest:
    """Immutable proof that SGLang's emitted slots equal neutral writes."""

    writes: tuple[ExternalTokenWrite, ...]
    last_use_pages: tuple[BackendPageAddress, ...]
    expected_locations_by_class: tuple[tuple[int, tuple[int, ...]], ...]
    arenas: tuple[SglangStructuredArena, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.writes, tuple) or not self.writes or any(
            not isinstance(item, ExternalTokenWrite) for item in self.writes
        ):
            raise TypeError("writes must be a nonempty tuple of ExternalTokenWrite values")
        if not isinstance(self.last_use_pages, tuple) or not self.last_use_pages or any(
            not isinstance(item, BackendPageAddress) for item in self.last_use_pages
        ):
            raise TypeError("last_use_pages must be a nonempty tuple of page addresses")
        if len(set(self.last_use_pages)) != len(self.last_use_pages):
            raise ValueError("last_use_pages must not contain duplicates")
        if not isinstance(self.arenas, tuple) or not self.arenas or any(
            not isinstance(item, SglangStructuredArena) for item in self.arenas
        ):
            raise TypeError("arenas must be a nonempty tuple of structured arenas")
        class_ids = tuple(item.class_id for item in self.arenas)
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("structured arena class ids must be unique")
        if (
            not isinstance(self.expected_locations_by_class, tuple)
            or tuple(item[0] for item in self.expected_locations_by_class) != class_ids
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[1], tuple)
                for item in self.expected_locations_by_class
            )
        ):
            raise TypeError(
                "expected_locations_by_class must follow structured arena order"
            )
        widths = {item.class_id: item.token_bytes for item in self.arenas}
        if any(
            write.destination.class_id not in widths
            or write.byte_count != widths.get(write.destination.class_id)
            for write in self.writes
        ):
            raise ValueError("manifest write width or class differs from its arena")
        if len({item.destination for item in self.writes}) != len(self.writes):
            raise ValueError("manifest aliases an external write destination")
        logical = {
            (item.context, item.destination.class_id, item.token_id)
            for item in self.writes
        }
        if len(logical) != len(self.writes):
            raise ValueError("manifest duplicates a logical class/token write")
        location_lengths = tuple(
            len(values) for _class_id, values in self.expected_locations_by_class
        )
        if not location_lengths or len(set(location_lengths)) != 1:
            raise ValueError("manifest class location cardinalities differ")
        if len(self.writes) != location_lengths[0] * len(self.arenas):
            raise ValueError("manifest write and location cardinalities differ")

    @property
    def structured_arenas(self) -> tuple[SglangStructuredArena, ...]:
        return self.arenas


def build_sglang_structured_arenas(
    config: Any, runtime: Any, token_to_kv_pool: Any
) -> tuple[SglangStructuredArena, ...]:
    """Describe the exact MHA tensors backing every compiled KV class.

    Only Full and ordered Full+SWA ``token_kv`` plans are accepted.  The
    returned registrations use neutral, dummy-free arena geometry even though
    every component tensor retains SGLang's one-page storage prefix.
    """

    classes = _config_classes(config)
    retentions = tuple(getattr(item, "retention", None) for item in classes)
    if retentions not in (("full",), ("full", "sliding")):
        raise RuntimeError(
            "structured SGLang arenas require Full or ordered Full+SWA classes"
        )
    if any(getattr(item, "storage", None) != "token_kv" for item in classes):
        raise RuntimeError("structured SGLang arenas do not support MLA storage")

    identities = _runtime_arenas(runtime, classes)
    pools = _physical_pools(token_to_kv_pool, retentions)
    result = tuple(
        _build_structured_arena(class_config, identity, pool, config)
        for class_config, identity, pool in zip(
            classes, identities, pools, strict=True
        )
    )
    registrations = tuple(item.registration for item in result)
    if len({item.pool_id for item in registrations}) != len(registrations):
        raise RuntimeError("structured arenas have duplicate pool ids")
    if len({(item.class_id, item.backend_domain) for item in registrations}) != len(
        registrations
    ):
        raise RuntimeError("structured arenas have duplicate class/domain identities")
    epochs = {item.engine_epoch for item in registrations}
    if len(epochs) != 1:
        raise RuntimeError("structured arenas do not share one engine epoch")
    page_ranges = sorted(
        (item.first_page_id, item.first_page_id + item.page_count)
        for item in registrations
    )
    if any(right_start < left_end for (_, left_end), (right_start, _) in zip(
        page_ranges, page_ranges[1:], strict=False
    )):
        raise RuntimeError("structured arena page-id ranges overlap")
    return result


def build_structured_arenas(
    config: Any, runtime: Any, token_to_kv_pool: Any
) -> tuple[SglangStructuredArena, ...]:
    """Compatibility spelling for :func:`build_sglang_structured_arenas`."""

    return build_sglang_structured_arenas(config, runtime, token_to_kv_pool)


def _config_classes(config: Any) -> tuple[Any, ...]:
    try:
        classes = tuple(config.classes)
    except Exception as error:
        raise RuntimeError("compiled KV classes are unreadable") from error
    if not classes:
        raise RuntimeError("compiled KV classes must not be empty")
    class_ids = tuple(_integer("class id", item.class_id) for item in classes)
    if len(set(class_ids)) != len(class_ids):
        raise RuntimeError("compiled KV class ids must be unique")
    return classes


def _runtime_arenas(runtime: Any, classes: Sequence[Any]) -> tuple[Any, ...]:
    mapping = getattr(runtime, "arenas_by_class", None)
    if not isinstance(mapping, Mapping):
        raise RuntimeError("runtime arena index is missing")
    expected_ids = tuple(int(item.class_id) for item in classes)
    if set(mapping) != set(expected_ids):
        raise RuntimeError("runtime arena index differs from compiled classes")
    identities = tuple(mapping[class_id] for class_id in expected_ids)
    ordered = getattr(runtime, "arenas", identities)
    try:
        ordered = tuple(ordered)
    except Exception as error:
        raise RuntimeError("runtime arenas are unreadable") from error
    if ordered != identities:
        raise RuntimeError("runtime arenas are not in compiled class order")
    return identities


def _physical_pools(root: Any, retentions: tuple[str, ...]) -> tuple[Any, ...]:
    if root is None:
        raise RuntimeError("SGLang KV pool is missing")
    if retentions == ("full", "sliding"):
        full = getattr(root, "full_kv_pool", None)
        sliding = getattr(root, "swa_kv_pool", None)
        if full is None or sliding is None or full is sliding:
            raise RuntimeError("SGLang Full+SWA physical pools are missing or aliased")
        return full, sliding

    if getattr(root, "swa_kv_pool", None) is not None:
        raise RuntimeError("Full-only plan received an SGLang SWA pool")
    has_direct_buffers = getattr(root, "k_buffer", None) is not None or getattr(
        root, "v_buffer", None
    ) is not None
    nested = getattr(root, "full_kv_pool", None)
    if has_direct_buffers and nested is not None:
        raise RuntimeError("SGLang Full pool routing is ambiguous")
    return (nested if nested is not None else root,)


def _build_structured_arena(
    class_config: Any, identity: Any, pool: Any, config: Any
) -> SglangStructuredArena:
    class_id = _integer("class id", class_config.class_id)
    page_tokens = _integer("page_tokens", config.page_tokens, positive=True)
    layers = _layer_ids(class_config)
    per_layer_bytes = _integer(
        "bytes_per_token_per_layer",
        class_config.bytes_per_token_per_layer,
        positive=True,
    )
    token_bytes = per_layer_bytes * len(layers)
    registration = _neutral_registration(
        identity, class_config, page_tokens, token_bytes
    )

    if pool is None:
        raise RuntimeError(f"SGLang class {class_id} physical pool is missing")
    if getattr(pool, "kv_cache_layout", None) != "nhd" or bool(
        getattr(pool, "use_hnd", False)
    ):
        raise RuntimeError(f"SGLang class {class_id} KV pool is not NHD")
    if bool(getattr(pool, "use_mla", False)) or bool(
        getattr(pool, "dsa_kv_cache_store_fp8", False)
    ):
        raise RuntimeError(f"SGLang class {class_id} KV pool is not plain MHA")
    quantized = getattr(pool, "is_quantized_kv_cache", False)
    quantized = quantized() if callable(quantized) else quantized
    quant_name = getattr(getattr(pool, "quant_method", None), "name", None)
    if bool(quantized) or quant_name not in (None, "unquantized"):
        raise RuntimeError(f"SGLang class {class_id} quantized KV is unsupported")
    if any(
        getattr(pool, name, None) is not None
        for name in ("k_scale_buffer", "v_scale_buffer")
    ):
        raise RuntimeError(f"SGLang class {class_id} FP8 scale storage is unsupported")

    pool_page_tokens = _integer(
        "SGLang pool page_size", getattr(pool, "page_size", None), positive=True
    )
    managed_tokens = _integer(
        "SGLang pool size", getattr(pool, "size", None), positive=True
    )
    if pool_page_tokens != page_tokens:
        raise RuntimeError(f"SGLang class {class_id} page size changed")
    if managed_tokens != registration.page_count * page_tokens:
        raise RuntimeError(f"SGLang class {class_id} capacity differs from its arena")
    layer_num = getattr(pool, "layer_num", len(layers))
    if _integer("SGLang pool layer_num", layer_num) != len(layers):
        raise RuntimeError(f"SGLang class {class_id} layer count changed")

    dtype = getattr(pool, "dtype", _MISSING)
    if dtype is _MISSING:
        raise RuntimeError(f"SGLang class {class_id} KV dtype is missing")
    store_dtype = getattr(pool, "store_dtype", dtype)
    if store_dtype != dtype:
        raise RuntimeError(f"SGLang class {class_id} uses packed or quantized KV storage")
    dtype_name = str(store_dtype).lower()
    if "float8" in dtype_name or "fp8" in dtype_name:
        raise RuntimeError(f"SGLang class {class_id} FP8 KV storage is unsupported")

    key_buffers = _buffer_sequence(pool, "k_buffer", len(layers), class_id)
    value_buffers = _buffer_sequence(pool, "v_buffer", len(layers), class_id)
    component_widths = _configured_component_widths(class_config)
    expected_rows = managed_tokens + page_tokens
    components: list[SglangArenaComponent] = []
    byte_offset = 0
    pointers: set[int] = set()
    canonical_device: tuple[str, int | None] | None = None
    observed_widths: tuple[int, int] | None = None
    for local_layer, global_layer in enumerate(layers):
        layer_widths = []
        for name, tensor in (
            ("key", key_buffers[local_layer]),
            ("value", value_buffers[local_layer]),
        ):
            width, device, pointer = _validate_component_tensor(
                tensor,
                name=f"class {class_id} layer {global_layer} {name}",
                rows=expected_rows,
                dtype=dtype,
                head_count=getattr(pool, "head_num", None),
                head_width=getattr(
                    pool, "head_dim" if name == "key" else "v_head_dim", None
                ),
            )
            if width == 1 or "float8" in str(getattr(tensor, "dtype", "")).lower():
                raise RuntimeError(
                    f"SGLang class {class_id} FP8 KV storage is unsupported"
                )
            if canonical_device is None:
                canonical_device = device
            elif not _same_device(canonical_device, device):
                raise RuntimeError("structured arena components span CUDA devices")
            if pointer is not None:
                if pointer in pointers:
                    raise RuntimeError("structured arena component buffers alias")
                pointers.add(pointer)
            layer_widths.append(width)
            components.append(
                SglangArenaComponent(
                    global_layer_id=global_layer,
                    local_layer_id=local_layer,
                    name=name,
                    tensor=tensor,
                    token_byte_offset=byte_offset,
                    token_byte_count=width,
                )
            )
            byte_offset += width
        pair = tuple(layer_widths)
        if observed_widths is None:
            observed_widths = pair  # type: ignore[assignment]
        elif pair != observed_widths:
            raise RuntimeError("SGLang per-layer KV component geometry changed")

    assert observed_widths is not None
    if sum(observed_widths) != per_layer_bytes:
        raise RuntimeError("SGLang aggregate KV width differs from the class plan")
    if component_widths is not None and observed_widths != component_widths:
        raise RuntimeError("SGLang K/V widths differ from class components")
    if byte_offset != token_bytes:
        raise RuntimeError("SGLang class token byte geometry changed")
    return SglangStructuredArena(
        registration=registration,
        retention=str(class_config.retention),
        storage=str(class_config.storage),
        storage_token_bias=page_tokens,
        components=tuple(components),
    )


def _layer_ids(class_config: Any) -> tuple[int, ...]:
    try:
        raw = tuple(class_config.layers)
    except Exception as error:
        raise RuntimeError("compiled KV layer ids are unreadable") from error
    layers = tuple(_integer("layer id", value) for value in raw)
    if not layers or layers != tuple(sorted(set(layers))):
        raise RuntimeError("compiled KV layer ids must be nonempty, unique, and ascending")
    return layers


def _neutral_registration(
    identity: Any, class_config: Any, page_tokens: int, token_bytes: int
) -> ArenaRegistration:
    class_id = _integer("class id", class_config.class_id)
    pairs = (
        ("class id", "class_id"),
        ("pool id", "pool_id"),
        ("backend domain", "backend_domain"),
    )
    for label, field in pairs:
        if _integer(f"arena {label}", getattr(identity, field, None)) != _integer(
            f"compiled {label}", getattr(class_config, field, None)
        ):
            raise RuntimeError(f"runtime arena {label} differs from its class")
    if _integer(
        "arena page_tokens", getattr(identity, "page_tokens", None), positive=True
    ) != page_tokens:
        raise RuntimeError("runtime arena page size differs from the plan")
    try:
        return ArenaRegistration(
            engine_epoch=_integer(
                "arena engine_epoch", identity.engine_epoch, positive=True
            ),
            pool_epoch=_integer(
                "arena pool_epoch", identity.pool_epoch, positive=True
            ),
            pool_id=_integer("arena pool_id", identity.pool_id, positive=True),
            class_id=class_id,
            backend_domain=_integer(
                "arena backend_domain", identity.backend_domain
            ),
            page_count=_integer(
                "arena page_count", identity.page_count, positive=True
            ),
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            backend_base_index=_integer(
                "arena backend_base_index", identity.backend_base_index
            ),
            first_page_id=_integer(
                "arena first_page_id", identity.first_page_id, positive=True
            ),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("runtime arena identity is invalid") from error


def _buffer_sequence(pool: Any, name: str, count: int, class_id: int) -> tuple[Any, ...]:
    value = getattr(pool, name, None)
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise RuntimeError(
            f"SGLang class {class_id} {name} does not match compiled layers"
        )
    if any(item is None for item in value):
        raise RuntimeError(f"SGLang class {class_id} {name} contains a missing tensor")
    return tuple(value)


def _configured_component_widths(class_config: Any) -> tuple[int, int] | None:
    try:
        components = tuple(class_config.components)
    except Exception as error:
        raise RuntimeError("compiled KV components are unreadable") from error
    if not components:
        return None
    if len(components) != 2:
        raise RuntimeError("token_kv class must describe exactly key and value")
    try:
        names = tuple(item[0] for item in components)
        widths = tuple(
            _integer("component byte width", item[1], positive=True)
            for item in components
        )
    except Exception as error:
        raise RuntimeError("compiled KV components are invalid") from error
    if names != _COMPONENT_NAMES:
        raise RuntimeError("token_kv class components must be ordered key then value")
    return widths  # type: ignore[return-value]


def _validate_component_tensor(
    tensor: Any,
    *,
    name: str,
    rows: int,
    dtype: Any,
    head_count: Any,
    head_width: Any,
) -> tuple[int, tuple[str, int | None], int | None]:
    try:
        shape = tuple(_integer(f"{name} shape", value, positive=True) for value in tensor.shape)
    except Exception as error:
        raise RuntimeError(f"SGLang {name} tensor shape is unreadable") from error
    if len(shape) != 3 or shape[0] != rows:
        raise RuntimeError(f"SGLang {name} tensor is not exact NHD storage")
    if head_count is not None and shape[1] != _integer(
        f"{name} head count", head_count, positive=True
    ):
        raise RuntimeError(f"SGLang {name} tensor head count changed")
    if head_width is not None and shape[2] != _integer(
        f"{name} head width", head_width, positive=True
    ):
        raise RuntimeError(f"SGLang {name} tensor head width changed")
    if getattr(tensor, "dtype", _MISSING) != dtype:
        raise RuntimeError(f"SGLang {name} tensor dtype changed")
    contiguous = getattr(tensor, "is_contiguous", None)
    if not callable(contiguous) or contiguous() is not True:
        raise RuntimeError(f"SGLang {name} tensor must be contiguous")
    if getattr(tensor, "requires_grad", False) is not False:
        raise RuntimeError(f"SGLang {name} tensor must not require gradients")
    element_size = getattr(tensor, "element_size", None)
    if not callable(element_size):
        raise RuntimeError(f"SGLang {name} tensor element size is missing")
    item_bytes = _integer(
        f"{name} tensor element size", element_size(), positive=True
    )
    token_bytes = shape[1] * shape[2] * item_bytes
    numel = getattr(tensor, "numel", None)
    if callable(numel) and _integer(f"{name} tensor elements", numel()) != (
        shape[0] * shape[1] * shape[2]
    ):
        raise RuntimeError(f"SGLang {name} tensor element count changed")
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None and _integer(f"{name} tensor bytes", nbytes) != (
        shape[0] * token_bytes
    ):
        raise RuntimeError(f"SGLang {name} tensor byte span changed")
    device = _device(getattr(tensor, "device", None), name)
    pointer_method = getattr(tensor, "data_ptr", None)
    pointer = None
    if callable(pointer_method):
        pointer = _integer(f"{name} tensor pointer", pointer_method(), positive=True)
    return token_bytes, device, pointer


def _device(value: Any, name: str) -> tuple[str, int | None]:
    encoded = str(value).lower()
    if not encoded.startswith("cuda"):
        raise RuntimeError(f"SGLang {name} tensor is not CUDA storage")
    pieces = encoded.split(":", 1)
    if len(pieces) == 1:
        return "cuda", None
    try:
        index = int(pieces[1])
    except ValueError as error:
        raise RuntimeError(f"SGLang {name} tensor has an invalid CUDA device") from error
    if index < 0:
        raise RuntimeError(f"SGLang {name} tensor has an invalid CUDA device")
    return "cuda", index


def _same_device(
    left: tuple[str, int | None], right: tuple[str, int | None]
) -> bool:
    return left[0] == right[0] and (
        left[1] is None or right[1] is None or left[1] == right[1]
    )


def build_external_append_manifest(
    batch_record: Any,
    plans: Sequence[Any],
    locations_by_class: Mapping[int, Any],
    runtime: Any,
    config: Any,
    arenas: Sequence[SglangStructuredArena] | Mapping[int, SglangStructuredArena],
) -> ExternalAppendManifest:
    """Map one lowered SGLang batch to canonical engine-neutral writes.

    The function does not issue tickets, record events, or mutate any supplied
    object.  It validates that every live SGLang slot is exactly the dummy-biased
    representation of the manager-issued neutral destination.
    """

    classes = _config_classes(config)
    arena_values = _manifest_arenas(arenas, classes)
    arenas_by_class = {item.class_id: item for item in arena_values}
    _validate_manifest_runtime_arenas(runtime, classes, arena_values)

    try:
        records = tuple(batch_record.records)
        keys = tuple(batch_record.keys)
        plan_values = tuple(plans)
    except Exception as error:
        raise RuntimeError("append batch records or plans are unreadable") from error
    if (
        not records
        or len(keys) != len(records)
        or len(plan_values) != len(records)
        or len(set(keys)) != len(keys)
    ):
        raise RuntimeError("append batch record and plan cardinalities differ")
    if any(getattr(record, "key", _MISSING) != key for key, record in zip(
        keys, records, strict=True
    )):
        raise RuntimeError("append batch keys differ from record order")

    total_appended = 0
    for plan in plan_values:
        previous = _integer("plan previous boundary", plan.previous_boundary)
        target = _integer("plan target boundary", plan.target_boundary)
        if target <= previous:
            raise RuntimeError("append plan must advance its logical boundary")
        total_appended += target - previous

    class_ids = tuple(int(item.class_id) for item in classes)
    if not isinstance(locations_by_class, Mapping) or set(locations_by_class) != set(
        class_ids
    ):
        raise RuntimeError("SGLang locations differ from compiled KV classes")
    expected_locations = tuple(
        (
            class_id,
            _location_vector(
                f"class {class_id} locations",
                locations_by_class[class_id],
                total_appended,
            ),
        )
        for class_id in class_ids
    )
    locations = dict(expected_locations)

    writes: list[ExternalTokenWrite] = []
    last_use: dict[BackendPageAddress, None] = {}
    class_cursors = {class_id: 0 for class_id in class_ids}
    for key, record, plan in zip(keys, records, plan_values, strict=True):
        prepared = getattr(record, "prepared", None)
        if prepared is None:
            raise RuntimeError("append batch record has no prepared step")
        if not _lease_equal(plan.request, prepared.request):
            raise RuntimeError("append plan request differs from its prepared step")
        if (
            _integer("prepared previous boundary", prepared.previous_boundary)
            != int(plan.previous_boundary)
            or _integer("prepared target boundary", prepared.target_boundary)
            != int(plan.target_boundary)
        ):
            raise RuntimeError("append plan boundaries differ from its prepared step")
        runtime_record = runtime.record_for(key)
        if getattr(runtime_record, "pending", None) is not record:
            raise RuntimeError("runtime pending record differs from the append batch")
        cursor = getattr(runtime_record, "cursor", None)
        if cursor is None or not _lease_equal(cursor.lease, plan.request):
            raise RuntimeError("runtime cursor differs from the append plan request")

        request = _public_request_lease(plan.request)
        step = _public_step_lease(prepared.step)
        context = OperationContext(request, step, DataPlaneOperation.APPEND)
        append_count = int(plan.target_boundary) - int(plan.previous_boundary)
        specs = _class_specs(plan, classes)
        current = _cursor_pages(cursor, plan.request, arenas_by_class)
        pending = _pending_pages(record, plan.request, arenas_by_class)
        for shadow in current.values():
            last_use.setdefault(_page_address(shadow, arenas_by_class), None)
        for shadow in pending.values():
            last_use.setdefault(_page_address(shadow, arenas_by_class), None)

        destinations = dict(current)
        for logical_key, shadow in pending.items():
            if logical_key in destinations:
                spec = specs[logical_key[0]]
                destination = getattr(getattr(spec, "tail_action", None), "destination", None)
                if destination is None or not _lease_equal(destination, shadow.page):
                    raise RuntimeError(
                        "pending page replaced a resident page without an exact tail action"
                    )
            destinations[logical_key] = shadow

        for delta in range(append_count):
            token_id = int(plan.previous_boundary) + delta
            for class_config in classes:
                class_id = int(class_config.class_id)
                spec = specs[class_id]
                physical_previous = _integer(
                    "class previous layout boundary",
                    spec.previous_layout_boundary,
                )
                physical_target = _integer(
                    "class target layout boundary", spec.target_layout_boundary
                )
                if physical_target - physical_previous != append_count:
                    raise RuntimeError(
                        "class physical append span differs from logical append span"
                    )
                physical_token = physical_previous + delta
                page_tokens = arena_values[0].page_tokens
                logical_ordinal, token_offset = divmod(physical_token, page_tokens)
                shadow = destinations.get((class_id, logical_ordinal))
                if shadow is None:
                    raise RuntimeError(
                        "append destination is absent from resident and pending pages"
                    )
                arena = arenas_by_class[class_id]
                address = _token_address(shadow, arena, token_offset)
                expected_slot = (
                    (address.backend_index - arena.registration.backend_base_index)
                    * page_tokens
                    + token_offset
                    + arena.storage_token_bias
                )
                vector_index = class_cursors[class_id] + delta
                if locations[class_id][vector_index] != expected_slot:
                    raise RuntimeError(
                        f"SGLang class {class_id} slot vector differs from manager mapping"
                    )
                writes.append(
                    ExternalTokenWrite(
                        context=context,
                        token_id=token_id,
                        destination=address,
                        byte_count=arena.token_bytes,
                    )
                )
        for class_id in class_ids:
            class_cursors[class_id] += append_count

    if any(class_cursors[class_id] != len(locations[class_id]) for class_id in class_ids):
        raise RuntimeError("SGLang location vector has unused entries")
    ordered_last_use = tuple(
        sorted(
            last_use,
            key=lambda item: (
                class_ids.index(item.class_id),
                item.backend_index,
                item.page.pool_id,
                item.page.page_id,
                item.page.generation,
            ),
        )
    )
    return ExternalAppendManifest(
        writes=tuple(writes),
        last_use_pages=ordered_last_use,
        expected_locations_by_class=expected_locations,
        arenas=arena_values,
    )


def _manifest_arenas(
    arenas: Sequence[SglangStructuredArena] | Mapping[int, SglangStructuredArena],
    classes: Sequence[Any],
) -> tuple[SglangStructuredArena, ...]:
    class_ids = tuple(int(item.class_id) for item in classes)
    if isinstance(arenas, Mapping):
        if set(arenas) != set(class_ids):
            raise RuntimeError("structured arenas differ from compiled classes")
        values = tuple(arenas[class_id] for class_id in class_ids)
    else:
        try:
            values = tuple(arenas)
        except Exception as error:
            raise RuntimeError("structured arenas are unreadable") from error
    if (
        len(values) != len(classes)
        or any(not isinstance(item, SglangStructuredArena) for item in values)
        or tuple(item.class_id for item in values) != class_ids
    ):
        raise RuntimeError("structured arenas are not in compiled class order")
    for class_config, arena in zip(classes, values, strict=True):
        if (
            arena.storage != "token_kv"
            or arena.retention != class_config.retention
            or arena.registration.token_bytes
            != int(class_config.bytes_per_token_per_layer) * len(class_config.layers)
        ):
            raise RuntimeError("structured arena differs from its compiled class")
    return values


def _validate_manifest_runtime_arenas(
    runtime: Any, classes: Sequence[Any], arenas: Sequence[SglangStructuredArena]
) -> None:
    identities = _runtime_arenas(runtime, classes)
    for identity, arena in zip(identities, arenas, strict=True):
        registration = arena.registration
        fields = (
            "engine_epoch",
            "pool_epoch",
            "pool_id",
            "class_id",
            "backend_domain",
            "page_count",
            "page_tokens",
            "backend_base_index",
            "first_page_id",
        )
        if any(
            _integer(f"runtime arena {name}", getattr(identity, name, None))
            != getattr(registration, name)
            for name in fields
        ):
            raise RuntimeError("structured arena identity became stale")


def _location_vector(name: str, value: Any, expected: int) -> tuple[int, ...]:
    try:
        ndim = getattr(value, "ndim", None)
        if ndim is not None and _integer(f"{name} rank", ndim) != 1:
            raise RuntimeError(f"{name} must be one-dimensional")
        current = value.detach() if callable(getattr(value, "detach", None)) else value
        current = current.cpu() if callable(getattr(current, "cpu", None)) else current
        raw = current.tolist() if callable(getattr(current, "tolist", None)) else list(current)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"{name} is unreadable") from error
    if not isinstance(raw, (list, tuple)) or len(raw) != expected:
        raise RuntimeError(f"{name} cardinality differs from appended tokens")
    return tuple(_integer(name, item) for item in raw)


def _class_specs(plan: Any, classes: Sequence[Any]) -> dict[int, Any]:
    try:
        specs = tuple(plan.class_specs)
    except Exception as error:
        raise RuntimeError("append plan class specs are unreadable") from error
    expected_ids = tuple(int(item.class_id) for item in classes)
    if len(specs) != len(classes) or tuple(
        _integer("plan class id", item.class_id) for item in specs
    ) != expected_ids:
        raise RuntimeError("append plan class specs are not in compiled order")
    result = {int(item.class_id): item for item in specs}
    for class_config, spec in zip(classes, specs, strict=True):
        if _integer("plan pool id", spec.pool_id, positive=True) != _integer(
            "compiled pool id", class_config.pool_id, positive=True
        ):
            raise RuntimeError("append plan class pool identity changed")
    return result


def _cursor_pages(
    cursor: Any, request: Any, arenas: Mapping[int, SglangStructuredArena]
) -> dict[tuple[int, int], Any]:
    pages = getattr(cursor, "pages", None)
    if not isinstance(pages, Mapping):
        raise RuntimeError("runtime cursor pages are unreadable")
    result: dict[tuple[int, int], Any] = {}
    for key, shadow in pages.items():
        logical_key = _shadow_key(shadow, request, arenas)
        if key != logical_key or logical_key in result:
            raise RuntimeError("runtime cursor page index is not canonical")
        result[logical_key] = shadow
    return result


def _pending_pages(
    record: Any, request: Any, arenas: Mapping[int, SglangStructuredArena]
) -> dict[tuple[int, int], Any]:
    try:
        pages = tuple(record.new_pages)
    except Exception as error:
        raise RuntimeError("pending append pages are unreadable") from error
    result: dict[tuple[int, int], Any] = {}
    for shadow in pages:
        key = _shadow_key(shadow, request, arenas)
        if key in result:
            raise RuntimeError("pending append aliases a logical page")
        result[key] = shadow
    return result


def _shadow_key(
    shadow: Any, request: Any, arenas: Mapping[int, SglangStructuredArena]
) -> tuple[int, int]:
    if not _lease_equal(getattr(shadow, "request", None), request):
        raise RuntimeError("page shadow belongs to another request")
    class_id = _integer("page shadow class id", getattr(shadow, "class_id", None))
    logical_ordinal = _integer(
        "page shadow logical ordinal", getattr(shadow, "logical_ordinal", None)
    )
    arena = arenas.get(class_id)
    if arena is None:
        raise RuntimeError("page shadow names an unknown class")
    _page_address(shadow, arenas)
    return class_id, logical_ordinal


def _page_address(
    shadow: Any, arenas: Mapping[int, SglangStructuredArena]
) -> BackendPageAddress:
    class_id = int(shadow.class_id)
    arena = arenas[class_id].registration
    page = _public_page_lease(shadow.page)
    backend_index = _integer(
        "page shadow backend index", shadow.backend_index
    )
    expected_index = arena.backend_base_index + page.page_id - arena.first_page_id
    if (
        page.engine_epoch != arena.engine_epoch
        or page.pool_epoch != arena.pool_epoch
        or page.pool_id != arena.pool_id
        or not arena.first_page_id
        <= page.page_id
        < arena.first_page_id + arena.page_count
        or backend_index != expected_index
    ):
        raise RuntimeError("page shadow differs from its structured arena")
    return BackendPageAddress(page, class_id, arena.backend_domain, backend_index)


def _token_address(
    shadow: Any, arena: SglangStructuredArena, token_offset: int
) -> BackendTokenAddress:
    page = _page_address(shadow, {arena.class_id: arena})
    if token_offset >= arena.page_tokens:
        raise RuntimeError("append token offset exceeds its arena page")
    return BackendTokenAddress(
        page=page.page,
        class_id=page.class_id,
        backend_domain=page.backend_domain,
        backend_index=page.backend_index,
        token_offset=token_offset,
    )


def _lease_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return all(
        getattr(left, name, _MISSING) == getattr(right, name, _MISSING)
        and getattr(left, name, _MISSING) is not _MISSING
        for name in ("engine_epoch", "slot", "generation")
    )


def _public_request_lease(value: Any) -> RequestLease:
    try:
        return RequestLease(
            _integer("request engine_epoch", value.engine_epoch, positive=True),
            _integer("request slot", value.slot),
            _integer("request generation", value.generation, positive=True),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("append request lease is invalid") from error


def _public_step_lease(value: Any) -> StepLease:
    try:
        return StepLease(
            _integer("step engine_epoch", value.engine_epoch, positive=True),
            _integer("step slot", value.slot),
            _integer("step generation", value.generation, positive=True),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("append step lease is invalid") from error


def _public_page_lease(value: Any) -> PageLease:
    try:
        return PageLease(
            _integer("page engine_epoch", value.engine_epoch, positive=True),
            _integer("page pool_epoch", value.pool_epoch, positive=True),
            _integer("page generation", value.generation, positive=True),
            _integer("page id", value.page_id, positive=True),
            _integer("page pool id", value.pool_id, positive=True),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("append page lease is invalid") from error


__all__ = [
    "ExternalAppendManifest",
    "SglangArenaComponent",
    "SglangStructuredArena",
    "build_external_append_manifest",
    "build_sglang_structured_arenas",
    "build_structured_arenas",
]
