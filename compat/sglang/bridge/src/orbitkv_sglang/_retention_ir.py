from __future__ import annotations

import hashlib
import struct
from typing import Any, Mapping


RETENTION_PROGRAM_SCHEMA = "orbitkv.retention-ir.v1"
LAYOUT_PROGRAM_SCHEMA = "orbitkv.layout-program.v1"
RETENTION_CAPABILITIES = frozenset(
    {
        "append_only_addressing",
        "block_domain_partitioning",
        "kv_head_partitioning",
        "periodic_addressing",
        "periodic_from_addressing",
        "pinned_addressing",
        "resettable_arena_addressing",
        "semantic_retirement",
        "token_manager",
    }
)

_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1


def validate_retention_program(raw: Any) -> dict[str, Any]:
    path = "RuntimeManifest.token_manager_plan.retention_program"
    program = _mapping(raw, path)
    _exact_keys(program, path, {"schema", "page_tokens", "states"})
    if program["schema"] != RETENTION_PROGRAM_SCHEMA:
        raise ValueError(f"{path}.schema must be {RETENTION_PROGRAM_SCHEMA!r}")
    _positive_integer(program["page_tokens"], f"{path}.page_tokens", _U64_MAX)
    states = program["states"]
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path}.states must be a non-empty list")

    names: set[str] = set()
    claims: dict[int, list[tuple[str, tuple[int, int] | None]]] = {}
    for index, raw_state in enumerate(states):
        state_path = f"{path}.states[{index}]"
        state = _mapping(raw_state, state_path)
        _exact_keys(
            state,
            state_path,
            {"name", "layers", "bytes_per_token_per_layer", "may_read"},
            {"kv_head_range"},
        )
        name = _nonempty_string(state["name"], f"{state_path}.name")
        if name in names:
            raise ValueError(f"{path} state names must be unique")
        names.add(name)
        layers = _layers(state["layers"], f"{state_path}.layers")
        head_range = (
            _kv_head_range(state["kv_head_range"], f"{state_path}.kv_head_range")
            if "kv_head_range" in state
            else None
        )
        _positive_integer(
            state["bytes_per_token_per_layer"],
            f"{state_path}.bytes_per_token_per_layer",
            _U64_MAX,
        )
        _predicate(state["may_read"], f"{state_path}.may_read")
        for layer in layers:
            previous = claims.setdefault(layer, [])
            for previous_name, previous_range in previous:
                if _head_ranges_overlap(previous_range, head_range):
                    qualifier = "KV head ranges" if head_range is not None else "layers"
                    raise ValueError(
                        f"{path} {qualifier} overlap between {previous_name!r} and {name!r}"
                    )
            previous.append((name, head_range))
    return program


def _integer_expression(raw: Any, path: str) -> None:
    expression = _mapping(raw, path)
    op = _nonempty_string(expression.get("op"), f"{path}.op")
    fields = {
        "query_position": {"op"},
        "key_position": {"op"},
        "constant": {"op", "value"},
        "add": {"op", "lhs", "rhs"},
        "sub": {"op", "lhs", "rhs"},
        "floor_div": {"op", "value", "divisor"},
        "mod": {"op", "value", "modulus"},
    }
    if op not in fields:
        raise ValueError(f"{path}.op is unsupported")
    _exact_keys(expression, path, fields[op])
    if op in ("query_position", "key_position"):
        return
    if op == "constant":
        _integer(expression["value"], f"{path}.value", _I64_MIN, _I64_MAX)
        return
    if op in ("add", "sub"):
        _integer_expression(expression["lhs"], f"{path}.lhs")
        _integer_expression(expression["rhs"], f"{path}.rhs")
        return
    operand = "divisor" if op == "floor_div" else "modulus"
    _integer_expression(expression["value"], f"{path}.value")
    _positive_integer(expression[operand], f"{path}.{operand}", _I64_MAX)


def _predicate(raw: Any, path: str) -> None:
    predicate = _mapping(raw, path)
    op = _nonempty_string(predicate.get("op"), f"{path}.op")
    fields = {
        "true": {"op"},
        "false": {"op"},
        "less_than": {"op", "lhs", "rhs"},
        "less_equal": {"op", "lhs", "rhs"},
        "equal": {"op", "lhs", "rhs"},
        "and": {"op", "terms"},
        "or": {"op", "terms"},
    }
    if op not in fields:
        raise ValueError(f"{path}.op is unsupported")
    _exact_keys(predicate, path, fields[op])
    if op in ("true", "false"):
        return
    if op in ("less_than", "less_equal", "equal"):
        _integer_expression(predicate["lhs"], f"{path}.lhs")
        _integer_expression(predicate["rhs"], f"{path}.rhs")
        return
    terms = predicate["terms"]
    if not isinstance(terms, list):
        raise ValueError(f"{path}.terms must be a list")
    for index, term in enumerate(terms):
        _predicate(term, f"{path}.terms[{index}]")


def validate_layout_program(
    raw: Any, retention: Mapping[str, Any]
) -> dict[str, Any]:
    path = "RuntimeManifest.token_manager_plan.layout"
    layout = _mapping(raw, path)
    _exact_keys(
        layout, path, {"schema", "plan_fingerprint", "page_tokens", "classes"}
    )
    if layout["schema"] != LAYOUT_PROGRAM_SCHEMA:
        raise ValueError(f"{path}.schema must be {LAYOUT_PROGRAM_SCHEMA!r}")
    _sha256_fingerprint(layout["plan_fingerprint"], f"{path}.plan_fingerprint")
    page_tokens = _positive_integer(
        layout["page_tokens"], f"{path}.page_tokens", _U64_MAX
    )
    if page_tokens != retention["page_tokens"]:
        raise ValueError(
            "RuntimeManifest token manager retention and layout use different page sizes"
        )
    classes = layout["classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{path}.classes must be a non-empty list")
    names: set[str] = set()
    for index, raw_class in enumerate(classes):
        class_path = f"{path}.classes[{index}]"
        item = _mapping(raw_class, class_path)
        _exact_keys(
            item,
            class_path,
            {
                "name",
                "layers",
                "bytes_per_token_per_layer",
                "address",
                "retirement",
                "minimum_slots_per_request",
            },
            {"kv_head_range", "block_domain"},
        )
        name = _nonempty_string(item["name"], f"{class_path}.name")
        if name in names:
            raise ValueError(f"{path} class names must be unique")
        names.add(name)
        _layers(item["layers"], f"{class_path}.layers")
        if "kv_head_range" in item:
            _kv_head_range(item["kv_head_range"], f"{class_path}.kv_head_range")
        _positive_integer(
            item["bytes_per_token_per_layer"],
            f"{class_path}.bytes_per_token_per_layer",
            _U64_MAX,
        )
        address_kind = _address(item["address"], f"{class_path}.address")
        retirement_kind = _retirement(
            item["retirement"], f"{class_path}.retirement"
        )
        slots = item["minimum_slots_per_request"]
        if slots is not None:
            _positive_integer(
                slots, f"{class_path}.minimum_slots_per_request", _U64_MAX
            )
        domain = None
        if "block_domain" in item:
            domain = _block_domain(item["block_domain"], f"{class_path}.block_domain")
        _validate_layout_opcode_pair(
            item, class_path, page_tokens, address_kind, retirement_kind, slots, domain
        )

    expected_plan_fingerprint = layout_plan_fingerprint(layout)
    if layout["plan_fingerprint"] != expected_plan_fingerprint:
        raise ValueError(
            f"{path}.plan_fingerprint does not match its compiled classes"
        )
    _validate_layout_projection(retention, layout)
    return layout


def _address(raw: Any, path: str) -> str:
    address = _mapping(raw, path)
    kind = _nonempty_string(address.get("kind"), f"{path}.kind")
    fields = {
        "append_only": {"kind"},
        "pinned": {"kind"},
        "periodic": {"kind", "period_blocks"},
        "periodic_from": {"kind", "period_blocks", "origin_block"},
        "resettable_arena": {"kind", "blocks_per_epoch"},
    }
    if kind not in fields:
        raise ValueError(f"{path}.kind is unsupported")
    _exact_keys(address, path, fields[kind])
    if kind in ("periodic", "periodic_from"):
        _positive_integer(address["period_blocks"], f"{path}.period_blocks", _U64_MAX)
    if kind == "periodic_from":
        _integer(address["origin_block"], f"{path}.origin_block", 0, _U64_MAX)
    if kind == "resettable_arena":
        _positive_integer(
            address["blocks_per_epoch"], f"{path}.blocks_per_epoch", _U64_MAX
        )
    return kind


def _retirement(raw: Any, path: str) -> str:
    retirement = _mapping(raw, path)
    kind = _nonempty_string(retirement.get("kind"), f"{path}.kind")
    fields = {
        "never": {"kind"},
        "block_end_plus": {"kind", "offset_tokens"},
        "epoch_end": {"kind", "blocks_per_epoch"},
    }
    if kind not in fields:
        raise ValueError(f"{path}.kind is unsupported")
    _exact_keys(retirement, path, fields[kind])
    if kind == "block_end_plus":
        _integer(
            retirement["offset_tokens"], f"{path}.offset_tokens", 0, _U64_MAX
        )
    if kind == "epoch_end":
        _positive_integer(
            retirement["blocks_per_epoch"],
            f"{path}.blocks_per_epoch",
            _U64_MAX,
        )
    return kind


def _block_domain(raw: Any, path: str) -> tuple[int, int | None]:
    domain = _mapping(raw, path)
    _exact_keys(domain, path, {"start_block"}, {"end_block_exclusive"})
    start = _integer(domain["start_block"], f"{path}.start_block", 0, _U64_MAX)
    end = None
    if "end_block_exclusive" in domain:
        end = _integer(
            domain["end_block_exclusive"],
            f"{path}.end_block_exclusive",
            0,
            _U64_MAX,
        )
        if start >= end:
            raise ValueError(f"{path} must be non-empty")
    if start == 0 and end is None:
        raise ValueError(f"{path} must be omitted for the whole block domain")
    return start, end


def _validate_layout_opcode_pair(
    item: Mapping[str, Any],
    path: str,
    page_tokens: int,
    address_kind: str,
    retirement_kind: str,
    slots: int | None,
    domain: tuple[int, int | None] | None,
) -> None:
    address = item["address"]
    retirement = item["retirement"]
    if address_kind == "append_only":
        if retirement_kind != "never" or slots is not None or domain is not None:
            raise ValueError(f"{path} has inconsistent append_only layout geometry")
        return
    if address_kind == "pinned":
        if retirement_kind != "never" or domain is None or domain[1] is None:
            raise ValueError(f"{path} has inconsistent pinned layout geometry")
        if slots != domain[1] - domain[0]:
            raise ValueError(f"{path} pinned slot count differs from its block domain")
        return
    if address_kind in ("periodic", "periodic_from"):
        period = address["period_blocks"]
        if retirement_kind != "block_end_plus" or slots != period:
            raise ValueError(f"{path} has inconsistent periodic layout geometry")
        offset = retirement["offset_tokens"]
        if offset == _U64_MAX:
            raise ValueError(
                f"{path} periodic slot count differs from its retirement window"
            )
        expected_period = _checked_u64(
            1 + _ceil_div_u64(offset, page_tokens, "sliding slot count"),
            "sliding slot count",
        )
        if period != expected_period:
            raise ValueError(f"{path} periodic slot count differs from its retirement window")
        if address_kind == "periodic" and domain is not None:
            raise ValueError(f"{path} periodic address must use the whole block domain")
        if address_kind == "periodic_from":
            if domain is None or address["origin_block"] != domain[0]:
                raise ValueError(f"{path} periodic_from origin differs from its block domain")
        return
    blocks = address["blocks_per_epoch"]
    if (
        retirement_kind != "epoch_end"
        or retirement["blocks_per_epoch"] != blocks
        or slots != blocks
        or domain is not None
        or blocks > _U64_MAX // page_tokens
    ):
        raise ValueError(f"{path} has inconsistent resettable_arena layout geometry")


def _validate_layout_projection(
    retention: Mapping[str, Any], layout: Mapping[str, Any]
) -> None:
    classes = layout["classes"]
    class_index = 0
    for state in retention["states"]:
        if class_index >= len(classes):
            raise ValueError(
                "RuntimeManifest token manager layout omits a retention state"
            )
        name = state["name"]
        first = classes[class_index]
        if first["name"] == name:
            projected = (first,)
            class_index += 1
        elif (
            class_index + 1 < len(classes)
            and first["name"] == f"{name}::sink"
            and classes[class_index + 1]["name"] == f"{name}::local"
        ):
            projected = (first, classes[class_index + 1])
            class_index += 2
        else:
            raise ValueError(
                "RuntimeManifest token manager layout class order differs from its retention program"
            )
        expected_head = state.get("kv_head_range")
        for item in projected:
            if (
                item["layers"] != state["layers"]
                or item.get("kv_head_range") != expected_head
                or item["bytes_per_token_per_layer"]
                != state["bytes_per_token_per_layer"]
            ):
                raise ValueError(
                    "RuntimeManifest token manager layout differs from its retention projection"
                )
        _validate_state_lowering(state, projected, layout["page_tokens"])
    if class_index != len(classes):
        raise ValueError(
            "RuntimeManifest token manager layout has classes without a retention state"
        )


def _validate_state_lowering(
    state: Mapping[str, Any],
    classes: tuple[Mapping[str, Any], ...],
    page_tokens: int,
) -> None:
    """Check the deterministic Retention IR subset emitted by the compiler."""

    inferred = _infer_retention(state["may_read"])
    if inferred[0] == "partitioned":
        if len(classes) != 2:
            raise ValueError(
                "RuntimeManifest token manager layout differs from its retention program"
            )
        sink_tokens, window_tokens = inferred[1], inferred[2]
        if sink_tokens % page_tokens != 0:
            raise ValueError(
                "RuntimeManifest retention sink boundary must be page aligned"
            )
        origin = sink_tokens // page_tokens
        offset = window_tokens - 1
        period = _checked_u64(
            1 + _ceil_div_u64(offset, page_tokens, "sliding slot count"),
            "sliding slot count",
        )
        expected = (
            (
                {"kind": "pinned"},
                {"kind": "never"},
                origin,
                {"start_block": 0, "end_block_exclusive": origin},
            ),
            (
                {"kind": "periodic_from", "period_blocks": period, "origin_block": origin},
                {"kind": "block_end_plus", "offset_tokens": window_tokens - 1},
                period,
                {"start_block": origin},
            ),
        )
    elif inferred[0] == "chunked":
        chunk_tokens = inferred[1]
        if chunk_tokens % page_tokens != 0:
            raise ValueError(
                "RuntimeManifest retention chunk size must be page aligned"
            )
        blocks = chunk_tokens // page_tokens
        expected = ((
            {"kind": "resettable_arena", "blocks_per_epoch": blocks},
            {"kind": "epoch_end", "blocks_per_epoch": blocks},
            blocks,
            None,
        ),)
    elif inferred[0] == "sliding":
        window_tokens = inferred[1]
        offset = window_tokens - 1
        period = _checked_u64(
            1 + _ceil_div_u64(offset, page_tokens, "sliding slot count"),
            "sliding slot count",
        )
        expected = ((
            {"kind": "periodic", "period_blocks": period},
            {"kind": "block_end_plus", "offset_tokens": window_tokens - 1},
            period,
            None,
        ),)
    else:
        expected = (({"kind": "append_only"}, {"kind": "never"}, None, None),)

    if len(classes) != len(expected):
        raise ValueError(
            "RuntimeManifest token manager layout differs from its retention program"
        )
    for item, (address, retirement, slots, domain) in zip(
        classes, expected, strict=True
    ):
        if (
            item["address"] != address
            or item["retirement"] != retirement
            or item["minimum_slots_per_request"] != slots
            or item.get("block_domain") != domain
        ):
            raise ValueError(
                "RuntimeManifest token manager layout does not match its retention program"
            )


def _infer_retention(predicate: Mapping[str, Any]) -> tuple[Any, ...]:
    chunk = _same_chunk(predicate)
    if chunk is not None:
        return ("chunked", chunk)
    dilated = _dilated_window(predicate)
    if dilated is not None:
        return ("sliding", dilated)
    partition = _sink_and_window(predicate)
    if partition is not None:
        return ("partitioned", *partition)
    bounds = _predicate_bounds(predicate)
    if not bounds[2]:
        raise ValueError("RuntimeManifest retention state has no legal readers")
    upper = bounds[1]
    if upper is None:
        return ("full",)
    if upper < 0:
        raise ValueError("RuntimeManifest retention state has no legal readers")
    return ("sliding", _checked_u64(upper + 1, "retention window"))


def _same_chunk(predicate: Mapping[str, Any]) -> int | None:
    if predicate["op"] != "equal":
        return None
    for query_side, key_side in (
        (predicate["lhs"], predicate["rhs"]),
        (predicate["rhs"], predicate["lhs"]),
    ):
        query_divisor = _chunk_divisor(query_side, "query_position")
        key_divisor = _chunk_divisor(key_side, "key_position")
        if query_divisor is not None and query_divisor == key_divisor:
            return query_divisor
    return None


def _chunk_divisor(expression: Mapping[str, Any], operand: str) -> int | None:
    if expression["op"] != "floor_div":
        return None
    inner = expression["value"]
    if inner == {"op": operand}:
        return expression["divisor"]
    return None


def _dilated_window(predicate: Mapping[str, Any]) -> int | None:
    if predicate["op"] != "and":
        return None
    modulus = None
    for term in predicate["terms"]:
        candidate = _zero_delta_modulus(term)
        if candidate is not None:
            if modulus is not None:
                return None
            modulus = candidate
    if modulus is None:
        return None
    lower, upper, satisfiable = _predicate_bounds(predicate)
    if not satisfiable or upper is None or upper < 0:
        return None
    maximum_delta = upper - upper % modulus
    if lower is not None and maximum_delta < lower:
        return None
    return _checked_u64(maximum_delta + 1, "retention window")


def _zero_delta_modulus(predicate: Mapping[str, Any]) -> int | None:
    if predicate["op"] != "equal":
        return None
    lhs, rhs = predicate["lhs"], predicate["rhs"]
    if rhs == {"op": "constant", "value": 0}:
        return _delta_modulus(lhs)
    if lhs == {"op": "constant", "value": 0}:
        return _delta_modulus(rhs)
    return None


def _delta_modulus(expression: Mapping[str, Any]) -> int | None:
    if expression["op"] != "mod":
        return None
    affine = _affine(expression["value"] )
    if affine[:3] == (1, -1, 0) and not affine[3]:
        return expression["modulus"]
    return None


def _sink_and_window(predicate: Mapping[str, Any]) -> tuple[int, int] | None:
    if predicate["op"] != "or":
        return None
    sink = None
    window = None
    for term in predicate["terms"]:
        prefix = _key_prefix_tokens(term)
        if prefix is not None:
            sink = prefix if sink is None else max(sink, prefix)
            continue
        _lower, upper, _satisfiable = _predicate_bounds(term)
        if upper is None:
            return None
        if upper >= 0:
            candidate = _checked_u64(upper + 1, "retention window")
            window = candidate if window is None else max(window, candidate)
    return (sink, window) if sink is not None and window is not None else None


def _key_prefix_tokens(predicate: Mapping[str, Any]) -> int | None:
    if predicate["op"] not in ("less_than", "less_equal"):
        return None
    query, key, constant, non_affine = _affine_difference(
        predicate["lhs"], predicate["rhs"]
    )
    if non_affine or query != 0 or key != 1:
        return None
    upper = _checked_neg_i64(constant, "retention expression")
    if predicate["op"] == "less_than":
        upper = _checked_i64(upper - 1, "retention expression")
    if upper < 0:
        return None
    return _checked_u64(upper + 1, "retention sink")


def _predicate_bounds(predicate: Mapping[str, Any]) -> tuple[int | None, int | None, bool]:
    op = predicate["op"]
    if op == "true":
        return 0, None, True
    if op == "false":
        return None, None, False
    if op in ("less_than", "less_equal"):
        return _comparison_bounds(
            predicate["lhs"], predicate["rhs"], op == "less_than"
        )
    if op == "equal":
        return _intersect_bounds(
            _comparison_bounds(predicate["lhs"], predicate["rhs"], False),
            _comparison_bounds(predicate["rhs"], predicate["lhs"], False),
        )
    result = (0, None, True) if op == "and" else (None, None, False)
    combine = _intersect_bounds if op == "and" else _union_bounds
    for term in predicate["terms"]:
        result = combine(result, _predicate_bounds(term))
    return result


def _comparison_bounds(
    lhs: Mapping[str, Any], rhs: Mapping[str, Any], strict: bool
) -> tuple[int | None, int | None, bool]:
    query, key, constant, non_affine = _affine_difference(lhs, rhs)
    if non_affine or query != -key:
        return 0, None, True
    if query == 0:
        valid = constant < 0 if strict else constant <= 0
        return (0, None, True) if valid else (None, None, False)
    if query == 1:
        upper = _checked_neg_i64(constant, "retention expression")
        if strict:
            upper = _checked_i64(upper - 1, "retention expression")
        return _intersect_bounds((0, None, True), (None, upper, True))
    if query == -1:
        lower = constant
        if strict:
            lower = _checked_i64(lower + 1, "retention expression")
        return _intersect_bounds((0, None, True), (lower, None, True))
    return 0, None, True


def _affine_difference(
    lhs: Mapping[str, Any], rhs: Mapping[str, Any]
) -> tuple[int, int, int, bool]:
    left = _affine(lhs)
    right = _affine(rhs)
    return (
        _checked_i64(left[0] - right[0], "retention expression"),
        _checked_i64(left[1] - right[1], "retention expression"),
        _checked_i64(left[2] - right[2], "retention expression"),
        left[3] or right[3],
    )


def _affine(expression: Mapping[str, Any]) -> tuple[int, int, int, bool]:
    op = expression["op"]
    if op == "query_position":
        return 1, 0, 0, False
    if op == "key_position":
        return 0, 1, 0, False
    if op == "constant":
        return 0, 0, expression["value"], False
    if op in ("add", "sub"):
        left = _affine(expression["lhs"])
        right = _affine(expression["rhs"])
        sign = 1 if op == "add" else -1
        return (
            _checked_i64(left[0] + sign * right[0], "retention expression"),
            _checked_i64(left[1] + sign * right[1], "retention expression"),
            _checked_i64(left[2] + sign * right[2], "retention expression"),
            left[3] or right[3],
        )
    _affine(expression["value"])
    return 0, 0, 0, True


def _intersect_bounds(
    left: tuple[int | None, int | None, bool],
    right: tuple[int | None, int | None, bool],
) -> tuple[int | None, int | None, bool]:
    if not left[2] or not right[2]:
        return None, None, False
    lower = _maximum_optional(left[0], right[0])
    upper = _minimum_optional(left[1], right[1])
    if lower is not None and upper is not None and lower > upper:
        return None, None, False
    return lower, upper, True


def _union_bounds(
    left: tuple[int | None, int | None, bool],
    right: tuple[int | None, int | None, bool],
) -> tuple[int | None, int | None, bool]:
    if not left[2]:
        return right
    if not right[2]:
        return left
    lower = _minimum_optional(left[0], right[0])
    upper = None if left[1] is None or right[1] is None else max(left[1], right[1])
    return lower, upper, True


def _maximum_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _minimum_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _checked_i64(value: int, calculation: str) -> int:
    if not _I64_MIN <= value <= _I64_MAX:
        raise ValueError(f"integer overflow while calculating {calculation}")
    return value


def _checked_neg_i64(value: int, calculation: str) -> int:
    if value == _I64_MIN:
        raise ValueError(f"integer overflow while calculating {calculation}")
    return -value


def _checked_u64(value: int, calculation: str) -> int:
    if not 0 <= value <= _U64_MAX:
        raise ValueError(f"integer overflow while calculating {calculation}")
    return value


def _ceil_div_u64(value: int, divisor: int, calculation: str) -> int:
    if value == 0:
        return 0
    return _checked_u64(value + divisor - 1, calculation) // divisor


def layout_plan_fingerprint(layout: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def update_u64(value: int) -> None:
        digest.update(struct.pack("<Q", value))

    def update_string(value: str) -> None:
        encoded = value.encode("utf-8")
        update_u64(len(encoded))
        digest.update(encoded)

    page_tokens = layout["page_tokens"]
    classes = layout["classes"]
    update_u64(page_tokens)
    update_u64(len(classes))
    for item in classes:
        update_string(item["name"])
        update_u64(len(item["layers"]))
        for layer in item["layers"]:
            update_u64(layer)
        update_u64(item["bytes_per_token_per_layer"])
        address = item["address"]
        retirement = item["retirement"]
        kind = address["kind"]
        if kind in ("append_only", "pinned"):
            retention_code = 0
            window_tokens = 0
        elif kind in ("periodic", "periodic_from"):
            retention_code = 1
            window_tokens = retirement["offset_tokens"] + 1
        else:
            retention_code = 2
            window_tokens = 0
        update_u64(retention_code)
        update_u64(window_tokens)
        if kind == "resettable_arena":
            update_u64(1)
            update_u64(address["blocks_per_epoch"] * page_tokens)
        update_u64(item["minimum_slots_per_request"] or 0)
        if "kv_head_range" in item:
            update_u64(1)
            update_u64(item["kv_head_range"]["start"])
            update_u64(item["kv_head_range"]["end_exclusive"])
        if "block_domain" in item:
            domain = item["block_domain"]
            update_u64(1)
            update_u64(domain["start_block"])
            update_u64(domain.get("end_block_exclusive", _U64_MAX))
    return "sha256:" + digest.hexdigest()


def validate_capability_requirements(raw: Any) -> tuple[str, ...]:
    path = "RuntimeManifest.capability_requirements"
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"{path} must be a list of strings")
    requirements = tuple(raw)
    if requirements != tuple(sorted(set(requirements))):
        raise ValueError(f"{path} must be sorted and unique")
    unknown = set(requirements) - RETENTION_CAPABILITIES
    if unknown:
        raise ValueError(
            f"{path} contains unsupported capabilities: {', '.join(sorted(unknown))}"
        )
    return requirements


def derive_capability_requirements(layout: Mapping[str, Any]) -> tuple[str, ...]:
    requirements = {"token_manager"}
    address_capabilities = {
        "append_only": "append_only_addressing",
        "pinned": "pinned_addressing",
        "periodic": "periodic_addressing",
        "periodic_from": "periodic_from_addressing",
        "resettable_arena": "resettable_arena_addressing",
    }
    for item in layout["classes"]:
        if "kv_head_range" in item:
            requirements.add("kv_head_partitioning")
        if "block_domain" in item:
            requirements.add("block_domain_partitioning")
        requirements.add(address_capabilities[item["address"]["kind"]])
        if item["retirement"]["kind"] != "never":
            requirements.add("semantic_retirement")
    return tuple(sorted(requirements))


def _layers(raw: Any, path: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must be a non-empty list")
    result = tuple(
        _integer(value, f"{path}[{index}]", 0, _U32_MAX)
        for index, value in enumerate(raw)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must contain unique layer ids")
    return result


def _kv_head_range(raw: Any, path: str) -> tuple[int, int]:
    value = _mapping(raw, path)
    _exact_keys(value, path, {"start", "end_exclusive"})
    start = _integer(value["start"], f"{path}.start", 0, _U32_MAX)
    end = _integer(
        value["end_exclusive"], f"{path}.end_exclusive", 0, _U32_MAX
    )
    if start >= end:
        raise ValueError(f"{path} must be non-empty")
    return start, end


def _head_ranges_overlap(
    left: tuple[int, int] | None, right: tuple[int, int] | None
) -> bool:
    if left is None or right is None:
        return True
    return left[0] < right[1] and right[0] < left[1]


def _mapping(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be an object")
    return raw


def _exact_keys(
    value: Mapping[str, Any],
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{path} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _positive_integer(value: Any, path: str, maximum: int) -> int:
    return _integer(value, path, 1, maximum)


def _sha256_fingerprint(value: Any, path: str) -> str:
    item = _nonempty_string(value, path)
    if (
        len(item) != len("sha256:") + 64
        or not item.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in item[7:])
    ):
        raise ValueError(f"{path} must be a lowercase sha256 fingerprint")
    return item


__all__ = [
    "LAYOUT_PROGRAM_SCHEMA",
    "RETENTION_PROGRAM_SCHEMA",
    "RETENTION_CAPABILITIES",
    "derive_capability_requirements",
    "layout_plan_fingerprint",
    "validate_capability_requirements",
    "validate_layout_program",
    "validate_retention_program",
]
