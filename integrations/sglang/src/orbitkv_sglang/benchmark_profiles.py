from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


PREFIX_REUSE = "prefix_reuse"
FRESH_PROMPT = "fresh_prompt"


def is_fresh_prompt(contract: dict[str, Any]) -> bool:
    profile = contract.get("workload_profile", PREFIX_REUSE)
    if profile not in (PREFIX_REUSE, FRESH_PROMPT):
        raise RuntimeError("checkpoint has an unsupported workload profile")
    return profile == FRESH_PROMPT


def fresh_input_ids(
    *, requests: int, prompt_tokens: int, vocab_size: int, seed: int, iteration: int,
    forbidden_token_ids: Sequence[int] = (),
    token_upper_bound: int | None = None,
) -> list[list[int]]:
    """Build one iteration of a collision-free request-private prompt matrix."""

    upper_bound = vocab_size if token_upper_bound is None else token_upper_bound
    if (requests <= 0 or prompt_tokens <= 0 or vocab_size <= 3
            or isinstance(upper_bound, bool) or not isinstance(upper_bound, int)
            or not 3 < upper_bound <= vocab_size
            or seed < 0 or iteration < 0):
        raise RuntimeError("invalid fresh-prompt dimensions")
    forbidden_values = tuple(forbidden_token_ids)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in forbidden_values
    ):
        raise RuntimeError("fresh-prompt token exclusions must be integers")
    forbidden = tuple(sorted({
        value for value in forbidden_values if 3 <= value < upper_bound
    }))
    domain_size = upper_bound - 3 - len(forbidden)
    if domain_size < requests:
        raise RuntimeError("checkpoint vocabulary is too small after exclusions")
    if iteration > (domain_size - requests) // requests:
        raise RuntimeError(
            "fresh-prompt matrix exceeds the collision-free token domain"
        )

    def allowed_token(index: int) -> int:
        token = 3 + index
        for blocked in forbidden:
            if blocked > token:
                break
            token += 1
        return token

    result = []
    for request in range(requests):
        ordinal = iteration * requests + request
        material = hashlib.shake_256(
            f"orbitkv-fresh-v1:{seed}:{iteration}:{request}".encode("ascii")
        ).digest(prompt_tokens * 4)
        prompt = [
            allowed_token(
                int.from_bytes(material[offset : offset + 4], "little")
                % domain_size
            )
            for offset in range(0, len(material), 4)
        ]
        prompt[0] = allowed_token((seed + ordinal) % domain_size)
        result.append(prompt)
    return result


def validate_fresh_prompt_evidence(
    record: dict[str, Any], label: str, *, page_tokens: int = 16
) -> None:
    """Rebuild and validate every persisted fresh-prompt input identity."""

    contract = record.get("checkpoint_contract")
    workload = record.get("workload")
    traces = record.get("request_traces")
    if (not isinstance(contract, dict) or not isinstance(workload, dict)
            or not isinstance(traces, list)):
        raise RuntimeError(f"{label} fresh-prompt evidence is malformed")
    try:
        prompts = [
            fresh_input_ids(
                requests=workload["requests"],
                prompt_tokens=workload["prompt_tokens"],
                vocab_size=contract["vocab_size"], seed=workload["seed"],
                iteration=iteration,
                forbidden_token_ids=tuple(contract["control_token_ids"].values()),
                token_upper_bound=contract["prompt_token_upper_bound"],
            )
            for iteration in range(workload["iterations"])
        ]
    except (AttributeError, KeyError, TypeError, RuntimeError) as error:
        raise RuntimeError(
            f"{label} fresh-prompt inputs cannot be reconstructed"
        ) from error
    expected = [[canonical_digest(prompt) for prompt in row] for row in prompts]
    trace_digests = (
        [[trace.get("submitted_input_ids_sha256") for trace in row] for row in traces]
        if all(isinstance(row, list)
               and all(isinstance(trace, dict) for trace in row) for row in traces)
        else None
    )
    if (workload.get("input_token_digests_by_iteration_sha256") != expected
            or trace_digests != expected):
        raise RuntimeError(f"{label} fresh-prompt input digests are invalid")
    try:
        validate_cached_token_evidence(traces, 0)
    except RuntimeError as error:
        raise RuntimeError(
            f"{label} fresh-prompt cached-token evidence is invalid"
        ) from error
    flat = [prompt for row in prompts for prompt in row]
    if workload.get("input_token_digest_sha256") != input_digest(flat):
        raise RuntimeError(f"{label} aggregate fresh-prompt digest is invalid")
    first_pages = [tuple(prompt[:page_tokens]) for prompt in flat]
    if (len(first_pages) != workload["requests"] * workload["iterations"]
            or len(set(first_pages)) != len(first_pages)):
        raise RuntimeError(f"{label} fresh-prompt pages are not unique")


def deterministic_input_ids(
    *, requests: int, prompt_tokens: int, vocab_size: int, seed: int,
    page_tokens: int = 16,
) -> list[list[int]]:
    if vocab_size - 3 < requests:
        raise RuntimeError("checkpoint vocabulary is too small")
    shared_count = (prompt_tokens - 1) // page_tokens * page_tokens
    material = hashlib.shake_256(
        f"orbitkv-prefix-shared-v1:{seed}".encode("ascii")
    ).digest(shared_count * 4)
    shared = [
        3 + int.from_bytes(material[offset : offset + 4], "little")
        % (vocab_size - 3) for offset in range(0, len(material), 4)
    ]
    result = []
    for request in range(requests):
        material = hashlib.shake_256(
            f"orbitkv-canonical-v1:{seed}:{request}".encode("ascii")
        ).digest((prompt_tokens - shared_count) * 4)
        suffix = [
            3 + int.from_bytes(material[offset : offset + 4], "little")
            % (vocab_size - 3) for offset in range(0, len(material), 4)
        ]
        suffix[0] = 3 + (seed + request) % (vocab_size - 3)
        result.append(shared + suffix)
    return result


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def input_digest(inputs: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(
        json.dumps(inputs, separators=(",", ":")).encode()
    ).hexdigest()


def token_digest(outputs: Sequence[Sequence[dict[str, Any]]]) -> str:
    return canonical_digest(
        [[output["output_ids"] for output in iteration] for iteration in outputs]
    )


def request_token_digests(
    outputs: Sequence[Sequence[dict[str, Any]]],
) -> list[list[str]]:
    return [
        [canonical_digest(output["output_ids"]) for output in iteration]
        for iteration in outputs
    ]


def request_traces(
    *, outputs: Sequence[Sequence[dict[str, Any]]],
    submitted_rids: Sequence[Sequence[str]],
    submitted_input_digests: Sequence[Sequence[str]],
) -> list[list[dict[str, Any]]]:
    if not (len(outputs) == len(submitted_rids) == len(submitted_input_digests)):
        raise RuntimeError("request trace iteration cardinality is inconsistent")
    traces = []
    for iteration_outputs, iteration_rids, iteration_inputs in zip(
        outputs, submitted_rids, submitted_input_digests, strict=True
    ):
        if not (len(iteration_outputs) == len(iteration_rids) == len(iteration_inputs)):
            raise RuntimeError("request trace batch cardinality is inconsistent")
        row = []
        for request_index, (output, rid, input_sha256) in enumerate(
            zip(iteration_outputs, iteration_rids, iteration_inputs, strict=True)
        ):
            ids = list(output["output_ids"])
            meta_info = output.get("meta_info")
            if not isinstance(meta_info, dict) or meta_info.get("id") != rid:
                raise RuntimeError("SGLang returned a foreign request id")
            row.append({
                "request_index": request_index, "submitted_rid": rid,
                "submitted_input_ids_sha256": input_sha256,
                "returned_rid": meta_info["id"],
                "cached_tokens": meta_info.get("cached_tokens"),
                "output_ids": ids,
                "output_ids_sha256": canonical_digest(ids),
            })
        traces.append(row)
    return traces


def verify_request_trace_stability(
    traces: Sequence[Sequence[dict[str, Any]]],
) -> None:
    if not traces:
        raise RuntimeError("qualification produced no request traces")
    width = len(traces[0])
    if width <= 0 or any(len(row) != width for row in traces):
        raise RuntimeError("request trace matrix is not rectangular")
    for request_index in range(width):
        expected = traces[0][request_index]["output_ids"]
        if any(row[request_index]["output_ids"] != expected for row in traces[1:]):
            raise RuntimeError(
                "deterministic inference changed output tokens across iterations "
                f"for request index {request_index}"
            )


def validate_cached_token_evidence(
    traces: Sequence[Sequence[dict[str, Any]]], expected: int
) -> None:
    if any(
        type(trace.get("cached_tokens")) is not int
        or trace["cached_tokens"] != expected
        for row in traces for trace in row
    ):
        raise RuntimeError(
            f"request trace cached-token evidence differs from {expected}"
        )


def expected_mirror_transactions(
    *, prompt_tokens: int, decode_tokens: int, completed_iterations: int,
    prefix_seeded: bool, global_cleanup: bool, fresh: bool,
    page_tokens: int = 16,
) -> int:
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                prompt_tokens, decode_tokens, completed_iterations, page_tokens
            )
        )
        or prompt_tokens <= 0
        or decode_tokens <= 0
        or completed_iterations < 0
        or page_tokens <= 0
    ):
        raise RuntimeError("invalid mirror-transaction dimensions")
    if fresh:
        initial_pages = (prompt_tokens + page_tokens - 1) // page_tokens
        final_boundary = prompt_tokens + decode_tokens - 1
        final_pages = (final_boundary + page_tokens - 1) // page_tokens
        # One grouped transaction publishes the initial prefill pages, each
        # later page boundary publishes another, and one grouped transaction
        # detaches the request batch at release. Requests at equal boundaries
        # share each transaction, so this count is intentionally not scaled by
        # batch size.
        return completed_iterations * (2 + final_pages - initial_pages)
    seed_batches = int(prefix_seeded)
    cleanup_batches = int(global_cleanup)
    return seed_batches * (1 + completed_iterations + cleanup_batches)


def expected_batch_counters(
    *, batch_size: int, completed_iterations: int, decode_tokens: int,
    hybrid: bool, prefix_seeded: bool, global_cleanup: bool, fresh: bool,
    prompt_tokens: int | None = None,
    legacy_equal_mirror_counters: bool = False,
) -> dict[str, int]:
    seed_batches = int(prefix_seeded)
    cleanup_batches = int(global_cleanup)
    forward_batches = seed_batches + completed_iterations * decode_tokens
    release_batches = completed_iterations
    request_count = completed_iterations * batch_size
    warm_request_calls = 0 if fresh else request_count
    acquisition_calls = completed_iterations if fresh else warm_request_calls
    cold_calls = warm_request_calls + release_batches + seed_batches + cleanup_batches
    if fresh and prompt_tokens is None:
        raise RuntimeError("fresh mirror accounting requires prompt_tokens")
    mirror_validation_transactions = expected_mirror_transactions(
        prompt_tokens=1 if prompt_tokens is None else prompt_tokens,
        decode_tokens=decode_tokens,
        completed_iterations=completed_iterations, prefix_seeded=prefix_seeded,
        global_cleanup=global_cleanup, fresh=fresh,
    )
    # Prefix-reuse cleanup mutates the mirror at every validation boundary.
    # Fresh-prompt validation also observes publication-only boundaries, while
    # the only device mutation is the grouped request release per iteration.
    mirror_syncs = (
        mirror_validation_transactions
        if legacy_equal_mirror_counters or not fresh
        else completed_iterations
    )
    return {
        "request_acquire_batch_calls": seed_batches + acquisition_calls,
        "request_fork_batch_calls": 0,
        "prepare_batch_calls": forward_batches,
        "submit_batch_calls": forward_batches,
        "complete_batch_calls": forward_batches,
        "release_batch_calls": release_batches,
        "recycle_requests_batch_calls": release_batches + seed_batches,
        "prefix_lookup_batch_calls": warm_request_calls,
        "prefix_attach_batch_calls": warm_request_calls,
        "prefix_publish_batch_calls": 0,
        "prefix_publish_release_batch_calls": seed_batches,
        "prefix_evict_batch_calls": cleanup_batches,
        "prefix_recycle_batch_calls": cleanup_batches,
        "token_views_batch_calls": 0,
        "mark_token_dispositions_batch_calls": 0,
        "prepare_relocation_batch_calls": 0,
        "submit_relocation_batch_calls": 0,
        "complete_relocation_batch_calls": 0,
        "abort_relocations_batch_calls": 0,
        "buffer_too_small_preflights": cold_calls,
        "cold_workspace_allocations": cold_calls,
        "forward_events": forward_batches,
        "completion_values": forward_batches,
        "prefix_matches": seed_batches + request_count,
        "prefix_hits": 0 if fresh else request_count,
        "prefix_publishes": seed_batches,
        "prefix_evictions": cleanup_batches,
        "prefix_global_alias_scans": cleanup_batches * int(hybrid),
        "mirror_validation_calls": mirror_validation_transactions,
        "mirror_syncs": mirror_syncs,
        "cow_copy_intents": 0,
        "cow_move_calls": 0,
        "cow_copied_tokens": 0,
        "token_disposition_batches": 0,
        "token_policy_evictions": 0,
        "relocation_batches": 0,
        "relocation_moves": 0,
        "relocation_reclaimed_pages": 0,
        "relocation_copy_events": 0,
        "relocation_copy_tokens": 0,
    }


def validate_fixed_state_activity(
    counters: dict[str, int], *, batch_size: int, completed_iterations: int,
    decode_tokens: int, stage: str,
) -> None:
    requests = batch_size * completed_iterations
    forwards = decode_tokens * completed_iterations
    expected = {
        "fixed_state_prepares": requests,
        "fixed_state_clears": requests,
        "fixed_state_copies": 0,
        "fixed_state_events": forwards,
        "fixed_state_retirements": requests,
        "fixed_state_acks": requests,
    }
    mismatches = {
        name: {"expected": value, "actual": counters[name]}
        for name, value in expected.items() if counters[name] != value
    }
    if mismatches:
        raise RuntimeError(
            f"OrbitKV fixed-state lifecycle counters disagree at {stage}: {mismatches}"
        )
