from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from shape_census import load_runs, summarize_runs, validate_run


def entry(
    *,
    phase: str = "decode",
    site: str = "model.layers.0.self_attn.q_proj",
    m: int = 1,
    n: int = 1536,
    k: int = 1536,
    calls: int = 7,
) -> dict:
    return {
        "phase": phase,
        "site": site,
        "observed_path": (
            "mistral_cuda_gemv" if m <= 8 else "candle_cuda_flattened_matmul"
        ),
        "device_kind": "cuda",
        "device_ordinal": 0,
        "m": m,
        "n": n,
        "k": k,
        "a_shape": [1, m, k],
        "a_stride": [m * k, k, 1],
        "a_offset_elements": 0,
        "weight_shape": [n, k],
        "weight_stride": [k, 1],
        "weight_offset_elements": 0,
        "a_dtype": "bf16",
        "weight_dtype": "bf16",
        "a_layout": "row_major_contiguous",
        "weight_layout": "row_major_contiguous_nk",
        "transpose_a": False,
        "transpose_weight": True,
        "post_ops": [],
        "host_calls": calls,
    }


def run(run_id: str = "run-1") -> dict:
    canonical_request = {
        "messages": [
            {
                "role": "user",
                "content": "Explain what a CUDA kernel does in one sentence.",
            }
        ],
        "sampler": {
            "temperature": 0.0,
            "max_output_tokens": 8,
            "top_logprobs": 5,
            "return_logprobs": True,
        },
    }
    canonical_request_bytes = json.dumps(
        canonical_request, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return {
        "schema": "oxide.gemm-shape-census.v1",
        "run_id": run_id,
        "source": {
            "producer": "mistral.rs",
            "repository": "https://github.com/feichai0017/mistral.rs",
            "commit": "a" * 40,
            "worktree_clean": True,
            "cargo_lock_sha256": "b" * 64,
            "binary_sha256": "c" * 64,
            "oxide_schema_commit": "d" * 40,
        },
        "hardware": {"gpu": "NVIDIA H20", "compute_capability": "9.0"},
        "environment": {
            "host_os": "linux",
            "host_arch": "x86_64",
            "rustc_version": "rustc 1.97.0-nightly",
            "cuda_toolkit_version": "13.1",
            "driver_version": "535.161.08",
            "cuda_arch": "sm_90",
        },
        "model": {
            "name": "Qwen2.5-1.5B-Instruct",
            "weights_sha256": "e" * 64,
            "config_sha256": "f" * 64,
            "tokenizer_sha256": "0" * 64,
            "tensor_parallel_size": 1,
        },
        "workload": {
            "scheduler": "single_request",
            "request_count": 1,
            "prompt_tokens": 13,
            "completion_tokens": 8,
            "prefill_forward_steps": 1,
            "decode_forward_steps": 7,
            "canonical_request": canonical_request,
            "canonical_request_sha256": hashlib.sha256(
                canonical_request_bytes
            ).hexdigest(),
            "cuda_graph_enabled": False,
        },
        "capture": {
            "boundary": "resolved_dense_linear_before_backend_selection",
            "count_semantics": "successful_linear_dispatches",
            "sample_rate": 1,
            "complete": True,
            "timed": False,
            "mistral_gemv_enabled": True,
        },
        "entries": [
            entry(),
            entry(
                phase="prefill",
                site="model.layers.0.self_attn.q_proj",
                m=13,
                calls=1,
            ),
        ],
        "accepted_claims": [
            "exact dense-linear host dispatch counts for the pinned model and workload"
        ],
        "excluded_claims": [
            "latency",
            "throughput",
            "CUDA kernel launch counts",
            "general model coverage",
            "production workload frequency",
        ],
    }


class ShapeCensusTests(unittest.TestCase):
    def test_summarizes_calls_flops_and_breakdowns(self) -> None:
        record = run()

        result = summarize_runs([record], [{"path": "run.jsonl", "sha256": "2" * 64}])

        self.assertEqual(result["totals"]["host_calls"], 8)
        self.assertEqual(result["totals"]["decode_host_calls_per_forward"], 1.0)
        self.assertEqual(result["shapes"][0]["m"], 1)
        self.assertEqual(result["shapes"][0]["call_share"], 7 / 8)
        self.assertEqual(result["shapes"][0]["cumulative_call_share"], 7 / 8)
        self.assertEqual(
            result["shapes"][0]["observed_path_calls"], {"mistral_cuda_gemv": 7}
        )
        self.assertTrue(result["shapes"][0]["oxide_logical_contract_compatible"])

    def test_merges_same_shape_across_sites_and_runs(self) -> None:
        first = run("first")
        second = run("second")
        second["entries"][0]["site"] = "model.layers.27.self_attn.o_proj"

        result = summarize_runs([first, second], [])

        shape = result["shapes"][0]
        self.assertEqual(shape["host_calls"], 14)
        self.assertEqual(shape["host_calls_per_run"], 7.0)
        self.assertEqual(len(shape["site_calls"]), 2)

    def test_rejects_contract_drift_across_runs(self) -> None:
        first = run("first")
        second = run("second")
        second["model"]["config_sha256"] = "9" * 64

        with self.assertRaisesRegex(ValueError, "incompatible.*model"):
            summarize_runs([first, second], [])

    def test_rejects_duplicate_run_id_through_direct_api(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate run_id"):
            summarize_runs([run(), run()], [])

    def test_rejects_claim_drift_across_runs(self) -> None:
        first = run("first")
        second = run("second")
        second["excluded_claims"] = ["latency"]

        with self.assertRaisesRegex(ValueError, "unsupported claim boundary"):
            summarize_runs([first, second], [])

    def test_rejects_boolean_dimension_and_zero_calls(self) -> None:
        bad_dimension = run()
        bad_dimension["entries"][0]["m"] = True
        with self.assertRaisesRegex(ValueError, "entries\[0\].m"):
            validate_run(bad_dimension, "record")

        zero_calls = run()
        zero_calls["entries"][0]["host_calls"] = 0
        with self.assertRaisesRegex(ValueError, "host_calls"):
            validate_run(zero_calls, "record")

    def test_rejects_dirty_incomplete_or_sampled_capture(self) -> None:
        dirty = run()
        dirty["source"]["worktree_clean"] = False
        with self.assertRaisesRegex(ValueError, "dirty source"):
            validate_run(dirty, "record")

        incomplete = run()
        incomplete["capture"]["complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_run(incomplete, "record")

        sampled = run()
        sampled["capture"]["sample_rate"] = 2
        with self.assertRaisesRegex(ValueError, "sampled"):
            validate_run(sampled, "record")

        graph = run()
        graph["workload"]["cuda_graph_enabled"] = True
        with self.assertRaisesRegex(ValueError, "Graph replay"):
            validate_run(graph, "record")

    def test_rejects_shape_stride_and_layout_mismatch(self) -> None:
        bad_shape = run()
        bad_shape["entries"][0]["a_shape"] = [1, 2, 1536]
        with self.assertRaisesRegex(ValueError, "activation shape"):
            validate_run(bad_shape, "record")

        bad_layout = run()
        bad_layout["entries"][0]["a_stride"] = [1536, 1536, 2]
        with self.assertRaisesRegex(ValueError, "a_layout"):
            validate_run(bad_layout, "record")

    def test_rejects_duplicate_entry(self) -> None:
        record = run()
        record["entries"].append(copy.deepcopy(record["entries"][0]))

        with self.assertRaisesRegex(ValueError, "duplicate census entry"):
            validate_run(record, "record")

    def test_rejects_unknown_fields_and_invalid_transpose(self) -> None:
        unknown = run()
        unknown["entries"][0]["output_dtype"] = "bf16"
        with self.assertRaisesRegex(ValueError, "unexpected fields: output_dtype"):
            validate_run(unknown, "record")

        invalid_transpose = run()
        invalid_transpose["entries"][0]["transpose_weight"] = False
        with self.assertRaisesRegex(ValueError, "weight must be transposed"):
            validate_run(invalid_transpose, "record")

    def test_keeps_distinct_strides_separate(self) -> None:
        record = run()
        strided = entry(site="model.layers.1.self_attn.q_proj")
        strided["a_shape"] = [1, 1, 1536]
        strided["a_stride"] = [3072, 3072, 1]
        strided["a_layout"] = "row_major_contiguous"
        record["entries"].append(strided)

        result = summarize_runs([record], [])

        self.assertEqual(len(result["shapes"]), 3)
        self.assertNotEqual(
            result["shapes"][0]["a_stride"], result["shapes"][1]["a_stride"]
        )

    def test_summary_order_does_not_depend_on_entry_order(self) -> None:
        first = run()
        offset = entry(site="model.layers.1.self_attn.q_proj")
        offset["a_offset_elements"] = 1
        first["entries"].append(offset)
        second = copy.deepcopy(first)
        second["entries"].reverse()

        self.assertEqual(
            summarize_runs([first], []),
            summarize_runs([second], []),
        )

    def test_marks_post_op_shape_incompatible(self) -> None:
        record = run()
        record["entries"][0]["post_ops"] = ["bias"]

        result = summarize_runs([record], [])

        self.assertFalse(result["shapes"][0]["oxide_logical_contract_compatible"])

    def test_rejects_unknown_post_op_and_usize_overflow(self) -> None:
        unknown_post_op = run()
        unknown_post_op["entries"][0]["post_ops"] = ["bias_v2_typo"]
        with self.assertRaisesRegex(ValueError, "unsupported value"):
            validate_run(unknown_post_op, "record")

        overflow = run()
        overflow["entries"][0].update(
            {
                "m": 1 << 63,
                "a_shape": [1 << 63, 1536],
                "a_stride": [1536, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "overflows 64-bit usize"):
            validate_run(overflow, "record")

        oversized_offset = run()
        oversized_offset["entries"][0]["a_offset_elements"] = 1 << 80
        with self.assertRaisesRegex(ValueError, "a_offset_elements"):
            validate_run(oversized_offset, "record")

        offset_span_overflow = run()
        offset_span_overflow["entries"][0]["a_offset_elements"] = (1 << 64) - 1
        with self.assertRaisesRegex(ValueError, "offset and span overflow"):
            validate_run(offset_span_overflow, "record")

        scalar_offset_overflow = run()
        scalar_offset_overflow["entries"][0].update(
            {
                "observed_path": "candle_matmul",
                "m": 1,
                "n": 1,
                "k": 1,
                "a_shape": [1, 1],
                "a_stride": [1, 1],
                "a_offset_elements": (1 << 64) - 1,
                "weight_shape": [1, 1],
                "weight_stride": [1, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "offset and span overflow"):
            validate_run(scalar_offset_overflow, "record")

    def test_rejects_observed_path_that_does_not_match_dispatch(self) -> None:
        wrong_gemv = run()
        wrong_gemv["entries"][0]["m"] = 9
        wrong_gemv["entries"][0]["a_shape"] = [1, 9, 1536]
        wrong_gemv["entries"][0]["a_stride"] = [9 * 1536, 1536, 1]
        with self.assertRaisesRegex(ValueError, "does not admit Mistral GEMV"):
            validate_run(wrong_gemv, "record")

        wrong_flattened = run()
        wrong_flattened["entries"][0]["observed_path"] = (
            "candle_cuda_flattened_matmul"
        )
        with self.assertRaisesRegex(ValueError, "does not admit flattened"):
            validate_run(wrong_flattened, "record")

        mixed_dtype = run()
        mixed_dtype["entries"][1]["weight_dtype"] = "f16"
        with self.assertRaisesRegex(ValueError, "dtypes must match"):
            validate_run(mixed_dtype, "record")

    def test_load_uses_one_snapshot_and_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            contents = (json.dumps(run()) + "\n").encode()
            path.write_bytes(contents)

            records, inputs = load_runs([str(path)])

            self.assertEqual(len(records), 1)
            self.assertEqual(inputs[0]["sha256"], hashlib.sha256(contents).hexdigest())

            duplicate = '{"schema":"wrong",' + json.dumps(run())[1:] + "\n"
            path.write_text(duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key 'schema'"):
                load_runs([str(path)])

    def test_rejects_phase_and_workload_mismatch(self) -> None:
        missing_prefill = run()
        missing_prefill["entries"] = [missing_prefill["entries"][0]]
        with self.assertRaisesRegex(ValueError, "no prefill calls"):
            validate_run(missing_prefill, "record")

        missing_decode = run()
        missing_decode["entries"] = [missing_decode["entries"][1]]
        with self.assertRaisesRegex(ValueError, "no decode calls"):
            validate_run(missing_decode, "record")

    def test_rejects_non_single_request_workload(self) -> None:
        batched = run()
        batched["workload"]["request_count"] = 2
        with self.assertRaisesRegex(ValueError, "requires one request"):
            validate_run(batched, "record")

        chunked = run()
        chunked["workload"]["prefill_forward_steps"] = 2
        with self.assertRaisesRegex(ValueError, "requires one prefill forward"):
            validate_run(chunked, "record")

        tensor_parallel = run()
        tensor_parallel["model"]["tensor_parallel_size"] = 2
        with self.assertRaisesRegex(ValueError, "tensor parallel size one"):
            validate_run(tensor_parallel, "record")

    def test_rejects_noncanonical_request_digest(self) -> None:
        record = run()
        record["workload"]["canonical_request_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "does not match canonical request"):
            validate_run(record, "record")


if __name__ == "__main__":
    unittest.main()
