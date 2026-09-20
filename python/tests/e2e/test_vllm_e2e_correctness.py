"""E2E correctness test: compare OrbitKV with equivalent vLLM reuse paths.

The single most important test for OrbitKV. Verifies the core contract:

    Given the same prompt + model + greedy sampling + prefix reuse plan,
    output must be identical with or without OrbitKV.

Covers all critical KV cache paths in one deterministic test:
- Cold save + warm load (basic round-trip)
- Multi-block prompts (block boundary alignment)
- Prefix extension (cached "A B C" -> request "A B C D E")
- Prefix rollback (cached "A B C D" -> request "A B")
- Multi-round decode (growing context across rounds)
- Cache metrics (directional assertions)

Usage:
    pytest python/tests/e2e/test_vllm_e2e_correctness.py -v -s

    Cache Manager is auto-started (via cargo run -r) and stopped by the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.vllm_helpers import (
    CacheManager,
    VLLMServer,
    _uses_linear_attention,
    adapt_prompt_for_hybrid_cache,
    call_openai_api,
    e2e_max_tokens,
    fetch_orbitkv_metrics,
    fetch_orbitkv_rpc_failures,
    fetch_vllm_prefix_cache_hits,
)

# ---------------------------------------------------------------------------
# Prompt design
#
# Each prompt group exercises a different cache path. Prompts are long enough
# to span multiple blocks (block_size=16 tokens typically) so that partial
# hits are meaningful, not degenerate single-block cases.
# ---------------------------------------------------------------------------

# Short prompt (likely < 1 full block) — edge case for incomplete blocks
SHORT_PROMPT = "2 + 2 ="

# Long prompt (multi-block) — tests block boundary alignment during save/load
LONG_PROMPT = (
    "Machine learning is a subset of artificial intelligence that focuses on "
    "building systems that can learn from and make decisions based on data. "
    "Deep learning, a further subset of machine learning, uses artificial "
    "neural networks with many layers to model complex patterns in large "
    "amounts of data. The key advantage of deep learning over traditional "
    "machine learning methods is its ability to automatically discover "
    "representations needed for feature detection or classification from "
    "raw data. This eliminates the need for manual feature engineering, "
    "which has traditionally been one of the most time-consuming aspects "
    "of applying machine learning to real-world problems. Convolutional "
    "neural networks have proven particularly effective for image recognition "
    "tasks, while recurrent neural networks and transformers have shown "
    "remarkable results in natural language processing. The transformer "
    "architecture, introduced in the paper Attention Is All You Need, has "
    "become the foundation for modern large language models. The main idea is"
)

# Prefix family: extension
# prefix_base is a strict prefix of prefix_extend.
# After caching prefix_base, requesting prefix_extend should partially hit.
PREFIX_BASE = (
    "The history of computing begins with mechanical calculators in the 17th "
    "century. Blaise Pascal built the Pascaline in 1642, which could perform "
    "addition and subtraction. Gottfried Wilhelm Leibniz later improved upon "
    "this design with the Stepped Reckoner, which could also multiply and "
    "divide. Charles Babbage conceived the Difference Engine in the 1820s and "
    "later designed the more ambitious Analytical Engine, which contained many "
    "features of modern computers including an arithmetic logic unit, control "
    "flow through conditional branching, and memory. Ada Lovelace wrote what "
    "is considered the first computer program for the Analytical Engine. The "
    "next major breakthrough came with"
)

PREFIX_EXTEND = (
    PREFIX_BASE + " the development of electronic computers in the mid-20th century. "
    "Alan Turing formalized the concept of computation with his theoretical "
    "Turing machine in 1936. During World War II, several electronic computing "
    "devices were built including Colossus and ENIAC. The invention of the "
    "transistor at Bell Labs in 1947 revolutionized electronics and led to "
    "increasingly powerful and compact computers. The integrated circuit, "
    "developed independently by Jack Kilby and Robert Noyce, further accelerated "
    "this trend. The impact of these innovations on modern society is"
)

# Prefix family: rollback
# rollback_long is cached first, then rollback_short (a strict prefix of
# rollback_long) requests a subset — the reverse of prefix extension.
ROLLBACK_LONG = (
    "In computer science, a hash table is a data structure that implements an "
    "associative array, also called a dictionary or map. A hash table uses a "
    "hash function to compute an index, also called a hash code, into an array "
    "of buckets or slots, from which the desired value can be found. During "
    "lookup, the key is hashed and the resulting hash indicates where the "
    "corresponding value is stored. Ideally, the hash function will assign each "
    "key to a unique bucket, but most hash table designs employ an imperfect "
    "hash function, which might cause hash collisions where the hash function "
    "generates the same index for more than one key. Such collisions are "
    "typically accommodated by chaining or open addressing. The main advantage is"
)

ROLLBACK_SHORT = (
    "In computer science, a hash table is a data structure that implements an "
    "associative array, also called a dictionary or map. A hash table uses a "
    "hash function to compute an index, also called a hash code, into an array "
    "of buckets or slots, from which the desired value can be found. During "
    "lookup, the key is hashed and the resulting hash indicates where the "
    "corresponding value is stored. Ideally, the hash function will assign each "
    "key to a unique bucket, but most hash table designs employ an imperfect "
    "hash function, which might cause hash collisions where the hash function "
    "generates the same index for more than one key. Such collisions are"
)

# Multi-round decode: three prompts where each is a prefix of the next.
# Simulates growing chat context across decode rounds.
_MULTI_ROUND_STEM = (
    "Quantum computing leverages quantum mechanical phenomena such as "
    "superposition and entanglement to perform computation. Unlike classical "
    "bits that exist in a state of either 0 or 1, quantum bits or qubits can "
    "exist in a superposition of both states simultaneously. This property "
    "allows quantum computers to explore many possible solutions at once. "
    "Quantum entanglement enables qubits that are entangled to be correlated "
    "with each other in ways that have no classical equivalent. When combined, "
    "these properties give quantum computers the potential to solve certain "
    "problems exponentially faster than classical computers. "
)

MULTI_ROUND = [
    _MULTI_ROUND_STEM + "The current state of quantum hardware is",
    (
        _MULTI_ROUND_STEM + "Key algorithms include Shor's algorithm for factoring large numbers "
        "and Grover's algorithm for searching unsorted databases. Current "
        "quantum computers face significant challenges including decoherence, "
        "error rates, and the need for extreme cooling. The path forward involves"
    ),
    (
        _MULTI_ROUND_STEM + "Key algorithms include Shor's algorithm for factoring large numbers "
        "and Grover's algorithm for searching unsorted databases. Current "
        "quantum computers face significant challenges including decoherence, "
        "error rates, and the need for extreme cooling. Major tech companies "
        "including Google, IBM, and Microsoft are investing heavily in quantum "
        "computing research. Google claimed quantum supremacy in 2019 with its "
        "Sycamore processor. IBM has been steadily increasing qubit counts with "
        "its Eagle, Osprey, and Condor processors. The most promising applications are"
    ),
]


# Ordered execution plan shared by the native-prefix and OrbitKV phases.
# Each entry: (label, prompt, cache_expectation)
# cache_expectation: "cold" = first time, "warm" = exact repeat, "partial" = prefix hit
EXECUTION_PLAN: list[tuple[str, str, str]] = [
    # Round 1: cold saves — fill the cache
    ("short_cold", SHORT_PROMPT, "cold"),
    ("long_cold", LONG_PROMPT, "cold"),
    ("prefix_base", PREFIX_BASE, "cold"),
    ("rollback_long", ROLLBACK_LONG, "cold"),
    ("multi_r1", MULTI_ROUND[0], "cold"),
    # Same-process warm hit proves vLLM's local HMA cache cannot mask OrbitKV.
    ("short_same_process", SHORT_PROMPT, "warm-same-process"),
    # Round 2: warm hits — exact same prompts
    ("short_warm", SHORT_PROMPT, "warm"),
    ("long_warm", LONG_PROMPT, "warm"),
    # Round 3: prefix operations
    ("prefix_extend", PREFIX_EXTEND, "partial"),
    ("rollback_short", ROLLBACK_SHORT, "partial"),
    # Round 4: multi-round growth
    ("multi_r2", MULTI_ROUND[1], "partial"),
    ("multi_r3", MULTI_ROUND[2], "partial"),
]

# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.gpu
class TestE2ECorrectness:
    """E2E correctness: baseline vLLM vs OrbitKV-enabled vLLM.

    Two-phase structure:
      Phase 1 — run the plan with native vLLM prefix caching, no OrbitKV.
      Phase 2 — run the plan with OrbitKV; restart vLLM before warm loads.
    The native phase keeps one vLLM process alive because its HBM prefix cache
    does not survive restart. The comparison is between reuse paths, while
    OrbitKV's restart separately proves Cache Manager persistence.
    The equality test walks every execution-plan label and reports the label
    that failed, so one expensive fixture run does not pretend each path is an
    independent test.
    """

    @pytest.fixture(scope="class")
    def log_dir(self, tmp_path_factory) -> Path:
        return tmp_path_factory.mktemp("e2e_logs")

    @pytest.fixture(scope="class")
    def orbitkv_server(
        self,
        log_dir: Path,
        orbitkv_use_hugepages: bool,
        orbitkv_pool_size: str,
    ):
        """Auto-start Cache Manager with prometheus metrics."""
        with CacheManager(
            log_file=log_dir / "orbitkv-cache-manager.log",
            pool_size=orbitkv_pool_size,
            use_hugepages=orbitkv_use_hugepages,
        ) as server:
            yield server

    @pytest.fixture(scope="class")
    def baseline_outputs(
        self,
        model: str,
        base_port: int,
        log_dir: Path,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        max_model_len: int | None,
    ) -> dict[str, str]:
        """Phase 1: execute the same plan using native vLLM prefix caching."""
        print("\n[Phase 1] Native vLLM prefix cache — executing cache plan")
        outputs: dict[str, str] = {}

        with VLLMServer(
            model,
            base_port,
            use_orbitkv=False,
            use_noop_connector=True,
            prefix_caching=True,
            log_file=log_dir / "baseline.log",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
        ):
            for label, prompt, expectation in EXECUTION_PLAN:
                if label == "long_warm":
                    before_long_warm = fetch_vllm_prefix_cache_hits(base_port)
                result = call_openai_api(
                    base_port,
                    model,
                    adapt_prompt_for_hybrid_cache(model, prompt),
                    max_tokens=e2e_max_tokens(model),
                )
                outputs[label] = result["text"]
                print(f"  [{label}] ({expectation}) {len(result['text'])} chars")
                if label == "long_warm":
                    native_hit_tokens = fetch_vllm_prefix_cache_hits(base_port) - before_long_warm
                    assert native_hit_tokens > 0, (
                        "native long_warm performed no prefix reuse; "
                        "the control path would be another cold prefill"
                    )

        print(f"[Phase 1] Done — {len(outputs)} native-path outputs collected\n")
        return outputs

    @pytest.fixture(scope="class")
    def orbitkv_results(
        self,
        model: str,
        base_port: int,
        orbitkv_server: CacheManager,
        orbitkv_transfer_backend: str,
        log_dir: Path,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        max_model_len: int | None,
    ) -> dict:
        """Phase 2: run execution plan through OrbitKV vLLM."""
        print("[Phase 2] OrbitKV vLLM — executing cache plan")
        orbitkv_port = base_port + 1
        outputs: dict[str, str] = {}
        metrics_port = orbitkv_server.metrics_port
        metrics_start = fetch_orbitkv_metrics(metrics_port)
        long_warm_load_bytes = 0.0

        with VLLMServer(
            model,
            orbitkv_port,
            use_orbitkv=True,
            orbitkv_port=orbitkv_server.cache_port,
            log_file=log_dir / "orbitkv.log",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            transfer_backend=orbitkv_transfer_backend,
        ):
            for label, prompt, expectation in EXECUTION_PLAN:
                if expectation not in {"cold", "warm-same-process"}:
                    continue
                result = call_openai_api(
                    orbitkv_port,
                    model,
                    adapt_prompt_for_hybrid_cache(model, prompt),
                    max_tokens=e2e_max_tokens(model),
                )
                outputs[label] = result["text"]
                print(f"  [{label}] ({expectation}) {len(result['text'])} chars")

            metrics_same_process = fetch_orbitkv_metrics(metrics_port)

        with VLLMServer(
            model,
            orbitkv_port,
            use_orbitkv=True,
            orbitkv_port=orbitkv_server.cache_port,
            log_file=log_dir / "orbitkv-load.log",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            transfer_backend=orbitkv_transfer_backend,
            server_label="OrbitKV load",
        ):
            for label, prompt, expectation in EXECUTION_PLAN:
                if expectation in {"cold", "warm-same-process"}:
                    continue
                if label == "long_warm":
                    before_long_warm = fetch_orbitkv_metrics(metrics_port)
                result = call_openai_api(
                    orbitkv_port,
                    model,
                    adapt_prompt_for_hybrid_cache(model, prompt),
                    max_tokens=e2e_max_tokens(model),
                )
                outputs[label] = result["text"]
                print(f"  [{label}] ({expectation}) {len(result['text'])} chars")
                if label == "long_warm":
                    after_long_warm = fetch_orbitkv_metrics(metrics_port)
                    long_warm_load_bytes = after_long_warm.get(
                        "orbitkv_load_bytes_total", 0
                    ) - before_long_warm.get("orbitkv_load_bytes_total", 0)

            metrics_end = fetch_orbitkv_metrics(metrics_port)

        print("[Phase 2] Done\n")
        return {
            "outputs": outputs,
            "metrics_start": metrics_start,
            "metrics_same_process": metrics_same_process,
            "metrics_end": metrics_end,
            "long_warm_load_bytes": long_warm_load_bytes,
            "connector_logs": (
                (log_dir / "orbitkv.log").read_text(errors="replace")
                + (log_dir / "orbitkv-load.log").read_text(errors="replace")
            ),
        }

    def test_execution_plan_outputs_match_baseline(self, baseline_outputs, orbitkv_results):
        """Each OrbitKV output must match native vLLM on the same reuse plan."""
        mismatches: list[str] = []
        for label, _prompt, expectation in EXECUTION_PLAN:
            baseline = baseline_outputs[label]
            orbitkv = orbitkv_results["outputs"][label]
            if baseline != orbitkv:
                mismatches.append(
                    f"[{label}/{expectation}] Output mismatch:\n"
                    f"  baseline ({len(baseline)} chars): {baseline[:120]}...\n"
                    f"  orbitkv ({len(orbitkv)} chars): {orbitkv[:120]}..."
                )

        assert not mismatches, "\n\n".join(mismatches)

    def test_cache_metrics(self, orbitkv_results):
        """Directional cache metrics — saves happened on cold, hits on warm/partial."""
        m_start = orbitkv_results["metrics_start"]
        m_end = orbitkv_results["metrics_end"]

        def delta(key: str) -> float:
            return m_end.get(key, 0) - m_start.get(key, 0)

        save_bytes = delta("orbitkv_save_bytes_total")
        insertions = delta("orbitkv_cache_block_insertions_total")
        load_bytes = delta("orbitkv_load_bytes_total")
        hits = delta("orbitkv_cache_block_hits_total")

        assert save_bytes > 0 or insertions > 0, (
            f"No SAVE activity: save_bytes={save_bytes}, insertions={insertions}"
        )
        assert load_bytes > 0 or hits > 0, f"No LOAD activity: load_bytes={load_bytes}, hits={hits}"

        print(
            f"\n[Metrics] saves={insertions:.0f} blocks ({save_bytes / 1e6:.1f}MB), "
            f"hits={hits:.0f} blocks ({load_bytes / 1e6:.1f}MB)"
        )

    def test_cross_process_warm_request_loads_kv(self, orbitkv_results):
        """A warm response must use saved bytes after the vLLM process restarts."""
        assert orbitkv_results["long_warm_load_bytes"] > 0, (
            "long_warm performed no KV restore after restart; text equality alone "
            "cannot distinguish a cache hit from local recomputation"
        )

    def test_cache_manager_channel(self, orbitkv_results):
        assert (
            "[OrbitKVConnector] Cache Manager channel: transport=iceoryx2"
            in orbitkv_results["connector_logs"]
        )

    def test_same_process_hma_load_uses_orbitkv(self, model: str, orbitkv_results):
        """The warm request in the first vLLM process must load from OrbitKV."""
        if not _uses_linear_attention(model):
            pytest.skip("same-process HMA assertion requires a hybrid linear-attention model")
        m_start = orbitkv_results["metrics_start"]
        m_end = orbitkv_results["metrics_same_process"]
        hit_delta = m_end.get("orbitkv_cache_block_hits_total", 0) - m_start.get(
            "orbitkv_cache_block_hits_total", 0
        )
        load_delta = m_end.get("orbitkv_load_bytes_total", 0) - m_start.get(
            "orbitkv_load_bytes_total", 0
        )
        assert hit_delta > 0 or load_delta > 0, (
            "same-process warm HMA request bypassed OrbitKV: "
            f"hit_delta={hit_delta}, load_delta={load_delta}"
        )

    def test_no_data_path_rpc_failures(self, orbitkv_results, orbitkv_server: CacheManager):
        """Every connector<->server RPC must return ok during a correct run.

        Generalizes the 0.22.5 empty-lease regression: that bug surfaced as
        release/"Client specified an invalid argument" on every cache miss, but
        the same gate also catches a failed load, save, or query_prefetch — none
        of which the output-equality test reliably exposes.

        The miss guard makes the gate non-vacuous: it proves the plan actually
        drove the zero-hit path the bug lived on, so a green run means something.
        """
        # orbitkv_results forces the real cache plan to run against this server.
        del orbitkv_results
        counters = fetch_orbitkv_metrics(orbitkv_server.metrics_port)
        assert counters.get("orbitkv_cache_block_misses_total", 0) > 0, (
            "execution plan exercised no cache miss; the RPC-health gate would be vacuous"
        )
        failures = fetch_orbitkv_rpc_failures(orbitkv_server.metrics_port)
        assert not failures, f"non-ok data-path RPCs during run: {failures}"

    def test_no_kv_load_failures(self, orbitkv_results, orbitkv_server: CacheManager):
        """KV loads must not fail.

        A load failure is silently masked by test_execution_plan_outputs_match_baseline:
        vLLM reports the failed blocks via get_block_ids_with_load_errors and
        recomputes them locally, so the final text still matches baseline. Only
        the failure counter exposes it, which is why this is a separate gate.
        """
        del orbitkv_results
        counters = fetch_orbitkv_metrics(orbitkv_server.metrics_port)
        loads_happened = (
            counters.get("orbitkv_cache_block_hits_total", 0) > 0
            or counters.get("orbitkv_load_bytes_total", 0) > 0
        )
        assert loads_happened, "plan performed no KV load; cannot assert load health"
        assert counters.get("orbitkv_load_failures_total", 0) == 0, (
            f"KV load failures during run: {counters.get('orbitkv_load_failures_total')}"
        )
