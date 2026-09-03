"""Wire-key sets and census fields for Engine E2E verification."""

RECORD_KEYS = frozenset(
    {
        "schema", "mode", "source", "environment", "accelerator",
        "model", "checkpoint", "runtime_manifest", "runtime_binding",
        "engine_args", "server_snapshots", "sampling_params", "workload",
        "outputs", "timings", "manager",
    }
)
ACCELERATOR_KEYS = frozenset(
    {
        "device_type", "device_name", "compute_capability",
        "total_memory_bytes", "runtime_version", "driver_version",
    }
)
COMPUTE_CAPABILITY_KEYS = frozenset({"major", "minor"})
SOURCE_KEYS = frozenset({"root", "release", "revision", "patch"})
PATCH_KEYS = frozenset({"status", "sha256"})
CHECKPOINT_KEYS = frozenset(
    {
        "load_format", "config_sha256", "index_files", "weight_files",
        "weight_bytes", "indexed_weight_files", "indexed_weight_bytes",
        "observed_indexed_weight_bytes", "indexed_weight_container_overhead_bytes",
        "missing_indexed_weights", "indexed_weights_complete",
    }
)
FILE_IDENTITY_KEYS = frozenset({"name", "bytes", "sha256"})
ENGINE_REQUIRED_KEYS = frozenset(
    {
        "model_path", "load_format", "dtype", "kv_cache_dtype",
        "skip_tokenizer_init", "trust_remote_code", "context_length",
        "page_size", "attention_backend", "disable_hybrid_swa_memory",
        "disable_overlap_schedule", "disable_radix_cache", "disable_cuda_graph",
        "enable_torch_compile", "enable_deterministic_inference",
        "sampling_backend", "chunked_prefill_size", "prefill_max_requests",
        "max_prefill_tokens", "enable_dynamic_chunking", "enable_mixed_chunk",
        "max_running_requests", "tp_size", "pp_size", "dp_size", "dcp_size",
        "enable_dp_attention", "speculative_algorithm", "disaggregation_mode",
        "enable_hierarchical_cache", "enable_streaming_session",
        "enable_unified_memory", "enable_pdmux", "enable_lmcache",
        "enable_flexkv", "enable_session_radix_cache", "enable_hisparse",
        "enable_page_major_kv_layout", "random_seed", "log_level",
        "max_total_tokens",
    }
)
ENGINE_OPTIONAL_KEYS = frozenset({"mem_fraction_static"})
SERVER_SNAPSHOT_KEYS = frozenset(
    {"stage", "resolved_engine", "orbitkv_manager_present"}
)
RESOLVED_ENGINE_KEYS = frozenset(
    {
        "page_size", "max_total_tokens", "attention_backend", "dtype",
        "kv_cache_dtype", "chunked_prefill_size", "max_prefill_tokens",
        "prefill_max_requests", "max_running_requests",
        "effective_max_running_requests_per_dp", "disable_overlap_schedule",
        "disable_radix_cache", "disable_cuda_graph", "enable_torch_compile",
        "enable_dynamic_chunking", "enable_mixed_chunk", "tp_size", "pp_size",
        "dp_size", "dcp_size",
    }
)
SAMPLING_KEYS = frozenset(
    {"temperature", "max_new_tokens", "min_new_tokens", "ignore_eos", "sampling_seed"}
)
WORKLOAD_KEYS = frozenset(
    {
        "prompt_tokens", "decode_tokens", "warmups", "iterations", "seed",
        "input_ids", "input_ids_sha256", "request_ids",
    }
)
WORKLOAD_PARTITION_KEYS = frozenset({"warmup", "measured"})
OUTPUT_KEYS = frozenset({"warmups", "iterations", "aggregate_sha256"})
WARMUP_OUTPUT_KEYS = frozenset(
    {
        "warmup", "request_id", "input_ids_sha256", "output_ids",
        "output_ids_sha256", "cached_tokens",
    }
)
ITERATION_OUTPUT_KEYS = (WARMUP_OUTPUT_KEYS - {"warmup"}) | {"iteration"}
TIMING_KEYS = frozenset(
    {
        "iteration_seconds", "median_seconds", "p95_seconds", "measured_seconds",
        "output_tokens_per_second", "total_tokens_per_second",
    }
)
MANIFEST_KEYS = frozenset(
    {
        "schema", "version", "fingerprint", "source", "token_manager_plan",
        "attention_state_plan", "capability_requirements",
    }
)
BINDING_KEYS = frozenset(
    {
        "schema", "version", "fingerprint", "manifest_fingerprint", "target",
        "admission_profile", "target_contract_fingerprint",
        "required_wire_version", "execution_topology", "execution_signature",
    }
)
EXECUTION_SIGNATURE_KEYS = frozenset(
    {
        "schema", "version", "fingerprint", "manifest_schema",
        "manifest_version", "manifest_fingerprint", "page_tokens",
        "token_classes", "token_states", "fixed_states",
    }
)
MANAGER_KEYS = frozenset({"wire_version", "post_workload_residency", "snapshots"})
POST_WORKLOAD_RESIDENCY_KEYS = frozenset(
    {
        "active_pages", "retiring_pages", "quarantined_pages",
        "total_request_page_refs", "total_prefix_page_refs", "total_reader_pins",
    }
)
MANAGER_SNAPSHOT_KEYS = frozenset(
    {
        "stage", "wire_version", "manager_input_fingerprint",
        "runtime_manifest_fingerprint", "runtime_binding_fingerprint",
        "lifecycle_route", "cache_policy", "direct_source_owner",
        "completion_evidence", "runtime_proof", "identities", "manager_stats",
        "arena_stats", "batch_counters", "swa_activity", "pressure",
    }
)
OWNER_KEYS = frozenset(
    {
        "module", "type", "owner_is_process_singleton", "config_is_canonical",
        "allocator_owned", "tree_cache_owned",
    }
)
COMPLETION_KEYS = frozenset({"event_backend", "pending_events", "completion_high_water"})
COMPLETION_POINT_KEYS = frozenset({"domain", "value"})
SWA_COUNTER_FIELDS = (
    "swa_retirement_certificates",
    "swa_pages_reclaimed",
    "swa_wrap_events",
    "swa_page_reuse_events",
)
SWA_ACTIVITY_KEYS = frozenset(
    {"status", "applicable", "source", "derived", *SWA_COUNTER_FIELDS}
)
PRESSURE_KEYS = frozenset({"schema", "enabled", "mode", "sample_count"})
RUNTIME_PROOF_KEYS = frozenset({"actual_attention_backend", "effective_scheduler"})
ACTUAL_BACKEND_KEYS = frozenset(
    {
        "backend_class", "backend_module", "prefill_backend", "decode_backend",
        "has_local_attention", "attention_chunk_size", "page_size",
        "compiled_layer_ids", "use_irope_layer_ids",
    }
)
EFFECTIVE_SCHEDULER_KEYS = frozenset(
    {"max_prefill_tokens", "max_running_requests", "effective_max_running_requests_per_dp"}
)
IDENTITY_KEYS = frozenset(
    {
        "engine_epoch", "pool_epoch", "pool_id", "class_id", "backend_domain",
        "page_count", "page_tokens", "backend_base_index", "first_page_id",
    }
)
ARENA_KEYS = frozenset(
    {
        "engine_epoch", "pool_epoch", "pool_id", "page_count", "class_id",
        "backend_domain", "first_page_id", "free_pages", "reserved_pages",
        "writing_pages", "active_pages", "retiring_pages", "quarantined_pages",
        "exhausted_pages", "request_page_refs", "prefix_page_refs", "reader_pins",
    }
)
MANAGER_STATS_KEYS = frozenset(
    {
        "active_requests", "active_snapshots", "active_prefixes",
        "evicted_prefixes", "prepared_steps", "submitted_steps", "free_pages",
        "reserved_pages", "writing_pages", "active_pages", "retiring_pages",
        "quarantined_pages", "exhausted_pages", "pending_reclamations",
        "total_request_page_refs", "total_prefix_page_refs", "total_reader_pins",
    }
)
DRAIN_FIELDS = (
    "active_requests", "active_snapshots", "active_prefixes", "prepared_steps",
    "submitted_steps", "reserved_pages", "writing_pages", "active_pages",
    "retiring_pages", "quarantined_pages", "pending_reclamations",
    "total_request_page_refs", "total_prefix_page_refs", "total_reader_pins",
)
PAGE_PHASE_FIELDS = (
    "free_pages", "reserved_pages", "writing_pages", "active_pages",
    "retiring_pages", "quarantined_pages", "exhausted_pages",
)
BATCH_COUNTER_KEYS = frozenset(
    {"forward_events", "completion_values", "event_queries", "event_waits", "fail_stop_count"}
)
PROGRESS_COUNTER_FIELDS = ("forward_events", "completion_values")
