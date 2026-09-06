# On-device greedy correctness diagnostic

Status: real-device, released-checkpoint correctness diagnostic. This record
does not qualify latency, throughput, memory, capacity, or production behavior.

The decoder language-model head feeds Luminal's fused dynamic-row argmax in the
same compiled graph as prefill and decode. During the diagnostic path, the
device token IDs matched the previous host `max_by` result for the final prefill
row and two decode rows. A following decode used the default execution API,
which reads only one `i32` token ID and does not copy full logits to the host.

The same two-bucket runtime and persistent K/V arena from the L1 record were
used. The run completed prefill plus three decode steps and four OrbitKV
publications. The final device-token-only decode was approximately 4 ms in this
single smoke. This is not a matched benchmark and does not establish a speedup.

The scope is greedy argmax only. Temperature, top-k, top-p, penalties, logprobs,
structured sampling, and distributed sampling remain unsupported by the native
model executor.

The test command is the released-checkpoint command documented in the bucketed
decoder record, selecting
`released_decoder_reuses_one_compiled_runtime_and_kv_arena`.
