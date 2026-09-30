#!/usr/bin/env bash
# Native vLLM Mooncake P/D with the explicit OrbitKV TENT factory.
# Requires the experimental engine revision documented in docs/pd.md.
set -euo pipefail

MODEL="${1:?Usage: scripts/run_pd_local.sh /path/to/model}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/vllm-native-pd/bin/python}"
ROUTER="${VLLM_ROUTER:-vllm-router}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8101}"
PREFILL_PORT="${PREFILL_PORT:-8102}"
export VLLM_MOONCAKE_BOOTSTRAP_PORT="${VLLM_MOONCAKE_BOOTSTRAP_PORT:-8998}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/orbitkv-native-pd.XXXXXX)}"
case "${MC_FORCE_TCP:-1}" in
    1) export MC_FORCE_TCP=1; PROTOCOL=tcp ;;
    0) unset MC_FORCE_TCP; PROTOCOL=rdma ;;
    *) echo "MC_FORCE_TCP must be 0 (RDMA) or 1 (TCP)" >&2; exit 2 ;;
esac

[[ -x "$PYTHON" ]] || { echo "Missing patched vLLM environment: $PYTHON" >&2; exit 1; }
[[ "$PREFILL_GPU" != "$DECODE_GPU" ]] || { echo "PREFILL_GPU and DECODE_GPU must differ" >&2; exit 1; }
[[ "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "STARTUP_TIMEOUT must be a positive number of seconds" >&2; exit 1; }
for command in setsid curl "$ROUTER"; do command -v "$command" >/dev/null; done
mkdir -p "$LOG_DIR"

kv_config() {
    "$PYTHON" - "$1" "$2" "$PROTOCOL" <<'PY'
import json
import sys

role, nic, protocol = sys.argv[1:]
print(json.dumps({
    "kv_connector": "MooncakeConnector",
    "kv_role": f"kv_{role}",
    "kv_connector_extra_config": {
        "mooncake_protocol": protocol,
        "device_name": nic,
        "transfer_engine_factory": "orbitkv.vllm.transport.TentTransferEngine",
    },
}))
PY
}

DECODE_CONFIG="$(kv_config consumer "${DECODE_NIC:-}")"
PREFILL_CONFIG="$(kv_config producer "${PREFILL_NIC:-}")"
pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    echo "Logs: $LOG_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env CUDA_VISIBLE_DEVICES="$DECODE_GPU" "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --host 127.0.0.1 --port "$DECODE_PORT" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
    --kv-transfer-config "$DECODE_CONFIG" > "$LOG_DIR/decode.log" 2>&1 &
pids+=("$!")
setsid env CUDA_VISIBLE_DEVICES="$PREFILL_GPU" "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --host 127.0.0.1 --port "$PREFILL_PORT" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
    --kv-transfer-config "$PREFILL_CONFIG" > "$LOG_DIR/prefill.log" 2>&1 &
pids+=("$!")

for index in 0 1; do
    if [[ "$index" == 0 ]]; then name=decode; port="$DECODE_PORT"; else name=prefill; port="$PREFILL_PORT"; fi
    deadline=$((SECONDS + STARTUP_TIMEOUT))
    until curl --max-time 2 -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        if ! kill -0 "${pids[$index]}" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "$name failed to become ready; see $LOG_DIR/$name.log" >&2
            exit 1
        fi
        sleep 1
    done
    echo "$name ready: http://127.0.0.1:$port"
done

setsid "$ROUTER" --vllm-pd-disaggregation --kv-connector mooncake \
    --prefill "http://127.0.0.1:$PREFILL_PORT" "$VLLM_MOONCAKE_BOOTSTRAP_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --host 127.0.0.1 --port "$ROUTER_PORT" > "$LOG_DIR/router.log" 2>&1 &
pids+=("$!")
echo "Native P/D router starting: http://127.0.0.1:$ROUTER_PORT; logs: $LOG_DIR"
wait -n "${pids[@]}"
