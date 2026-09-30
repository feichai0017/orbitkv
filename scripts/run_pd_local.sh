#!/usr/bin/env bash
# Two-GPU vLLM P/D example. The proxy runs on CPU.
# Usage: PREFILL_GPU=0 DECODE_GPU=1 scripts/run_pd_local.sh /path/to/model
set -euo pipefail

MODEL="${1:?Usage: scripts/run_pd_local.sh /path/to/model}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/vllm-release/bin/python}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PROXY_PORT="${PROXY_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8101}"
PREFILL_PORT="${PREFILL_PORT:-8102}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/orbitkv-pd.XXXXXX)}"
# The native TENT ABI treats any presence of MC_FORCE_TCP as forcing TCP.
case "${MC_FORCE_TCP:-1}" in
    1) export MC_FORCE_TCP=1 ;;
    0) unset MC_FORCE_TCP ;;
    *) echo "MC_FORCE_TCP must be 0 (RDMA) or 1 (TCP)" >&2; exit 2 ;;
esac
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

[[ -x "$PYTHON" ]] || { echo "Missing vLLM environment: $PYTHON" >&2; exit 1; }
[[ "$PREFILL_GPU" != "$DECODE_GPU" ]] || { echo "PREFILL_GPU and DECODE_GPU must differ" >&2; exit 1; }
[[ "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "STARTUP_TIMEOUT must be a positive number of seconds" >&2; exit 1; }
for command in setsid curl; do command -v "$command" >/dev/null; done
mkdir -p "$LOG_DIR"

kv_config() {
    "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys

role, nic = sys.argv[1:]
extra = {"orbitkv.pd.mooncake.bind_host": "127.0.0.1"}
if nic:
    extra["orbitkv.pd.mooncake.rank_map"] = {"0": {"nic": nic}}
print(json.dumps({
    "kv_connector": f"Pd{role.title()}Connector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "orbitkv.vllm.pd",
    "engine_id": role,
    "kv_connector_extra_config": extra,
}))
PY
}

DECODE_CONFIG="$(kv_config decode "${DECODE_NIC:-}")"
PREFILL_CONFIG="$(kv_config prefill "${PREFILL_NIC:-}")"
pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    # Only process groups created by this script are signalled.
    for _ in {1..50}; do
        live=0
        for pid in "${pids[@]}"; do kill -0 -- "-$pid" 2>/dev/null && live=1; done
        [[ "$live" == 0 ]] && break
        sleep 0.1
    done
    for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
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

setsid "$PYTHON" -m orbitkv.vllm.pd.proxy \
    --listen-host 127.0.0.1 --listen-port "$PROXY_PORT" \
    --prefill-url "http://127.0.0.1:$PREFILL_PORT" \
    --decode-url "http://127.0.0.1:$DECODE_PORT" \
    --timeout-s 600 > "$LOG_DIR/proxy.log" 2>&1 &
pids+=("$!")
echo "P/D proxy starting: http://127.0.0.1:$PROXY_PORT/v1/chat/completions; logs: $LOG_DIR"
# Any service exit ends this deployment and drains the other process groups.
wait -n "${pids[@]}"
