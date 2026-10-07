#!/usr/bin/env bash

set -euo pipefail

wrapper="${RUSTC_WRAPPER:-}"
if [[ -z "$wrapper" ]]; then
    rustc -vV
    exit 0
fi

probe_log="${RUNNER_TEMP:-/tmp}/orbitkv-sccache-probe.log"
set +e
timeout 30 "$wrapper" rustc -vV >"$probe_log" 2>&1
probe_status=$?
set -e
if [[ "$probe_status" -eq 0 ]]; then
    cat "$probe_log"
    exit 0
fi

cat "$probe_log" >&2
echo "::warning::Rust cache probe failed with exit $probe_status; using direct rustc"
if [[ -z "${GITHUB_ENV:-}" ]]; then
    echo "GITHUB_ENV is required to disable the wrapper for later CI steps" >&2
    exit 2
fi
printf 'RUSTC_WRAPPER=\nSCCACHE_GHA_ENABLED=false\n' >>"$GITHUB_ENV"
RUSTC_WRAPPER= rustc -vV
