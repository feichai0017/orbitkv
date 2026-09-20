#!/bin/bash
# Automated build script for orbitkv Python package with embedded binary
# Usage: ./scripts/build-wheel.sh [--release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/python"

# Parse arguments
RELEASE_ARGS=()
PROFILE="debug"
if [[ "${1:-}" == "--release" ]]; then
    RELEASE_ARGS=(--release)
    PROFILE="release"
    shift
fi

# Remaining args are feature flags (e.g. --no-default-features --features cuda-13)
EXTRA_ARGS=("$@")
VARIANT="cu12"
if [[ " ${EXTRA_ARGS[*]} " == *" cuda-13"* ]]; then
    VARIANT="cu13"
fi

# PEP 621 package names are static. Build the CUDA 13 distribution from a
# temporary manifest edit, then restore the developer's original manifest.
if [[ "$VARIANT" == "cu13" ]]; then
    MANIFEST_BACKUP="$(mktemp "$PYTHON_DIR/pyproject.toml.XXXXXX")"
    cp -p "$PYTHON_DIR/pyproject.toml" "$MANIFEST_BACKUP"
    restore_manifest() {
        mv -f "$MANIFEST_BACKUP" "$PYTHON_DIR/pyproject.toml"
    }
    trap restore_manifest EXIT
    python3 - "$PYTHON_DIR/pyproject.toml" <<'PY'
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
source = manifest.read_text()
old = 'name = "orbitkv-llm"'
if source.count(old) != 1:
    raise SystemExit("expected exactly one orbitkv-llm package name")
manifest.write_text(source.replace(old, 'name = "orbitkv-llm-cu13"', 1))
PY
fi

echo "==> Building binaries ($PROFILE mode)..."
cd "$PROJECT_ROOT"
cargo build "${RELEASE_ARGS[@]}" "${EXTRA_ARGS[@]}" -p orbitkv-py --bin orbitkv-cache-manager-py --bin orbitkv-metaserver-py

echo "==> Copying binaries to Python package..."
for bin in orbitkv-cache-manager-py orbitkv-metaserver-py; do
    install -m 755 "$PROJECT_ROOT/target/$PROFILE/$bin" "$PYTHON_DIR/orbitkv/$bin"
    if [[ "$PROFILE" == "release" ]]; then
        strip --strip-unneeded "$PYTHON_DIR/orbitkv/$bin"
    fi
done

echo "==> Copying Mooncake runtime libraries..."
MOONCAKE_VARIANT="cpu"
if [[ " ${EXTRA_ARGS[*]} " == *" cuda-12"* || " ${EXTRA_ARGS[*]} " == *" cuda-13"* || " ${EXTRA_ARGS[*]} " != *" --no-default-features "* ]]; then
    MOONCAKE_VARIANT="cuda"
fi
MOONCAKE_LINK_DIR="$PROJECT_ROOT/.orbitkv/mooncake/$MOONCAKE_VARIANT/lib"
if [[ ! -d "$MOONCAKE_LINK_DIR" ]]; then
    echo "Mooncake native runtime directory not found: $MOONCAKE_LINK_DIR" >&2
    exit 1
fi
for lib in libtransfer_engine.so libmooncake_common.so libasio.so; do
    cp "$MOONCAKE_LINK_DIR/$lib" "$PYTHON_DIR/orbitkv/$lib"
done
for bin in orbitkv-cache-manager-py orbitkv-metaserver-py; do
    patchelf --set-rpath '$ORIGIN' "$PYTHON_DIR/orbitkv/$bin"
done

echo "==> Building Python wheel with maturin..."
cd "$PYTHON_DIR"
if command -v maturin >/dev/null 2>&1; then
    maturin build "${RELEASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    uvx maturin build "${RELEASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi

echo ""
echo "==> Done! Wheel built at:"
WHEEL="$(find "$PROJECT_ROOT/target/wheels" -maxdepth 1 -type f \
    -name 'orbitkv*.whl' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$WHEEL" ]]; then
    echo "Built wheel was not found under target/wheels" >&2
    exit 1
fi
python3 "$SCRIPT_DIR/check-wheel.py" "$WHEEL" --variant "$VARIANT" --install-smoke
ls -lh "$WHEEL"
echo ""
echo "To install: pip install $WHEEL"
