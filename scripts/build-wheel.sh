#!/bin/bash
# Automated build script for orbitkv Python package with embedded binary
# Usage: ./scripts/build-wheel.sh [--release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/python"
BUILD_PYTHON="$(command -v "${PYO3_PYTHON:-python3}")"
export PYO3_PYTHON="$BUILD_PYTHON"
VERSION="$("$BUILD_PYTHON" "$SCRIPT_DIR/check-versions.py")"
mkdir -p "$PROJECT_ROOT/target/wheels"
BUILD_DIR="$(mktemp -d "$PROJECT_ROOT/target/wheel-build.XXXXXX")"
MANIFEST_BACKUP=""
cleanup() {
    if [[ -n "$MANIFEST_BACKUP" ]]; then
        mv -f "$MANIFEST_BACKUP" "$PYTHON_DIR/pyproject.toml"
    fi
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

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
    "$BUILD_PYTHON" - "$PYTHON_DIR/pyproject.toml" <<'PY'
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
cargo build "${RELEASE_ARGS[@]}" "${EXTRA_ARGS[@]}" -p orbitkv-py --bin orbitkv-cache-manager-py

echo "==> Copying binaries to Python package..."
for bin in orbitkv-cache-manager-py; do
    install -m 755 "$PROJECT_ROOT/target/$PROFILE/$bin" "$PYTHON_DIR/orbitkv/$bin"
    if [[ "$PROFILE" == "release" ]]; then
        strip --strip-unneeded "$PYTHON_DIR/orbitkv/$bin"
    fi
done

echo "==> Copying Mooncake runtime libraries..."
rm -f "$PYTHON_DIR/orbitkv/libtransfer_engine.so"
MOONCAKE_VARIANT="cpu"
if [[ " ${EXTRA_ARGS[*]} " == *" cuda-12"* || " ${EXTRA_ARGS[*]} " == *" cuda-13"* || " ${EXTRA_ARGS[*]} " != *" --no-default-features "* ]]; then
    MOONCAKE_VARIANT="cuda"
fi
MOONCAKE_LINK_DIR="$PROJECT_ROOT/.orbitkv/mooncake/$MOONCAKE_VARIANT/lib"
if [[ ! -d "$MOONCAKE_LINK_DIR" ]]; then
    echo "Mooncake native runtime directory not found: $MOONCAKE_LINK_DIR" >&2
    exit 1
fi
for lib in libtent_shared.so libmooncake_common.so libasio.so; do
    cp "$MOONCAKE_LINK_DIR/$lib" "$PYTHON_DIR/orbitkv/$lib"
done
for bin in orbitkv-cache-manager-py; do
    patchelf --set-rpath '$ORIGIN' "$PYTHON_DIR/orbitkv/$bin"
done

echo "==> Building Python wheel with maturin..."
cd "$PYTHON_DIR"
if command -v maturin >/dev/null 2>&1; then
    maturin build --interpreter "$BUILD_PYTHON" --compatibility linux --out "$BUILD_DIR" "${RELEASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    uvx maturin build --interpreter "$BUILD_PYTHON" --compatibility linux --out "$BUILD_DIR" "${RELEASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi

echo "==> Repairing dependencies of every bundled ELF file..."
UNREPAIRED=("$BUILD_DIR"/*.whl)
if [[ ${#UNREPAIRED[@]} -ne 1 || ! -f "${UNREPAIRED[0]}" ]]; then
    echo "Expected exactly one newly built wheel" >&2
    exit 1
fi
if "$BUILD_PYTHON" -c 'import auditwheel, wheel' >/dev/null 2>&1; then
    "$BUILD_PYTHON" "$SCRIPT_DIR/repair-wheel.py" "${UNREPAIRED[0]}" "$BUILD_DIR/repaired"
else
    uv run --isolated --no-project --with 'auditwheel==6.8.2' --with 'patchelf>=0.14.5' --with wheel \
        python "$SCRIPT_DIR/repair-wheel.py" "${UNREPAIRED[0]}" "$BUILD_DIR/repaired"
fi
REPAIRED=("$BUILD_DIR/repaired"/*.whl)
if [[ ${#REPAIRED[@]} -ne 1 || ! -f "${REPAIRED[0]}" ]]; then
    echo "Expected exactly one repaired wheel" >&2
    exit 1
fi
WHEEL="$PROJECT_ROOT/target/wheels/$(basename "${REPAIRED[0]}")"
mv "${REPAIRED[0]}" "$WHEEL"
echo "==> Validating the repaired wheel..."
"$BUILD_PYTHON" "$SCRIPT_DIR/check-wheel.py" "$WHEEL" --variant "$VARIANT" --version "$VERSION" --install-smoke
ls -lh "$WHEEL"
echo ""
echo "To install: pip install $WHEEL"
