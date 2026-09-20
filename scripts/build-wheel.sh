#!/bin/bash
# Automated build script for orbitkv Python package with embedded binary
# Usage: ./scripts/build-wheel.sh [--release]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/python"

# Parse arguments
RELEASE_FLAG=""
PROFILE="debug"
CARGO_FEATURES=""
if [[ "$1" == "--release" ]]; then
    RELEASE_FLAG="--release"
    PROFILE="release"
    shift
fi

# Remaining args are feature flags (e.g. --no-default-features --features cuda-13)
EXTRA_ARGS=("$@")

echo "==> Building binaries ($PROFILE mode)..."
cd "$PROJECT_ROOT"
cargo build $RELEASE_FLAG "${EXTRA_ARGS[@]}" -p orbitkv-py --bin orbitkv-server-py --bin orbitkv-metaserver-py

echo "==> Copying binaries to Python package..."
for bin in orbitkv-server-py orbitkv-metaserver-py; do
    cp "$PROJECT_ROOT/target/$PROFILE/$bin" "$PYTHON_DIR/orbitkv/$bin"
    chmod +x "$PYTHON_DIR/orbitkv/$bin"
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
for bin in orbitkv-server-py orbitkv-metaserver-py; do
    patchelf --set-rpath '$ORIGIN' "$PYTHON_DIR/orbitkv/$bin"
done

echo "==> Building Python wheel with maturin..."
cd "$PYTHON_DIR"
if command -v maturin >/dev/null 2>&1; then
    maturin build $RELEASE_FLAG "${EXTRA_ARGS[@]}"
else
    uvx maturin build $RELEASE_FLAG "${EXTRA_ARGS[@]}"
fi

echo ""
echo "==> Done! Wheel built at:"
WHEEL="$(find "$PROJECT_ROOT/target/wheels" -maxdepth 1 -type f \
    -name 'orbitkv*.whl' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$WHEEL" ]]; then
    echo "Built wheel was not found under target/wheels" >&2
    exit 1
fi
ls -lh "$WHEEL"
echo ""
echo "To install: pip install $WHEEL"
