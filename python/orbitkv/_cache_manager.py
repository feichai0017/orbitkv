#!/usr/bin/env python3
"""Launch the OrbitKV Cache Manager bundled with the wheel."""

import subprocess
import sys

from orbitkv._bin_utils import binary_env, find_binary

_BINARY = "orbitkv-cache-manager-py"


def get_cache_manager_binary() -> str:
    return find_binary(_BINARY)


def main():
    binary = get_cache_manager_binary()
    try:
        result = subprocess.run([binary] + sys.argv[1:], check=False, env=binary_env())
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Error: {_BINARY} binary not found at {binary}", file=sys.stderr)
        print(
            "Run `cargo build -r --bin orbitkv-cache-manager-py` or reinstall orbitkv.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
