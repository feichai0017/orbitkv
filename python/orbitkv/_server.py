#!/usr/bin/env python3
"""Wrapper script to launch orbitkv-server binary."""

import subprocess
import sys

from orbitkv._bin_utils import binary_env, find_binary

_BINARY = "orbitkv-server-py"


def main():
    binary = find_binary(_BINARY)
    try:
        result = subprocess.run([binary] + sys.argv[1:], check=False, env=binary_env())
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Error: {_BINARY} binary not found at {binary}", file=sys.stderr)
        print(
            "Run `cargo build -r --bin orbitkv-server-py` or reinstall orbitkv.", file=sys.stderr
        )
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
