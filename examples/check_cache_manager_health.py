"""Check that the node-local Cache Manager is reachable."""

from __future__ import annotations

import argparse
import sys

from orbitkv import ChannelClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Cache Manager health")
    parser.add_argument(
        "--socket",
        default="/tmp/orbitkv-50055.sock",
        help="Cache Manager bootstrap Unix socket (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = ChannelClient(args.socket)
    ok, message = client.health()
    client.close()
    status = "healthy" if ok else "unhealthy"
    print(f"Cache Manager {status}.")
    if message:
        print(f"Message: {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
