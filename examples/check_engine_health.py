#!/usr/bin/env python3
"""Simple helper to verify the gRPC engine server is reachable."""

from __future__ import annotations

import argparse
import sys

from orbitkv import EngineRpcClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check engine-server health RPC")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:50055",
        help="gRPC endpoint for the engine server (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = EngineRpcClient(args.endpoint)
    ok, message = client.health()
    status = "healthy" if ok else "unhealthy"
    print(f"Engine server {status}.")
    if message:
        print(f"Message: {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
