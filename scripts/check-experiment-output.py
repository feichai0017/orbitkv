#!/usr/bin/env python3
"""Reject tracked files in reserved experiment-output directories, even if force-added."""

import subprocess
from pathlib import PurePosixPath

# These are output locations, not filename heuristics. In particular, test
# fixtures and source modules named result(s) remain valid tracked inputs.
OUTPUT_DIRECTORIES = (
    PurePosixPath("benches/results"),
    PurePosixPath("benches/runs"),
    PurePosixPath("results"),
    PurePosixPath("runs"),
)


def main() -> int:
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    rejected = [
        name
        for name in tracked
        if name
        and any(PurePosixPath(name).is_relative_to(path) for path in OUTPUT_DIRECTORIES)
    ]
    if rejected:
        print(
            "Generated experiment output must be archived outside the source checkout:"
        )
        print("\n".join(f"  {name}" for name in rejected))
        return 1
    print("Tracked-file experiment-output check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
