#!/usr/bin/env python
"""Run all e2e test scripts and report results.

Usage:
    python scripts/e2e_all.py
"""

import subprocess
import sys
import time

SCRIPTS = [
    ("text_grid mode", "scripts/e2e_text_grid.py"),
    ("vision_grid mode", "scripts/e2e_vision_grid.py"),
    ("vision_drag mode", "scripts/e2e_vision_drag.py"),
    ("reward modes", "scripts/e2e_reward_modes.py"),
    ("max_steps cutoff", "scripts/e2e_max_steps.py"),
]


def main():
    results = []
    total_start = time.monotonic()

    for name, script in SCRIPTS:
        print(f"\n{'=' * 60}")
        print(f"  Running: {name} ({script})")
        print(f"{'=' * 60}\n")

        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=False,
        )
        elapsed = time.monotonic() - start
        passed = proc.returncode == 0
        results.append((name, passed, elapsed))

    total_elapsed = time.monotonic() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print("  E2E Test Summary")
    print(f"{'=' * 60}")
    all_passed = True
    for name, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        mark = "✓" if passed else "✗"
        print(f"  {mark} {name:30s} {status}  ({elapsed:.1f}s)")
        if not passed:
            all_passed = False

    print(f"\n  Total time: {total_elapsed:.1f}s")
    if all_passed:
        print("\n  All e2e tests PASSED")
    else:
        print("\n  Some e2e tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
