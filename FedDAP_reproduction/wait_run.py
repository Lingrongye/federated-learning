"""Bounded completion gate: a background PID is NOT successful completion."""
import argparse
import json
from pathlib import Path
import time


def wait(directory, kind="train", timeout=300, interval=2):
    if timeout <= 0 or interval <= 0 or kind not in ("train", "download"):
        raise ValueError("Require positive timeout/interval and known kind")
    root = Path(directory)
    success = root / ("summary.json" if kind == "train" else "download_manifest.json")
    failure = root / ("failure.json" if kind == "train" else "download_failure.json")
    deadline = time.monotonic() + timeout
    while True:
        # JSON files can be observed mid-write: retry until complete JSON exists.
        if failure.exists():
            try:
                detail = json.loads(failure.read_text())
            except json.JSONDecodeError:
                detail = None
            if detail is not None:
                raise RuntimeError(f"{kind} failed: {json.dumps(detail)}")
        if success.exists() and not failure.exists():
            try:
                result = json.loads(success.read_text())
            except json.JSONDecodeError:
                result = None
            if result is not None:
                if result.get("status") != "completed":
                    raise RuntimeError(f"{kind} has non-completed final status: {result.get('status')}")
                print(json.dumps({"completion_gate": "PASS", "kind": kind,
                                  "directory": str(root)}), flush=True)
                return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{kind} not confirmed complete within {timeout}s: {root}")
        time.sleep(min(interval, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--kind", choices=("train", "download"), default="train")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    wait(args.directory, args.kind, args.timeout)
