"""Start one detached real smoke from an immutable JSON configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    root = Path(__file__).resolve().parents[1]
    diag = Path(config["arguments"]["dump_diag"]).resolve()
    if diag.exists():
        raise FileExistsError(f"Refusing to reuse any diagnostic directory: {diag}")
    for domain in ("MNIST", "MNIST_M", "SVHN"):
        for relative in ("partitions/train_part0.pkl", "test.pkl"):
            path = Path(config["arguments"]["data_root"]) / domain / relative
            if not path.is_file():
                raise FileNotFoundError(path)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(root / "FedDAP_reproduction/run_digits.py")]
    for key, value in config["arguments"].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    environment = os.environ.copy()
    environment.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with open(output / "train.log", "xb") as log:
        child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    with open(output / "launch.json", "x", encoding="utf-8") as stream:
        json.dump({"pid": child.pid, "command": command, "config": config,
                   "revision": subprocess.check_output(
                       ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()},
                  stream, indent=2)
    print(json.dumps({"pid": child.pid, "output": str(output), "command": command}), flush=True)


if __name__ == "__main__":
    main()
