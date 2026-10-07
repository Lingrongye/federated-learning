"""Download pinned original files; read all four Digits domains without pickle."""
import argparse
import bz2
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DOMAINS = ("MNIST", "USPS", "SVHN", "SYN")
SYN_REVISION = "91ff02225a73db330883843bc41ee3aca59a19d9"
RESOURCES = [
    ("MNIST/train-images-idx3-ubyte.gz",
     ["https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"],
     "md5", "f68b3c2dcbeaaa9fbdd348bbdeb94873", None),
    ("MNIST/train-labels-idx1-ubyte.gz",
     ["https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"],
     "md5", "d53e105ee54ea40749a09fcbcd1e9432", None),
    ("MNIST/t10k-images-idx3-ubyte.gz",
     ["https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"],
     "md5", "9fb629c4189551a2d022fa330f9573f3", None),
    ("MNIST/t10k-labels-idx1-ubyte.gz",
     ["https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"],
     "md5", "ec29112dd5afa0611ce80d1b7f02629c", None),
    ("USPS/usps.bz2",
     ["https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"],
     "md5", "ec16c51db3855ca6c91edd34d0e9b197", None),
    ("USPS/usps.t.bz2",
     ["https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2"],
     "md5", "8ea070ee2aca1ac39742fdd1ef5ed118", None),
    ("SVHN/train_32x32.mat",
     ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat"],
     "md5", "e26dedcc434d2e4c54c9b2d4a06d8373", None),
    ("SVHN/test_32x32.mat",
     ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat"],
     "md5", "eb5a983be6a315427106f1b164d9cef3", None),
    ("SYN/synth_train_32x32.mat",
     [f"https://media.githubusercontent.com/media/domainadaptation/datasets/{SYN_REVISION}/synth/synth_train_32x32.mat"],
     "sha256", "e0e0924ecd0c0b5a55ed6485aaa7d39937af982397717b89a3d91777aca9433c", 895578032),
    ("SYN/synth_test_32x32.mat",
     [f"https://media.githubusercontent.com/media/domainadaptation/datasets/{SYN_REVISION}/synth/synth_test_32x32.mat"],
     "sha256", "6285fe9a1c6f27b379a47263907a37899b59ce86948fd343a5b904435cde5bcc", 17851156),
]


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def resource_paths(domain, train):
    split = "train" if train else "test"
    if domain == "MNIST":
        prefix = "train" if train else "t10k"
        return [f"MNIST/{prefix}-images-idx3-ubyte.gz",
                f"MNIST/{prefix}-labels-idx1-ubyte.gz"]
    if domain == "USPS":
        return [f"USPS/{'usps.bz2' if train else 'usps.t.bz2'}"]
    if domain in ("SVHN", "SYN"):
        prefix = "synth_" if domain == "SYN" else ""
        return [f"{domain}/{prefix}{split}_32x32.mat"]
    raise ValueError(f"Unknown domain: {domain}")


def read_arrays(root, domain, train):
    paths = [Path(root) / name for name in resource_paths(domain, train)]
    if domain == "MNIST":
        with gzip.open(paths[0], "rb") as stream:
            magic, count, height, width = struct.unpack(">IIII", stream.read(16))
            if magic != 2051 or (height, width) != (28, 28):
                raise ValueError("Invalid MNIST image header")
            images = np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, height, width)
        with gzip.open(paths[1], "rb") as stream:
            magic, count = struct.unpack(">II", stream.read(8))
            if magic != 2049:
                raise ValueError("Invalid MNIST label header")
            labels = np.frombuffer(stream.read(), dtype=np.uint8).astype(np.int64)
            if len(labels) != count:
                raise ValueError("MNIST label count mismatch")
    elif domain == "USPS":
        with bz2.open(paths[0], "rt") as stream:
            rows = [line.split() for line in stream]
        pixels = np.array([[float(pair.split(":")[1]) for pair in row[1:]]
                           for row in rows], dtype=np.float32)
        if not np.isfinite(pixels).all() or np.any(np.abs(pixels) > 1):
            raise ValueError("Invalid USPS pixel range")
        images = ((pixels.reshape(-1, 16, 16) + 1) / 2 * 255).astype(np.uint8)
        # Exactly torchvision USPS: LIBSVM classes 1..10 become digits 0..9.
        labels = np.array([int(row[0]) - 1 for row in rows], dtype=np.int64)
    else:
        from scipy.io import loadmat
        item = loadmat(paths[0])
        source = item["X"]
        if source.dtype != np.uint8 or source.shape[:3] != (32, 32, 3):
            raise ValueError("Expected uint8 MAT X[32,32,3,N]")
        images = source.transpose(3, 0, 1, 2)
        labels = np.asarray(item["y"], dtype=np.int64).reshape(-1)
        # Both published MAT datasets use SVHN's 10 for digit zero.
        labels = np.where(labels == 10, 0, labels)
    if len(images) != len(labels) or not len(labels):
        raise ValueError("Empty/mismatched images and labels")
    if not np.isin(labels, np.arange(10)).all():
        raise ValueError("Labels outside digits 0..9")
    return images, labels


class RawDigits(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images, self.labels = images, labels
        self.indices = np.asarray(indices, dtype=np.int64)
        if not len(self.indices) or self.indices.min() < 0 or self.indices.max() >= len(labels):
            raise ValueError("Invalid original-data indices")
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        return self.transform(Image.fromarray(self.images[index]).convert("RGB")), int(self.labels[index])


def verify_sources(root):
    root = Path(root)
    manifest = json.loads((root / "download_manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Original download did not complete")
    records = {item["relative"]: item for item in manifest["files"]}
    if set(records) != {item[0] for item in RESOURCES}:
        raise ValueError("Source manifest does not contain all originals")
    for name, _, algorithm, expected, size in RESOURCES:
        path = root / name
        record = records[name]
        if digest(path, algorithm) != expected or digest(path) != record["sha256"]:
            raise ValueError(f"Original source checksum mismatch: {name}")
        if path.stat().st_size != record["bytes"] or (size and path.stat().st_size != size):
            raise ValueError(f"Original source size mismatch: {name}")
    return manifest


def download(root):
    """Reserve a NEW root. Interrupted partial files are retained, never reused."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    files = []
    with open(root / "download_events.jsonl", "x", encoding="utf-8") as events:
        def event(value):
            events.write(json.dumps(value) + "\n")
            events.flush()
            os.fsync(events.fileno())
            print(json.dumps(value), flush=True)
        try:
            for name, urls, algorithm, expected, size in RESOURCES:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                succeeded = False
                for attempt, url in enumerate(urls, 1):
                    partial = destination.with_name(destination.name + f".part_attempt{attempt}")
                    event({"file": name, "url": url, "status": "downloading"})
                    try:
                        request = urllib.request.Request(url, headers={"User-Agent": "FedDAP-reproduction/1"})
                        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "xb") as stream:
                            final_url = response.url
                            received = 0
                            next_report = 32 * 1024 ** 2
                            while chunk := response.read(1024 * 1024):
                                stream.write(chunk)
                                received += len(chunk)
                                if received >= next_report:
                                    event({"file": name, "received_bytes": received})
                                    next_report += 32 * 1024 ** 2
                            stream.flush()
                            os.fsync(stream.fileno())
                        if digest(partial, algorithm) != expected or (size and received != size):
                            raise ValueError(f"Checksum/size mismatch: {name}")
                        # Link is exclusive, unlike replace(). Keep original transfer evidence.
                        os.link(partial, destination)
                        record = {"relative": name, "requested_url": url, "resolved_url": final_url,
                                  "checksum_algorithm": algorithm, "expected_checksum": expected,
                                  "sha256": digest(destination), "bytes": received}
                        files.append(record)
                        event({**record, "status": "verified"})
                        succeeded = True
                        break
                    except Exception as error:
                        event({"file": name, "url": url, "status": "failed", "error": repr(error)})
                if not succeeded:
                    raise RuntimeError(f"No verified download available: {name}")
            counts = {}
            for domain in DOMAINS:
                counts[domain] = {}
                for train in (True, False):
                    images, labels = read_arrays(root, domain, train)
                    counts[domain]["train" if train else "test"] = {
                        "count": len(labels), "image_shape": list(images.shape[1:]),
                        "label_counts": np.bincount(labels, minlength=10).tolist()}
            manifest = {"status": "completed", "files": files, "splits": counts,
                        "revision": subprocess.check_output(
                            ["git", "-C", str(Path(__file__).resolve().parents[1]),
                             "rev-parse", "HEAD"], text=True).strip(),
                        "elapsed_seconds": time.monotonic() - started,
                        "synth_mirror_revision": SYN_REVISION,
                        "synth_provenance_limit": "Research-library full MAT mirror; author Drive link unavailable; not proven identical to FedDAP's unprovided ImageFolder conversion"}
            with open(root / "download_manifest.json", "x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
            event({"status": "DOWNLOAD_PASS", "splits": counts})
        except BaseException as error:
            with open(root / "download_failure.json", "x", encoding="utf-8") as stream:
                json.dump({"error": repr(error), "verified_files": files}, stream, indent=2)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--background-log-dir")
    args = parser.parse_args()
    if args.background_log_dir:
        log_dir = Path(args.background_log_dir)
        log_dir.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", args.root]
        with open(log_dir / "download.log", "xb") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        with open(log_dir / "launch.json", "x", encoding="utf-8") as stream:
            json.dump({"pid": child.pid, "command": command}, stream, indent=2)
        print(json.dumps({"pid": child.pid, "log_dir": str(log_dir)}), flush=True)
    else:
        download(args.root)
