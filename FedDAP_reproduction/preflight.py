"""Read-only real cache and environment inspection BEFORE reserving a run."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import torch
import torchvision

from run_digits import CachedDigits, DOMAINS, sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    files = []
    for domain in DOMAINS:
        for split, name in (("train", "partitions/train_part0.pkl"), ("test", "test.pkl")):
            source = Path(args.data_root) / domain / name
            images, labels = np.load(source, allow_pickle=True)
            data = CachedDigits(source, [0])
            image, label = data[0]
            assert image.shape == (3, 32, 32) and torch.isfinite(image).all()
            files.append({
                "domain": domain, "split": split, "source": str(source),
                "image_shape": np.asarray(images).shape, "dtype": str(np.asarray(images).dtype),
                "count": len(images), "labels": np.unique(labels).tolist(),
                "sample_shape_after_transform": list(image.shape),
                "sha256": sha256(source),
            })
    print(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "files": files,
    }, indent=2), flush=True)
