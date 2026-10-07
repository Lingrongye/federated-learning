"""Verified original-source adapters; keep the training/core algorithm unchanged."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from raw_data import DOMAINS, RawDigits, digest, read_arrays, resource_paths, verify_sources


def transform(augment):
    steps = [transforms.Resize((32, 32))]
    if augment:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    return transforms.Compose(steps + [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def loaders(args):
    provenance = verify_sources(args.data_root)
    rng = np.random.default_rng(args.seed)
    train, proto, test = [], [], []
    manifest = {"clients": [], "test": [], "data_format": "raw",
                "download_manifest": provenance,
                "download_manifest_sha256": digest(Path(args.data_root) / "download_manifest.json"),
                "split_policy": "published train/test files; seeded disjoint per-domain client subsets"}
    for domain in DOMAINS:
        images, labels = read_arrays(args.data_root, domain, True)
        required = args.clients_per_domain * args.train_samples
        if required > len(labels):
            raise ValueError(f"Training request exceeds original {domain} split")
        selected = rng.choice(len(labels), required, replace=False)
        sources = [str(Path(args.data_root) / name) for name in resource_paths(domain, True)]
        for client in range(args.clients_per_domain):
            indices = selected[client * args.train_samples:(client + 1) * args.train_samples]
            generator = torch.Generator().manual_seed(args.seed + len(train))
            train.append(DataLoader(RawDigits(images, labels, indices, transform(True)),
                                    batch_size=args.batch_size, shuffle=True,
                                    generator=generator, num_workers=0))
            proto.append(DataLoader(RawDigits(images, labels, indices, transform(False)),
                                    batch_size=args.batch_size, shuffle=False, num_workers=0))
            manifest["clients"].append({
                "domain": domain, "source": sources, "split": "train",
                "indices": indices.tolist(), "count": len(indices),
                "source_count": len(labels),
                "label_counts": np.bincount(labels[indices], minlength=10).tolist(),
            })
        images, labels = read_arrays(args.data_root, domain, False)
        if args.test_samples > len(labels):
            raise ValueError(f"Testing request exceeds original {domain} split")
        indices = rng.choice(len(labels), args.test_samples, replace=False)
        test.append(DataLoader(RawDigits(images, labels, indices, transform(False)),
                               batch_size=args.batch_size, shuffle=False, num_workers=0))
        manifest["test"].append({
            "domain": domain, "split": "test",
            "source": [str(Path(args.data_root) / name) for name in resource_paths(domain, False)],
            "indices": indices.tolist(), "count": len(indices), "source_count": len(labels),
            "label_counts": np.bincount(labels[indices], minlength=10).tolist(),
        })
    return train, proto, test, manifest
