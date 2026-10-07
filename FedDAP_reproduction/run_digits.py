"""Real cached Digits engineering smoke; deliberately NOT a paper-results CLI."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from core import (aggregate_states, alignment_losses, attention_aggregate,
                  extract_prototypes)

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("MNIST", "MNIST_M", "SVHN")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backbone():
    # This tracked F2DC file is executable-AST identical to the FedDAP upstream
    # backbone (only two comments and whitespace differ), verified in test_core.
    source = ROOT / "F2DC/backbone/ResNet.py"
    spec = importlib.util.spec_from_file_location("feddap_upstream_resnet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resnet10(10), source


class CachedDigits(Dataset):
    """Read ONLY the user's trusted existing FedPLVM cache; transform once."""
    def __init__(self, source, indices, augment=False):
        # Existing cache uses NumPy's object container; never load untrusted pkl.
        images, labels = np.load(source, allow_pickle=True)
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.indices = np.asarray(indices, dtype=np.int64)
        if len(images) != len(self.labels) or len(self.indices) == 0:
            raise ValueError("Malformed/empty cache")
        if self.indices.min() < 0 or self.indices.max() >= len(images):
            raise ValueError("Invalid sample indices")
        if not np.isin(self.labels, np.arange(10)).all():
            raise ValueError("Digits labels must be in [0,9]")
        steps = [transforms.Resize((32, 32))]
        if augment:
            steps += [transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip()]
        steps += [transforms.ToTensor(),
                  transforms.Normalize((0.485, 0.456, 0.406),
                                       (0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        array = np.asarray(self.images[index])
        if array.dtype != np.uint8:
            raise ValueError(f"Expected original uint8 image, got {array.dtype}")
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        image = Image.fromarray(array).convert("RGB")
        return self.transform(image), int(self.labels[index])


def loaders(args):
    """Seeded disjoint per-domain subsets; record every assigned raw index."""
    rng = np.random.default_rng(args.seed)
    train, proto, test, manifest = [], [], [], {"clients": [], "test": []}
    for domain in DOMAINS:
        train_file = Path(args.data_root) / domain / "partitions/train_part0.pkl"
        test_file = Path(args.data_root) / domain / "test.pkl"
        train_images, _ = np.load(train_file, allow_pickle=True)
        test_images, _ = np.load(test_file, allow_pickle=True)
        required = args.clients_per_domain * args.train_samples
        if required > len(train_images) or args.test_samples > len(test_images):
            raise ValueError(f"Requested subset exceeds available {domain} data")
        selected = rng.choice(len(train_images), required, replace=False)
        for client in range(args.clients_per_domain):
            indices = selected[client * args.train_samples:(client + 1) * args.train_samples]
            generator = torch.Generator().manual_seed(args.seed + len(train))
            train.append(DataLoader(CachedDigits(train_file, indices, True),
                                    batch_size=args.batch_size, shuffle=True,
                                    generator=generator, num_workers=0))
            proto.append(DataLoader(CachedDigits(train_file, indices),
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=0))
            manifest["clients"].append({
                "domain": domain, "source": str(train_file),
                "sha256": sha256(train_file), "indices": indices.tolist(),
                "count": len(indices),
            })
        indices = rng.choice(len(test_images), args.test_samples, replace=False)
        test.append(DataLoader(CachedDigits(test_file, indices),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=0))
        manifest["test"].append({
            "domain": domain, "source": str(test_file),
            "sha256": sha256(test_file), "indices": indices.tolist(),
            "count": len(indices),
        })
    return train, proto, test, manifest


def cpu_state(net):
    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


def json_write(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def save_npz(path, **arrays):
    # Exclusive creation even if filename already exists after a failed run.
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **arrays)


def pack_prototypes(prototypes):
    keys = sorted(prototypes)
    return {
        "prototype_classes": np.array([key[0] for key in keys], dtype=np.int64),
        "prototype_domains": np.array([key[1] for key in keys]),
        "prototype_values": np.stack([prototypes[key].cpu().numpy() for key in keys]),
    }


@torch.no_grad()
def evaluate(net, test, device):
    net.eval()
    correct, counts, features, labels, domains = [], [], [], [], []
    for domain, loader in zip(DOMAINS, test):
        hits = count = 0
        for images, target in loader:
            feature = net.features(images.to(device))
            prediction = net.classifier(feature).argmax(1).cpu()
            hits += int((prediction == target).sum())
            count += len(target)
            features.append(feature.cpu().numpy())
            labels.append(target.numpy())
            domains.extend([domain] * len(target))
        correct.append(hits)
        counts.append(count)
    accuracy = 100 * np.asarray(correct) / np.asarray(counts)
    return accuracy, {
        "test_correct": np.array(correct), "test_count": np.array(counts),
        "features": np.concatenate(features), "labels": np.concatenate(labels),
        "feature_domains": np.array(domains),
    }


def gradient_norm(loss, features):
    value = torch.autograd.grad(loss, features, retain_graph=True)[0]
    return float(value.norm().detach())


def validate_args(args):
    for field in ("rounds", "local_epochs", "clients_per_domain", "train_samples",
                  "test_samples", "batch_size"):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.rounds < 2:
        raise ValueError("Require >=2 rounds so DPA/CPCL are actually exercised")
    if args.clients_per_domain < 3:
        raise ValueError("Require >=3 clients/domain to exercise nontrivial attention")
    if args.batch_size < 2 or args.train_samples < args.batch_size:
        raise ValueError("BatchNorm training requires a usable batch")
    if args.lr <= 0 or args.tau_cross <= 0 or args.tau_agg <= 0:
        raise ValueError("Learning rate and temperatures must be positive")
    if args.lambda_dpa <= 0 or args.lambda_cpcl <= 0:
        raise ValueError("Smoke must exercise BOTH alignment objectives")


def _run_reserved(args, output):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, no silent CPU fallback")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    net, source = backbone()
    device = torch.device(args.device)
    if device.type == "cuda":
        # Small smoke on a shared host: cap THIS process's torch allocator only.
        # Never stop other users' jobs; one client model is used at a time.
        torch.cuda.set_per_process_memory_fraction(0.10, device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes < 4 * 1024 ** 3:
            raise RuntimeError("Shared GPU has <4GiB free; do not start training")
    else:
        free_bytes = total_bytes = None
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    meta = {
        "experiment": "EXP-155", "status": "starting", "args": vars(args),
        "revision": revision, "protocol": "digits-engineering-smoke-NOT-paper-results",
        "domains": DOMAINS, "backbone": "upstream-compatible ResNet10 nf64, scratch",
        "backbone_source": str(source), "backbone_sha256": sha256(source),
        "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "cuda": torch.version.cuda, "device": str(device),
        "gpu_free_bytes_at_start": free_bytes, "gpu_total_bytes": total_bytes,
        "torch_allocator_fraction_cap": 0.10 if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "prototype_pass": "final encoder, eval BN, deterministic transform, all local samples",
        "dpa_reduction": "batch mean, missing anchors zero (upstream convention)",
        "integer_buffers": "maximum counter; floating BN buffers sample-weighted",
        "privacy_noise": False, "first_round": "CE-only, initially empty prototype dictionary",
    }
    json_write(output / "meta.json", meta)
    started = time.monotonic()
    history, prototypes = [], {}
    try:
        train, proto, test, manifest = loaders(args)
        json_write(output / "data_manifest.json", manifest)
        domains = [item["domain"] for item in manifest["clients"]]
        counts = [len(loader.dataset) for loader in train]
        net.to(device)
        state = cpu_state(net)
        best = -float("inf")
        for round_id in range(1, args.rounds + 1):
            states, local_prototypes, prototype_counts, client_records = [], [], [], []
            old_prototypes = prototypes
            reference_prototypes = {key: value.to(device) for key, value in old_prototypes.items()}
            for client, loader in enumerate(train):
                net.load_state_dict(state)
                net.train()
                optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                            momentum=0.9, weight_decay=1e-5)
                totals = np.zeros(3)
                seen = valid_dpa = valid_cpcl = 0
                dpa_grad = cpcl_grad = None
                for epoch in range(args.local_epochs):
                    for images, labels in loader:
                        images, labels = images.to(device), labels.to(device)
                        features = net.features(images)
                        ce = F.cross_entropy(net.classifier(features), labels)
                        dpa, cpcl, ndpa, ncpcl = alignment_losses(
                            features, labels, domains[client], reference_prototypes, args.tau_cross
                        )
                        if dpa_grad is None:
                            dpa_grad = gradient_norm(dpa, features)
                            cpcl_grad = gradient_norm(cpcl, features)
                        loss = ce + args.lambda_dpa * dpa + args.lambda_cpcl * cpcl
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite objective")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if any(parameter.grad is not None and
                               not torch.isfinite(parameter.grad).all()
                               for parameter in net.parameters()):
                            raise FloatingPointError("Nonfinite model gradient")
                        optimizer.step()
                        totals += np.array([ce.item(), dpa.item(), cpcl.item()]) * len(labels)
                        seen += len(labels)
                        valid_dpa += ndpa
                        valid_cpcl += ncpcl
                # Crucially AFTER the final optimizer step, not stale features
                # accumulated inside the final training epoch.
                local, local_counts = extract_prototypes(
                    net, proto[client], domains[client], device
                )
                if sum(local_counts.values()) != counts[client]:
                    raise AssertionError("Prototype pass dropped local samples")
                local_prototypes.append(local)
                prototype_counts.append(local_counts)
                states.append(cpu_state(net))
                client_records.append({
                    "client": client, "domain": domains[client], "samples": seen,
                    "ce": totals[0] / seen, "dpa": totals[1] / seen,
                    "cpcl": totals[2] / seen, "valid_dpa": valid_dpa,
                    "valid_cpcl": valid_cpcl, "dpa_feature_grad": dpa_grad,
                    "cpcl_feature_grad": cpcl_grad,
                })
                print(json.dumps({"round": round_id, **client_records[-1]}), flush=True)
            prototypes, attention = attention_aggregate(local_prototypes, args.tau_agg)
            state, weights = aggregate_states(states, counts)
            net.load_state_dict(state)
            accuracy, evaluated = evaluate(net, test, device)
            average = float(accuracy.mean())
            record = {
                "round": round_id, "avg_accuracy": average,
                "domain_accuracy": dict(zip(DOMAINS, accuracy.tolist())),
                "global_prototype_count": len(prototypes),
                "previous_prototype_count": len(old_prototypes),
                "fedavg_weights": weights, "clients": client_records,
                "attention": {f"{key[0]}:{key[1]}": value for key, value in attention.items()},
                "prototype_counts": [{f"{key[0]}:{key[1]}": value
                                      for key, value in item.items()}
                                     for item in prototype_counts],
                "elapsed_seconds": time.monotonic() - started,
            }
            round_arrays = {
                "round": round_id, "domain_accuracy": accuracy, "avg_accuracy": average,
                "test_correct": evaluated["test_correct"], "test_count": evaluated["test_count"],
                "fedavg_weights": np.array(weights), "client_sample_counts": np.array(counts),
                "client_domains": np.array(domains), "record_json": json.dumps(record),
                **pack_prototypes(prototypes),
            }
            save_npz(output / f"round_{round_id:03d}.npz", **round_arrays)
            with open(output / "proto_logs.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            history.append(record)
            snapshot = {
                **round_arrays, **evaluated,
                **{f"model::{key}": value.numpy() for key, value in state.items()},
                **{f"local_proto::{client}::{key[0]}::{key[1]}": value.numpy()
                   for client, item in enumerate(local_prototypes)
                   for key, value in item.items()},
            }
            if average > best:
                best = average
                save_npz(output / f"best_R{round_id:03d}.npz", **snapshot)
            if round_id == args.rounds:
                save_npz(output / f"final_R{round_id:03d}.npz", **snapshot)
            print("ROUND_COMPLETE " + json.dumps({
                "round": round_id, "accuracy": record["domain_accuracy"],
                "avg": average, "prototypes": len(prototypes)}), flush=True)
        second = history[1]["clients"]
        if any(item["valid_dpa"] == 0 or item["valid_cpcl"] == 0 or
               item["dpa_feature_grad"] <= 0 or item["cpcl_feature_grad"] <= 0
               for item in second):
            raise AssertionError("Second round did not exercise both losses/gradients")
        summary = {
            "status": "completed", "revision": revision,
            "rounds": len(history), "seed": args.seed, "domains": DOMAINS,
            "best_avg": max(item["avg_accuracy"] for item in history),
            "last_avg": history[-1]["avg_accuracy"],
            "last5_mean": (float(np.mean([item["avg_accuracy"] for item in history[-5:]]))
                           if len(history) >= 5 else None),
            "elapsed_seconds": time.monotonic() - started, "paper_result_reproduced": False,
            "peak_torch_gpu_bytes": (torch.cuda.max_memory_allocated(device)
                                     if device.type == "cuda" else None),
            "second_round_nonzero_dpa_cpcl_gradients_all_clients": True,
            "diag": str(output),
        }
        json_write(output / "summary.json", summary)
        print("SMOKE_PASS " + json.dumps(summary), flush=True)
    except BaseException as error:
        json_write(output / "failure.json", {
            "error": repr(error), "completed_rounds": len(history),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


def run(args):
    validate_args(args)
    # No overwrite, including failures. Reserve before environment/data setup.
    output = Path(args.dump_diag).resolve()
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "attempt.json", {"args": vars(args), "experiment": "EXP-155"})
    try:
        _run_reserved(args, output)
    except BaseException as error:
        if not (output / "failure.json").exists():
            json_write(output / "failure.json", {
                "error": repr(error), "phase": "environment-or-metadata-setup",
                "completed_rounds": 0,
            })
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--data-root", required=True)
    result.add_argument("--dump-diag", required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=2)
    result.add_argument("--rounds", type=int, default=2)
    result.add_argument("--local-epochs", type=int, default=1)
    result.add_argument("--clients-per-domain", type=int, default=3)
    result.add_argument("--train-samples", type=int, default=128)
    result.add_argument("--test-samples", type=int, default=128)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--lr", type=float, default=0.01)
    result.add_argument("--lambda-dpa", type=float, default=1)
    result.add_argument("--lambda-cpcl", type=float, default=1)
    result.add_argument("--tau-cross", type=float, default=0.02)
    result.add_argument("--tau-agg", type=float, default=0.001)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
