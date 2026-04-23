"""Analyze using exact file name matching."""
import re, os
import numpy as np

LOG_DIR = "/root/autodl-tmp/federated-learning/FDSE_CVPR25/task/PACS_c4/log"

# Find all logs from 19-44 (ablation runs)
all_files = os.listdir(LOG_DIR)
ablation_files = [f for f in all_files if f.startswith("2026-04-06-19-44")]

# Map experiment by algopara substring
v_map = {
    "V1: low weight": "1.0|0.01|0.5|",  # actually 0.1|0.01|0.5
    "V2: no HSIC": "0.5|0.0|0.5|",
    "V3: long warmup": "0.5|0.05|0.5|",
}
# Correct V1 pattern
v_map = {
    "V1: low weight (0.1|0.01|0.5)": "_0.1|0.01|0.5|",
    "V2: no HSIC (0.5|0.0|0.5)": "_0.5|0.0|0.5|",
    "V3: long warmup (0.5|0.05|0.5, w50)": "_0.5|0.05|0.5|",
}

original_files = {
    "FedDSA original": ("2026-04-06-09-50", "feddsa"),
    "FedAvg": ("2026-04-06-09-55", "fedavg"),
    "FedBN": ("2026-04-06-09-55", "fedbn"),
    "FedProto": ("2026-04-06-09-55", "fedproto"),
    "FDSE": ("2026-04-06-09-55", "fdse"),
}

def parse_log(path):
    metrics = {"mean": [], "std": [], "min": [], "max": [], "rounds": []}
    cur_round = 0
    with open(path) as f:
        for line in f:
            m = re.search(r"Round (\d+)", line)
            if m: cur_round = int(m.group(1))
            for k, p in [("mean", r"mean_local_test_accuracy\s+([\d.]+)"),
                         ("std", r"std_local_test_accuracy\s+([\d.]+)"),
                         ("min", r"min_local_test_accuracy\s+([\d.]+)"),
                         ("max", r"max_local_test_accuracy\s+([\d.]+)")]:
                m2 = re.search(p, line)
                if m2:
                    metrics[k].append(float(m2.group(1)))
                    if k == "mean":
                        metrics["rounds"].append(cur_round)
    return metrics

def find_file(prefix, algo, exclude_v=False):
    """Find log file matching prefix and algorithm."""
    candidates = [f for f in all_files if f.startswith(prefix) and algo in f]
    if not candidates: return None
    # Filter by file size (largest = most complete)
    candidates.sort(key=lambda f: os.path.getsize(os.path.join(LOG_DIR, f)), reverse=True)
    return os.path.join(LOG_DIR, candidates[0])

def find_ablation(substr):
    """Find ablation file by algopara substring."""
    candidates = [f for f in ablation_files if substr in f]
    if not candidates: return None
    return os.path.join(LOG_DIR, candidates[0])

print("=" * 90)
print("PACS Experiment Results - Final Analysis")
print("=" * 90)
print()

results = {}

# Original experiments
for label, (prefix, algo) in original_files.items():
    path = find_file(prefix, algo)
    if not path:
        print(f"  {label}: NOT FOUND")
        continue
    m = parse_log(path)
    if not m["mean"]:
        print(f"  {label}: NO DATA")
        continue
    acc = np.array(m["mean"])
    results[label] = {
        "best": acc.max(),
        "best_round": m["rounds"][int(acc.argmax())] if m["rounds"] else 0,
        "last": acc[-1],
        "min_d": m["min"][-1] if m["min"] else 0,
        "max_d": m["max"][-1] if m["max"] else 0,
        "drops": sum(1 for i in range(1, len(acc)) if acc[i] < acc[i-1] - 0.05),
        "stab": acc[-20:].std() if len(acc) >= 20 else 0,
        "total_rounds": m["rounds"][-1] if m["rounds"] else len(acc),
        "curve": acc,
    }

# Ablation variants
for label, substr in v_map.items():
    path = find_ablation(substr)
    if not path:
        print(f"  {label}: NOT FOUND ({substr})")
        continue
    m = parse_log(path)
    if not m["mean"]:
        print(f"  {label}: NO DATA")
        continue
    acc = np.array(m["mean"])
    results[label] = {
        "best": acc.max(),
        "best_round": m["rounds"][int(acc.argmax())] if m["rounds"] else 0,
        "last": acc[-1],
        "min_d": m["min"][-1] if m["min"] else 0,
        "max_d": m["max"][-1] if m["max"] else 0,
        "drops": sum(1 for i in range(1, len(acc)) if acc[i] < acc[i-1] - 0.05),
        "stab": acc[-20:].std() if len(acc) >= 20 else 0,
        "total_rounds": m["rounds"][-1] if m["rounds"] else len(acc),
        "curve": acc,
    }

# Print results
print(f"{'Method':<40} {'Best':>7} {'@Round':>7} {'Last':>7} {'MinD':>7} {'MaxD':>7} {'Drops':>6} {'Stab':>8} {'Rnds':>6}")
print("-" * 100)
for name, r in results.items():
    print(f"{name:<40} {r['best']:>7.4f} {r['best_round']:>7} {r['last']:>7.4f} {r['min_d']:>7.4f} {r['max_d']:>7.4f} {r['drops']:>6} {r['stab']:>8.4f} {r['total_rounds']:>6}")

print()
print("=" * 90)
print("RANKINGS")
print("=" * 90)
print("By Best Accuracy:")
for i, (n, r) in enumerate(sorted(results.items(), key=lambda x: x[1]["best"], reverse=True), 1):
    print(f"  {i}. {n}: {r['best']:.4f}")

print()
print("By Last Accuracy (final stable performance):")
for i, (n, r) in enumerate(sorted(results.items(), key=lambda x: x[1]["last"], reverse=True), 1):
    print(f"  {i}. {n}: {r['last']:.4f}")

print()
print("By Stability (lower=better, last 20 rounds std):")
for i, (n, r) in enumerate(sorted(results.items(), key=lambda x: x[1]["stab"]), 1):
    print(f"  {i}. {n}: {r['stab']:.4f} (drops={r['drops']})")
