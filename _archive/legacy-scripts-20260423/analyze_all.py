"""Analyze all FDSE framework PACS experiment results."""
import re
import os
import numpy as np

LOG_DIR = "/root/autodl-tmp/federated-learning/FDSE_CVPR25/task/PACS_c4/log"

# Map: (label, log file pattern)
experiments = [
    ("FedDSA (original)", "2026-04-06-09-50*feddsa*R500*"),
    ("FedAvg", "2026-04-06-09-55*fedavg*R500*"),
    ("FedBN", "2026-04-06-09-55*fedbn*R500*"),
    ("FedProto", "2026-04-06-09-55*fedproto*R500*"),
    ("FDSE", "2026-04-06-09-55*fdse*R500*"),
    ("V1: low weight", "2026-04-06-19-44*0.1.0.01*R200*"),
    ("V2: no HSIC", "2026-04-06-19-44*0.5.0.0.0.5*R200*"),
    ("V3: long warmup", "2026-04-06-19-44*0.5.0.05*50*R200*"),
]

def find_log(pattern):
    import glob
    files = glob.glob(os.path.join(LOG_DIR, pattern))
    if files:
        return sorted(files, key=os.path.getmtime, reverse=True)[0]
    # try without wildcard prefix
    return None

def parse_log(path):
    if not path or not os.path.exists(path):
        return None

    metrics = {
        "mean_acc": [], "std_acc": [], "min_acc": [], "max_acc": [],
        "mean_loss": [], "rounds": []
    }

    current_round = 0
    with open(path, "r") as f:
        for line in f:
            m = re.search(r"Round (\d+)", line)
            if m:
                current_round = int(m.group(1))

            for key, pattern in [
                ("mean_acc", r"mean_local_test_accuracy\s+([\d.]+)"),
                ("std_acc", r"std_local_test_accuracy\s+([\d.]+)"),
                ("min_acc", r"min_local_test_accuracy\s+([\d.]+)"),
                ("max_acc", r"max_local_test_accuracy\s+([\d.]+)"),
                ("mean_loss", r"mean_local_test_loss\s+([\d.]+)"),
            ]:
                m2 = re.search(pattern, line)
                if m2:
                    metrics[key].append(float(m2.group(1)))
                    if key == "mean_acc":
                        metrics["rounds"].append(current_round)

    return metrics

print("=" * 80)
print("PACS Experiment Results Analysis (FDSE Framework, AlexNet from scratch)")
print("=" * 80)
print()

# 1. Raw data table
print("1. FINAL RESULTS TABLE")
print("-" * 80)
print("{:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}".format(
    "Method", "Best", "Last", "Std", "Min", "Max", "Gap", "Rounds"))
print("-" * 80)

all_results = {}
for label, pattern in experiments:
    path = find_log(pattern)
    if not path:
        print("{:<22} LOG NOT FOUND".format(label))
        continue

    m = parse_log(path)
    if not m or len(m["mean_acc"]) == 0:
        print("{:<22} NO DATA".format(label))
        continue

    acc = np.array(m["mean_acc"])
    best = acc.max()
    best_round = m["rounds"][int(acc.argmax())] if m["rounds"] else "?"
    last = acc[-1]
    std_last = m["std_acc"][-1] if m["std_acc"] else 0
    min_last = m["min_acc"][-1] if m["min_acc"] else 0
    max_last = m["max_acc"][-1] if m["max_acc"] else 0
    gap = max_last - min_last
    total_rounds = m["rounds"][-1] if m["rounds"] else len(acc)

    # Count major drops
    drops = sum(1 for i in range(1, len(acc)) if acc[i] < acc[i-1] - 0.05)

    # Last 20 stability
    last20 = acc[-20:] if len(acc) >= 20 else acc
    stability = last20.std()

    all_results[label] = {
        "best": best, "best_round": best_round, "last": last,
        "std": std_last, "min": min_last, "max": max_last,
        "gap": gap, "drops": drops, "stability": stability,
        "total_rounds": total_rounds, "curve": acc
    }

    print("{:<22} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>6}".format(
        label, best, last, std_last, min_last, max_last, gap, total_rounds))

print()

# 2. Stability analysis
print("2. STABILITY ANALYSIS")
print("-" * 80)
print("{:<22} {:>10} {:>12} {:>15}".format("Method", "Drops>5%", "Last20 std", "Best-Last gap"))
print("-" * 80)
for label, r in all_results.items():
    bl_gap = r["best"] - r["last"]
    print("{:<22} {:>10} {:>12.4f} {:>15.4f}".format(
        label, r["drops"], r["stability"], bl_gap))

print()

# 3. Convergence speed
print("3. CONVERGENCE ANALYSIS")
print("-" * 80)
for label, r in all_results.items():
    curve = r["curve"]
    # Round to reach 70%
    r70 = next((i for i, v in enumerate(curve) if v >= 0.70), -1)
    # Round to reach 80%
    r80 = next((i for i, v in enumerate(curve) if v >= 0.80), -1)
    print("{:<22} 70%@Round {:>4}  80%@Round {:>4}  Best {:.4f}@Round {}".format(
        label, r70 if r70>=0 else "N/A", r80 if r80>=0 else "N/A",
        r["best"], r["best_round"]))

print()

# 4. Ablation comparison
print("4. ABLATION: FedDSA VARIANTS vs ORIGINAL")
print("-" * 80)
orig = all_results.get("FedDSA (original)")
if orig:
    for label in ["V1: low weight", "V2: no HSIC", "V3: long warmup"]:
        v = all_results.get(label)
        if v:
            delta = v["best"] - orig["best"]
            drop_diff = v["drops"] - orig["drops"]
            print("{:<22} Best={:.4f} ({:+.4f})  Drops={} ({:+d})  Stability={:.4f}".format(
                label, v["best"], delta, v["drops"], drop_diff, v["stability"]))

print()
print("5. KEY FINDINGS")
print("-" * 80)

# Auto-generate findings
if all_results:
    sorted_by_best = sorted(all_results.items(), key=lambda x: x[1]["best"], reverse=True)
    print("   Ranking by Best Accuracy:")
    for i, (name, r) in enumerate(sorted_by_best):
        print("     {}. {} = {:.4f}".format(i+1, name, r["best"]))

    best_method = sorted_by_best[0][0]
    best_val = sorted_by_best[0][1]["best"]

    dsa_orig = all_results.get("FedDSA (original)")
    fdse = all_results.get("FDSE")
    if dsa_orig and fdse:
        diff = dsa_orig["best"] - fdse["best"]
        print()
        print("   FedDSA vs FDSE: {:.4f} vs {:.4f} ({:+.4f})".format(
            dsa_orig["best"], fdse["best"], diff))
        print("   FedDSA stability (drops={}), FDSE stability (drops={})".format(
            dsa_orig["drops"], fdse["drops"]))
