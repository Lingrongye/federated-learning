"""Check status of all currently running experiments."""
import os, re

LOG_DIR = "/root/autodl-tmp/federated-learning/FDSE_CVPR25/task/PACS_c4/log"

def parse(path):
    accs = []
    cur_r = 0
    with open(path) as f:
        for line in f:
            m = re.search(r"Round (\d+)", line)
            if m: cur_r = int(m.group(1))
            m2 = re.search(r"mean_local_test_accuracy\s+([\d.]+)", line)
            if m2:
                accs.append((cur_r, float(m2.group(1))))
    return accs

def find_log(must_in, must_not=None):
    must_not = must_not or []
    files = []
    for f in os.listdir(LOG_DIR):
        if not (f.startswith("2026-04-07") or f.startswith("2026-04-08")): continue
        if not all(s in f for s in must_in): continue
        if any(s in f for s in must_not): continue
        files.append(f)
    if not files: return None
    # Sort by file size (largest first) to prefer fully-trained logs over crashed ones
    files.sort(key=lambda f: os.path.getsize(os.path.join(LOG_DIR, f)), reverse=True)
    return os.path.join(LOG_DIR, files[0])

# Stability batch (already running)
stability = [
    ("EXP-022 lr=0.05 fast decay",  ["feddsa_algopara_1.0|0.0|1.0|0.1|50", "LR5.00e-02"], ["plus","auto","triplet","cka","pcgrad","stable"]),
    ("EXP-023 EMA=0.9",             ["feddsa_stable_algopara_1.0|0.0|1.0|0.1|50|5|128|0.9"], []),
    ("EXP-024 Beta(0.5) soft aug",  ["feddsa_stable_algopara_1.0|0.0|1.0|0.1|50|5|128|0.0|0.5"], []),
]

# New loss experiments
loss_exps = [
    ("EXP-025 No InfoNCE",          ["feddsa_algopara_1.0|0.0|0.0|0.1|50"], ["plus","auto","triplet","cka","pcgrad","stable"]),
    ("EXP-028 Uncertainty Weight",  ["feddsa_auto_algopara"], []),
    ("EXP-029 PCGrad",              ["feddsa_pcgrad_algopara_1.0|0.0|1.0"], []),
    ("EXP-030 Triplet Loss",        ["feddsa_triplet_algopara"], []),
    ("EXP-031 CKA Loss",            ["feddsa_cka_algopara"], []),
]

print("=" * 90)
print("Current Status of All Running Experiments")
print("=" * 90)

print("\n[Stability Batch (3 experiments)]")
print(f"{'Experiment':<35} {'Round':>6} {'Current':>9} {'Best':>9} {'@Round':>7}")
print("-" * 90)
for label, must_in, must_not in stability:
    path = find_log(must_in, must_not)
    if not path:
        print(f"{label:<35} NOT FOUND")
        continue
    accs = parse(path)
    if not accs:
        print(f"{label:<35} STARTING...")
        continue
    cur_round = accs[-1][0]
    cur_acc = accs[-1][1]
    best_round, best_acc = max(accs, key=lambda x: x[1])
    print(f"{label:<35} {cur_round:>6} {cur_acc:>9.4f} {best_acc:>9.4f} {best_round:>7}")

print("\n[Loss Improvement Batch (5 experiments)]")
print(f"{'Experiment':<35} {'Round':>6} {'Current':>9} {'Best':>9} {'@Round':>7}")
print("-" * 90)
for label, must_in, must_not in loss_exps:
    path = find_log(must_in, must_not)
    if not path:
        print(f"{label:<35} NOT FOUND")
        continue
    accs = parse(path)
    if not accs:
        print(f"{label:<35} STARTING...")
        continue
    cur_round = accs[-1][0]
    cur_acc = accs[-1][1]
    best_round, best_acc = max(accs, key=lambda x: x[1])
    print(f"{label:<35} {cur_round:>6} {cur_acc:>9.4f} {best_acc:>9.4f} {best_round:>7}")

print()
print("Reference:")
print("  FDSE (paper):                 R500   0.8216")
print("  EXP-017 V4 no HSIC (current best): 0.8224")
