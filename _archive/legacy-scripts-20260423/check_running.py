"""Check status of all currently running experiments."""
import os, re, glob

LOG_DIR = "/root/autodl-tmp/federated-learning/FDSE_CVPR25/task/PACS_c4/log"

# Map experiment label to substring identifier in filename
experiments = [
    # (label, [must_contain_all_substrings], [must_NOT_contain])
    ("EXP-014 V4 strong+w50",         ["1.0|0.1|1.0|0.1|50|5|128Mfeddsa_R200", "LR1.00e-01", "_S2_"], ["plus"]),
    ("EXP-015 FedDSA+ stage50/100",   ["1.0|0.1|1.0|0.1|50|100|10|1.0|5|128Mfeddsa_plus_R200", "_S2_"], []),
    ("EXP-016 V4 seed=15",            ["1.0|0.1|1.0|0.1|50|5|128Mfeddsa_R200", "LR1.00e-01", "_S15_"], ["plus"]),
    ("EXP-017 V4 no HSIC",            ["1.0|0.0|1.0|0.1|50|5|128Mfeddsa_R200", "_S2_"], []),
    ("EXP-018 FedDSA+ stage80/150",   ["1.0|0.1|1.0|0.1|80|150|15|1.0|5|128Mfeddsa_plus_R200", "_S2_"], []),
    ("EXP-019 V4 lr=0.05",            ["1.0|0.1|1.0|0.1|50|5|128Mfeddsa_R200", "LR5.00e-02", "_S2_"], ["plus"]),
    ("EXP-020 V4 tau=0.5",            ["1.0|0.1|1.0|0.5|50|5|128Mfeddsa_R200", "_S2_"], []),
    ("EXP-021 V4 orth=2.0",           ["2.0|0.1|1.0|0.1|50|5|128Mfeddsa_R200", "_S2_"], []),
]

def find_log(must_contain, must_not):
    files = []
    for f in os.listdir(LOG_DIR):
        if not f.startswith("2026-04-07"): continue
        if not all(s in f for s in must_contain): continue
        if any(s in f for s in must_not): continue
        files.append(f)
    if not files: return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)), reverse=True)
    return os.path.join(LOG_DIR, files[0])

def parse_log(path):
    if not path: return None
    accs = []
    cur_round = 0
    with open(path) as f:
        for line in f:
            m = re.search(r"Round (\d+)", line)
            if m: cur_round = int(m.group(1))
            m2 = re.search(r"mean_local_test_accuracy\s+([\d.]+)", line)
            if m2:
                accs.append((cur_round, float(m2.group(1))))
    return accs

print("=" * 90)
print("Current Status of 8 Running Experiments")
print("=" * 90)
print(f"{'Experiment':<32} {'Round':>6} {'Current':>9} {'Best':>9} {'@Round':>7}")
print("-" * 90)

for label, must_in, must_not in experiments:
    path = find_log(must_in, must_not)
    if not path:
        print(f"{label:<32} NOT FOUND")
        continue
    accs = parse_log(path)
    if not accs:
        print(f"{label:<32} NO DATA YET")
        continue

    cur_round = accs[-1][0]
    cur_acc = accs[-1][1]
    best_round, best_acc = max(accs, key=lambda x: x[1])
    print(f"{label:<32} {cur_round:>6} {cur_acc:>9.4f} {best_acc:>9.4f} {best_round:>7}")

print()
print("Reference baseline:")
print(f"  FDSE (paper):                 R500   0.8216 - target")
print(f"  FedDSA original (EXP-006):    R500   0.8115 - we beat?")
print(f"  V3 long warmup (EXP-013):     R200   0.8168 - V4 should beat this")
