"""Compare stability between FedDSA variants and FDSE."""
import os, re
import numpy as np
import glob

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

def analyze(path, label, target_round=None):
    if not os.path.exists(path):
        print(f"  {label}: FILE NOT FOUND")
        return None
    accs = parse(path)
    if not accs:
        return None

    acc_arr = np.array([a[1] for a in accs])
    total_rounds = accs[-1][0]

    # If target_round specified, only look up to that round
    if target_round:
        cutoff_accs = [(r, a) for r, a in accs if r <= target_round]
        if cutoff_accs:
            cut_arr = np.array([a[1] for a in cutoff_accs])
            best_r, best = max(cutoff_accs, key=lambda x: x[1])
            last = cutoff_accs[-1][1]
            drops = sum(1 for i in range(1, len(cut_arr)) if cut_arr[i] < cut_arr[i-1] - 0.05)
            last20 = cut_arr[-20:] if len(cut_arr) >= 20 else cut_arr
            stab = last20.std()
            best_last_gap = best - last
            last20_mean = last20.mean()

            print(f"  {label} (up to R{target_round}):")
            print(f"    Best: {best:.4f} @ R{best_r}")
            print(f"    Last: {last:.4f}")
            print(f"    Best-Last gap: {best_last_gap:.4f}")
            print(f"    Drops>5%: {drops}")
            print(f"    Last 20 rounds: mean={last20_mean:.4f}, std={stab:.4f}")
            return best, last, drops, stab

    # Full run
    best_r, best = max(accs, key=lambda x: x[1])
    last = accs[-1][1]
    drops = sum(1 for i in range(1, len(acc_arr)) if acc_arr[i] < acc_arr[i-1] - 0.05)
    last20 = acc_arr[-20:] if len(acc_arr) >= 20 else acc_arr
    stab = last20.std()
    best_last_gap = best - last
    last20_mean = last20.mean()

    # Also compute after-peak drop rate
    peak_idx = int(acc_arr.argmax())
    if peak_idx < len(acc_arr) - 1:
        after_peak = acc_arr[peak_idx:]
        max_drop_after_peak = (best - after_peak.min())
    else:
        max_drop_after_peak = 0

    print(f"  {label} (full run R{total_rounds}):")
    print(f"    Best: {best:.4f} @ R{best_r}")
    print(f"    Last: {last:.4f}")
    print(f"    Best-Last gap: {best_last_gap:.4f}")
    print(f"    Drops>5%: {drops}")
    print(f"    Last 20 rounds: mean={last20_mean:.4f}, std={stab:.4f}")
    print(f"    Max drop after peak: {max_drop_after_peak:.4f}")
    return best, last, drops, stab

# Find logs by substring
def find(substr_list, must_not=None):
    must_not = must_not or []
    for f in os.listdir(LOG_DIR):
        if all(s in f for s in substr_list) and not any(s in f for s in must_not):
            return os.path.join(LOG_DIR, f)
    return None

print("=" * 80)
print("Stability Comparison: FedDSA (our best) vs FDSE (paper baseline)")
print("=" * 80)
print()

# FDSE - 500 rounds
print("[FDSE - 500 rounds baseline]")
fdse_500 = find(["fdse_algopara", "R500", "_S2_", "2026-04-06-09-55"])
analyze(fdse_500, "FDSE full 500 rounds")
print()

# FDSE truncated to 200 for fair comparison
print("[FDSE - truncated to 200 rounds (fair comparison with FedDSA 200R)]")
analyze(fdse_500, "FDSE @ R200", target_round=200)
print()

# EXP-017 our best
print("[EXP-017 FedDSA V4 no HSIC - 200 rounds]")
exp017 = find(["feddsa_algopara_1.0|0.0|1.0", "R200", "_S2_", "LR1.00e-01"])
analyze(exp017, "EXP-017 no HSIC")
print()

# FedDSA original for comparison
print("[EXP-006 FedDSA original - 500 rounds]")
orig = find(["feddsa_algopara_1.0|0.1|1.0", "R500", "_S2_", "2026-04-06-09-50"])
analyze(orig, "EXP-006 original")
print()

# FedBN as another reference
print("[EXP-008 FedBN - 258 rounds (early stopped)]")
fedbn = find(["fedbn", "R500", "_S2_", "2026-04-06-09-55"])
analyze(fedbn, "EXP-008 FedBN")
