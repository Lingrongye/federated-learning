"""
看 sketch domain 训练全过程 acc trajectory.
4 method × 2 seed.
"""
import os, numpy as np
from glob import glob

ROOT = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_pacs"
DOMAINS = ["photo", "art", "cartoon", "sketch"]
METHODS = {
    "F2DC vanilla": {15: "diag_f2dc_pacs_s15", 333: "diag_f2dc_pacs_s333"},
    "F2DC + DaA":   {15: "diag_f2dc_daa_pacs_s15", 333: "diag_f2dc_daa_pacs_s333"},
    "PG-DFC":       {15: "diag_pgdfc_pacs_s15", 333: "diag_pgdfc_pacs_s333"},
    "PG-DFC + DaA": {15: "diag_pgdfc_daa_pacs_s15", 333: "diag_pgdfc_daa_pacs_s333"},
}


def load_traj(d):
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    return np.stack([np.load(f, allow_pickle=True)["per_domain_acc"] for f in files])


# 收集
traj = {m: {} for m in METHODS}
for m, dirs in METHODS.items():
    for s, d in dirs.items():
        traj[m][s] = load_traj(d)  # (100, 4)

sketch_idx = DOMAINS.index("sketch")

# Sketch trajectory at key rounds
print("=" * 90)
print("PACS sketch domain Acc trajectory (4 method × 2 seed)")
print("=" * 90)
print(f"{'round':>5s} | " + " ".join([f"{m:>14s}" for m in METHODS]))
print("-" * 90)
for r in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100]:
    row = f"{r:>5d} | "
    for m in METHODS:
        # 2-seed mean
        vals = [traj[m][s][r-1, sketch_idx] for s in [15, 333]]
        row += f"{np.mean(vals):14.2f} "
    print(row)

print("\n=" * 1 + "=" * 89)
print("Sketch best round + best acc + final acc (per seed)")
print("=" * 90)
for m in METHODS:
    for s in [15, 333]:
        sk = traj[m][s][:, sketch_idx]
        best_r = int(sk.argmax()) + 1
        print(f"  {m:>14s} s={s}: best={sk.max():.2f} @R{best_r}, final R100={sk[-1]:.2f}, gap={sk.max()-sk[-1]:5.2f}")
    print()

# Per-round Δ (F2DC vs F2DC+DaA)
print("=" * 90)
print("DaA effect on sketch (F2DC+DaA − F2DC vanilla, 2-seed mean)")
print("=" * 90)
print(f"{'round':>5s} | F2DC sketch | F2DC+DaA sketch | Δ (DaA effect)")
print("-" * 90)
for r in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    a = np.mean([traj["F2DC vanilla"][s][r-1, sketch_idx] for s in [15, 333]])
    b = np.mean([traj["F2DC + DaA"][s][r-1, sketch_idx] for s in [15, 333]])
    print(f"{r:>5d} | {a:11.2f} | {b:15.2f} | {b-a:+6.2f}")

# 同时看 art / photo (DaA 救了哪些, 拖了哪些)
print("\n" + "=" * 90)
print("DaA effect on each domain (F2DC+DaA − F2DC vanilla, R100 final, 2-seed mean)")
print("=" * 90)
for di, dn in enumerate(DOMAINS):
    a = np.mean([traj["F2DC vanilla"][s][-1, di] for s in [15, 333]])
    b = np.mean([traj["F2DC + DaA"][s][-1, di] for s in [15, 333]])
    a_best = np.mean([traj["F2DC vanilla"][s][:, di].max() for s in [15, 333]])
    b_best = np.mean([traj["F2DC + DaA"][s][:, di].max() for s in [15, 333]])
    print(f"  {dn:>8s}: vanilla R100={a:.2f} (best {a_best:.2f}) → +DaA R100={b:.2f} (best {b_best:.2f}) | ΔLast={b-a:+6.2f} ΔBest={b_best-a_best:+6.2f}")
