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

def find(must_in, must_not=None):
    must_not = must_not or []
    files = []
    for f in os.listdir(LOG_DIR):
        if not (f.startswith("2026-04-07") or f.startswith("2026-04-08")): continue
        if not all(s in f for s in must_in): continue
        if any(s in f for s in must_not): continue
        files.append(f)
    if not files: return None
    files.sort(key=lambda f: os.path.getsize(os.path.join(LOG_DIR, f)), reverse=True)
    return os.path.join(LOG_DIR, files[0])

experiments = [
    ("EXP-028b Uncertainty (fix)", ["feddsa_auto_algopara_0.1|50", "2026-04-08-10"], []),
    ("EXP-032 PCGrad+orth=2+HSIC0", ["feddsa_pcgrad_algopara_2.0|0.0|1.0|0.1|50"], []),
    ("EXP-033 PCGrad+warmup=80",    ["feddsa_pcgrad_algopara_2.0|0.0|1.0|0.1|80"], []),
    ("EXP-035 seed=15 (V4 no HSIC)", ["feddsa_algopara_1.0|0.0|1.0|0.1|50", "_S15_"], ["plus","auto","pcgrad","triplet","cka","stable","vae","asym","multilayer"]),
    ("EXP-036 seed=333 (V4 no HSIC)", ["feddsa_algopara_1.0|0.0|1.0|0.1|50", "_S333_"], ["plus","auto","pcgrad","triplet","cka","stable","vae","asym","multilayer"]),
    ("EXP-040 MultiLayer Style",    ["feddsa_multilayer_algopara"], []),
    ("EXP-041 VAE Style Head",      ["feddsa_vae_algopara"], []),
    ("EXP-042 Asymmetric Heads",    ["feddsa_asym_algopara"], []),
]

print("=" * 95)
print("Current Status")
print("=" * 95)
print(f"{'Experiment':<35} {'Round':>6} {'Current':>9} {'Best':>9} {'@Round':>7}")
print("-" * 95)

for label, must_in, must_not in experiments:
    path = find(must_in, must_not)
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
print("  FDSE (paper):        82.16%")
print("  EXP-017 (our SOTA):  82.24%")
