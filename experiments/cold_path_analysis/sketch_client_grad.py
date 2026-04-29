"""
看 sketch 3 个 client 的 grad_l2 trajectory (client 训练强度代理).
对比 F2DC vanilla vs F2DC+DaA, 看 DaA 是否压制 sketch client 的训练.
"""
import os, json, numpy as np
from glob import glob

ROOT = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_pacs"
DOMAINS = ["photo", "art", "cartoon", "sketch"]


def load_per_round(d, key):
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    arr = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        arr.append(z[key])
    return np.stack(arr)  # (R, K)


def load_layer_l2_per_client(d, layer_pattern):
    """提 layer_l2_pickle 里某层 (e.g. layer4.0.conv1.weight) 各 client 的 trajectory"""
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    R = len(files); K = 10
    arr = np.zeros((R, K))
    for ri, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        pkl = z["layer_l2_pickle"][0]
        if isinstance(pkl, str): pkl = json.loads(pkl)
        for cid, layers in pkl.items():
            cid = int(cid)
            for lk, lv in layers.items():
                if layer_pattern in lk:
                    arr[ri, cid] = float(lv); break
    return arr


def get_domain_per_client(d):
    z = np.load(sorted(glob(os.path.join(ROOT, d, "round_*.npz")))[0], allow_pickle=True)
    return list(z["domain_per_client"])


# 用 s=15 + s=333 mean
methods = {
    "F2DC vanilla":    {"15": "diag_f2dc_pacs_s15", "333": "diag_f2dc_pacs_s333"},
    "F2DC + DaA":      {"15": "diag_f2dc_daa_pacs_s15", "333": "diag_f2dc_daa_pacs_s333"},
    "PG-DFC":          {"15": "diag_pgdfc_pacs_s15", "333": "diag_pgdfc_pacs_s333"},
    "PG-DFC + DaA":    {"15": "diag_pgdfc_daa_pacs_s15", "333": "diag_pgdfc_daa_pacs_s333"},
}

dom_per_client = get_domain_per_client("diag_f2dc_pacs_s15")
print(f"Client → domain mapping (PACS 2-3-2-3):")
for ci, dn in enumerate(dom_per_client):
    print(f"  client {ci}: {dn}")

# 找 sketch client id
sketch_clients = [i for i, d in enumerate(dom_per_client) if d == "sketch"]
photo_clients = [i for i, d in enumerate(dom_per_client) if d == "photo"]
art_clients = [i for i, d in enumerate(dom_per_client) if d == "art"]
cartoon_clients = [i for i, d in enumerate(dom_per_client) if d == "cartoon"]
print(f"\nSketch clients: {sketch_clients}")

# 收集 grad_l2 (R, K) 2-seed mean
data_grad = {}
for m, dirs in methods.items():
    arr15 = load_per_round(dirs["15"], "grad_l2")
    arr333 = load_per_round(dirs["333"], "grad_l2")
    data_grad[m] = (arr15 + arr333) / 2.0

print("\n" + "=" * 95)
print("Sketch 3 个 client 的 grad_l2 trajectory (2-seed mean, F2DC vanilla vs F2DC+DaA)")
print("=" * 95)
print(f"{'round':>5s} | " + " ".join([f"sk_c{i}_van  sk_c{i}_DaA  Δ" for i in sketch_clients]))
print("-" * 95)
for r in [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    row = f"{r:>5d} | "
    for ci in sketch_clients:
        v = data_grad["F2DC vanilla"][r-1, ci]
        d = data_grad["F2DC + DaA"][r-1, ci]
        row += f"{v:7.3f}  {d:7.3f}  {d-v:+6.3f}  "
    print(row)

# 也看 photo / art client 看 DaA 是不是给它们升强度
print("\n" + "=" * 95)
print("Photo + Art client grad_l2 (DaA 救的 domain, 看是否被升强度)")
print("=" * 95)
print(f"{'round':>5s} | " + " ".join([f"{dn[0]}_c{i}_van  {dn[0]}_c{i}_DaA  Δ" for ci, dn in [(c, "photo") for c in photo_clients] + [(c, "art") for c in art_clients] for i in [ci]]))
print("-" * 95)
target_clients = photo_clients + art_clients
for r in [1, 5, 10, 20, 30, 50, 70, 100]:
    row = f"{r:>5d} | "
    for ci in target_clients:
        v = data_grad["F2DC vanilla"][r-1, ci]
        d = data_grad["F2DC + DaA"][r-1, ci]
        row += f"{v:7.3f}  {d:7.3f}  {d-v:+6.3f}  "
    print(row)

# 按 domain 平均 grad_l2 看哪些 domain 被 DaA 调强/调弱
print("\n" + "=" * 95)
print("Per-domain mean grad_l2 trajectory (跨 client 平均, R[10,30,50,70,100])")
print("=" * 95)
print(f"{'round':>5s} | " + " ".join([f"{dn:>9s}" for dn in DOMAINS]) + "   |  vs vanilla Δ")
for r in [10, 30, 50, 70, 90, 100]:
    print(f"\n  R={r}:")
    for m in ["F2DC vanilla", "F2DC + DaA"]:
        row = f"  {m:>13s} |"
        per_dom = []
        for dn in DOMAINS:
            cids = [i for i, d in enumerate(dom_per_client) if d == dn]
            v = np.mean([data_grad[m][r-1, c] for c in cids])
            per_dom.append(v)
            row += f" {v:9.3f}"
        print(row)
    delta_row = f"  {'Δ DaA-van':>13s} |"
    for dn in DOMAINS:
        cids = [i for i, d in enumerate(dom_per_client) if d == dn]
        v = np.mean([data_grad["F2DC vanilla"][r-1, c] for c in cids])
        d = np.mean([data_grad["F2DC + DaA"][r-1, c] for c in cids])
        delta_row += f" {d-v:+9.3f}"
    print(delta_row)

# layer4.0.conv1.weight (深层) per-client trajectory — 深层 drift 看 client 学到啥
print("\n" + "=" * 95)
print("Layer4.0.conv1.weight per-client drift (深层语义 conv, 2-seed mean)")
print("=" * 95)
data_l4 = {}
for m, dirs in methods.items():
    arr15 = load_layer_l2_per_client(dirs["15"], "layer4.0.conv1.weight")
    arr333 = load_layer_l2_per_client(dirs["333"], "layer4.0.conv1.weight")
    data_l4[m] = (arr15 + arr333) / 2.0
print(f"{'round':>5s} | sketch client (mean of 3) | photo (mean of 2)")
print("-" * 95)
for r in [10, 30, 50, 70, 90, 100]:
    sk_van = np.mean([data_l4["F2DC vanilla"][r-1, c] for c in sketch_clients])
    sk_daa = np.mean([data_l4["F2DC + DaA"][r-1, c] for c in sketch_clients])
    ph_van = np.mean([data_l4["F2DC vanilla"][r-1, c] for c in photo_clients])
    ph_daa = np.mean([data_l4["F2DC + DaA"][r-1, c] for c in photo_clients])
    print(f"R{r:>3d} | sketch van={sk_van:.3f} DaA={sk_daa:.3f} Δ={sk_daa-sk_van:+.3f} | photo van={ph_van:.3f} DaA={ph_daa:.3f} Δ={ph_daa-ph_van:+.3f}")
