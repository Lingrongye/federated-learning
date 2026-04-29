"""
专项诊断: 为什么 PG-DFC prototype guidance 在 PACS 反而不起作用?
对比 PG-DFC vs F2DC vanilla (都不带 DaA), 看 prototype 注入到底改变了什么.
"""
import os, json, numpy as np
from glob import glob
from sklearn.metrics import silhouette_score

ROOT = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_pacs"
DOMAINS = ["photo", "art", "cartoon", "sketch"]
PG_DIRS = ["diag_pgdfc_pacs_s15", "diag_pgdfc_pacs_s333"]
F2_DIRS = ["diag_f2dc_pacs_s15", "diag_f2dc_pacs_s333"]


def per_round_acc(d):
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    return np.stack([np.load(f, allow_pickle=True)["per_domain_acc"] for f in files])


def proto_cos_traj_per_client(d):
    """每 round 每 client 的 local_proto vs leave-one-out consensus cos sim"""
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    R = len(files)
    cos_traj = np.full((R, 10), np.nan)
    for ri, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        if "local_protos" not in z.files:
            return None
        lp = z["local_protos"].astype(np.float32)  # (10, 7, 512)
        K, C, D = lp.shape
        total = lp.sum(axis=0)  # (C, dim)
        for k in range(K):
            others = (total - lp[k]) / max(K - 1, 1)
            sims = []
            for c in range(C):
                if np.linalg.norm(lp[k, c]) < 1e-6 or np.linalg.norm(others[c]) < 1e-6:
                    continue
                sims.append((lp[k, c] * others[c]).sum() /
                            (np.linalg.norm(lp[k, c]) * np.linalg.norm(others[c])))
            cos_traj[ri, k] = np.mean(sims) if sims else np.nan
    return cos_traj


def grad_l2_per_client(d):
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    arr = np.stack([np.load(f, allow_pickle=True)["grad_l2"] for f in files])
    return arr  # (R, K)


def aligned_grad_l2(d):
    """按 client_id 对齐"""
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    R = len(files)
    al = np.zeros((R, 10))
    for ri, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        oc = z["online_clients"]
        g = z["grad_l2"]
        for pos, cid in enumerate(oc):
            al[ri, cid] = g[pos]
    return al


def heavy_load(d, kind):
    files = sorted(glob(os.path.join(ROOT, d, f"{kind}_*.npz")))
    if not files: return None
    return np.load(files[0] if kind == "best" else files[-1], allow_pickle=True)


def per_class_acc(heavy):
    if heavy is None: return None
    conf = heavy["confusion"].item()
    out = {}
    for dn in DOMAINS:
        if dn in conf:
            c = conf[dn]
            diag = np.diag(c).astype(np.float32)
            rs = c.sum(1).astype(np.float32)
            out[dn] = np.where(rs > 0, diag / np.maximum(rs, 1), 0.0)
    return out


def feat_silhouette(heavy):
    if heavy is None: return None, None
    fd = heavy["features"].item(); ld = heavy["labels"].item()
    X, y, dom = [], [], []
    for di, dn in enumerate(DOMAINS):
        if dn not in fd: continue
        f = fd[dn].astype(np.float32); l = ld[dn].astype(np.int32)
        X.append(f); y.append(l); dom.append(np.full(len(f), di))
    X = np.concatenate(X); y = np.concatenate(y); dom = np.concatenate(dom)
    if len(X) > 1500:
        idx = np.random.RandomState(0).choice(len(X), 1500, replace=False)
        X, y, dom = X[idx], y[idx], dom[idx]
    return float(silhouette_score(X, y)), float(silhouette_score(X, dom))


# 1. AVG trajectory + sketch 单独
print("=" * 95)
print("【维度 5】PACS Acc trajectory: PG-DFC vs F2DC vanilla (2-seed mean, 都不带 DaA)")
print("=" * 95)
pg_traj = (per_round_acc(PG_DIRS[0]) + per_round_acc(PG_DIRS[1])) / 2
f2_traj = (per_round_acc(F2_DIRS[0]) + per_round_acc(F2_DIRS[1])) / 2
sketch_idx = DOMAINS.index("sketch")
print(f"{'round':>5s} | {'AVG PG':>7s} {'AVG F2DC':>9s} {'Δ':>6s} | {'sketch PG':>9s} {'sketch F2DC':>11s} {'Δ':>6s}")
for r in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    pg_avg = pg_traj[r-1].mean(); f2_avg = f2_traj[r-1].mean()
    pg_sk = pg_traj[r-1, sketch_idx]; f2_sk = f2_traj[r-1, sketch_idx]
    print(f"{r:>5d} | {pg_avg:7.2f} {f2_avg:9.2f} {pg_avg-f2_avg:+6.2f} | {pg_sk:9.2f} {f2_sk:11.2f} {pg_sk-f2_sk:+6.2f}")

# 2. Per-domain final + per-class shift
print("\n" + "=" * 95)
print("【维度 6】Per-domain Last + per-class shift (PG-DFC vs F2DC vanilla, 看 PG 损失在哪)")
print("=" * 95)
print(f"{'Domain':>10s} | {'PG vanilla Last':>16s} {'F2DC vanilla Last':>18s} {'Δ Last':>7s}")
for di, dn in enumerate(DOMAINS):
    pg_l = pg_traj[-1, di]; f2_l = f2_traj[-1, di]
    print(f"{dn:>10s} | {pg_l:16.2f} {f2_l:18.2f} {pg_l-f2_l:+7.2f}")

print("\nPer-class accuracy at R100 (final), sketch domain (PG vs F2DC):")
for s in ["s15", "s333"]:
    hf_pg = heavy_load(f"diag_pgdfc_pacs_{s}", "final")
    hf_f2 = heavy_load(f"diag_f2dc_pacs_{s}", "final")
    pca_pg = per_class_acc(hf_pg)
    pca_f2 = per_class_acc(hf_f2)
    if pca_pg and pca_f2 and "sketch" in pca_pg:
        diff = (pca_pg["sketch"] - pca_f2["sketch"]) * 100
        print(f"  {s} sketch per-class Δ (PG - F2DC): " + " ".join([f"{x:+5.1f}" for x in diff]))

# 3. Prototype cos sim trajectory (only PG has local_protos)
print("\n" + "=" * 95)
print("【维度 2】PG-DFC client→consensus prototype cos sim (LOO, sketch client 单独)")
print("=" * 95)
cos_pg = (proto_cos_traj_per_client(PG_DIRS[0]) + proto_cos_traj_per_client(PG_DIRS[1])) / 2
sketch_clients = [7, 8, 9]
print(f"{'round':>5s} | sketch_c7  sketch_c8  sketch_c9 | photo_c0  art_c2  cartoon_c5")
for r in [10, 20, 30, 50, 70, 100]:
    print(f"{r:>5d} | "
          f"{cos_pg[r-1, 7]:9.4f}  {cos_pg[r-1, 8]:9.4f}  {cos_pg[r-1, 9]:9.4f}  | "
          f"{cos_pg[r-1, 0]:8.4f}  {cos_pg[r-1, 2]:6.4f}  {cos_pg[r-1, 5]:9.4f}")

# 4. Grad L2 per-client (sketch client 训练强度)
print("\n" + "=" * 95)
print("【维度 1】Sketch 3 client grad_l2 trajectory (按 client_id 对齐, 2-seed mean)")
print("=" * 95)
g_pg = (aligned_grad_l2(PG_DIRS[0]) + aligned_grad_l2(PG_DIRS[1])) / 2
g_f2 = (aligned_grad_l2(F2_DIRS[0]) + aligned_grad_l2(F2_DIRS[1])) / 2
print(f"{'round':>5s} | sk7 PG   F2DC   Δ | sk8 PG   F2DC   Δ | sk9 PG   F2DC   Δ")
for r in [10, 20, 30, 50, 70, 100]:
    row = f"{r:>5d} | "
    for c in sketch_clients:
        row += f"{g_pg[r-1,c]:.3f}  {g_f2[r-1,c]:.3f}  {g_pg[r-1,c]-g_f2[r-1,c]:+.3f} | "
    print(row)

# 5. Final feature silhouette (PG vs F2DC vanilla, 2-seed mean)
print("\n" + "=" * 95)
print("【维度 3】Final feature silhouette (heavy reconstruct)")
print("=" * 95)
for s in ["s15", "s333"]:
    hf_pg = heavy_load(f"diag_pgdfc_pacs_{s}", "final")
    hf_f2 = heavy_load(f"diag_f2dc_pacs_{s}", "final")
    sc_pg, sd_pg = feat_silhouette(hf_pg)
    sc_f2, sd_f2 = feat_silhouette(hf_f2)
    print(f"  {s}: sil_class PG={sc_pg:.4f}  F2DC={sc_f2:.4f}  Δ={sc_pg-sc_f2:+.4f}")
    print(f"  {s}: sil_dom   PG={sd_pg:.4f}  F2DC={sd_f2:.4f}  Δ={sd_pg-sd_f2:+.4f}")

# 6. Mode collapse — PG-DFC sketch class proto 跟其他 class proto 是否变近
print("\n" + "=" * 95)
print("【维度 2】Sketch class prototype 跟其他 class prototype pairwise cos sim")
print("(看 PG-DFC 是否把 sketch class 跟其他 class 同化)")
print("=" * 95)


def class_pairwise_per_domain(heavy, target_dom="sketch"):
    if heavy is None: return None
    feats = heavy["features"].item()
    labels = heavy["labels"].item()
    if target_dom not in feats: return None
    f = feats[target_dom].astype(np.float32)
    l = labels[target_dom].astype(np.int32)
    C = int(l.max() + 1)
    proto = np.zeros((C, f.shape[1]), dtype=np.float32)
    for c in range(C):
        m = l == c
        if m.sum() > 0: proto[c] = f[m].mean(0)
    pairs = []
    for i in range(C):
        for j in range(i+1, C):
            pairs.append((proto[i] * proto[j]).sum() / (np.linalg.norm(proto[i]) * np.linalg.norm(proto[j])))
    return np.mean(pairs)

print(f"{'seed':>5s} {'method':>15s} | sketch class间 pairwise cos sim (越低越好)")
for s in ["s15", "s333"]:
    hf_pg = heavy_load(f"diag_pgdfc_pacs_{s}", "final")
    hf_f2 = heavy_load(f"diag_f2dc_pacs_{s}", "final")
    p_pg = class_pairwise_per_domain(hf_pg, "sketch")
    p_f2 = class_pairwise_per_domain(hf_f2, "sketch")
    print(f"  {s:>5s} {'PG vanilla':>15s} | {p_pg:.4f}")
    print(f"  {s:>5s} {'F2DC vanilla':>15s} | {p_f2:.4f}")
    print(f"  {s:>5s} {'Δ PG - F2DC':>15s} | {p_pg-p_f2:+.4f}")

# 7. CKA (PG vs F2DC vanilla representation 相似度)
print("\n" + "=" * 95)
print("【维度 7】CKA PG-DFC vs F2DC vanilla (看 PG-DFC 改造 backbone 程度)")
print("=" * 95)


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Kx = X @ X.T; Ky = Y @ Y.T
    return float((Kx*Ky).sum() / (np.sqrt((Kx*Kx).sum()*(Ky*Ky).sum())+1e-9))


for s in ["s15", "s333"]:
    hf_pg = heavy_load(f"diag_pgdfc_pacs_{s}", "final")
    hf_f2 = heavy_load(f"diag_f2dc_pacs_{s}", "final")
    if hf_pg is None or hf_f2 is None: continue
    fa = hf_pg["features"].item(); fb = hf_f2["features"].item()
    Xa, Xb = [], []
    for dn in DOMAINS:
        n = min(len(fa[dn]), len(fb[dn]))
        Xa.append(fa[dn][:n].astype(np.float32))
        Xb.append(fb[dn][:n].astype(np.float32))
    Xa = np.concatenate(Xa); Xb = np.concatenate(Xb)
    if len(Xa) > 800:
        idx = np.random.RandomState(0).choice(len(Xa), 800, replace=False)
        Xa, Xb = Xa[idx], Xb[idx]
    print(f"  {s}: CKA(final) = {cka(Xa, Xb):.4f}")
