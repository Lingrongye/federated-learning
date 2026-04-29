"""
单 seed (s=15) 直接对比 PG-DFC+DaA vs F2DC+DaA 在 office 上.
不预设结论, 用诊断体系所有可算指标各跑一遍, 看哪些指标 PG-DFC+DaA 真正"差".
"""
import os, json, numpy as np
from glob import glob
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_office"
A_DIR = os.path.join(ROOT, "diag_pgdfc_daa_office_s15")  # PG-DFC+DaA
B_DIR = os.path.join(ROOT, "diag_f2dc_daa_office_s15")   # F2DC+DaA
DOMAINS = ["caltech", "amazon", "webcam", "dslr"]


def per_round_acc(d):
    files = sorted(glob(os.path.join(d, "round_*.npz")))
    rs, accs = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        rs.append(int(z["round"]))
        accs.append(z["per_domain_acc"])
    return np.array(rs), np.stack(accs)


def aggregate_grad_drift(d):
    files = sorted(glob(os.path.join(d, "round_*.npz")))
    grad_l2, layer_l2 = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        grad_l2.append(z["grad_l2"].mean())
        pkl = z["layer_l2_pickle"][0]
        if isinstance(pkl, str):
            pkl = json.loads(pkl)
        avg_per_layer = {}
        for cid, layers in pkl.items():
            for lk, lv in layers.items():
                avg_per_layer.setdefault(lk, []).append(float(lv))
        layer_l2.append({k: np.mean(v) for k, v in avg_per_layer.items()})
    return np.array(grad_l2), layer_l2


def daa_dispatch(d):
    files = sorted(glob(os.path.join(d, "round_*.npz")))
    ratios = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        s, fr = z["sample_shares"], z["daa_freqs"]
        if fr.sum() == 0:
            return None, None
        ratios.append(fr / np.maximum(s, 1e-9))
    z0 = np.load(files[0], allow_pickle=True)
    return np.mean(ratios, axis=0), list(z0["domain_per_client"])


def heavy_load(d, kind):
    files = sorted(glob(os.path.join(d, f"{kind}_*.npz")))
    if not files: return None
    return np.load(files[0] if kind == "best" else files[-1], allow_pickle=True)


def feat_silhouette(heavy):
    if heavy is None: return None, None
    feats_d = heavy["features"].item()
    labels_d = heavy["labels"].item()
    X, y, dom = [], [], []
    for di, dn in enumerate(DOMAINS):
        if dn not in feats_d: continue
        f = feats_d[dn].astype(np.float32)
        l = labels_d[dn].astype(np.int32)
        X.append(f); y.append(l)
        dom.append(np.full(len(f), di, dtype=np.int32))
    X = np.concatenate(X); y = np.concatenate(y); dom = np.concatenate(dom)
    if len(X) > 1500:
        idx = np.random.RandomState(0).choice(len(X), 1500, replace=False)
        X, y, dom = X[idx], y[idx], dom[idx]
    return float(silhouette_score(X, y)), float(silhouette_score(X, dom))


def per_class_acc(heavy):
    if heavy is None: return None
    conf = heavy["confusion"].item()
    out = {}
    for dn in DOMAINS:
        if dn not in conf: continue
        c = conf[dn]
        diag = np.diag(c).astype(np.float32)
        rs = c.sum(axis=1).astype(np.float32)
        out[dn] = np.where(rs > 0, diag / np.maximum(rs, 1), 0.0)
    return out


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Kx = X @ X.T; Ky = Y @ Y.T
    return float((Kx*Ky).sum() / (np.sqrt((Kx*Kx).sum() * (Ky*Ky).sum()) + 1e-9))


def cross_method_cka(ha, hb):
    fa = ha["features"].item(); fb = hb["features"].item()
    Xa, Xb = [], []
    for dn in DOMAINS:
        if dn not in fa or dn not in fb: continue
        n = min(len(fa[dn]), len(fb[dn]))
        Xa.append(fa[dn][:n].astype(np.float32))
        Xb.append(fb[dn][:n].astype(np.float32))
    Xa = np.concatenate(Xa); Xb = np.concatenate(Xb)
    if len(Xa) > 800:
        idx = np.random.RandomState(0).choice(len(Xa), 800, replace=False)
        Xa, Xb = Xa[idx], Xb[idx]
    return cka(Xa, Xb)


def per_domain_proto(heavy):
    """每 domain mean feature per class (heavy reconstruct)"""
    if heavy is None: return None
    feats = heavy["features"].item()
    labels = heavy["labels"].item()
    all_l = np.concatenate([labels[d] for d in DOMAINS if d in labels])
    C = int(all_l.max() + 1)
    K, dim = len(DOMAINS), feats[DOMAINS[0]].shape[1]
    proto = np.zeros((K, C, dim), dtype=np.float32)
    valid = np.zeros((K, C), dtype=bool)
    for k, dn in enumerate(DOMAINS):
        if dn not in feats: continue
        f, l = feats[dn].astype(np.float32), labels[dn].astype(np.int32)
        for c in range(C):
            m = l == c
            if m.sum() > 0:
                proto[k, c] = f[m].mean(0); valid[k, c] = True
    return proto, valid


def cos(a, b):
    return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


def proto_pairwise_inter_class(proto, valid):
    """同 domain 内 class 之间的 pairwise cos sim — 越低类区分越好"""
    K, C, _ = proto.shape
    pairs = []
    for k in range(K):
        vc = np.where(valid[k])[0]
        for i in range(len(vc)):
            for j in range(i+1, len(vc)):
                pairs.append(cos(proto[k, vc[i]], proto[k, vc[j]]))
    return np.mean(pairs)


def proto_intra_class_inter_domain(proto, valid):
    """同 class 跨 domain 的 cos sim — 越高表示 class 跨 domain 越一致 (好)"""
    K, C, _ = proto.shape
    res = []
    for c in range(C):
        valid_k = np.where(valid[:, c])[0]
        for i in range(len(valid_k)):
            for j in range(i+1, len(valid_k)):
                res.append(cos(proto[valid_k[i], c], proto[valid_k[j], c]))
    return np.mean(res)


# ========== MAIN ==========
print("=" * 90)
print("V100 office s=15 单 seed 直接对比 PG-DFC+DaA vs F2DC+DaA")
print("=" * 90)

rs_a, acc_a = per_round_acc(A_DIR)  # PG-DFC+DaA
rs_b, acc_b = per_round_acc(B_DIR)  # F2DC+DaA

# 维度 5: Acc trajectory
print("\n【维度 5】Per-domain Acc Best & Last")
print(f"{'domain':>10s} | {'PG-DFC+DaA Best/Last':>22s} | {'F2DC+DaA Best/Last':>22s} | {'Δ Best':>7s} {'Δ Last':>7s}")
print("-" * 90)
for di, dn in enumerate(DOMAINS):
    bA, lA = acc_a[:, di].max(), acc_a[-1, di]
    bB, lB = acc_b[:, di].max(), acc_b[-1, di]
    print(f"{dn:>10s} | {bA:6.2f} / {lA:6.2f}      | {bB:6.2f} / {lB:6.2f}      | {bA-bB:+6.2f} {lA-lB:+6.2f}")
print(f"{'AVG':>10s} | {acc_a.max(0).mean():6.2f} / {acc_a[-1].mean():6.2f}      | {acc_b.max(0).mean():6.2f} / {acc_b[-1].mean():6.2f}      | {acc_a.max(0).mean()-acc_b.max(0).mean():+6.2f} {acc_a[-1].mean()-acc_b[-1].mean():+6.2f}")
print(f"{'gap (B-L)':>10s} | {acc_a.max(0).mean()-acc_a[-1].mean():22.2f} | {acc_b.max(0).mean()-acc_b[-1].mean():22.2f} |")

# 维度 5: Per-round AVG curve key points
print("\n【维度 5】Acc trajectory 关键点 (AVG over 4 domain)")
key_rs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"{'round':>5s} | {'PG-DFC+DaA':>11s} | {'F2DC+DaA':>10s} | Δ")
for r in key_rs:
    if r > len(rs_a): break
    a = acc_a[r-1].mean(); b = acc_b[r-1].mean()
    print(f"{r:>5d} | {a:11.2f} | {b:10.2f} | {a-b:+.2f}")

# 维度 1: DaA dispatch
print("\n【维度 1】DaA dispatch ratio (avg over 100 round)")
ra, dom_a = daa_dispatch(A_DIR)
rb, dom_b = daa_dispatch(B_DIR)
print(f"{'client':>6s} {'domain':>8s} | PG-DFC+DaA | F2DC+DaA | Δ")
for k in range(len(ra)):
    print(f"{k:>6d} {dom_a[k]:>8s} | {ra[k]:.3f}      | {rb[k]:.3f}    | {ra[k]-rb[k]:+.3f}")

# 维度 1: grad_l2 + layer drift
print("\n【维度 1】Grad L2 (avg) + per-layer drift (final round avg)")
gA, layA = aggregate_grad_drift(A_DIR)
gB, layB = aggregate_grad_drift(B_DIR)
print(f"  Grad L2 mean over 100 rounds: PG-DFC+DaA={gA.mean():.3f}  F2DC+DaA={gB.mean():.3f}  Δ={gA.mean()-gB.mean():+.3f}")
print(f"  Grad L2 final 10 rounds:       PG-DFC+DaA={gA[-10:].mean():.3f}  F2DC+DaA={gB[-10:].mean():.3f}  Δ={gA[-10:].mean()-gB[-10:].mean():+.3f}")
print(f"  Per-layer drift (avg over 100 rounds, all clients):")
all_layers = sorted(set(list(layA[0].keys()) + list(layB[0].keys())))
print(f"  {'layer':<35s} | PG-DFC+DaA | F2DC+DaA | Δ")
for layer in all_layers:
    a_vals = [d.get(layer, 0) for d in layA]
    b_vals = [d.get(layer, 0) for d in layB]
    print(f"  {layer:<35s} | {np.mean(a_vals):.4f}     | {np.mean(b_vals):.4f}   | {np.mean(a_vals)-np.mean(b_vals):+.4f}")

# 维度 3: silhouette
print("\n【维度 3】Feature silhouette (best vs final, 高 sil_class 好, 低 sil_domain = domain-invariant)")
for snap in ["best", "final"]:
    hA = heavy_load(A_DIR, snap)
    hB = heavy_load(B_DIR, snap)
    scA, sdA = feat_silhouette(hA)
    scB, sdB = feat_silhouette(hB)
    print(f"  {snap:>5s}: sil_class  PG={scA:.4f}  F2DC={scB:.4f}  Δ={scA-scB:+.4f}")
    print(f"  {snap:>5s}: sil_domain PG={sdA:.4f}  F2DC={sdB:.4f}  Δ={sdA-sdB:+.4f}")

# 维度 6: per-class shift
print("\n【维度 6】Per-class accuracy best→final shift (检测训练后期掉哪些 class)")
hA_b = heavy_load(A_DIR, "best"); hA_f = heavy_load(A_DIR, "final")
hB_b = heavy_load(B_DIR, "best"); hB_f = heavy_load(B_DIR, "final")
pca_b = per_class_acc(hA_b); pca_f = per_class_acc(hA_f)
pcb_b = per_class_acc(hB_b); pcb_f = per_class_acc(hB_f)
for dn in DOMAINS:
    if dn in pca_b and dn in pca_f and dn in pcb_b and dn in pcb_f:
        shift_a = (pca_f[dn] - pca_b[dn]) * 100
        shift_b = (pcb_f[dn] - pcb_b[dn]) * 100
        print(f"  {dn:>8s} PG-DFC+DaA shift: " + " ".join([f"{s:+5.1f}" for s in shift_a]))
        print(f"  {dn:>8s} F2DC+DaA   shift: " + " ".join([f"{s:+5.1f}" for s in shift_b]))

# 维度 7: cross-method CKA (PG vs F2DC representation similarity)
print("\n【维度 7】Cross-method CKA (representation similarity)")
print(f"  CKA(best):  PG-DFC+DaA vs F2DC+DaA = {cross_method_cka(hA_b, hB_b):.4f}")
print(f"  CKA(final): PG-DFC+DaA vs F2DC+DaA = {cross_method_cka(hA_f, hB_f):.4f}")

# 维度 2: per-domain proto pairwise (heavy reconstruct, mode collapse + class consistency)
print("\n【维度 2】Per-domain proto pairwise (heavy reconstruct, final R100)")
print("  inter-class pairwise: 越低越好 (类间区分大)")
print("  intra-class inter-domain: 越高越好 (同 class 跨 domain 一致)")
proA, vA = per_domain_proto(hA_f)
proB, vB = per_domain_proto(hB_f)
ic_a = proto_pairwise_inter_class(proA, vA)
ic_b = proto_pairwise_inter_class(proB, vB)
icd_a = proto_intra_class_inter_domain(proA, vA)
icd_b = proto_intra_class_inter_domain(proB, vB)
print(f"  inter-class pairwise:    PG={ic_a:.4f}  F2DC={ic_b:.4f}  Δ={ic_a-ic_b:+.4f}  ({'PG 类区分差' if ic_a > ic_b else 'PG 类区分好'})")
print(f"  intra-class inter-domain: PG={icd_a:.4f}  F2DC={icd_b:.4f}  Δ={icd_a-icd_b:+.4f}  ({'PG 跨域一致好' if icd_a > icd_b else 'PG 跨域一致差'})")

# 综合诊断打分
print("\n" + "=" * 90)
print("【综合】PG-DFC+DaA vs F2DC+DaA 各维度判定 (单 seed s=15 office)")
print("=" * 90)
print(f"  AVG_Best:  PG {acc_a.max(0).mean():.2f} vs F2DC {acc_b.max(0).mean():.2f} → {'PG 赢' if acc_a.max(0).mean()>acc_b.max(0).mean() else 'PG 输'}")
print(f"  AVG_Last:  PG {acc_a[-1].mean():.2f} vs F2DC {acc_b[-1].mean():.2f} → {'PG 赢' if acc_a[-1].mean()>acc_b[-1].mean() else 'PG 输'}")
gap_a = acc_a.max(0).mean() - acc_a[-1].mean()
gap_b = acc_b.max(0).mean() - acc_b[-1].mean()
print(f"  Stability gap: PG {gap_a:.2f} vs F2DC {gap_b:.2f} → {'PG 更稳' if gap_a<gap_b else 'PG 更不稳'}")
