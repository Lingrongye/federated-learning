"""
3 seed (s=2/s=15/s=333) 平均诊断指标对比 PG-DFC+DaA vs F2DC+DaA on office.
不预设结论, 全部 7 维度指标算一遍.
"""
import os, json, numpy as np
from glob import glob
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_office"
A_DIRS = {2: "diag_pgdfc_daa_office_s2", 15: "diag_pgdfc_daa_office_s15", 333: "diag_pgdfc_daa_office_s333"}
B_DIRS = {2: "diag_f2dc_daa_office_s2", 15: "diag_f2dc_daa_office_s15", 333: "diag_f2dc_daa_office_s333"}
DOMAINS = ["caltech", "amazon", "webcam", "dslr"]


def per_round_acc(d):
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    rs, accs = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        rs.append(int(z["round"])); accs.append(z["per_domain_acc"])
    return np.array(rs), np.stack(accs)


def daa_dispatch(d):
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    ratios = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        s, fr = z["sample_shares"], z["daa_freqs"]
        if fr.sum() == 0: return None, None
        ratios.append(fr / np.maximum(s, 1e-9))
    z0 = np.load(files[0], allow_pickle=True)
    return np.mean(ratios, axis=0), list(z0["domain_per_client"])


def aggregate_grad_drift(d):
    files = sorted(glob(os.path.join(ROOT, d, "round_*.npz")))
    grad_l2, layer_l2 = [], {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        grad_l2.append(z["grad_l2"].mean())
        pkl = z["layer_l2_pickle"][0]
        if isinstance(pkl, str): pkl = json.loads(pkl)
        for cid, lr in pkl.items():
            for lk, lv in lr.items():
                layer_l2.setdefault(lk, []).append(float(lv))
    return np.array(grad_l2), {k: np.mean(v) for k, v in layer_l2.items()}


def heavy_load(d, kind):
    files = sorted(glob(os.path.join(ROOT, d, f"{kind}_*.npz")))
    if not files: return None
    return np.load(files[0] if kind == "best" else files[-1], allow_pickle=True)


def feat_silhouette(heavy):
    if heavy is None: return None, None
    fd = heavy["features"].item(); ld = heavy["labels"].item()
    X, y, dom = [], [], []
    for di, dn in enumerate(DOMAINS):
        if dn not in fd: continue
        f = fd[dn].astype(np.float32); l = ld[dn].astype(np.int32)
        X.append(f); y.append(l); dom.append(np.full(len(f), di, dtype=np.int32))
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
        c = conf[dn]; diag = np.diag(c).astype(np.float32); rs = c.sum(1).astype(np.float32)
        out[dn] = np.where(rs > 0, diag / np.maximum(rs, 1), 0.0)
    return out


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Kx = X @ X.T; Ky = Y @ Y.T
    return float((Kx*Ky).sum() / (np.sqrt((Kx*Kx).sum()*(Ky*Ky).sum())+1e-9))


def cross_method_cka(ha, hb):
    if ha is None or hb is None: return None
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
    if heavy is None: return None, None
    feats = heavy["features"].item(); labels = heavy["labels"].item()
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
            if m.sum() > 0: proto[k, c] = f[m].mean(0); valid[k, c] = True
    return proto, valid


def cos(a, b): return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


def proto_pairwise_inter_class(proto, valid):
    if proto is None: return None
    K, C, _ = proto.shape; pairs = []
    for k in range(K):
        vc = np.where(valid[k])[0]
        for i in range(len(vc)):
            for j in range(i+1, len(vc)):
                pairs.append(cos(proto[k, vc[i]], proto[k, vc[j]]))
    return np.mean(pairs)


def proto_intra_class_inter_domain(proto, valid):
    if proto is None: return None
    K, C, _ = proto.shape; res = []
    for c in range(C):
        valid_k = np.where(valid[:, c])[0]
        for i in range(len(valid_k)):
            for j in range(i+1, len(valid_k)):
                res.append(cos(proto[valid_k[i], c], proto[valid_k[j], c]))
    return np.mean(res)


# 收集 3 seed 数据
data = {"PG-DFC+DaA": {}, "F2DC+DaA": {}}
for method, dirs in [("PG-DFC+DaA", A_DIRS), ("F2DC+DaA", B_DIRS)]:
    for seed, d in dirs.items():
        rs, accs = per_round_acc(d)
        gA, layA = aggregate_grad_drift(d)
        ratio, dom_clients = daa_dispatch(d)
        hb = heavy_load(d, "best"); hf = heavy_load(d, "final")
        sb_c, sb_d = feat_silhouette(hb)
        sf_c, sf_d = feat_silhouette(hf)
        proto_b, valid_b = per_domain_proto(hb)
        proto_f, valid_f = per_domain_proto(hf)
        data[method][seed] = {
            "acc_traj": accs, "rs": rs,
            "grad_l2_mean": gA.mean(), "grad_l2_final10": gA[-10:].mean(),
            "layer_drift": layA,
            "daa_ratio": ratio, "dom_clients": dom_clients,
            "sil_class_best": sb_c, "sil_domain_best": sb_d,
            "sil_class_final": sf_c, "sil_domain_final": sf_d,
            "best_per_class_acc": per_class_acc(hb), "final_per_class_acc": per_class_acc(hf),
            "proto_inter_class_final": proto_pairwise_inter_class(proto_f, valid_f),
            "proto_intra_class_final": proto_intra_class_inter_domain(proto_f, valid_f),
            "heavy_best": hb, "heavy_final": hf,
        }


def avg3(method, key):
    return np.mean([data[method][s][key] for s in [2, 15, 333]])


def avg_per_dom(method, what):
    """what: 'best' = max each domain across rounds; 'last' = last round"""
    arr = []
    for s in [2, 15, 333]:
        accs = data[method][s]["acc_traj"]
        if what == "best": arr.append(accs.max(axis=0))
        else: arr.append(accs[-1])
    return np.mean(arr, axis=0)


def avg_traj_at_round(method, r):
    arr = []
    for s in [2, 15, 333]:
        accs = data[method][s]["acc_traj"]
        if r-1 < len(accs): arr.append(accs[r-1].mean())
    return np.mean(arr) if arr else None


print("=" * 95)
print("V100 office 3-seed (s=2/s=15/s=333) 平均诊断指标对比")
print("=" * 95)

# 维度 5: 主表
print("\n【维度 5】Per-domain Best/Last (3-seed mean)")
print(f"{'domain':>10s} | {'PG-DFC+DaA Best/Last':>22s} | {'F2DC+DaA Best/Last':>22s} | Δ Best  Δ Last")
print("-" * 95)
pgB = avg_per_dom("PG-DFC+DaA", "best"); pgL = avg_per_dom("PG-DFC+DaA", "last")
f2B = avg_per_dom("F2DC+DaA", "best");   f2L = avg_per_dom("F2DC+DaA", "last")
for di, dn in enumerate(DOMAINS):
    print(f"{dn:>10s} | {pgB[di]:6.2f} / {pgL[di]:6.2f}        | {f2B[di]:6.2f} / {f2L[di]:6.2f}        | {pgB[di]-f2B[di]:+6.2f} {pgL[di]-f2L[di]:+6.2f}")
print(f"{'AVG':>10s} | {pgB.mean():6.2f} / {pgL.mean():6.2f}        | {f2B.mean():6.2f} / {f2L.mean():6.2f}        | {pgB.mean()-f2B.mean():+6.2f} {pgL.mean()-f2L.mean():+6.2f}")
print(f"{'gap (B-L)':>10s} | {pgB.mean()-pgL.mean():22.2f} | {f2B.mean()-f2L.mean():22.2f} | gap Δ {(pgB.mean()-pgL.mean())-(f2B.mean()-f2L.mean()):+.2f}")

# Acc trajectory
print("\n【维度 5】AVG acc trajectory (3-seed mean)")
print(f"{'round':>5s} | {'PG-DFC+DaA':>11s} | {'F2DC+DaA':>10s} | Δ")
for r in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    a, b = avg_traj_at_round("PG-DFC+DaA", r), avg_traj_at_round("F2DC+DaA", r)
    if a and b: print(f"{r:>5d} | {a:11.2f} | {b:10.2f} | {a-b:+.2f}")

# DaA dispatch
print("\n【维度 1】DaA dispatch ratio (3-seed mean)")
ra3 = np.mean([data["PG-DFC+DaA"][s]["daa_ratio"] for s in [2, 15, 333]], axis=0)
rb3 = np.mean([data["F2DC+DaA"][s]["daa_ratio"] for s in [2, 15, 333]], axis=0)
dom = data["PG-DFC+DaA"][2]["dom_clients"]
print(f"{'client':>6s} {'domain':>8s} | PG-DFC+DaA | F2DC+DaA | Δ")
for k in range(len(ra3)):
    print(f"{k:>6d} {dom[k]:>8s} | {ra3[k]:.3f}      | {rb3[k]:.3f}    | {ra3[k]-rb3[k]:+.3f}")

# Grad L2 + layer drift
print("\n【维度 1】Grad L2 + per-layer drift (3-seed mean)")
print(f"  Grad L2 mean (100R):       PG={avg3('PG-DFC+DaA','grad_l2_mean'):.3f}  F2DC={avg3('F2DC+DaA','grad_l2_mean'):.3f}  Δ={avg3('PG-DFC+DaA','grad_l2_mean')-avg3('F2DC+DaA','grad_l2_mean'):+.3f}")
print(f"  Grad L2 final 10R:         PG={avg3('PG-DFC+DaA','grad_l2_final10'):.3f}  F2DC={avg3('F2DC+DaA','grad_l2_final10'):.3f}  Δ={avg3('PG-DFC+DaA','grad_l2_final10')-avg3('F2DC+DaA','grad_l2_final10'):+.3f}")
print(f"  Per-layer drift:")
all_layers = sorted(data["PG-DFC+DaA"][2]["layer_drift"].keys())
for layer in all_layers:
    a = np.mean([data["PG-DFC+DaA"][s]["layer_drift"].get(layer, 0) for s in [2, 15, 333]])
    b = np.mean([data["F2DC+DaA"][s]["layer_drift"].get(layer, 0) for s in [2, 15, 333]])
    print(f"  {layer:<35s} | PG={a:.4f}  F2DC={b:.4f}  Δ={a-b:+.4f}")

# Silhouette
print("\n【维度 3】Feature silhouette (3-seed mean)")
print(f"  best  sil_class:  PG={avg3('PG-DFC+DaA','sil_class_best'):.4f}  F2DC={avg3('F2DC+DaA','sil_class_best'):.4f}  Δ={avg3('PG-DFC+DaA','sil_class_best')-avg3('F2DC+DaA','sil_class_best'):+.4f}")
print(f"  best  sil_domain: PG={avg3('PG-DFC+DaA','sil_domain_best'):.4f}  F2DC={avg3('F2DC+DaA','sil_domain_best'):.4f}  Δ={avg3('PG-DFC+DaA','sil_domain_best')-avg3('F2DC+DaA','sil_domain_best'):+.4f}")
print(f"  final sil_class:  PG={avg3('PG-DFC+DaA','sil_class_final'):.4f}  F2DC={avg3('F2DC+DaA','sil_class_final'):.4f}  Δ={avg3('PG-DFC+DaA','sil_class_final')-avg3('F2DC+DaA','sil_class_final'):+.4f}")
print(f"  final sil_domain: PG={avg3('PG-DFC+DaA','sil_domain_final'):.4f}  F2DC={avg3('F2DC+DaA','sil_domain_final'):.4f}  Δ={avg3('PG-DFC+DaA','sil_domain_final')-avg3('F2DC+DaA','sil_domain_final'):+.4f}")

# CKA (3-seed)
print("\n【维度 7】Cross-method CKA (3-seed mean)")
cka_b = np.mean([cross_method_cka(data["PG-DFC+DaA"][s]["heavy_best"], data["F2DC+DaA"][s]["heavy_best"]) for s in [2,15,333]])
cka_f = np.mean([cross_method_cka(data["PG-DFC+DaA"][s]["heavy_final"], data["F2DC+DaA"][s]["heavy_final"]) for s in [2,15,333]])
print(f"  CKA(best):  {cka_b:.4f}")
print(f"  CKA(final): {cka_f:.4f}")

# Per-domain proto pairwise
print("\n【维度 2】Per-domain proto pairwise (heavy reconstruct, 3-seed mean)")
ic_a = avg3("PG-DFC+DaA", "proto_inter_class_final")
ic_b = avg3("F2DC+DaA",  "proto_inter_class_final")
icd_a = avg3("PG-DFC+DaA", "proto_intra_class_final")
icd_b = avg3("F2DC+DaA",  "proto_intra_class_final")
print(f"  inter-class pairwise (越低越好): PG={ic_a:.4f}  F2DC={ic_b:.4f}  Δ={ic_a-ic_b:+.4f}")
print(f"  intra-class inter-domain (越高越好): PG={icd_a:.4f}  F2DC={icd_b:.4f}  Δ={icd_a-icd_b:+.4f}")

# Per-class shift averaged across 3 seeds
print("\n【维度 6】Per-class best→final shift (3-seed mean, 看后期掉哪些 class)")
for dn in DOMAINS:
    shifts_a, shifts_b = [], []
    for s in [2, 15, 333]:
        b_a = data["PG-DFC+DaA"][s]["best_per_class_acc"]; f_a = data["PG-DFC+DaA"][s]["final_per_class_acc"]
        b_b = data["F2DC+DaA"][s]["best_per_class_acc"];   f_b = data["F2DC+DaA"][s]["final_per_class_acc"]
        if dn in b_a and dn in f_a: shifts_a.append((f_a[dn]-b_a[dn])*100)
        if dn in b_b and dn in f_b: shifts_b.append((f_b[dn]-b_b[dn])*100)
    if shifts_a:
        sa = np.mean(shifts_a, axis=0); sb = np.mean(shifts_b, axis=0)
        print(f"  {dn:>8s} PG-DFC+DaA: " + " ".join([f"{x:+5.1f}" for x in sa]))
        print(f"  {dn:>8s} F2DC+DaA  : " + " ".join([f"{x:+5.1f}" for x in sb]))
        print(f"  {dn:>8s} Δ shift    : " + " ".join([f"{x:+5.1f}" for x in sa-sb]))

# 综合 verdict
print("\n" + "=" * 95)
print("【综合】PG-DFC+DaA vs F2DC+DaA 3-seed mean 各维度判定")
print("=" * 95)
b_a, b_b = pgB.mean(), f2B.mean()
l_a, l_b = pgL.mean(), f2L.mean()
print(f"  AVG_Best:  PG {b_a:.2f} vs F2DC {b_b:.2f} → {'⭐ PG 赢' if b_a>b_b else 'PG 输'} {b_a-b_b:+.2f}pp")
print(f"  AVG_Last:  PG {l_a:.2f} vs F2DC {l_b:.2f} → {'⭐ PG 赢' if l_a>l_b else 'PG 输'} {l_a-l_b:+.2f}pp")
print(f"  Stability gap: PG {b_a-l_a:.2f} vs F2DC {b_b-l_b:.2f} → {'⭐ PG 更稳' if b_a-l_a<b_b-l_b else 'PG 更不稳'} Δ={(b_a-l_a)-(b_b-l_b):+.2f}pp")
