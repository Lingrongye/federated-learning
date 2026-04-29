"""
PACS 2-seed (s=15, s=333) 全维度诊断对比 — 找 PG-DFC+DaA 反输的真实机制.
不预设结论, 4 method × 7 维指标全跑.
"""
import os, json, numpy as np
from glob import glob
from sklearn.metrics import silhouette_score

ROOT = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_pacs"
DOMAINS = ["photo", "art", "cartoon", "sketch"]

METHODS = {
    "F2DC":         {15: "diag_f2dc_pacs_s15",       333: "diag_f2dc_pacs_s333"},
    "F2DC+DaA":     {15: "diag_f2dc_daa_pacs_s15",   333: "diag_f2dc_daa_pacs_s333"},
    "PG-DFC":       {15: "diag_pgdfc_pacs_s15",      333: "diag_pgdfc_pacs_s333"},
    "PG-DFC+DaA":   {15: "diag_pgdfc_daa_pacs_s15",  333: "diag_pgdfc_daa_pacs_s333"},
}


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


# 收集 4 method × 2 seed 数据
data = {m: {} for m in METHODS}
for method, dirs in METHODS.items():
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


def avg2(method, key):
    vals = [data[method][s][key] for s in [15, 333]]
    return np.mean([v for v in vals if v is not None])


def avg_per_dom(method, what):
    arr = []
    for s in [15, 333]:
        accs = data[method][s]["acc_traj"]
        if what == "best": arr.append(accs.max(axis=0))
        else: arr.append(accs[-1])
    return np.mean(arr, axis=0)


print("=" * 100)
print("PACS sc 高质量数据 2-seed (s=15+s=333) 全维度诊断对比")
print("=" * 100)

print("\n【维度 5】Per-domain Best/Last (2-seed mean)")
print(f"{'Method':>15s} | " + " ".join([f"{d:>8s}B/L" for d in DOMAINS]) + f" | {'AVG_B':>6s} {'AVG_L':>6s} {'gap':>5s}")
print("-" * 100)
for method in ["F2DC", "F2DC+DaA", "PG-DFC", "PG-DFC+DaA"]:
    pgB = avg_per_dom(method, "best"); pgL = avg_per_dom(method, "last")
    row = f"{method:>15s} | "
    for di, dn in enumerate(DOMAINS):
        row += f"{pgB[di]:5.2f}/{pgL[di]:5.2f} "
    row += f"| {pgB.mean():6.2f} {pgL.mean():6.2f} {pgB.mean()-pgL.mean():5.2f}"
    print(row)

# Δ 对比 PG-DFC+DaA vs F2DC+DaA + vs F2DC vanilla
print("\n【维度 5】关键 Δ 对比")
pg_b = avg_per_dom("PG-DFC+DaA", "best"); pg_l = avg_per_dom("PG-DFC+DaA", "last")
f2_b = avg_per_dom("F2DC+DaA", "best"); f2_l = avg_per_dom("F2DC+DaA", "last")
fv_b = avg_per_dom("F2DC", "best"); fv_l = avg_per_dom("F2DC", "last")
print(f"  Δ PG-DFC+DaA vs F2DC+DaA   per-dom Best: " + " ".join([f"{d}={pg_b[i]-f2_b[i]:+5.2f}" for i,d in enumerate(DOMAINS)]) + f" | AVG {pg_b.mean()-f2_b.mean():+5.2f}")
print(f"  Δ PG-DFC+DaA vs F2DC+DaA   per-dom Last: " + " ".join([f"{d}={pg_l[i]-f2_l[i]:+5.2f}" for i,d in enumerate(DOMAINS)]) + f" | AVG {pg_l.mean()-f2_l.mean():+5.2f}")
print(f"  Δ PG-DFC+DaA vs F2DC vanilla per-dom Best: " + " ".join([f"{d}={pg_b[i]-fv_b[i]:+5.2f}" for i,d in enumerate(DOMAINS)]) + f" | AVG {pg_b.mean()-fv_b.mean():+5.2f}")

# DaA dispatch
print("\n【维度 1】DaA dispatch ratio (2-seed mean)")
ra2 = np.mean([data["PG-DFC+DaA"][s]["daa_ratio"] for s in [15, 333]], axis=0)
rb2 = np.mean([data["F2DC+DaA"][s]["daa_ratio"] for s in [15, 333]], axis=0)
dom = data["PG-DFC+DaA"][15]["dom_clients"]
print(f"  按 domain 聚合 (mean ratio):")
dom_ratios_pg = {}; dom_ratios_f2 = {}
for k in range(len(ra2)):
    dom_ratios_pg.setdefault(dom[k], []).append(ra2[k])
    dom_ratios_f2.setdefault(dom[k], []).append(rb2[k])
for dn in DOMAINS:
    if dn in dom_ratios_pg:
        n_clients = len(dom_ratios_pg[dn])
        m_pg = np.mean(dom_ratios_pg[dn])
        m_f2 = np.mean(dom_ratios_f2[dn])
        print(f"    {dn:>8s} ({n_clients} client): PG-DFC+DaA ratio={m_pg:.3f}  F2DC+DaA ratio={m_f2:.3f}  Δ={m_pg-m_f2:+.3f}")

# Grad L2 + layer drift
print("\n【维度 1】Grad L2 + per-layer drift (2-seed mean)")
print(f"  Grad L2 mean:     PG-DFC+DaA={avg2('PG-DFC+DaA','grad_l2_mean'):.3f}  F2DC+DaA={avg2('F2DC+DaA','grad_l2_mean'):.3f}  Δ={avg2('PG-DFC+DaA','grad_l2_mean')-avg2('F2DC+DaA','grad_l2_mean'):+.3f}")
print(f"  Grad L2 final 10: PG-DFC+DaA={avg2('PG-DFC+DaA','grad_l2_final10'):.3f}  F2DC+DaA={avg2('F2DC+DaA','grad_l2_final10'):.3f}  Δ={avg2('PG-DFC+DaA','grad_l2_final10')-avg2('F2DC+DaA','grad_l2_final10'):+.3f}")
print(f"  Per-layer drift (avg over 100R):")
all_layers = sorted(data["PG-DFC+DaA"][15]["layer_drift"].keys())
for layer in all_layers:
    a = np.mean([data["PG-DFC+DaA"][s]["layer_drift"].get(layer, 0) for s in [15, 333]])
    b = np.mean([data["F2DC+DaA"][s]["layer_drift"].get(layer, 0) for s in [15, 333]])
    print(f"    {layer:<35s} | PG={a:.4f}  F2DC={b:.4f}  Δ={a-b:+.4f}")

# Silhouette
print("\n【维度 3】Feature silhouette (best vs final, 2-seed mean)")
for snap in ["best", "final"]:
    ck_a = avg2("PG-DFC+DaA", f"sil_class_{snap}"); ck_b = avg2("F2DC+DaA", f"sil_class_{snap}")
    dk_a = avg2("PG-DFC+DaA", f"sil_domain_{snap}"); dk_b = avg2("F2DC+DaA", f"sil_domain_{snap}")
    print(f"  {snap:>5s}: sil_class  PG={ck_a:.4f}  F2DC={ck_b:.4f}  Δ={ck_a-ck_b:+.4f}")
    print(f"  {snap:>5s}: sil_domain PG={dk_a:.4f}  F2DC={dk_b:.4f}  Δ={dk_a-dk_b:+.4f}")

# CKA
print("\n【维度 7】Cross-method CKA (2-seed mean)")
cka_b = np.mean([cross_method_cka(data["PG-DFC+DaA"][s]["heavy_best"], data["F2DC+DaA"][s]["heavy_best"]) for s in [15,333]])
cka_f = np.mean([cross_method_cka(data["PG-DFC+DaA"][s]["heavy_final"], data["F2DC+DaA"][s]["heavy_final"]) for s in [15,333]])
print(f"  CKA(best):  PG-DFC+DaA vs F2DC+DaA = {cka_b:.4f}")
print(f"  CKA(final): PG-DFC+DaA vs F2DC+DaA = {cka_f:.4f}")

# Per-domain proto pairwise
print("\n【维度 2】Per-domain proto pairwise (heavy reconstruct, final, 2-seed mean)")
ic_a = avg2("PG-DFC+DaA", "proto_inter_class_final")
ic_b = avg2("F2DC+DaA",  "proto_inter_class_final")
icd_a = avg2("PG-DFC+DaA", "proto_intra_class_final")
icd_b = avg2("F2DC+DaA",  "proto_intra_class_final")
print(f"  inter-class pairwise (越低越好): PG={ic_a:.4f}  F2DC={ic_b:.4f}  Δ={ic_a-ic_b:+.4f}")
print(f"  intra-class inter-domain (越高越好): PG={icd_a:.4f}  F2DC={icd_b:.4f}  Δ={icd_a-icd_b:+.4f}")

# Per-class shift averaged
print("\n【维度 6】Per-class best→final shift (2-seed mean, 看后期掉哪些 class)")
for dn in DOMAINS:
    shifts_a, shifts_b = [], []
    for s in [15, 333]:
        b_a = data["PG-DFC+DaA"][s]["best_per_class_acc"]; f_a = data["PG-DFC+DaA"][s]["final_per_class_acc"]
        b_b = data["F2DC+DaA"][s]["best_per_class_acc"];   f_b = data["F2DC+DaA"][s]["final_per_class_acc"]
        if dn in b_a and dn in f_a: shifts_a.append((f_a[dn]-b_a[dn])*100)
        if dn in b_b and dn in f_b: shifts_b.append((f_b[dn]-b_b[dn])*100)
    if shifts_a:
        sa = np.mean(shifts_a, axis=0); sb = np.mean(shifts_b, axis=0)
        print(f"  {dn:>8s} PG-DFC+DaA shift: " + " ".join([f"{x:+5.1f}" for x in sa]))
        print(f"  {dn:>8s} F2DC+DaA   shift: " + " ".join([f"{x:+5.1f}" for x in sb]))
        print(f"  {dn:>8s} Δ shift          : " + " ".join([f"{x:+5.1f}" for x in sa-sb]))

# 综合 verdict
print("\n" + "=" * 100)
print("【综合】PACS sc 数据 4 method 排名 (2-seed mean)")
print("=" * 100)
all_b = {m: avg_per_dom(m, "best").mean() for m in METHODS}
all_l = {m: avg_per_dom(m, "last").mean() for m in METHODS}
sorted_m = sorted(METHODS.keys(), key=lambda m: -all_b[m])
print(f"\n排名 (按 AVG_B 高→低):")
for i, m in enumerate(sorted_m):
    medal = ["🥇","🥈","🥉","❌"][i]
    print(f"  {medal} {m:>15s}: AVG_B={all_b[m]:6.2f}  AVG_L={all_l[m]:6.2f}  gap={all_b[m]-all_l[m]:5.2f}")
