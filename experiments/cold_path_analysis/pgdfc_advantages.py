"""
找 PG-DFC 真正的优点 — 公平对比 (PG-DFC vs F2DC, 同 setup), 看哪些指标 PG 系统性更好.
"""
import os, json, numpy as np
from glob import glob
from sklearn.metrics import silhouette_score


def heavy_load(d, kind):
    files = sorted(glob(os.path.join(d, f"{kind}_*.npz")))
    if not files: return None
    return np.load(files[0] if kind == "best" else files[-1], allow_pickle=True)


def per_round_acc(d):
    files = sorted(glob(os.path.join(d, "round_*.npz")))
    return np.stack([np.load(f, allow_pickle=True)["per_domain_acc"] for f in files])


def feat_silhouette(heavy, doms):
    if heavy is None: return None, None
    fd = heavy["features"].item(); ld = heavy["labels"].item()
    X, y, dom = [], [], []
    for di, dn in enumerate(doms):
        if dn not in fd: continue
        X.append(fd[dn].astype(np.float32))
        y.append(ld[dn].astype(np.int32))
        dom.append(np.full(len(fd[dn]), di))
    X = np.concatenate(X); y = np.concatenate(y); dom = np.concatenate(dom)
    if len(X) > 1500:
        idx = np.random.RandomState(0).choice(len(X), 1500, replace=False)
        X, y, dom = X[idx], y[idx], dom[idx]
    return float(silhouette_score(X, y)), float(silhouette_score(X, dom))


def per_class_acc(heavy, doms):
    if heavy is None: return None
    conf = heavy["confusion"].item()
    out = {}
    for dn in doms:
        if dn in conf:
            c = conf[dn]; diag = np.diag(c).astype(np.float32); rs = c.sum(1).astype(np.float32)
            out[dn] = np.where(rs > 0, diag / np.maximum(rs, 1), 0.0)
    return out


def per_domain_proto(heavy, doms):
    if heavy is None: return None, None
    feats = heavy["features"].item(); labels = heavy["labels"].item()
    all_l = np.concatenate([labels[d] for d in doms if d in labels])
    C = int(all_l.max() + 1)
    K, dim = len(doms), feats[doms[0]].shape[1]
    proto = np.zeros((K, C, dim), dtype=np.float32)
    valid = np.zeros((K, C), dtype=bool)
    for k, dn in enumerate(doms):
        if dn not in feats: continue
        f, l = feats[dn].astype(np.float32), labels[dn].astype(np.int32)
        for c in range(C):
            m = l == c
            if m.sum() > 0: proto[k, c] = f[m].mean(0); valid[k, c] = True
    return proto, valid


def cos(a, b): return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


def proto_intra_class_inter_domain(proto, valid):
    """同 class 跨 domain 的 cos sim — 越高表示 class 跨 domain 一致 (good)"""
    K, C, _ = proto.shape; res = []
    for c in range(C):
        valid_k = np.where(valid[:, c])[0]
        for i in range(len(valid_k)):
            for j in range(i+1, len(valid_k)):
                res.append(cos(proto[valid_k[i], c], proto[valid_k[j], c]))
    return np.mean(res)


# Office (V100 fixed allocation, 跟 PACS 一样公平 setup)
print("=" * 95)
print("Office (V100 fixed allocation, 3-seed mean) — PG-DFC vs F2DC 各项对比")
print("=" * 95)
OFFICE = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_office"
DOMS_O = ["caltech", "amazon", "webcam", "dslr"]

# vanilla 对比 (不带 DaA, 看纯 PG-DFC backbone 效果)
metrics = {"PG vanilla": [], "F2DC vanilla": [], "PG+DaA": [], "F2DC+DaA": []}
for s in [2, 15, 333]:
    for label, sub in [("PG vanilla", f"diag_pgdfc_office_s{s}" if s != 333 else None),
                        ("F2DC vanilla", f"diag_f2dc_office_s{s}" if s != 333 else None),
                        ("PG+DaA", f"diag_pgdfc_daa_office_s{s}"),
                        ("F2DC+DaA", f"diag_f2dc_daa_office_s{s}")]:
        if sub is None: continue
        d = os.path.join(OFFICE, sub)
        if not os.path.isdir(d): continue
        acc = per_round_acc(d)
        hf = heavy_load(d, "final")
        sc, sd = feat_silhouette(hf, DOMS_O)
        proto, valid = per_domain_proto(hf, DOMS_O)
        intra = proto_intra_class_inter_domain(proto, valid)
        metrics[label].append({
            "best_avg": acc.max(0).mean(),
            "last_avg": acc[-1].mean(),
            "gap": acc.max(0).mean() - acc[-1].mean(),
            "best_dslr": acc[:, 3].max(),
            "last_dslr": acc[-1, 3],
            "sil_class_final": sc,
            "sil_dom_final": sd,
            "intra_class_consistency": intra,
        })


def avg_dict(arr, k):
    vals = [a[k] for a in arr]
    return np.mean(vals)


print(f"{'指标':>30s} | {'PG vanilla':>11s} {'F2DC vanilla':>13s} | {'PG+DaA':>8s} {'F2DC+DaA':>10s}")
print("-" * 95)
for k in ["best_avg", "last_avg", "gap", "best_dslr", "last_dslr",
          "sil_class_final", "sil_dom_final", "intra_class_consistency"]:
    pg_v = avg_dict(metrics["PG vanilla"], k)
    f2_v = avg_dict(metrics["F2DC vanilla"], k)
    pg_d = avg_dict(metrics["PG+DaA"], k)
    f2_d = avg_dict(metrics["F2DC+DaA"], k)
    win_v = "✅PG" if (k.startswith("best") or k.startswith("last") or k == "sil_class_final" or k == "intra_class_consistency") and pg_v > f2_v else ("✅PG" if (k == "gap" or k == "sil_dom_final") and pg_v < f2_v else "❌F2")
    win_d = "✅PG" if (k.startswith("best") or k.startswith("last") or k == "sil_class_final" or k == "intra_class_consistency") and pg_d > f2_d else ("✅PG" if (k == "gap" or k == "sil_dom_final") and pg_d < f2_d else "❌F2")
    print(f"{k:>30s} | {pg_v:11.4f} {f2_v:13.4f} {win_v} | {pg_d:8.4f} {f2_d:10.4f} {win_d}")

# PACS (sc fixed allocation, 2-seed mean)
print("\n" + "=" * 95)
print("PACS (sc fixed allocation, 2-seed mean) — PG-DFC vs F2DC 各项对比")
print("=" * 95)
PACS = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_pacs"
DOMS_P = ["photo", "art", "cartoon", "sketch"]

metrics_p = {"PG vanilla": [], "F2DC vanilla": [], "PG+DaA": [], "F2DC+DaA": []}
for s in [15, 333]:
    for label, sub in [("PG vanilla", f"diag_pgdfc_pacs_s{s}"),
                        ("F2DC vanilla", f"diag_f2dc_pacs_s{s}"),
                        ("PG+DaA", f"diag_pgdfc_daa_pacs_s{s}"),
                        ("F2DC+DaA", f"diag_f2dc_daa_pacs_s{s}")]:
        d = os.path.join(PACS, sub)
        if not os.path.isdir(d): continue
        acc = per_round_acc(d)
        hf = heavy_load(d, "final")
        sc, sd = feat_silhouette(hf, DOMS_P)
        proto, valid = per_domain_proto(hf, DOMS_P)
        intra = proto_intra_class_inter_domain(proto, valid)
        # 找最稀有 domain (sample share 最少的)
        # PACS 的 photo 是最稀有 (photo client 2 个 sample 最少)
        metrics_p[label].append({
            "best_avg": acc.max(0).mean(),
            "last_avg": acc[-1].mean(),
            "gap": acc.max(0).mean() - acc[-1].mean(),
            "best_photo": acc[:, 0].max(),
            "last_photo": acc[-1, 0],
            "best_art": acc[:, 1].max(),
            "last_art": acc[-1, 1],
            "sil_class_final": sc,
            "sil_dom_final": sd,
            "intra_class_consistency": intra,
        })

print(f"{'指标':>30s} | {'PG vanilla':>11s} {'F2DC vanilla':>13s} | {'PG+DaA':>8s} {'F2DC+DaA':>10s}")
print("-" * 95)
for k in ["best_avg", "last_avg", "gap", "best_photo", "last_photo", "best_art", "last_art",
          "sil_class_final", "sil_dom_final", "intra_class_consistency"]:
    pg_v = avg_dict(metrics_p["PG vanilla"], k)
    f2_v = avg_dict(metrics_p["F2DC vanilla"], k)
    pg_d = avg_dict(metrics_p["PG+DaA"], k)
    f2_d = avg_dict(metrics_p["F2DC+DaA"], k)
    win_v = "✅PG" if (k.startswith("best") or k.startswith("last") or k == "sil_class_final" or k == "intra_class_consistency") and pg_v > f2_v else ("✅PG" if (k == "gap" or k == "sil_dom_final") and pg_v < f2_v else "❌F2")
    win_d = "✅PG" if (k.startswith("best") or k.startswith("last") or k == "sil_class_final" or k == "intra_class_consistency") and pg_d > f2_d else ("✅PG" if (k == "gap" or k == "sil_dom_final") and pg_d < f2_d else "❌F2")
    print(f"{k:>30s} | {pg_v:11.4f} {f2_v:13.4f} {win_v} | {pg_d:8.4f} {f2_d:10.4f} {win_d}")

print("\n" + "=" * 95)
print("综合: PG-DFC 在哪些维度系统性赢")
print("=" * 95)
print("看 ✅PG 出现的频次 (越多说明 PG 在该指标越稳定赢)")
