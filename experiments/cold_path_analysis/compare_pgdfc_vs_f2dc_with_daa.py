"""
全维度对比 PG-DFC+DaA vs F2DC+DaA on Office (s=2 + s=15).
覆盖诊断体系 7 个维度. 输出 markdown 给 obsidian.
"""
import json
import os
import sys
import numpy as np
from glob import glob
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = "/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_office"

METHODS = {
    "F2DC":         {"f2dc_office_s2": 2, "f2dc_office_s15": 15},
    "F2DC+DaA":     {"f2dc_daa_office_s2": 2, "f2dc_daa_office_s15": 15, "f2dc_daa_office_s333": 333},
    "PG-DFC":       {"pgdfc_office_s2": 2, "pgdfc_office_s15": 15},
    "PG-DFC+DaA":   {"pgdfc_daa_office_s2": 2, "pgdfc_daa_office_s15": 15, "pgdfc_daa_office_s333": 333},
}

DOMAINS = ["caltech", "amazon", "webcam", "dslr"]


def load_round(diag_dir, r):
    return np.load(os.path.join(diag_dir, f"round_{r:03d}.npz"), allow_pickle=True)


def load_heavy(diag_dir, kind):
    files = sorted(glob(os.path.join(diag_dir, f"{kind}_*.npz")))
    if not files:
        return None
    return np.load(files[-1] if kind == "final" else files[0], allow_pickle=True)


def per_round_acc_traj(diag_dir):
    """Returns (rounds, acc_per_domain (R, D))"""
    rs, accs = [], []
    files = sorted(glob(os.path.join(diag_dir, "round_*.npz")))
    for f in files:
        d = np.load(f, allow_pickle=True)
        rs.append(int(d["round"]))
        accs.append(d["per_domain_acc"])
    return np.array(rs), np.stack(accs)


def domain_per_client(diag_dir):
    d = load_round(diag_dir, 1)
    return list(d["domain_per_client"]), list(d["sample_shares"]), list(d["all_dataset_names"])


def daa_dispatch_summary(diag_dir):
    """avg over rounds: daa_freq / sample_share per client"""
    files = sorted(glob(os.path.join(diag_dir, "round_*.npz")))
    ratios = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        s = d["sample_shares"]
        f_arr = d["daa_freqs"]
        if f_arr.sum() == 0:  # vanilla, no DaA
            return None
        ratios.append(f_arr / np.maximum(s, 1e-9))
    return np.mean(ratios, axis=0)  # (K,)


def effective_contribution(diag_dir):
    """avg over rounds: weight * grad_l2 per client"""
    files = sorted(glob(os.path.join(diag_dir, "round_*.npz")))
    contribs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        f_arr = d["daa_freqs"]
        if f_arr.sum() == 0:
            f_arr = d["sample_shares"]
        g = d["grad_l2"]
        contribs.append(f_arr * g)
    return np.mean(contribs, axis=0)  # (K,)


def proto_cos_sim_traj(diag_dir):
    """For each round: avg cos sim between local_protos[i,c] and consensus across clients.
       Returns (R, K) per-client cos sim averaged over class. None if no local_protos.
    """
    files = sorted(glob(os.path.join(diag_dir, "round_*.npz")))
    traj = []
    rs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "local_protos" not in d.files:
            return None, None
        rs.append(int(d["round"]))
        lp = d["local_protos"].astype(np.float32)  # (K, C, dim)
        consensus = lp.mean(axis=0)  # (C, dim)
        K, C, dim = lp.shape
        cos_per_client = []
        for k in range(K):
            sim = (lp[k] * consensus).sum(-1) / (
                np.linalg.norm(lp[k], axis=-1) * np.linalg.norm(consensus, axis=-1) + 1e-9
            )
            cos_per_client.append(sim.mean())
        traj.append(cos_per_client)
    return np.array(rs), np.array(traj)  # (R,K)


def mode_collapse_score(diag_dir, round_idx):
    """off-diagonal pairwise cos sim of consensus protos at round_idx. None if no protos."""
    d = load_round(diag_dir, round_idx)
    if "local_protos" not in d.files:
        return None
    lp = d["local_protos"].astype(np.float32).mean(axis=0)  # (C, dim) consensus
    sim = cosine_similarity(lp)
    C = sim.shape[0]
    off = (sim.sum() - np.trace(sim)) / (C * (C - 1))
    return off


def per_layer_drift_avg(diag_dir):
    """Average layer_l2 across rounds + clients, returns per-layer mean dict"""
    files = sorted(glob(os.path.join(diag_dir, "round_*.npz")))
    by_layer = {}
    n = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        pkl = d["layer_l2_pickle"][0]
        if isinstance(pkl, str):
            pkl = json.loads(pkl)
        for cli, layers in pkl.items():
            for layer, v in layers.items():
                by_layer[layer] = by_layer.get(layer, 0.0) + float(v)
                n[layer] = n.get(layer, 0) + 1
    return {k: by_layer[k] / max(n[k], 1) for k in by_layer}


def heavy_silhouette(heavy):
    """sil_class, sil_domain over all features"""
    if heavy is None:
        return None, None
    feats_d = heavy["features"].item()
    labels_d = heavy["labels"].item()
    X, y, dom = [], [], []
    for di, dn in enumerate(DOMAINS):
        if dn not in feats_d:
            continue
        f = feats_d[dn].astype(np.float32)
        l = labels_d[dn].astype(np.int32)
        X.append(f); y.append(l)
        dom.append(np.full(len(f), di, dtype=np.int32))
    X = np.concatenate(X); y = np.concatenate(y); dom = np.concatenate(dom)
    # Subsample if too big
    if len(X) > 1500:
        idx = np.random.RandomState(0).choice(len(X), 1500, replace=False)
        X, y, dom = X[idx], y[idx], dom[idx]
    sc = silhouette_score(X, y) if len(set(y)) > 1 else float("nan")
    sd = silhouette_score(X, dom) if len(set(dom)) > 1 else float("nan")
    return float(sc), float(sd)


def per_domain_per_class_acc(heavy):
    if heavy is None:
        return None
    conf_d = heavy["confusion"].item()
    out = {}
    for dn in DOMAINS:
        if dn not in conf_d:
            continue
        c = conf_d[dn]
        diag = np.diag(c).astype(np.float32)
        row_sum = c.sum(axis=1).astype(np.float32)
        per_class = np.where(row_sum > 0, diag / np.maximum(row_sum, 1), 0.0)
        out[dn] = per_class
    return out


def overall_acc_from_conf(heavy):
    if heavy is None:
        return None
    conf_d = heavy["confusion"].item()
    accs = {}
    for dn, c in conf_d.items():
        diag = np.diag(c).sum()
        total = c.sum()
        accs[dn] = float(diag / max(total, 1))
    return accs


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Kx = X @ X.T
    Ky = Y @ Y.T
    return float((Kx * Ky).sum() / (np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()) + 1e-9))


def cross_method_cka(heavy_a, heavy_b):
    if heavy_a is None or heavy_b is None:
        return None
    fa = heavy_a["features"].item()
    fb = heavy_b["features"].item()
    Xa, Xb = [], []
    for dn in DOMAINS:
        if dn not in fa or dn not in fb:
            continue
        n = min(len(fa[dn]), len(fb[dn]))
        Xa.append(fa[dn][:n].astype(np.float32))
        Xb.append(fb[dn][:n].astype(np.float32))
    Xa = np.concatenate(Xa); Xb = np.concatenate(Xb)
    if len(Xa) > 800:
        idx = np.random.RandomState(0).choice(len(Xa), 800, replace=False)
        Xa, Xb = Xa[idx], Xb[idx]
    return cka(Xa, Xb)


# ============ MAIN COMPARISON ============

def main():
    print("=" * 90)
    print("PG-DFC+DaA vs F2DC+DaA on Office: 全维度诊断对比")
    print("=" * 90)

    # 1. 收集所有实验数据
    all_data = {}  # method -> seed -> {acc_traj, daa_dispatch, ...}
    for method, runs in METHODS.items():
        for run_name, seed in runs.items():
            d = os.path.join(ROOT, f"diag_{run_name}")
            if not os.path.isdir(d):
                continue
            rs, accs = per_round_acc_traj(d)
            heavy_best = load_heavy(d, "best")
            heavy_final = load_heavy(d, "final")
            domain_clients, shares, dnames = domain_per_client(d)
            entry = {
                "dir": d,
                "rounds": rs,
                "acc_traj": accs,  # (R, D)
                "best_round": heavy_best["round"].item() if heavy_best is not None else None,
                "best_acc": float(heavy_best["current_acc"]) if heavy_best is not None else None,
                "final_acc": float(heavy_final["current_acc"]) if heavy_final is not None else None,
                "domain_per_client": domain_clients,
                "sample_shares": shares,
                "domain_names": dnames,
                "best_per_domain_class": per_domain_per_class_acc(heavy_best),
                "final_per_domain_class": per_domain_per_class_acc(heavy_final),
                "best_overall_acc": overall_acc_from_conf(heavy_best),
                "final_overall_acc": overall_acc_from_conf(heavy_final),
                "sil_best": heavy_silhouette(heavy_best),
                "sil_final": heavy_silhouette(heavy_final),
                "daa_dispatch": daa_dispatch_summary(d),
                "eff_contrib": effective_contribution(d),
                "proto_cos_traj": proto_cos_sim_traj(d),
                "mode_collapse_r1": mode_collapse_score(d, 1),
                "mode_collapse_r50": mode_collapse_score(d, 50),
                "mode_collapse_r100": mode_collapse_score(d, 100),
                "layer_drift": per_layer_drift_avg(d),
                "heavy_best": heavy_best,
                "heavy_final": heavy_final,
            }
            all_data.setdefault(method, {})[seed] = entry
            print(f"  loaded {method} seed={seed}: best=R{entry['best_round']} acc={entry['best_acc']:.4f}")

    # 2. 主表 acc 对比 (best per domain, AVG, Last)
    print("\n" + "=" * 90)
    print("【维度 5】Acc 主表对比 (4 method × 2-3 seed)")
    print("=" * 90)
    print(f"{'Method':18s} {'seed':4s} | {'caltech':>7s} {'amazon':>7s} {'webcam':>7s} {'dslr':>7s} | {'AVG_B':>6s} {'AVG_L':>6s} {'gap':>5s}")
    print("-" * 90)

    summary_rows = []
    for method in METHODS:
        if method not in all_data:
            continue
        for seed in sorted(all_data[method].keys()):
            e = all_data[method][seed]
            # Best per domain across all rounds
            best_per_dom = e["acc_traj"].max(axis=0)  # (D,)
            last_per_dom = e["acc_traj"][-1]  # (D,)
            avg_b = best_per_dom.mean()
            avg_l = last_per_dom.mean()
            row = f"{method:18s} {seed:4d} | "
            for di, dn in enumerate(e["domain_names"]):
                row += f"{best_per_dom[di]:7.2f} "
            row += f"| {avg_b:6.2f} {avg_l:6.2f} {avg_b-avg_l:5.2f}"
            print(row)
            summary_rows.append((method, seed, list(best_per_dom), list(last_per_dom), avg_b, avg_l))

    # 3. DaA dispatch 对比 (PG-DFC+DaA vs F2DC+DaA)
    print("\n" + "=" * 90)
    print("【维度 1】DaA dispatch ratio (daa_freq / sample_share, avg across rounds)")
    print("=" * 90)
    for method in ["F2DC+DaA", "PG-DFC+DaA"]:
        if method not in all_data:
            continue
        seeds = sorted(all_data[method].keys())
        e = all_data[method][seeds[0]]  # use first seed (consistent across seeds)
        ratio = e["daa_dispatch"]
        if ratio is None:
            continue
        print(f"\n{method} (seed={seeds[0]}): 10 client dispatch ratio")
        domain_clients = e["domain_per_client"]
        for k, (dom, r) in enumerate(zip(domain_clients, ratio)):
            tag = " ⬆升权" if r > 1.05 else (" ⬇降权" if r < 0.95 else " ≈持平")
            print(f"  client {k} ({dom:8s}): {r:.3f}{tag}")

    # 4. Effective contribution 对比
    print("\n" + "=" * 90)
    print("【维度 1】Effective contribution = weight × ‖Δw‖ (avg across rounds)")
    print("=" * 90)
    print(f"{'method':14s} | " + " ".join([f"c{i}" for i in range(10)]))
    for method in ["F2DC", "F2DC+DaA", "PG-DFC", "PG-DFC+DaA"]:
        if method not in all_data:
            continue
        seeds = sorted(all_data[method].keys())
        e = all_data[method][seeds[0]]
        ec = e["eff_contrib"]
        print(f"{method:14s} | " + " ".join([f"{v:.3f}" for v in ec]))

    # per-domain bucket
    print("\n  按 domain 聚合 (sum effective_contribution by domain):")
    for method in ["F2DC", "F2DC+DaA", "PG-DFC", "PG-DFC+DaA"]:
        if method not in all_data:
            continue
        seeds = sorted(all_data[method].keys())
        e = all_data[method][seeds[0]]
        ec = e["eff_contrib"]
        dom_clients = e["domain_per_client"]
        bucket = {}
        for k, dn in enumerate(dom_clients):
            bucket[dn] = bucket.get(dn, 0) + ec[k]
        s = " ".join([f"{dn}={bucket[dn]:.3f}" for dn in DOMAINS if dn in bucket])
        print(f"  {method:14s}: {s}")

    # 5. Per-layer drift
    print("\n" + "=" * 90)
    print("【维度 1】Per-layer drift ‖w_client - w_global‖ (avg, 浅 → 深)")
    print("=" * 90)
    layer_keys = None
    for method in ["F2DC", "F2DC+DaA", "PG-DFC", "PG-DFC+DaA"]:
        if method not in all_data:
            continue
        seeds = sorted(all_data[method].keys())
        e = all_data[method][seeds[0]]
        ld = e["layer_drift"]
        if layer_keys is None:
            layer_keys = sorted(ld.keys())
        print(f"  {method:14s}: " + " ".join([f"{ld.get(k,0):.3f}" for k in layer_keys]))

    # 6. Prototype cos sim trajectory (early vs late)
    print("\n" + "=" * 90)
    print("【维度 2】Prototype consensus alignment (avg cos sim across clients)")
    print("=" * 90)
    print(f"{'method':14s} | r10    r50    r100   |  Δ(end-start)  | mode_collapse_r100")
    for method in ["F2DC", "F2DC+DaA", "PG-DFC", "PG-DFC+DaA"]:
        if method not in all_data:
            continue
        seeds = sorted(all_data[method].keys())
        e = all_data[method][seeds[0]]
        rs, traj = e["proto_cos_traj"]  # (R, K) or (None, None)
        if rs is None:
            print(f"  {method:14s}: (无 local_protos dump)")
            continue
        avg_per_round = traj.mean(axis=1)
        idx10 = min(10, len(rs)-1)
        idx50 = min(50, len(rs)-1)
        idx100 = len(rs) - 1
        c10, c50, c100 = avg_per_round[idx10], avg_per_round[idx50], avg_per_round[idx100]
        delta = c100 - c10
        mc = e["mode_collapse_r100"]
        mc_s = f"{mc:.3f}" if mc is not None else "—"
        print(f"  {method:14s}: {c10:.3f}  {c50:.3f}  {c100:.3f}  | {delta:+.3f}        | {mc_s}")

    # 7. Per-client proto cos sim breakdown (best round)
    print("\n" + "=" * 90)
    print("【维度 2】Per-client proto cos sim @ R100 (验证 PG-DFC 是否同化 client)")
    print("=" * 90)
    for method in ["F2DC+DaA", "PG-DFC+DaA"]:
        if method not in all_data:
            continue
        seeds = sorted(all_data[method].keys())
        e = all_data[method][seeds[0]]
        rs, traj = e["proto_cos_traj"]
        if rs is None:
            print(f"\n  {method}: (无 local_protos dump)")
            continue
        last = traj[-1]
        dom_clients = e["domain_per_client"]
        print(f"\n  {method} (seed={seeds[0]}) @ R100:")
        for k, (dom, c) in enumerate(zip(dom_clients, last)):
            tag = " ⚠ 同化" if c > 0.85 else (" 健康" if c > 0.5 else " 偏离")
            print(f"    client {k} ({dom:8s}): cos={c:.3f}{tag}")

    # 8. Silhouette 对比 (best vs final, class vs domain)
    print("\n" + "=" * 90)
    print("【维度 3 + 6】Feature space silhouette (best 跟 final 对比)")
    print("=" * 90)
    print(f"{'method':14s} {'seed':4s} | sil_class B/F     | sil_domain B/F (低=domain-invariant)")
    for method in METHODS:
        if method not in all_data:
            continue
        for seed in sorted(all_data[method].keys()):
            e = all_data[method][seed]
            scb, sdb = e["sil_best"]
            scf, sdf = e["sil_final"]
            print(f"  {method:14s} {seed:4d} | {scb:.3f} → {scf:.3f}  ({scf-scb:+.3f}) | {sdb:.3f} → {sdf:.3f}  ({sdf-sdb:+.3f})")

    # 9. CKA cross-method (PG-DFC+DaA vs F2DC+DaA)
    print("\n" + "=" * 90)
    print("【维度 7】Cross-method CKA (PG-DFC+DaA vs F2DC+DaA, same seed)")
    print("=" * 90)
    for seed in [2, 15]:
        if "F2DC+DaA" in all_data and "PG-DFC+DaA" in all_data:
            ea = all_data["F2DC+DaA"].get(seed)
            eb = all_data["PG-DFC+DaA"].get(seed)
            if ea is None or eb is None:
                continue
            cka_best = cross_method_cka(ea["heavy_best"], eb["heavy_best"])
            cka_final = cross_method_cka(ea["heavy_final"], eb["heavy_final"])
            print(f"  seed={seed}: CKA(best)={cka_best:.3f}  CKA(final)={cka_final:.3f}")

    # 10. Per-class shift best→final (训练后期掉了哪些 class)
    print("\n" + "=" * 90)
    print("【维度 6】Per-class acc shift best→final (PG-DFC+DaA vs F2DC+DaA, seed=15)")
    print("=" * 90)
    for method in ["F2DC+DaA", "PG-DFC+DaA"]:
        if method not in all_data or 15 not in all_data[method]:
            continue
        e = all_data[method][15]
        bp = e["best_per_domain_class"]
        fp = e["final_per_domain_class"]
        if bp is None or fp is None:
            continue
        print(f"\n  {method} seed=15:")
        for dn in DOMAINS:
            if dn not in bp:
                continue
            shift = (fp[dn] - bp[dn]) * 100
            print(f"    {dn:8s}: per-class shift = " + " ".join([f"{s:+5.1f}" for s in shift]))

    # 11. Per-domain best vs last gap (verify training stability)
    print("\n" + "=" * 90)
    print("【维度 5】Best vs Last gap per domain (训练稳定性)")
    print("=" * 90)
    print(f"{'method':14s} {'seed':4s} | " + " ".join([f"{d:>7s}" for d in DOMAINS]) + " | total_gap")
    for method in ["F2DC", "F2DC+DaA", "PG-DFC", "PG-DFC+DaA"]:
        if method not in all_data:
            continue
        for seed in sorted(all_data[method].keys()):
            e = all_data[method][seed]
            best = e["acc_traj"].max(axis=0)
            last = e["acc_traj"][-1]
            gap = best - last
            row = f"  {method:14s} {seed:4d} | "
            for di in range(len(e["domain_names"])):
                row += f"{gap[di]:7.2f} "
            row += f"| {gap.mean():7.2f}"
            print(row)


if __name__ == "__main__":
    main()
