"""
深度诊断: PG-DFC + DaA 为什么没明显超过 F2DC + DaA?
从 diag dump 实证算指标 (不画图, 只输出数字 + 解读).
"""
import os, json
import numpy as np
from pathlib import Path
from collections import defaultdict

DIAG_ROOT = Path("/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_office")
DOMAINS = ["caltech", "amazon", "webcam", "dslr"]


def load_rounds(diag_dir):
    return [dict(np.load(f, allow_pickle=True)) for f in sorted(diag_dir.glob("round_*.npz"))]

def load_heavy(diag_dir):
    h = {}
    for f in sorted(diag_dir.glob("best_*.npz")):
        h[f.stem] = dict(np.load(f, allow_pickle=True))
    final = list(diag_dir.glob("final_*.npz"))
    if final: h['final'] = dict(np.load(final[0], allow_pickle=True))
    return h

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-6 and nb > 1e-6 else 0.0


# 4 method × s=15 (主表 seed)
EXPS = {
    'vanilla_F2DC': DIAG_ROOT / "diag_f2dc_office_s15",
    'F2DC+DaA': DIAG_ROOT / "diag_f2dc_daa_office_s15",
    'vanilla_PG-DFC': DIAG_ROOT / "diag_pgdfc_office_s15",
    'PG-DFC+DaA': DIAG_ROOT / "diag_pgdfc_daa_office_s15",
}

# Note: V100 上 s=15 数据系统性低 (PG-DFC 55.91 跟 sc5 63.80 差 8pp)
# 但 cos sim / effective contrib / drift 等 diag metric 应该仍有解读价值


print("=" * 80)
print("DIAG 1: 4 method best acc trajectory (V100 s=15) — 验证训练动态")
print("=" * 80)

for name, d in EXPS.items():
    rounds = load_rounds(d)
    if not rounds: print(f"{name}: NO DATA"); continue
    accs = [float(np.mean(r['per_domain_acc'])) for r in rounds]
    best = max(accs)
    best_r = accs.index(best) + 1
    last = accs[-1]

    # 后 30 round mean (训练后期稳定 acc)
    late_mean = np.mean(accs[-30:]) if len(accs) >= 30 else np.mean(accs)

    # max increase per round (early) vs late
    early_slope = (accs[20] - accs[0]) / 20 if len(accs) > 20 else 0
    late_slope = (accs[-1] - accs[-30]) / 30 if len(accs) >= 30 else 0

    print(f"  {name:18s}: best={best:.2f}@R{best_r:3d}, last={last:.2f}, late30_mean={late_mean:.2f}")
    print(f"  {' '*18}  early_slope (R0→R20) = {early_slope:+.3f}/round, late_slope = {late_slope:+.4f}/round")


print("\n" + "=" * 80)
print("DIAG 2: 各 method 的 client-global proto cos sim")
print("(高 cos = client 被 prototype 同化)")
print("=" * 80)

for name, d in EXPS.items():
    rounds = load_rounds(d)
    if not rounds or 'global_proto' not in rounds[0]:
        print(f"{name}: NO global_proto data"); continue
    K = rounds[0]['local_protos'].shape[0] if 'local_protos' in rounds[0] else 0
    if K == 0: print(f"{name}: NO local_protos"); continue

    # mean cos per client over all rounds + over all classes
    client_mean_cos = []
    domain_per_client = [str(rounds[0]['domain_per_client'][ki]) for ki in range(K)]
    for ki in range(K):
        cos_per_round = []
        for r in rounds:
            gp = r['global_proto'].astype(np.float32)
            lp = r['local_protos'][ki].astype(np.float32)
            class_coses = [cos(lp[c], gp[c]) for c in range(min(gp.shape[0], lp.shape[0]))
                          if np.linalg.norm(lp[c]) > 1e-6]
            if class_coses:
                cos_per_round.append(np.mean(class_coses))
        if cos_per_round:
            client_mean_cos.append((ki, domain_per_client[ki], np.mean(cos_per_round)))

    print(f"\n  {name}:")
    for ki, dom, c in client_mean_cos:
        bar = '█' * int(c * 30)
        marker = '⚠️ 同化' if c > 0.85 else '✓ 健康' if c > 0.5 else '⚠️ 太散'
        print(f"    c{ki:2d} ({dom:8s}): cos={c:.3f}  {bar:30s}  {marker}")

    overall_mean = np.mean([c for _, _, c in client_mean_cos])
    print(f"    {'overall mean':18s}: {overall_mean:.3f}")


print("\n" + "=" * 80)
print("DIAG 3: 各 method per-domain best round acc (实证 DaA 修哪些 domain)")
print("=" * 80)

# 取 best round 的 per-domain acc
print(f"\n  {'Method':18s} | {'caltech':>8s} | {'amazon':>8s} | {'webcam':>8s} | {'dslr':>8s} | mean")
print(f"  {'-'*18} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | -----")
records = {}
for name, d in EXPS.items():
    rounds = load_rounds(d)
    if not rounds: continue
    accs = [float(np.mean(r['per_domain_acc'])) for r in rounds]
    best_r = np.argmax(accs)
    per_dom = rounds[best_r]['per_domain_acc']
    records[name] = per_dom
    print(f"  {name:18s} | {per_dom[0]:>8.2f} | {per_dom[1]:>8.2f} | {per_dom[2]:>8.2f} | {per_dom[3]:>8.2f} | {accs[best_r]:.2f}")

print(f"\n  ===Δ vs vanilla===")
print(f"  {'Δ':18s} | {'caltech':>8s} | {'amazon':>8s} | {'webcam':>8s} | {'dslr':>8s}")
for daa_method, vanilla_method in [('F2DC+DaA', 'vanilla_F2DC'), ('PG-DFC+DaA', 'vanilla_PG-DFC')]:
    if daa_method in records and vanilla_method in records:
        delta = records[daa_method] - records[vanilla_method]
        print(f"  {daa_method:18s} | {delta[0]:>+8.2f} | {delta[1]:>+8.2f} | {delta[2]:>+8.2f} | {delta[3]:>+8.2f}")


print("\n" + "=" * 80)
print("DIAG 4: client effective contribution (α × ‖w_i - w_g‖) 分布")
print("(F2DC+DaA vs PG-DFC+DaA — 看 DaA 实际改变 dispatch 多少)")
print("=" * 80)

for name in ['F2DC+DaA', 'PG-DFC+DaA']:
    d = EXPS[name]
    rounds = load_rounds(d)
    if not rounds: continue
    K = len(rounds[0]['sample_shares'])
    domain_per_client = [str(rounds[0]['domain_per_client'][ki]) for ki in range(K)]

    # 取 last 30 round mean per client
    contrib_per_client = []
    for ki in range(K):
        contribs = [r['daa_freqs'][ki] * r['grad_l2'][ki] for r in rounds[-30:]]
        contrib_per_client.append((ki, domain_per_client[ki], np.mean(contribs)))

    total_contrib = sum(c for _, _, c in contrib_per_client)
    print(f"\n  {name} (last 30 round mean contrib):")
    for ki, dom, c in contrib_per_client:
        ratio = c / total_contrib if total_contrib > 0 else 0
        bar = '█' * int(ratio * 100)
        print(f"    c{ki:2d} ({dom:8s}): contrib={c:.4f} ({ratio*100:5.1f}%)  {bar}")


print("\n" + "=" * 80)
print("DIAG 5: per-layer drift trajectory mean (R0-30 vs R30-60 vs R60-100)")
print("(对比 4 method 各 layer 训练动态)")
print("=" * 80)

for name, d in EXPS.items():
    rounds = load_rounds(d)
    if not rounds: continue
    layer_per_round = []
    for r in rounds:
        try: layer_per_round.append(json.loads(r['layer_l2_pickle'][0]))
        except: layer_per_round.append({})

    if not layer_per_round[0]: continue
    first_client = list(layer_per_round[0].keys())[0]
    layer_names = list(layer_per_round[0][first_client].keys())

    print(f"\n  {name}:")
    for layer in layer_names:
        # mean drift over clients per round
        per_round = []
        for d_round in layer_per_round:
            client_vals = [d_round[c].get(layer, 0) for c in d_round if isinstance(d_round[c], dict)]
            per_round.append(np.mean(client_vals) if client_vals else 0)

        early = np.mean(per_round[:30]) if len(per_round) >= 30 else np.mean(per_round)
        mid = np.mean(per_round[30:60]) if len(per_round) >= 60 else 0
        late = np.mean(per_round[-30:]) if len(per_round) >= 30 else 0

        layer_short = layer.split('.')[0][-15:]
        print(f"    {layer_short:15s}: R0-30={early:.4f}  R30-60={mid:.4f}  R60-100={late:.4f}")


print("\n" + "=" * 80)
print("DIAG 6: 关键对比 — F2DC+DaA vs PG-DFC+DaA (看 prototype guidance 加 DaA 后表现)")
print("=" * 80)

f2dc_daa_rounds = load_rounds(EXPS['F2DC+DaA'])
pgdfc_daa_rounds = load_rounds(EXPS['PG-DFC+DaA'])

if f2dc_daa_rounds and pgdfc_daa_rounds:
    f2dc_accs = np.array([float(np.mean(r['per_domain_acc'])) for r in f2dc_daa_rounds])
    pgdfc_accs = np.array([float(np.mean(r['per_domain_acc'])) for r in pgdfc_daa_rounds])

    # 哪个 round 区间 F2DC+DaA 比 PG-DFC+DaA 强?
    diff = pgdfc_accs - f2dc_accs[:len(pgdfc_accs)]
    print(f"\n  PG-DFC+DaA - F2DC+DaA acc diff over rounds:")
    print(f"    R1-30:    mean Δ = {np.mean(diff[:30]):+.3f}")
    print(f"    R30-60:   mean Δ = {np.mean(diff[30:60]):+.3f}")
    print(f"    R60-100:  mean Δ = {np.mean(diff[60:]):+.3f}")

    # final round
    print(f"\n  best acc: F2DC+DaA={max(f2dc_accs):.2f}@R{np.argmax(f2dc_accs)+1}, PG-DFC+DaA={max(pgdfc_accs):.2f}@R{np.argmax(pgdfc_accs)+1}")
    print(f"  last R100: F2DC+DaA={f2dc_accs[-1]:.2f}, PG-DFC+DaA={pgdfc_accs[-1]:.2f}")
    print(f"  best vs last gap: F2DC+DaA = {max(f2dc_accs) - f2dc_accs[-1]:.2f}, PG-DFC+DaA = {max(pgdfc_accs) - pgdfc_accs[-1]:.2f}")

    # 早期 vs 后期 slope
    print(f"\n  slope: F2DC+DaA early(R0→R20) = {(f2dc_accs[20]-f2dc_accs[0])/20:+.3f}/r, PG-DFC+DaA = {(pgdfc_accs[20]-pgdfc_accs[0])/20:+.3f}/r")


print("\n" + "=" * 80)
print("DIAG 7: feature space silhouette (best round) - representation 质量")
print("=" * 80)

try:
    from sklearn.metrics import silhouette_score
    for name, d in EXPS.items():
        h = load_heavy(d)
        best_keys = [k for k in h if k.startswith('best_')]
        if not best_keys: continue
        latest = sorted(best_keys, key=lambda x: int(x.split('R')[-1]))[-1]
        data = h[latest]
        features = data['features'].item()
        labels = data['labels'].item()
        all_X, all_class, all_dom = [], [], []
        for dom_i, dom in enumerate(DOMAINS):
            if dom in features:
                f = features[dom].astype(np.float32)
                if len(f) > 300:
                    idx = np.random.RandomState(42).choice(len(f), 300, replace=False)
                    f = f[idx]; lbl = labels[dom][idx]
                else: lbl = labels[dom]
                all_X.append(f); all_class.extend(lbl); all_dom.extend([dom_i]*len(f))
        X = np.concatenate(all_X)
        sil_class = silhouette_score(X, all_class)
        sil_domain = silhouette_score(X, all_dom)
        print(f"  {name:18s}: sil_class = {sil_class:.4f} (高=class 区分好), sil_domain = {sil_domain:.4f} (低=domain-invariant)")
except ImportError:
    print("  sklearn 不可用")


print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
