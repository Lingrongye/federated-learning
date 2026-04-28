"""
Cold path 诊断分析 — V100 office 10 个 R100 实验.

输入: diag_office/diag_{method}_office_{seed}/
- round_001..100.npz (light)
- best_R*.npz / final_R100.npz (heavy)
- meta.json

输出: cold_path_analysis/figs/ 下若干 PNG + summary.json
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

DIAG_ROOT = Path("/Users/changdao/联邦学习/experiments/cold_path_analysis/diag_office")
FIGS_DIR = Path("/Users/changdao/联邦学习/experiments/cold_path_analysis/figs")
FIGS_DIR.mkdir(exist_ok=True, parents=True)

DOMAINS = ["caltech", "amazon", "webcam", "dslr"]
N_CLASSES = 10

# 4 method × 3 seed = 10 实验 (vanilla 没 s=333)
METHODS = ['f2dc', 'f2dc_daa', 'pgdfc', 'pgdfc_daa']
COLORS = {'f2dc': '#4477AA', 'f2dc_daa': '#228833', 'pgdfc': '#EE6677', 'pgdfc_daa': '#CC0000'}
NAMES = {'f2dc': 'vanilla F2DC', 'f2dc_daa': 'F2DC + DaA', 'pgdfc': 'vanilla PG-DFC', 'pgdfc_daa': 'PG-DFC + DaA'}


def parse_dirname(name):
    """diag_pgdfc_daa_office_s15 → ('pgdfc_daa', 15)"""
    if not name.startswith('diag_'): return None
    name = name[5:]
    if not name.endswith(name.split('_')[-1]): return None
    parts = name.split('_office_')
    if len(parts) != 2: return None
    method, seed_str = parts
    if seed_str.startswith('s'): seed = int(seed_str[1:])
    else: seed = int(seed_str)
    if method not in METHODS: return None
    return method, seed


def load_light_trajectory(diag_dir):
    """加载所有 round_NNN.npz, 返回按 round 排序的 list of dict."""
    rounds = []
    for f in sorted(diag_dir.glob("round_*.npz")):
        data = dict(np.load(f, allow_pickle=True))
        rounds.append(data)
    return rounds


def load_heavy_snapshots(diag_dir):
    """加载 best_*.npz + final_*.npz, 返回 dict[name -> data]."""
    out = {}
    for f in sorted(diag_dir.glob("best_*.npz")):
        out[f.stem] = dict(np.load(f, allow_pickle=True))
    final = list(diag_dir.glob("final_*.npz"))
    if final:
        out['final'] = dict(np.load(final[0], allow_pickle=True))
    return out


def discover_experiments():
    """返回 {method: {seed: diag_dir}}"""
    exps = defaultdict(dict)
    for d in DIAG_ROOT.iterdir():
        if not d.is_dir(): continue
        parsed = parse_dirname(d.name)
        if parsed is None: continue
        method, seed = parsed
        exps[method][seed] = d
    return exps


# ============================================================
# 指标 1: 收敛曲线 per-domain accuracy trajectory
# ============================================================
def plot_convergence(exps):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for di, dom in enumerate(DOMAINS):
        ax = axes[di]
        for method in METHODS:
            if method not in exps: continue
            # 每 method 用 s=15 (主 seed)
            seeds = exps[method]
            seed = 15 if 15 in seeds else (333 if 333 in seeds else min(seeds.keys()))
            rounds = load_light_trajectory(seeds[seed])
            if not rounds: continue
            accs = [r['per_domain_acc'][di] if di < len(r['per_domain_acc']) else 0 for r in rounds]
            ax.plot(np.arange(1, len(accs)+1), accs, label=NAMES[method], color=COLORS[method], linewidth=1.5)
        ax.set_title(f'Office: {dom}', fontsize=12)
        ax.set_xlabel('Round');
        if di == 0: ax.set_ylabel('Acc [%]')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    plt.suptitle('Per-domain Convergence (s=15)', fontsize=14)
    plt.tight_layout()
    out = FIGS_DIR / "01_per_domain_convergence.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ============================================================
# 指标 2: DaA dispatch ratio (sample_share vs daa_freq) per client
# ============================================================
def plot_daa_dispatch(exps):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ai, method in enumerate(['f2dc_daa', 'pgdfc_daa']):
        if method not in exps: continue
        ax = axes[ai]
        seed = 15 if 15 in exps[method] else min(exps[method].keys())
        rounds = load_light_trajectory(exps[method][seed])
        if not rounds: continue
        K = len(rounds[0]['sample_shares'])
        for ki in range(K):
            ratios = [r['daa_freqs'][ki] / r['sample_shares'][ki] if r['sample_shares'][ki] > 1e-6 else 1
                      for r in rounds]
            domain = str(rounds[0]['domain_per_client'][ki]) if 'domain_per_client' in rounds[0] else f'c{ki}'
            ax.plot(np.arange(1, len(ratios)+1), ratios, label=f'c{ki}({domain})', alpha=0.7, linewidth=0.8)
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{NAMES[method]}: DaA freq / FedAvg freq')
        ax.set_xlabel('Round'); ax.set_ylabel('Reweight ratio')
        ax.legend(fontsize=6, ncol=2, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.suptitle('DaA Dispatch Ratio per Client (>1 升权, <1 降权)', fontsize=13)
    plt.tight_layout()
    out = FIGS_DIR / "02_daa_dispatch_ratio.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ============================================================
# 指标 3: client-global proto cos sim trajectory (per client)
# ============================================================
def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6: return 0.0
    return float(np.dot(a, b) / (na * nb))


def plot_proto_cos(exps):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for ai, method in enumerate(METHODS):
        if method not in exps:
            axes[ai].set_title(f'{NAMES[method]} (no data)')
            continue
        ax = axes[ai]
        seed = 15 if 15 in exps[method] else min(exps[method].keys())
        rounds = load_light_trajectory(exps[method][seed])
        if not rounds: continue
        # 每 client 算 mean cos sim over all classes per round
        K = None
        traj_per_client = defaultdict(list)
        for r in rounds:
            if 'global_proto' not in r or 'local_protos' not in r:
                continue
            gp = r['global_proto'].astype(np.float32)  # (C, D)
            lp = r['local_protos'].astype(np.float32)  # (K, C, D)
            K = lp.shape[0]
            for ki in range(K):
                cosines = [cos_sim(lp[ki, c], gp[c]) for c in range(min(N_CLASSES, gp.shape[0]))
                           if np.linalg.norm(lp[ki, c]) > 1e-6]
                if cosines:
                    traj_per_client[ki].append(np.mean(cosines))
                else:
                    traj_per_client[ki].append(0)
        if K and traj_per_client:
            for ki in range(K):
                domain = str(rounds[0]['domain_per_client'][ki]) if 'domain_per_client' in rounds[0] else f'c{ki}'
                ax.plot(traj_per_client[ki][:100], label=f'c{ki}({domain})', alpha=0.7, linewidth=0.8)
        ax.set_title(NAMES[method])
        ax.set_xlabel('Round');
        if ai == 0: ax.set_ylabel('mean cos(local_proto, global_proto)')
        ax.set_ylim([0.5, 1.05])
        ax.axhline(0.85, color='red', linestyle='--', alpha=0.4, label='同化 threshold')
        ax.legend(fontsize=6, ncol=2, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.suptitle('Client-Global Prototype Cos Sim Trajectory (高 = 被同化)', fontsize=13)
    plt.tight_layout()
    out = FIGS_DIR / "03_proto_cos_trajectory.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ============================================================
# 指标 4: effective contribution α_i × ‖w_i - w_g‖
# ============================================================
def plot_effective_contrib(exps):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for ai, method in enumerate(METHODS):
        if method not in exps:
            axes[ai].set_title(f'{NAMES[method]} (no data)')
            continue
        ax = axes[ai]
        seed = 15 if 15 in exps[method] else min(exps[method].keys())
        rounds = load_light_trajectory(exps[method][seed])
        if not rounds: continue
        K = len(rounds[0]['sample_shares'])
        for ki in range(K):
            # 用 daa_freqs (有 DaA) 或 sample_shares (无 DaA) 作 freq
            uses_daa = 'daa' in method
            if uses_daa:
                contribs = [r['daa_freqs'][ki] * r['grad_l2'][ki] for r in rounds]
            else:
                contribs = [r['sample_shares'][ki] * r['grad_l2'][ki] for r in rounds]
            domain = str(rounds[0]['domain_per_client'][ki]) if 'domain_per_client' in rounds[0] else f'c{ki}'
            ax.plot(contribs[:100], label=f'c{ki}({domain})', alpha=0.7, linewidth=0.8)
        ax.set_title(NAMES[method])
        ax.set_xlabel('Round');
        if ai == 0: ax.set_ylabel('effective contrib (α × ‖w_i - w_g‖)')
        ax.legend(fontsize=6, ncol=2, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.suptitle('Effective Client Contribution per Round', fontsize=13)
    plt.tight_layout()
    out = FIGS_DIR / "04_effective_contribution.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ============================================================
# 指标 5: per-class confusion at best round (4 method 对比)
# ============================================================
def plot_confusion(exps):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ai, method in enumerate(METHODS):
        if method not in exps:
            axes[ai].set_title(f'{NAMES[method]} (no data)'); continue
        seed = 15 if 15 in exps[method] else min(exps[method].keys())
        heavy = load_heavy_snapshots(exps[method][seed])
        # 找 best (取 round 最大的 best_*)
        best_keys = [k for k in heavy if k.startswith('best_')]
        if not best_keys: continue
        latest_best = sorted(best_keys, key=lambda x: int(x.split('R')[-1]))[-1]
        data = heavy[latest_best]
        confusion = data['confusion'].item()  # dict[domain -> (C, C)]
        # 合并 4 domain
        merged = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
        for dom in DOMAINS:
            if dom in confusion:
                merged += confusion[dom]
        # normalize per row (true class)
        row_sum = merged.sum(axis=1, keepdims=True)
        normed = np.where(row_sum > 0, merged / row_sum, 0)
        ax = axes[ai]
        im = ax.imshow(normed, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{NAMES[method]} ({latest_best})')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                v = normed[i, j]
                if v > 0.05:
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                            color='white' if v > 0.5 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle('Confusion Matrix at Best Round (4 method × Office)', fontsize=13)
    plt.tight_layout()
    out = FIGS_DIR / "05_confusion_best_round.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ============================================================
# 指标 6: t-SNE on best round features (4 method)
# ============================================================
def plot_tsne(exps):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  ⚠️ sklearn 不可用, 跳过 t-SNE"); return
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    domain_colors = {'caltech':'#4477AA', 'amazon':'#228833', 'webcam':'#EE6677', 'dslr':'#CCBB44'}
    for ai, method in enumerate(METHODS):
        ax = axes[ai]
        if method not in exps:
            ax.set_title(f'{NAMES[method]} (no data)'); continue
        seed = 15 if 15 in exps[method] else min(exps[method].keys())
        heavy = load_heavy_snapshots(exps[method][seed])
        best_keys = [k for k in heavy if k.startswith('best_')]
        if not best_keys: continue
        latest_best = sorted(best_keys, key=lambda x: int(x.split('R')[-1]))[-1]
        data = heavy[latest_best]
        features = data['features'].item()  # dict[domain -> (N, 512)]
        all_X, all_dom, all_y = [], [], []
        for dom in DOMAINS:
            if dom in features:
                f = features[dom].astype(np.float32)
                # 取每域最多 200 sample (加速 t-SNE)
                if len(f) > 200:
                    idx = np.random.RandomState(42).choice(len(f), 200, replace=False)
                    f = f[idx]
                else:
                    idx = np.arange(len(f))
                all_X.append(f)
                all_dom.extend([dom] * len(f))
                lbl = data['labels'].item()[dom][idx] if dom in data['labels'].item() else np.zeros(len(f))
                all_y.extend(lbl)
        if not all_X: continue
        X = np.concatenate(all_X)
        embedding = TSNE(n_components=2, perplexity=30, random_state=42, init='pca').fit_transform(X)
        for dom in DOMAINS:
            mask = np.array([d == dom for d in all_dom])
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=domain_colors[dom],
                       label=dom, alpha=0.6, s=8)
        ax.set_title(f'{NAMES[method]}', fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle('t-SNE Feature Space at Best Round (color = domain)', fontsize=13)
    plt.tight_layout()
    out = FIGS_DIR / "06_tsne_best_round.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ============================================================
# 指标 7: per-layer drift trajectory
# ============================================================
def plot_per_layer_drift(exps):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for ai, method in enumerate(METHODS):
        if method not in exps:
            axes[ai].set_title(f'{NAMES[method]} (no data)'); continue
        ax = axes[ai]
        seed = 15 if 15 in exps[method] else min(exps[method].keys())
        rounds = load_light_trajectory(exps[method][seed])
        if not rounds: continue
        # 每 round 解 layer_l2_pickle (json)
        layer_per_round = []
        for r in rounds:
            try:
                d = json.loads(r['layer_l2_pickle'][0])
                layer_per_round.append(d)
            except: layer_per_round.append({})
        # 找 layer names from first round
        if not layer_per_round or not layer_per_round[0]: continue
        first_client = list(layer_per_round[0].keys())[0]
        layer_names = list(layer_per_round[0][first_client].keys())
        for layer_name in layer_names:
            # mean over clients per round
            vals = []
            for d in layer_per_round:
                client_vals = [d[c].get(layer_name, 0) for c in d if isinstance(d[c], dict)]
                vals.append(np.mean(client_vals) if client_vals else 0)
            ax.plot(vals[:100], label=layer_name.split('.')[0][-12:], linewidth=1.0, alpha=0.8)
        ax.set_title(NAMES[method])
        ax.set_xlabel('Round')
        if ai == 0: ax.set_ylabel('mean ‖w_i^layer - w_g‖')
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.suptitle('Per-Layer Drift over Rounds', fontsize=13)
    plt.tight_layout()
    out = FIGS_DIR / "07_per_layer_drift.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ============================================================
# 指标 8: cross-seed stability (best 跨 seed 对比)
# ============================================================
def plot_cross_seed_stability(exps):
    summary = {}
    for method in METHODS:
        if method not in exps: continue
        for seed, diag_dir in exps[method].items():
            heavy = load_heavy_snapshots(diag_dir)
            best_keys = [k for k in heavy if k.startswith('best_')]
            if not best_keys: continue
            latest = sorted(best_keys, key=lambda x: int(x.split('R')[-1]))[-1]
            best_acc = float(heavy[latest]['current_acc'])
            summary[(method, seed)] = best_acc

    methods_present = sorted(set(m for m, _ in summary.keys()))
    seeds_present = sorted(set(s for _, s in summary.keys()))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.18
    x = np.arange(len(methods_present))
    for si, seed in enumerate(seeds_present):
        vals = [summary.get((m, seed), 0) for m in methods_present]
        ax.bar(x + si * width, vals, width, label=f's={seed}')
        for xi, v in enumerate(vals):
            if v > 0:
                ax.text(xi + si * width, v + 0.3, f'{v:.1f}', ha='center', fontsize=7)
    ax.set_xticks(x + width * (len(seeds_present)-1) / 2)
    ax.set_xticklabels([NAMES[m] for m in methods_present])
    ax.set_ylabel('Best Acc [%]'); ax.set_title('Cross-seed Best Acc (Office)')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    out = FIGS_DIR / "08_cross_seed_stability.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")
    return summary


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    exps = discover_experiments()
    print(f"发现 {sum(len(v) for v in exps.values())} 个 experiments:")
    for m, seeds in exps.items():
        print(f"  {m}: seeds = {sorted(seeds.keys())}")

    print("\n[出图]")
    plot_convergence(exps)
    plot_daa_dispatch(exps)
    plot_proto_cos(exps)
    plot_effective_contrib(exps)
    plot_confusion(exps)
    plot_tsne(exps)
    plot_per_layer_drift(exps)
    summary = plot_cross_seed_stability(exps)

    # summary
    with open(FIGS_DIR / "summary.json", 'w') as f:
        json.dump({f"{k[0]}_s{k[1]}": v for k, v in summary.items()}, f, indent=2)
    print(f"\n✅ ALL DONE → {FIGS_DIR}")
