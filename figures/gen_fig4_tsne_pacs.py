"""
Fig 4 t-SNE on PACS — F2DC paper Fig 4 style 复现

数据规格 (best_RXXX.npz / final_R100.npz):
  features: dict per domain (photo/art/cartoon/sketch), each (N, 512) fp16
  labels: dict per domain (N,) int32
  domain_names: array(['photo', 'art', 'cartoon', 'sketch'])

F2DC paper 风格:
  - 7 color = 7 PACS classes
  - 4 shape = 4 domains (photo=○, art=□, cartoon=△, sketch=◇)
  - black-outlined stars = per-class semantic centers (cross-domain mean)
  - clean square panels, no axis ticks/labels, small method title above
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# === paths (4 method, 都用 highest best npz, F2DC paper Fig 4 style 5-panel 简化版 4-panel) ===
PANELS = [
    ("FedAvg",          "/Users/changdao/联邦学习/experiments/ablation/EXP-153_baselines_for_tsne/diag/fedavg_pacs_s15/best_R099.npz"),
    ("FedBN",           "/Users/changdao/联邦学习/experiments/ablation/EXP-153_baselines_for_tsne/diag/fedbn_pacs_s15/best_R072.npz"),
    ("F2DC + DaA",      "/Users/changdao/联邦学习/experiments/ablation/EXP-153_baselines_for_tsne/diag/f2dc_daa_pacs_s15/best_R094.npz"),
    ("Ours (DSE+LAB)",  "/Users/changdao/联邦学习/experiments/ablation/EXP-149_f2dc_dse_lab_pacs/diag_pacs_s333_rho03_dselab_rerun_sub1/best_R092.npz"),
]
OUT_DIR = "/Users/changdao/联邦学习/figures"

# === style (F2DC Fig 4 风格) ===
DOMAIN_MARKERS = {'photo': 'o', 'art': 's', 'cartoon': '^', 'sketch': 'D'}
PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
NUM_CLASSES = 7
CLASS_COLORS = plt.cm.tab10.colors[:NUM_CLASSES]  # 7 distinct colors
SAMPLES_PER_DOMAIN_PER_CLASS = 100  # 每域每类 100, 4 域 × 7 类 × 100 = 2800/panel (3.4× 原来)

def load_features(npz_path):
    """加载并整合: 返回 (features, labels, domains, semantic_centers)"""
    z = np.load(npz_path, allow_pickle=True)
    feats_dict = z['features'].item()
    labels_dict = z['labels'].item()
    dnames = list(z['domain_names'])

    all_feats, all_labels, all_domains = [], [], []
    for di, dn in enumerate(dnames):
        f = feats_dict[dn].astype(np.float32)  # (N, 512)
        l = labels_dict[dn].astype(np.int64)
        rng = np.random.RandomState(42)
        idx_list = []
        for c in range(NUM_CLASSES):
            cls_idx = np.where(l == c)[0]
            if len(cls_idx) > SAMPLES_PER_DOMAIN_PER_CLASS:
                cls_idx = rng.choice(cls_idx, SAMPLES_PER_DOMAIN_PER_CLASS, replace=False)
            idx_list.append(cls_idx)
        sel = np.concatenate(idx_list)
        all_feats.append(f[sel])
        all_labels.append(l[sel])
        all_domains.append(np.full(len(sel), di))
    feats = np.vstack(all_feats)
    labels = np.concatenate(all_labels)
    domains = np.concatenate(all_domains)

    # 语义中心: 每 class 跨域 mean (基于 全 test set, 不只 sample, 更稳)
    all_f_full = np.vstack([feats_dict[dn].astype(np.float32) for dn in dnames])
    all_l_full = np.concatenate([labels_dict[dn].astype(np.int64) for dn in dnames])
    centers = np.array([all_f_full[all_l_full == c].mean(0) for c in range(NUM_CLASSES)])
    return feats, labels, domains, centers, dnames

def plot_tsne_panel(ax, feats, labels, domains, centers, dnames, title):
    """绘单个 panel: tsne 后散点 + stars"""
    print(f"  [{title}] running t-SNE on {feats.shape[0]} samples + {centers.shape[0]} centers ...")
    # Stack feats + centers, 一起 t-SNE 才能在同一 2D space 比较
    X = np.vstack([feats, centers])
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca')
    Z = tsne.fit_transform(X)
    Z_pts, Z_centers = Z[:len(feats)], Z[len(feats):]

    # 散点 (color=class, marker=domain)
    for di, dn in enumerate(dnames):
        marker = DOMAIN_MARKERS[dn]
        for c in range(NUM_CLASSES):
            mask = (domains == di) & (labels == c)
            if not mask.any(): continue
            ax.scatter(Z_pts[mask, 0], Z_pts[mask, 1],
                       c=[CLASS_COLORS[c]], marker=marker,
                       s=18, alpha=0.7, linewidths=0)
    # Semantic centers (black-outlined stars per class)
    for c in range(NUM_CLASSES):
        ax.scatter(Z_centers[c, 0], Z_centers[c, 1],
                   c=[CLASS_COLORS[c]], marker='*',
                   s=350, edgecolors='black', linewidths=1.2, zorder=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_color('black')
        ax.spines[s].set_linewidth(0.8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=6)

# === main ===
fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 4.2), constrained_layout=True)

for i, (title, path) in enumerate(PANELS):
    print(f"\n[{title}] Loading {path.split('/')[-1]} ...")
    feats, labels, domains, centers, dnames = load_features(path)
    plot_tsne_panel(axes[i], feats, labels, domains, centers, dnames, title)

# Optional shared legend below
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
class_handles = [mpatches.Patch(color=CLASS_COLORS[c], label=PACS_CLASSES[c]) for c in range(NUM_CLASSES)]
domain_handles = [mlines.Line2D([], [], color='gray', marker=m, linestyle='None', markersize=8,
                                  label=dn) for dn, m in DOMAIN_MARKERS.items()]
star_handle = mlines.Line2D([], [], color='gray', marker='*', linestyle='None', markersize=14,
                             markeredgecolor='black', label='semantic center')
fig.legend(handles=class_handles + domain_handles + [star_handle],
           loc='lower center', ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))

out = os.path.join(OUT_DIR, "fig4_tsne_pacs_4panels.pdf")
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
out_png = out.replace('.pdf', '.png')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved: {out_png}")
