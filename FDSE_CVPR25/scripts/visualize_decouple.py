"""可视化 FedDSA 双头的共享 trunk 程度(诊断 shared-trunk 诅咒)。

输入: global_model.pt (含 semantic_head.0.weight + style_head.0.weight, 都是 [128, 1024])
输出: PNG 三联图 + 控制台数字

用法:
    python visualize_decouple.py --ckpt xxx/global_model.pt --out figs/ --label run_name
"""
import argparse, os, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    # 兼容 plain (semantic_head.0.weight) 和 VIB (semantic_head.mu_head.0.weight)
    if 'semantic_head.0.weight' in sd:
        W_sem = sd['semantic_head.0.weight'].float()
        arch = 'plain'
    elif 'semantic_head.mu_head.0.weight' in sd:
        W_sem = sd['semantic_head.mu_head.0.weight'].float()
        arch = 'vib'
    else:
        raise KeyError(f"no semantic_head weight found. keys: {list(sd.keys())[:30]}")
    W_sty = sd['style_head.0.weight'].float()
    print(f"  arch: {arch}")

    # 行归一化(消除尺度影响)
    Ws = W_sem / (W_sem.norm(dim=1, keepdim=True) + 1e-8)
    Wt = W_sty / (W_sty.norm(dim=1, keepdim=True) + 1e-8)

    # 指标
    corr = (Ws @ Wt.T).numpy()                        # [128, 128] 行向量 cos 矩阵
    sem_usage = W_sem.abs().mean(dim=0).numpy()       # [1024] 每个 trunk channel 被 sem 的平均依赖
    sty_usage = W_sty.abs().mean(dim=0).numpy()       # [1024]

    metrics = {
        'corr_abs_mean': float(np.abs(corr).mean()),
        'corr_abs_max': float(np.abs(corr).max()),
        'corr_diag_abs_mean': float(np.abs(np.diag(corr)).mean()),
        'channel_usage_cos': float(
            (sem_usage @ sty_usage) /
            (np.linalg.norm(sem_usage) * np.linalg.norm(sty_usage) + 1e-8)
        ),
        'channel_usage_pearson': float(np.corrcoef(sem_usage, sty_usage)[0, 1]),
    }
    # 两路都"重度使用"的 channel 重叠率
    heavy_sem = sem_usage > np.percentile(sem_usage, 75)
    heavy_sty = sty_usage > np.percentile(sty_usage, 75)
    overlap = int((heavy_sem & heavy_sty).sum())
    metrics['heavy_overlap'] = overlap
    metrics['heavy_overlap_ratio'] = overlap / (1024 * 0.25)   # vs 随机 = 1.0
    return corr, sem_usage, sty_usage, metrics


def plot(corr, sem_usage, sty_usage, metrics, out_png, label):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # 1. 相关性热力图
    im = axes[0].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title(f'[{label}] W_sem rows · W_sty rows (cos)\n'
                      f'|avg|={metrics["corr_abs_mean"]:.3f}  '
                      f'|max|={metrics["corr_abs_max"]:.3f}')
    axes[0].set_xlabel('sty row idx'); axes[0].set_ylabel('sem row idx')
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # 2. per-channel usage 曲线
    axes[1].plot(sem_usage, label='sem usage', alpha=0.75, lw=0.8)
    axes[1].plot(sty_usage, label='sty usage', alpha=0.75, lw=0.8)
    axes[1].set_title(f'per-trunk-channel |weight| mean\n'
                      f'pearson={metrics["channel_usage_pearson"]:.3f}')
    axes[1].set_xlabel('trunk channel (0~1023)')
    axes[1].legend()

    # 3. channel overlap scatter
    axes[2].scatter(sem_usage, sty_usage, alpha=0.35, s=10)
    mx = max(sem_usage.max(), sty_usage.max())
    axes[2].plot([0, mx], [0, mx], 'r--', alpha=0.5, lw=1,
                 label=f'y=x (perfect share)')
    axes[2].set_xlabel('sem usage'); axes[2].set_ylabel('sty usage')
    axes[2].set_title(f'channel-usage overlap\n'
                      f'cos={metrics["channel_usage_cos"]:.3f}  '
                      f'heavy-overlap={metrics["heavy_overlap"]}/256')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='path to global_model.pt')
    ap.add_argument('--out', default='./figs')
    ap.add_argument('--label', default=None, help='tag printed on figure')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    label = args.label or os.path.basename(os.path.dirname(args.ckpt))
    print(f'=== [{label}] ===')

    corr, sem_usage, sty_usage, metrics = analyze(args.ckpt)

    # 控制台打印
    for k, v in metrics.items():
        print(f'  {k:28s} = {v:.4f}' if isinstance(v, float) else f'  {k:28s} = {v}')

    # 画图
    png = os.path.join(args.out, f'decouple_{label}.png')
    plot(corr, sem_usage, sty_usage, metrics, png, label)
    print(f'  saved: {png}')

    # dump json
    with open(os.path.join(args.out, f'decouple_{label}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
