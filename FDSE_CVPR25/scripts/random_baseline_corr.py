"""Sanity check: 如果 W_sem / W_sty 是完全随机的初始化矩阵,
corr_abs_mean 和 usage pearson 会是多少?

用来判断我们实测 0.025 / 0 到底是 L_orth 的功劳,还是高维随机天然基线。"""
import torch, numpy as np

torch.manual_seed(42)
rows = []
for trial in range(10):
    # Kaiming-like 初始化 (PyTorch Linear 默认)
    W_sem = torch.empty(128, 1024).normal_(0, 1/np.sqrt(1024))
    W_sty = torch.empty(128, 1024).normal_(0, 1/np.sqrt(1024))

    Ws = W_sem / W_sem.norm(dim=1, keepdim=True)
    Wt = W_sty / W_sty.norm(dim=1, keepdim=True)
    corr = (Ws @ Wt.T).numpy()

    sem_usage = W_sem.abs().mean(dim=0).numpy()
    sty_usage = W_sty.abs().mean(dim=0).numpy()

    row = {
        'corr_abs_mean': float(np.abs(corr).mean()),
        'corr_abs_max': float(np.abs(corr).max()),
        'pearson': float(np.corrcoef(sem_usage, sty_usage)[0, 1]),
        'heavy_overlap': int((
            (sem_usage > np.percentile(sem_usage, 75)) &
            (sty_usage > np.percentile(sty_usage, 75))
        ).sum()),
    }
    rows.append(row)

# Aggregate
import json
keys = rows[0].keys()
print(f"{'metric':25s} | mean    | std     | min     | max")
print('-' * 72)
for k in keys:
    vals = [r[k] for r in rows]
    print(f"{k:25s} | {np.mean(vals):7.4f} | {np.std(vals):7.4f} | "
          f"{np.min(vals):7.4f} | {np.max(vals):7.4f}")

print('\n=== 对比我们实测的 PACS s=2 (VIB) ===')
print(f"  corr_abs_mean   实测 0.0254  vs 随机 mean {np.mean([r['corr_abs_mean'] for r in rows]):.4f}")
print(f"  corr_abs_max    实测 0.1191  vs 随机 mean {np.mean([r['corr_abs_max'] for r in rows]):.4f}")
print(f"  pearson         实测 0.0400  vs 随机 mean {np.mean([r['pearson'] for r in rows]):.4f}")
print(f"  heavy_overlap   实测 64      vs 随机 mean {np.mean([r['heavy_overlap'] for r in rows]):.1f}")
