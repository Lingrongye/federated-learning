"""
PACS convergence v3 — zoom + best 标注 + 视觉优化
- ylim 30-80 (砍掉 R0-R5 起步段, 突出训练阶段差距)
- 加 Ours best round R91=75.37 的星标
- F2DC+DaA / FDSE / Ours 三色突出 (paper main competitor)
- 其他 baseline 淡灰化
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

base = '/Users/changdao/联邦学习/experiments/ablation'
F2DC_LOGS = '/Users/changdao/联邦学习/F2DC/_phaseC_v2_logs'
OUT = Path('/Users/changdao/联邦学习/figures')


def parse_log(path):
    with open(path) as f:
        text = f.read()
    rounds = {}
    for line in text.splitlines():
        m = re.match(r'^The (\d+) Communcation Accuracy:\s*([\d.]+)', line)
        if m:
            rounds[int(m.group(1))] = float(m.group(2))
    return rounds


# 三层视觉: faint baselines / 灰系主对手 / 红强调 Ours
RUNS = [
    # 5 个 weak baselines — faint 浅色
    ('FedAvg', f'{base}/EXP-130_F2DC_baseline_main_table/sc4_v2_logs/fedavg_fl_pacs_s15.log',
     '#c6dbef', '-', 1.2, 'FedAvg', False),
    ('FedBN', f'{base}/EXP-132_baselines_3algo/logs/fedbn_pacs_s15.log',
     '#9ecae1', '-', 1.2, 'FedBN', False),
    ('FedProx', f'{base}/EXP-132_baselines_3algo/logs/fedprox_pacs_s333.log',
     '#fdd0a2', '-', 1.2, 'FedProx', False),
    ('FedProto', f'{base}/EXP-132_baselines_3algo/logs/fedproto_pacs_s15.log',
     '#c7e9c0', '-', 1.2, 'FedProto (mem-leak crash)', False),
    ('MOON', f'{F2DC_LOGS}/moon_fl_pacs_s15.log',
     '#dadaeb', '-', 1.2, 'MOON', False),
    # 2 个 main competitor — 强调
    ('FDSE', f'{F2DC_LOGS}/fdse_fl_pacs_s15.log',
     '#fb6a4a', '-', 1.8, "FDSE [CVPR'25]", True),
    ('F2DC', f'{F2DC_LOGS}/f2dc_fl_pacs_s15.log',
     '#737373', ':', 1.8, "F2DC [CVPR'26]", True),
    ('F2DC+DaA', f'{base}/EXP-133_DaA_validation/logs/f2dc_daa_pacs_s15.log',
     '#252525', '--', 2.0, "F2DC+DaA [CVPR'26]", True),
    # Ours — 大红粗
    ('Ours', f'{base}/EXP-149_f2dc_dse_lab_pacs/logs/pacs_s333_rho03_rerun_sub1.log',
     '#cb181d', '-', 2.8, 'Ours: F2DC+DSE+LAB', True),
]


def smooth(y, k=3):
    if k <= 1: return y
    out = np.copy(y).astype(float)
    for i in range(1, len(y)):
        out[i] = (1 - 1.0/k) * out[i-1] + (1.0/k) * y[i]
    return out


fig, ax = plt.subplots(figsize=(11, 6.5))

# 画线
ours_data = None
for name, path, color, ls, lw, label, emphasized in RUNS:
    rounds = parse_log(path)
    if not rounds:
        print(f"  WARN no data: {name}"); continue
    rs = sorted(rounds.keys())
    ys = np.array([rounds[r] for r in rs])
    ys_smooth = smooth(ys, k=3)
    alpha = 0.95 if emphasized else 0.55
    ax.plot(rs, ys_smooth, color=color, linestyle=ls, linewidth=lw, label=label, alpha=alpha)
    if name == 'Ours':
        ours_data = (rs, ys, ys_smooth)

# 加 Ours best 星标 + acc 标注
if ours_data:
    rs, ys_raw, ys_smooth = ours_data
    best_idx = int(np.argmax(ys_raw))
    best_r = rs[best_idx]
    best_acc = ys_raw[best_idx]
    # 星标
    ax.scatter([best_r], [best_acc], marker='*', s=350, color='#cb181d', edgecolor='white',
               linewidth=1.8, zorder=10)
    # acc 文本
    ax.annotate(f'best = {best_acc:.2f}',
                xy=(best_r, best_acc), xytext=(best_r - 12, best_acc + 2.5),
                fontsize=11, fontweight='bold', color='#cb181d',
                arrowprops=dict(arrowstyle='->', color='#cb181d', lw=1.4))

ax.set_xlim(0, 100)
ax.set_ylim(30, 80)
ax.set_xlabel('Communication Round', fontsize=12)
ax.set_ylabel('Average Accuracy [%]', fontsize=12)
ax.set_title('PACS Convergence', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(fontsize=9.5, loc='lower right', framealpha=0.94, ncol=1)

plt.tight_layout()
out = OUT / 'convergence_pacs_v3.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ saved: {out}")
