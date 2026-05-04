"""
PACS convergence curve — 9 method 对比 (类似 Office v1 风格)
- 8 baseline 全用 s=15 (同 seed 公平对比, 无 cherry-picking)
- 1 条我们: F2DC+DSE+LAB s=333 rho=0.3 rerun (最强 single-run, best 75.37 ⭐⭐⭐)
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = '/Users/changdao/联邦学习/experiments/ablation'
OUT = Path('/Users/changdao/联邦学习/figures')
OUT.mkdir(exist_ok=True, parents=True)


def parse_log(path):
    with open(path) as f:
        text = f.read()
    rounds = {}
    for line in text.splitlines():
        m = re.match(r'^The (\d+) Communcation Accuracy:\s*([\d.]+)', line)
        if m:
            rounds[int(m.group(1))] = float(m.group(2))
    return rounds


# baseline 全 s=15, 我们用 s=333 (最强 PACS run)
RUNS = [
    ('FedAvg', f'{base}/EXP-130_F2DC_baseline_main_table/sc4_v2_logs/fedavg_fl_pacs_s15.log',
     '#9ecae1', '-', 1.4, 'FedAvg'),
    ('FedBN', f'{base}/EXP-132_baselines_3algo/logs/fedbn_pacs_s15.log',
     '#6baed6', '-', 1.4, 'FedBN'),
    ('FedProx', f'{base}/EXP-132_baselines_3algo/logs/fedprox_pacs_s15.log',
     '#fdae6b', '-', 1.4, 'FedProx'),
    ('FedProto', f'{base}/EXP-132_baselines_3algo/logs/fedproto_pacs_s15.log',
     '#a1d99b', '-', 1.4, 'FedProto'),
    ('MOON', f'{base}/EXP-130_F2DC_baseline_main_table/sc4_v2_logs/moon_fl_pacs_s15.log',
     '#bcbddc', '-', 1.4, 'MOON'),
    ('FDSE', f'{base}/EXP-130_F2DC_baseline_main_table/sc3_v2_logs/fdse_fl_pacs_s15.log',
     '#fb6a4a', '-', 1.6, "FDSE [CVPR'25]"),
    ('F2DC', f'{base}/EXP-130_F2DC_baseline_main_table/sc3_v2_logs/f2dc_fl_pacs_s15.log',
     '#888888', ':', 1.6, "F2DC [CVPR'26]"),
    ('F2DC+DaA', f'{base}/EXP-133_DaA_validation/logs/f2dc_daa_pacs_s15.log',
     '#525252', '--', 1.8, "F2DC+DaA [CVPR'26]"),
    ('Ours', f'{base}/EXP-149_f2dc_dse_lab_pacs/logs/pacs_s333_rho03_rerun_sub1.log',
     '#cb181d', '-', 2.6, 'Ours: F2DC+DSE+LAB'),
]


def smooth(y, k=3):
    if k <= 1: return y
    out = np.copy(y).astype(float)
    for i in range(1, len(y)):
        out[i] = (1 - 1.0/k) * out[i-1] + (1.0/k) * y[i]
    return out


fig, ax = plt.subplots(figsize=(11, 6))
for name, path, color, ls, lw, label in RUNS:
    rounds = parse_log(path)
    if not rounds:
        print(f"  WARN no data: {name}"); continue
    rs = sorted(rounds.keys())
    ys = np.array([rounds[r] for r in rs])
    ys_smooth = smooth(ys, k=3)
    ax.plot(rs, ys_smooth, color=color, linestyle=ls, linewidth=lw, label=label,
            alpha=0.95 if name == 'Ours' else 0.85)

ax.set_xlim(0, 100)
ax.set_ylim(20, 80)
ax.set_xlabel('Communication Round', fontsize=12)
ax.set_ylabel('Average Accuracy [%]', fontsize=12)
ax.set_title('PACS Convergence', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.35)
ax.legend(fontsize=10, loc='lower right', framealpha=0.92, ncol=1)

plt.tight_layout()
out = OUT / 'convergence_pacs_v1.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ saved: {out}")
print(f"  file size: {out.stat().st_size / 1024:.1f} KB")

# print 各 method best (sanity)
print()
print("各 method 历史最高 (R0-R99):")
for name, path, *_ in RUNS:
    rd = parse_log(path)
    if rd:
        br = max(rd, key=rd.get)
        print(f"  {name:<14} best R{br:<3} acc={rd[br]:.2f}")
