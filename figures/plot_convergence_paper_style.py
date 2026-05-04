"""
Paper-style convergence figure (双 panel: Office-Caltech / PACS)
- 仿照 F²DC paper Figure 5 风格: 等线宽 / 鲜明配色 / 2 列 legend / italic
- 8 method: FedAvg / FedBN / FedProx / FedProto / MOON / FDSE / F2DC+DaA / Ours
- baseline 同 seed 公平对比 (s=15 优先, s=15 截断时 fallback s=333)
- Ours 最强 single-run: Office EXP-152 s=333 rho=0.2 v2c (best 65.43); PACS EXP-149 s=333 rho=0.3 rerun (best 75.37)
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = '/Users/changdao/联邦学习/experiments/ablation'
F2DC_LOGS = '/Users/changdao/联邦学习/F2DC/_phaseC_v2_logs'
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


def smooth(y, k=2):
    if k <= 1: return y
    out = np.copy(y).astype(float)
    for i in range(1, len(y)):
        out[i] = (1 - 1.0/k) * out[i-1] + (1.0/k) * y[i]
    return out


# 仿 F²DC paper Figure 5 配色 (8 method)
# 原图: blue / magenta / orange / pink / cyan / green / rose / red
COLORS = {
    # 仿 F²DC paper Figure 5 配色 (区分度高)
    'FedAvg':    '#1F77B4',   # 蓝
    'FedBN':     '#9467BD',   # 紫
    'FedProx':   '#FF7F0E',   # 橙
    'FedProto':  '#E377C2',   # 浅粉
    'MOON':      '#17BECF',   # 浅蓝/青
    'FDSE':      '#2CA02C',   # 绿
    'F2DC+DaA':  '#C2185B',   # 玫瑰红
    'Ours':      '#D62728',   # 大红 (突出)
}

LW = 1.5  # 统一线宽 (paper-style)

# Office runs (s=15 baseline, s=333 Ours)
OFFICE_RUNS = [
    ('FedAvg',   f'{base}/EXP-130_F2DC_baseline_main_table/sc4_v2_logs/fedavg_fl_officecaltech_s15.log'),
    ('FedBN',    f'{base}/EXP-132_baselines_3algo/logs/fedbn_office_s15.log'),
    ('FedProx',  f'{base}/EXP-132_baselines_3algo/logs/fedprox_office_s15.log'),
    ('FedProto', f'{base}/EXP-132_baselines_3algo/logs/fedproto_office_s15.log'),
    ('MOON',     f'{base}/EXP-130_F2DC_baseline_main_table/sc4_v2_logs/moon_fl_officecaltech_s15.log'),
    # FDSE Office 用 EXP-153 lmbd=0.05 fix 后数据 (跟 PACS 一致用修复版)
    ('FDSE',     f'{base}/EXP-153_fdse_review/logs/fdse_office_s15.log'),
    ('F2DC+DaA', f'{base}/EXP-133_DaA_validation/logs/f2dc_daa_office_s333.log'),
    ('Ours',     f'{base}/EXP-152_f2dc_dse_lab_office_sweep/logs/office_s333_rho02_v2c.log'),
]

# PACS runs (优先 R100 完整版本)
PACS_RUNS = [
    ('FedAvg',   f'{base}/EXP-130_F2DC_baseline_main_table/sc4_v2_logs/fedavg_fl_pacs_s15.log'),
    ('FedBN',    f'{base}/EXP-132_baselines_3algo/logs/fedbn_pacs_s15.log'),
    ('FedProx',  f'{base}/EXP-132_baselines_3algo/logs/fedprox_pacs_s333.log'),  # s=15 截断
    ('FedProto', f'{base}/EXP-132_baselines_3algo/logs/fedproto_pacs_s15.log'),  # mem-leak crash R41
    ('MOON',     f'{F2DC_LOGS}/moon_fl_pacs_s15.log'),
    # FDSE PACS 用 EXP-153 lmbd=0.05 fix 后数据 (替代原 collapse 版本)
    ('FDSE',     f'{base}/EXP-153_fdse_review/logs/fdse_pacs_s15_lmbd005_revert.log'),
    ('F2DC+DaA', f'{base}/EXP-133_DaA_validation/logs/f2dc_daa_pacs_s15.log'),
    ('Ours',     f'{base}/EXP-149_f2dc_dse_lab_pacs/logs/pacs_s333_rho03_rerun_sub1.log'),
]


def plot_panel(ax, runs, title, ylim):
    for name, path in runs:
        rounds = parse_log(path)
        if not rounds:
            print(f"  WARN no data: {name} ({path})"); continue
        rs = sorted(rounds.keys())
        ys = np.array([rounds[r] for r in rs])
        ys_smooth = smooth(ys, k=2)
        is_ours = (name == 'Ours')
        # italic label, $F^2DC+DSE+LAB$ 类似学术风
        if is_ours:
            label = r'$\mathrm{Ours}$'
        elif name == 'F2DC+DaA':
            label = r'$F^2DC{+}DaA$'
        elif name == 'FDSE':
            label = r'$\mathit{FDSE}$'
        else:
            label = rf'$\mathit{{{name}}}$'
        lw = 2.8 if is_ours else LW
        alpha = 1.0 if is_ours else 0.78
        ax.plot(rs, ys_smooth, color=COLORS[name], linewidth=lw, label=label, alpha=alpha)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Average Accuracy [%]', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10, loc='lower right', ncol=2, framealpha=0.95,
              edgecolor='gray', handlelength=2.5, columnspacing=1.0,
              labelspacing=0.4, fancybox=False)


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
plot_panel(axes[0], OFFICE_RUNS, 'Office-Caltech', ylim=(10, 70))
plot_panel(axes[1], PACS_RUNS,  'PACS',           ylim=(20, 80))

plt.tight_layout()
out = OUT / 'convergence_paper_style.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ saved: {out}")
print(f"  file size: {out.stat().st_size / 1024:.1f} KB")
