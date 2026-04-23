# -*- coding: utf-8 -*-
"""
EXP-123 Stage A — PACS Art domain 诊断, 从已有 record 提取信号.

目标: 用 orth_only vs FDSE 3-seed R200 record 数据回答:
1. Art 收敛 pattern vs 其他 domain (什么时候 peak, 后期是否退化)
2. per-domain acc 分布 (3-seed mean ± std), 证实 Art std 大
3. orth_only vs FDSE 在 Art 上的差距 (FDSE 有没有比我们更好?)
4. domain gap per round (最佳 vs 最差 domain 差距随时间变化)
5. last-20-round stability (各 domain 后期稳不稳)

输出: report + PNG 图表到 experiments/ablation/EXP-123_art_diagnostic/stageA/
"""

import sys, os, io, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 display
import matplotlib.pyplot as plt

# Windows console utf-8
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------------------------------------
# 路径 + 配置
# ----------------------------------------------------------
RECORD_DIR = r"D:\桌面文件\联邦学习\FDSE_CVPR25\task\PACS_c4\record"
OUT_DIR = r"D:\桌面文件\联邦学习\experiments\ablation\EXP-123_art_diagnostic\stageA"
os.makedirs(OUT_DIR, exist_ok=True)

# PACS 4 client alphabetical 顺序 (flgo 默认)
DOMAINS = ['Art', 'Cartoon', 'Photo', 'Sketch']
DOMAIN_COLORS = {'Art': '#e74c3c', 'Cartoon': '#3498db', 'Photo': '#2ecc71', 'Sketch': '#9b59b6'}

# 目标方法 pattern (只选 LR=0.05 R=200 的, 对齐 EXP-109)
# orth_only: feddsa_scheduled mode=0 纯正交头 (sm0, 不带 se1/sas/scpr 扩展)
#   正则: lo1.0_..._sm0_bp60_bw30_cr80_gli10_lm1.0_al0.25_  (后面直接 Mfeddsa)
#   排除 *_se1_*, *_sas*, *_scpr* 等扩展变体
PATTERNS = {
    'orth_only': {
        'include': 'feddsa_scheduled_lo1.0_lh0.0_ls1.0_tau0.2_sdn5_pd128_sm0_bp60_bw30_cr80_gli10_lm1.0_al0.25_Mfeddsa_scheduled',
        'exclude_tokens': ['_se1_', '_sas', '_scpr', '_sas_tau', '_scpr_tau'],
        'lr': '5.00e-02',  # LR=0.05
        'color': '#e74c3c',
    },
    'fdse': {
        'include': 'fdse_lmbd0.5_tau0.5_beta0.001_Mfdse',
        'exclude_tokens': [],
        'lr': '5.00e-02',
        'color': '#3498db',
    },
}
SEEDS = [2, 15, 333]


def seed_of(fn):
    for s in SEEDS:
        if f'_S{s}_' in fn:
            return s
    return None


def match_file(fn, pattern):
    """fn 是否匹配 pattern 条件."""
    if pattern['include'] not in fn:
        return False
    for bad in pattern['exclude_tokens']:
        if bad in fn:
            return False
    if f"_LR{pattern['lr']}_" not in fn:
        return False
    return True


def find_records():
    """返回 {method: {seed: filepath}} dict."""
    all_files = sorted(os.listdir(RECORD_DIR))
    result = {}
    for method, pat in PATTERNS.items():
        result[method] = {}
        for fn in all_files:
            if not fn.endswith('.json'):
                continue
            s = seed_of(fn)
            if s is None:
                continue
            if not match_file(fn, pat):
                continue
            if s in result[method]:
                # 多个匹配, 选 fn 最短的 (假设最简配置)
                if len(fn) < len(result[method][s]):
                    result[method][s] = fn
            else:
                result[method][s] = fn
    return result


def load_record(fp):
    with open(fp) as f:
        return json.load(f)


def extract_curves(d):
    """从 JSON 提取 per-round per-domain accuracy.

    Returns:
        avg_curve: list of mean_local_test_accuracy (R+1 个)
        per_client_curve: [R+1][4] array, [round][client]
    """
    avg_curve = d.get('mean_local_test_accuracy', [])
    per_client_dist = d.get('local_test_accuracy_dist', [])
    if per_client_dist and isinstance(per_client_dist[0], list):
        per_client = np.array(per_client_dist) * 100  # (R+1, 4)
    else:
        per_client = None
    return np.array(avg_curve) * 100, per_client


# ----------------------------------------------------------
# 开始分析
# ----------------------------------------------------------
print('=' * 60)
print('EXP-123 Stage A — PACS Art domain 诊断')
print('=' * 60)
records = find_records()
for method, sm in records.items():
    print(f'{method}: {sorted(sm.keys())} ({len(sm)} seeds)')

# ----------------------------------------------------------
# 1. 每个 (method, seed) 加载 per-domain curve
# ----------------------------------------------------------
print('\n--- Step 1: Load records ---')
curves = {}  # {method: {seed: per_client_curve (R+1, 4)}}
for method, sm in records.items():
    curves[method] = {}
    for s, fn in sm.items():
        d = load_record(os.path.join(RECORD_DIR, fn))
        avg_c, pc = extract_curves(d)
        if pc is None:
            print(f'  [warn] {method}_s{s}: no per-client data')
            continue
        print(f'  {method}_s{s}: R={pc.shape[0]}, '
              f'AVG Best {np.max(np.mean(pc, axis=1)):.2f}  '
              f'AVG Last {np.mean(pc[-1]):.2f}')
        curves[method][s] = pc

# ----------------------------------------------------------
# 2. 计算 3-seed mean/std per-domain 曲线
# ----------------------------------------------------------
print('\n--- Step 2: 3-seed mean ± std per-domain ---')
summary = {}  # {method: {'mean': (R+1, 4), 'std': (R+1, 4), 'max_best': (4,), 'max_last': (4,)}}
for method, sm in curves.items():
    if len(sm) < 3:
        print(f'  [warn] {method} 只有 {len(sm)} seed, 跳过')
        continue
    all_pc = np.stack([sm[s] for s in SEEDS if s in sm])  # (3, R+1, 4)
    mean_pc = all_pc.mean(axis=0)  # (R+1, 4)
    std_pc = all_pc.std(axis=0)
    per_domain_best = all_pc.max(axis=1).mean(axis=0)   # per-domain mean of (3-seed max)
    per_domain_last = all_pc[:, -1, :].mean(axis=0)     # 3-seed mean of last
    per_domain_best_std = all_pc.max(axis=1).std(axis=0)
    per_domain_last_std = all_pc[:, -1, :].std(axis=0)
    summary[method] = {
        'mean_curve': mean_pc,
        'std_curve': std_pc,
        'per_domain_best': per_domain_best,
        'per_domain_last': per_domain_last,
        'per_domain_best_std': per_domain_best_std,
        'per_domain_last_std': per_domain_last_std,
    }
    print(f'\n  == {method} ==')
    print(f'  {"":10s} {"Best mean":>10s} {"±std":>7s} '
          f'{"Last mean":>10s} {"±std":>7s}')
    for i, d in enumerate(DOMAINS):
        print(f'  {d:10s} {per_domain_best[i]:>10.2f} '
              f'{per_domain_best_std[i]:>7.2f}  '
              f'{per_domain_last[i]:>10.2f} '
              f'{per_domain_last_std[i]:>7.2f}')
    print(f'  {"Avg":10s} {per_domain_best.mean():>10.2f} '
          f'{per_domain_best_std.mean():>7.2f}  '
          f'{per_domain_last.mean():>10.2f}')

# ----------------------------------------------------------
# 3. 画图 1: per-domain accuracy curve (mean ± std)
# ----------------------------------------------------------
print('\n--- Step 3: Plot per-domain curves ---')
fig, axes = plt.subplots(1, len(summary), figsize=(7 * len(summary), 5), sharey=True)
if len(summary) == 1:
    axes = [axes]
for ax, (method, s) in zip(axes, summary.items()):
    R = s['mean_curve'].shape[0]
    rounds = np.arange(R)
    for i, d in enumerate(DOMAINS):
        ax.plot(rounds, s['mean_curve'][:, i], label=d, color=DOMAIN_COLORS[d], linewidth=1.5)
        ax.fill_between(rounds,
                        s['mean_curve'][:, i] - s['std_curve'][:, i],
                        s['mean_curve'][:, i] + s['std_curve'][:, i],
                        color=DOMAIN_COLORS[d], alpha=0.2)
    ax.set_title(f'{method} (3-seed PACS R=200)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 100])
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'fig1_per_domain_curves.png')
plt.savefig(plot_path, dpi=120)
plt.close()
print(f'  saved: {plot_path}')

# ----------------------------------------------------------
# 4. Domain gap per round (best − worst)
# ----------------------------------------------------------
print('\n--- Step 4: Domain gap (max − min) per round ---')
fig, ax = plt.subplots(figsize=(10, 5))
for method, s in summary.items():
    mc = s['mean_curve']  # (R+1, 4)
    gap = mc.max(axis=1) - mc.min(axis=1)  # (R+1,)
    ax.plot(np.arange(len(gap)), gap, label=method,
            color=PATTERNS[method]['color'], linewidth=1.8)
ax.set_xlabel('Round')
ax.set_ylabel('Domain gap (max - min) %')
ax.set_title('Domain gap over training rounds')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'fig2_domain_gap.png')
plt.savefig(plot_path, dpi=120)
plt.close()
print(f'  saved: {plot_path}')

# ----------------------------------------------------------
# 5. Peak round per domain per method (何时收敛?)
# ----------------------------------------------------------
print('\n--- Step 5: Peak round per domain ---')
for method, s in summary.items():
    mc = s['mean_curve']
    peak_rounds = mc.argmax(axis=0)
    print(f'\n  {method} peak round:')
    for i, d in enumerate(DOMAINS):
        print(f'    {d:10s}: R={peak_rounds[i]:3d}  (acc {mc[peak_rounds[i], i]:.2f}%)')

# ----------------------------------------------------------
# 6. Last-20-round stability (std of last 20 rounds per domain)
# ----------------------------------------------------------
print('\n--- Step 6: Last-20-round stability (std of per-round acc) ---')
for method, s in summary.items():
    mc = s['mean_curve']
    last20 = mc[-20:]
    stab = last20.std(axis=0)
    print(f'\n  {method} last-20 std:')
    for i, d in enumerate(DOMAINS):
        print(f'    {d:10s}: std {stab[i]:.2f}% '
              f'(range [{last20[:, i].min():.2f}, {last20[:, i].max():.2f}])')

# ----------------------------------------------------------
# 7. orth_only vs FDSE 对比 (per-domain delta)
# ----------------------------------------------------------
if 'orth_only' in summary and 'fdse' in summary:
    print('\n--- Step 7: orth_only vs FDSE per-domain ---')
    oo_best = summary['orth_only']['per_domain_best']
    fd_best = summary['fdse']['per_domain_best']
    oo_last = summary['orth_only']['per_domain_last']
    fd_last = summary['fdse']['per_domain_last']
    print(f"\n  {'Domain':10s} {'orth_only B':>12s} {'FDSE B':>10s} {'Δ B':>8s} "
          f"{'orth_only L':>12s} {'FDSE L':>10s} {'Δ L':>8s}")
    for i, d in enumerate(DOMAINS):
        dB = oo_best[i] - fd_best[i]
        dL = oo_last[i] - fd_last[i]
        marker_B = '✅' if dB > 0 else ('❌' if dB < 0 else '=')
        marker_L = '✅' if dL > 0 else ('❌' if dL < 0 else '=')
        print(f'  {d:10s} {oo_best[i]:>12.2f} {fd_best[i]:>10.2f} '
              f'{dB:>+7.2f} {marker_B}  '
              f'{oo_last[i]:>12.2f} {fd_last[i]:>10.2f} {dL:>+7.2f} {marker_L}')

# ----------------------------------------------------------
# 8. 输出简易报告 markdown
# ----------------------------------------------------------
print('\n--- Step 8: Generate markdown report ---')
report_lines = []
report_lines.append('# EXP-123 Stage A — PACS Art Diagnostic Report\n')
report_lines.append(f'生成时间: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
report_lines.append('\n## Data sources\n')
for method, sm in records.items():
    report_lines.append(f'- **{method}**: {len(sm)} seeds {sorted(sm.keys())}\n')

report_lines.append('\n## 3-seed Per-domain Best / Last\n')
report_lines.append('| Method | Domain | Best ± std | Last ± std |\n')
report_lines.append('|---|---|:---:|:---:|\n')
for method, s in summary.items():
    for i, d in enumerate(DOMAINS):
        report_lines.append(f"| {method} | {d} | "
                            f"{s['per_domain_best'][i]:.2f} ± {s['per_domain_best_std'][i]:.2f} | "
                            f"{s['per_domain_last'][i]:.2f} ± {s['per_domain_last_std'][i]:.2f} |\n")

report_lines.append('\n## Peak round\n')
report_lines.append('| Method | Art | Cartoon | Photo | Sketch |\n')
report_lines.append('|---|:-:|:-:|:-:|:-:|\n')
for method, s in summary.items():
    mc = s['mean_curve']
    peak = mc.argmax(axis=0)
    row = '| ' + method + ' | ' + ' | '.join(f'R{peak[i]}' for i in range(4)) + ' |\n'
    report_lines.append(row)

report_lines.append('\n## Last-20-round stability (std)\n')
report_lines.append('| Method | Art | Cartoon | Photo | Sketch |\n')
report_lines.append('|---|:-:|:-:|:-:|:-:|\n')
for method, s in summary.items():
    last20 = s['mean_curve'][-20:]
    stab = last20.std(axis=0)
    row = '| ' + method + ' | ' + ' | '.join(f'{stab[i]:.2f}' for i in range(4)) + ' |\n'
    report_lines.append(row)

if 'orth_only' in summary and 'fdse' in summary:
    report_lines.append('\n## orth_only vs FDSE per-domain Δ\n')
    oo = summary['orth_only']['per_domain_best']
    fd = summary['fdse']['per_domain_best']
    report_lines.append('| Domain | orth_only | FDSE | Δ (orth − FDSE) |\n')
    report_lines.append('|---|:-:|:-:|:-:|\n')
    for i, d in enumerate(DOMAINS):
        delta = oo[i] - fd[i]
        sign = '✅' if delta > 0 else ('❌' if delta < 0 else '=')
        report_lines.append(f'| {d} | {oo[i]:.2f} | {fd[i]:.2f} | {delta:+.2f} {sign} |\n')

report_lines.append('\n## Figures\n')
report_lines.append('- `fig1_per_domain_curves.png` — per-domain curve 3-seed mean ± std\n')
report_lines.append('- `fig2_domain_gap.png` — domain gap (max - min) over rounds\n')

report_path = os.path.join(OUT_DIR, 'report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)
print(f'  report saved: {report_path}')
print('\n' + '=' * 60)
print('Stage A 完成 — 看 report.md + 2 张图')
print('=' * 60)
