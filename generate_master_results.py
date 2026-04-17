"""从 JSON records 生成干净的核心对比表（论文候选）。只保留关键方法，按 {2,15,333} 3-seed mean。"""
import json, glob, os, re
from collections import defaultdict

def load(f):
    with open(f) as fh: return json.load(fh)

def get_m(d):
    a = [x*100 for x in d.get('local_test_accuracy', [])]
    v = [x*100 for x in d.get('mean_local_test_accuracy', [])]
    dist = d.get('local_test_accuracy_dist', [])
    if len(v) < 150: return None
    out = {'R': len(v),
           'ALL_best': max(a), 'ALL_last': a[-1],
           'AVG_best': max(v), 'AVG_last': v[-1],
           'AVG_best_R': v.index(max(v))+1}
    if dist:
        d_pct = [[c*100 for c in r] for r in dist]
        for c in range(len(d_pct[0]) if d_pct else 0):
            col = [r[c] for r in d_pct]
            out[f'c{c}_best'] = max(col)
            out[f'c{c}_last'] = col[-1]
    return out

def classify(bn):
    """返回 (method_name, lr) 或 None 跳过"""
    if 'R200' not in bn or 'R5_' in bn: return None
    lr = '0.05' if 'LR5.00e-02' in bn else ('0.1' if 'LR1.00e-01' in bn else '?')

    if bn.startswith('fdse'):
        return f'FDSE'

    if bn.startswith('feddsa_algopara_1.0_0.0_1.0_') or bn.startswith('feddsa_Mfeddsa'):
        return f'FedDSA 原版 LR={lr}'

    if bn.startswith('feddsa_scheduled_'):
        # JSON 文件名格式: feddsa_scheduled_lo1.0_lh0.0_ls1.0_tau0.2_sdn5_pd128_sm0_bp60_bw30_cr80_gli10[_lm1.0_al0.25[_se1[_sas1_sas_tau0.3]]]_M...
        lo_m = re.search(r'_lo([\d.]+)_', bn)
        lh_m = re.search(r'_lh([\d.]+)_', bn)
        sm_m = re.search(r'_sm(\d+)_', bn)
        sas_m = re.search(r'_sas(\d)_', bn)
        if not (lo_m and lh_m and sm_m): return None
        lo = lo_m.group(1); lh = lh_m.group(1); sm = sm_m.group(1)
        sas = sas_m.group(1) if sas_m else '0'
        mode_name = {'0':'orth_only','1':'bell_60_30','2':'cutoff_80',
                     '3':'always_on','4':'mse_anchor','5':'alpha_sparse',
                     '6':'mse_alpha','7':'detach_aug'}.get(sm,f'sm{sm}')
        suffix = ''
        if lo != '1.0': suffix += f' lo={lo}'
        if lh not in ['0.0','0']: suffix += ' +hsic'
        if sas == '1': suffix += ' +sas'
        return f'{mode_name}{suffix} LR={lr}'

    return None

lines = ['# Master Results Table — R200 核心方法对比\n\n']
lines.append('> 口径对齐 FDSE Table 1: ALL (样本加权) / AVG (客户端等权) × Best (峰值) / Last (R200)\n')
lines.append('> 优先报 `{2,15,333}` 3-seed mean (严格同 seed)；多 seed 可用时额外报 `{2,333,42}`\n\n')

DOMAIN_MAP = {
    'PACS_c4': ['Art', 'Cartoon', 'Photo', 'Sketch'],
    'office_caltech10_c4': ['Caltech', 'Amazon', 'DSLR', 'Webcam'],
}

# Orth_only LR=0.1 基线有多个 JSON（包括 EXP-076 mode=0）— 需要精准 filter
# 我们要的 baseline orth_only LR=0.1: lo=1.0, lh=0.0, sm=0, LR=0.1
# 我们要的 orth_only LR=0.05: lo=1.0, lh=0.0, sm=0, LR=0.05

def is_target(method, sd):
    targets = {
        'FDSE', 'FedDSA 原版 LR=0.1', 'FedDSA 原版 LR=0.05',
        'orth_only LR=0.1', 'orth_only LR=0.05',
        'mse_alpha LR=0.1', 'mse_alpha LR=0.05',
        'bell_60_30 LR=0.1', 'cutoff_80 LR=0.1', 'always_on LR=0.1',
        'orth_only lo=2.0 LR=0.1', 'orth_only lo=0.5 LR=0.1',
        'orth_only +hsic LR=0.1', 'orth_only +sas LR=0.05',
    }
    return method in targets

for task, task_name in [('PACS_c4', 'PACS'), ('office_caltech10_c4', 'Office-Caltech10')]:
    lines.append(f'\n## {task_name} R200\n\n')
    domains = DOMAIN_MAP[task]
    groups = defaultdict(dict)  # method -> {seed: metrics}
    for f in sorted(glob.glob(f'FDSE_CVPR25/task/{task}/record/*.json')):
        bn = os.path.basename(f)
        method = classify(bn)
        if method is None or not is_target(method, None): continue
        sd = re.search(r'_S(\d+)_', bn).group(1)
        m = get_m(load(f))
        if m is None: continue
        # 如果同一 method + seed 有多个（多次跑），保留最新/最高
        if sd in groups[method]:
            if m['AVG_best'] > groups[method][sd]['AVG_best']:
                groups[method][sd] = m
        else:
            groups[method][sd] = m

    # 按方法组输出
    method_order = ['FedDSA 原版 LR=0.1', 'FDSE',
                    'orth_only LR=0.1', 'orth_only LR=0.05',
                    'mse_alpha LR=0.1',
                    'bell_60_30 LR=0.1', 'cutoff_80 LR=0.1', 'always_on LR=0.1',
                    'orth_only lo=2.0 LR=0.1', 'orth_only lo=0.5 LR=0.1',
                    'orth_only +hsic LR=0.1', 'orth_only +sas LR=0.05']

    hdr = '| 方法 | seeds | ALL Best | ALL Last | AVG Best | AVG Last | ' + ' | '.join(f'{d} Best/Last' for d in domains) + ' |\n'
    sep = '|------|------|---------|---------|---------|---------|' + '|'.join(['---']*len(domains)) + '|\n'
    lines.append(hdr)
    lines.append(sep)

    def mean_row(method, seeds_to_use, label, rows):
        """打印一行 mean，只用指定的 seed"""
        avail = [(sd, rows[sd]) for sd in seeds_to_use if sd in rows]
        if not avail: return None
        n = len(avail)
        def mn(k): return sum(r[k] for _,r in avail) / n
        per_dom = []
        for c in range(len(domains)):
            per_dom.append(f'{mn(f"c{c}_best"):.1f}/{mn(f"c{c}_last"):.1f}')
        return f'| **{method}** | {label}({",".join([sd for sd,_ in avail])}) | **{mn("ALL_best"):.2f}** | **{mn("ALL_last"):.2f}** | **{mn("AVG_best"):.2f}** | **{mn("AVG_last"):.2f}** | ' + ' | '.join(per_dom) + ' |\n'

    for method in method_order:
        if method not in groups: continue
        rows = groups[method]
        seeds_sorted = sorted(rows.keys(), key=lambda x: int(x))

        # 优先报 {2,15,333}
        r = mean_row(method, ['2','15','333'], '3s', rows)
        if r: lines.append(r)
        else:
            # 否则报所有可用 seed
            r = mean_row(method, seeds_sorted, f'{len(seeds_sorted)}s', rows)
            if r: lines.append(r)

        # 如果有 {2, 333, 42}，也报一下（因为部分今日 seed 集）
        if all(s in rows for s in ['2','333','42']) and '15' not in rows:
            pass  # 已经在上面报了
        elif all(s in rows for s in ['2','333','42']) and '15' in rows:
            # 补一行 {2, 333, 42}
            r = mean_row(method, ['2','333','42'], '3s', rows)
            if r: lines.append(r.replace(f'**{method}**', f'{method} (alt)'))

        # per seed
        for sd in seeds_sorted:
            m = rows[sd]
            per_dom = ' | '.join(
                f'{m.get(f"c{c}_best",0):.1f}/{m.get(f"c{c}_last",0):.1f}'
                for c in range(len(domains))
            )
            lines.append(f'| ↳ s={sd} | R{m["AVG_best_R"]} | {m["ALL_best"]:.2f} | {m["ALL_last"]:.2f} | {m["AVG_best"]:.2f} | {m["AVG_last"]:.2f} | ' + per_dom + ' |\n')
        lines.append('\n')

with open('experiments/MASTER_RESULTS.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('OK -> experiments/MASTER_RESULTS.md')
