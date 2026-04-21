"""Summarize EXP-113 probe results into comparison table.

Reads all probe JSONs in experiments/ablation/EXP-113_vib_vsc_supcon/office/probes/
and aggregates into 3-seed mean tables.
"""
import json
import glob
import os
import sys
from pathlib import Path


OUTDIR = Path('/root/autodl-tmp/federated-learning/experiments/ablation/EXP-113_vib_vsc_supcon/office/probes')


def load_probes(outdir):
    results = {}  # {method: {seed: probe_dict}}
    for f in sorted(outdir.glob('*.json')):
        name = f.stem  # e.g. A_vib_s2
        parts = name.split('_s')
        if len(parts) != 2:
            continue
        method, seed = parts
        with open(f) as fp:
            d = json.load(fp)
        results.setdefault(method, {})[seed] = d
    return results


def fmt_probe(target, sub):
    """Print target probe results row-by-method."""
    if not sub:
        return ''
    r = sub.get(target, {})
    # capacity probe output: {'linear': {'test':...}, 'mlp_16': {...}, 'mlp_64': ..., ...}
    lin = r.get('linear', {}).get('test', None)
    m16 = r.get('mlp_16', {}).get('test', None)
    m64 = r.get('mlp_64', {}).get('test', None)
    m128 = r.get('mlp_128', {}).get('test', None)
    m256 = r.get('mlp_256', {}).get('test', None)
    def f(v): return f'{v:.3f}' if v is not None else '?'
    return f'{f(lin)}  {f(m16)}  {f(m64)}  {f(m128)}  {f(m256)}'


def method_order():
    return ['orth_uc1', 'A_vib', 'B_vsc', 'C_supcon']


def main():
    results = load_probes(OUTDIR)

    for target in ['probe_sty_class', 'probe_sem_class', 'probe_sty_domain', 'probe_sem_domain']:
        print(f'\n=== {target} (linear / m16 / m64 / m128 / m256) ===')
        for method in method_order():
            if method not in results:
                continue
            sub = results[method]
            for seed in sorted(sub.keys()):
                row = fmt_probe(target, sub[seed])
                print(f'  {method:10s} {seed:4s}: {row}')
            # 3-seed mean
            metrics = ['linear', 'mlp_16', 'mlp_64', 'mlp_128', 'mlp_256']
            means = []
            for m in metrics:
                vals = []
                for seed in sub:
                    v = sub[seed].get(target, {}).get(m, {}).get('test', None)
                    if v is not None:
                        vals.append(v)
                if vals:
                    means.append(sum(vals) / len(vals))
                else:
                    means.append(None)
            mean_line = '  '.join(f'{m:.3f}' if m is not None else '?' for m in means)
            print(f'  {method:10s} mean: {mean_line}')
            print()


if __name__ == '__main__':
    main()
