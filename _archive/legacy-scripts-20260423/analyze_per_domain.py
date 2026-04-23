import json, glob, os, re, sys

def load(f):
    with open(f) as fh: d = json.load(fh)
    return d.get('local_test_accuracy_dist', [])

out_lines = []

out_lines.append('=== PACS per-domain AVG Best/Last (clients: C0=Photo C1=Art C2=Cartoon C3=Sketch) ===\n')

# orth_only LR=0.05
out_lines.append('\n--- orth_only LR=0.05 ---\n')
for f in sorted(glob.glob('FDSE_CVPR25/task/PACS_c4/record/*.json')):
    if 'sm0' not in f or 'LR5.00e-02' not in f or 'R200' not in f: continue
    if 'lh0.0' not in f: continue
    sd = re.search(r'_S(\d+)_', f).group(1)
    d = load(f)
    if not d: continue
    dist = [[c*100 for c in r] for r in d]
    last = dist[-1]
    best = [max(r[c] for r in dist) for c in range(4)]
    out_lines.append('s={:>3}: Last P={:.1f} A={:.1f} C={:.1f} S={:.1f} | Best P={:.1f} A={:.1f} C={:.1f} S={:.1f}\n'.format(
        sd, last[0],last[1],last[2],last[3], best[0],best[1],best[2],best[3]))

# orth_only LR=0.1
out_lines.append('\n--- orth_only LR=0.1 ---\n')
for f in sorted(glob.glob('FDSE_CVPR25/task/PACS_c4/record/*.json')):
    if 'sm0' not in f or 'LR1.00e-01' not in f or 'R200' not in f: continue
    if 'lh0.0' not in f: continue
    sd = re.search(r'_S(\d+)_', f).group(1)
    d = load(f)
    if not d: continue
    dist = [[c*100 for c in r] for r in d]
    last = dist[-1]
    best = [max(r[c] for r in dist) for c in range(4)]
    out_lines.append('s={:>3}: Last P={:.1f} A={:.1f} C={:.1f} S={:.1f} | Best P={:.1f} A={:.1f} C={:.1f} S={:.1f}\n'.format(
        sd, last[0],last[1],last[2],last[3], best[0],best[1],best[2],best[3]))

# FDSE
out_lines.append('\n--- FDSE (LR=0.05) ---\n')
for f in sorted(glob.glob('FDSE_CVPR25/task/PACS_c4/record/fdse_*R200*.json')):
    sd = re.search(r'_S(\d+)_', f).group(1)
    d = load(f)
    if not d: continue
    dist = [[c*100 for c in r] for r in d]
    last = dist[-1]
    best = [max(r[c] for r in dist) for c in range(4)]
    out_lines.append('s={:>5}: Last P={:.1f} A={:.1f} C={:.1f} S={:.1f} | Best P={:.1f} A={:.1f} C={:.1f} S={:.1f}\n'.format(
        sd, last[0],last[1],last[2],last[3], best[0],best[1],best[2],best[3]))

out_lines.append('\n\n=== Office per-domain (clients: C0=Caltech C1=Amazon C2=DSLR(157样本) C3=Webcam) ===\n')

out_lines.append('\n--- orth_only LR=0.05 ---\n')
for f in sorted(glob.glob('FDSE_CVPR25/task/office_caltech10_c4/record/*.json')):
    if 'sm0' not in f or 'LR5.00e-02' not in f or 'R200' not in f: continue
    if 'lh0.0' not in f: continue
    sd = re.search(r'_S(\d+)_', f).group(1)
    d = load(f)
    if not d: continue
    dist = [[c*100 for c in r] for r in d]
    last = dist[-1]
    best = [max(r[c] for r in dist) for c in range(4)]
    out_lines.append('s={:>3}: Last C={:.1f} A={:.1f} D={:.1f} W={:.1f} | Best C={:.1f} A={:.1f} D={:.1f} W={:.1f}\n'.format(
        sd, last[0],last[1],last[2],last[3], best[0],best[1],best[2],best[3]))

out_lines.append('\n--- orth_only LR=0.1 ---\n')
for f in sorted(glob.glob('FDSE_CVPR25/task/office_caltech10_c4/record/*.json')):
    if 'sm0' not in f or 'LR1.00e-01' not in f or 'R200' not in f: continue
    if 'lh0.0' not in f: continue
    sd = re.search(r'_S(\d+)_', f).group(1)
    d = load(f)
    if not d: continue
    dist = [[c*100 for c in r] for r in d]
    last = dist[-1]
    best = [max(r[c] for r in dist) for c in range(4)]
    out_lines.append('s={:>3}: Last C={:.1f} A={:.1f} D={:.1f} W={:.1f} | Best C={:.1f} A={:.1f} D={:.1f} W={:.1f}\n'.format(
        sd, last[0],last[1],last[2],last[3], best[0],best[1],best[2],best[3]))

out_lines.append('\n--- FDSE ---\n')
for f in sorted(glob.glob('FDSE_CVPR25/task/office_caltech10_c4/record/fdse_*R200*.json')):
    sd = re.search(r'_S(\d+)_', f).group(1)
    d = load(f)
    if not d: continue
    dist = [[c*100 for c in r] for r in d]
    last = dist[-1]
    best = [max(r[c] for r in dist) for c in range(4)]
    out_lines.append('s={:>3}: Last C={:.1f} A={:.1f} D={:.1f} W={:.1f} | Best C={:.1f} A={:.1f} D={:.1f} W={:.1f}\n'.format(
        sd, last[0],last[1],last[2],last[3], best[0],best[1],best[2],best[3]))

# 写到文件
with open('per_domain_analysis.txt', 'w', encoding='utf-8') as f:
    f.writelines(out_lines)

print('DONE, written to per_domain_analysis.txt')
