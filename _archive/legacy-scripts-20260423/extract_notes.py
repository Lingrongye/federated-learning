# -*- coding: utf-8 -*-
import os, glob, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

exp_dir = 'experiments'
for note_path in sorted(glob.glob(os.path.join(exp_dir, '**', 'NOTE.md'), recursive=True)):
    dirname = os.path.basename(os.path.dirname(note_path))
    m = re.match(r'EXP-(\d+)', dirname)
    if not m:
        continue
    exp_num = int(m.group(1))
    with open(note_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    title = dirname
    for line in lines[:5]:
        if line.startswith('# '):
            title = line[2:].strip()
            break
    desc = ''
    for line in lines:
        s = line.strip()
        if s and not s.startswith('#') and not s.startswith('---') and not s.startswith('|') and not s.startswith('```'):
            desc = s[:150]
            break
    print(f'{exp_num:03d}|||{title}|||{desc}')
