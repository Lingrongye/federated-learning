import pathlib
base = pathlib.Path(r'D:\桌面文件\联邦学习\refine-logs\2026-04-22_FedDSA-Swap')
tmpl = (base / 'review-prompt-r1.txt').read_text(encoding='utf-8')
prop = (base / 'round-0-initial-proposal.md').read_text(encoding='utf-8')
combined = tmpl.replace('<paste the round-0-initial-proposal.md content here>', prop)
(base / 'combined-prompt-r1.txt').write_text(combined, encoding='utf-8')
print('combined length:', len(combined), 'chars')
