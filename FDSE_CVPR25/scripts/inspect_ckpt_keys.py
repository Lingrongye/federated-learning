"""快速 inspect 一个 ckpt 的 key 结构,确认 head 权重命名"""
import sys, torch

ckpt_path = sys.argv[1]
sd = torch.load(ckpt_path, map_location='cpu')
if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']

keys = list(sd.keys())
print(f"total keys: {len(keys)}")
print(f"first 10 keys: {keys[:10]}")
print()

keywords = ['head', 'sem', 'sty', 'classifier', 'corrector']
for kw in keywords:
    matched = [k for k in keys if kw in k.lower()]
    if matched:
        print(f"=== keys with '{kw}' ===")
        for k in matched[:15]:
            shape = tuple(sd[k].shape) if hasattr(sd[k], 'shape') else type(sd[k]).__name__
            print(f"  {k:60s} {shape}")
        print()
