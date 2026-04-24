# -*- coding: utf-8 -*-
"""Debug sgpa model loading — 打印 state_dict 结构 + 对比 load 前后."""
import sys, os
from pathlib import Path

FDSE_ROOT = Path(__file__).parent.parent / 'FDSE_CVPR25'
sys.path.insert(0, str(FDSE_ROOT))

import torch
from algorithm.feddsa_sgpa import FedDSASGPAModel
import json

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else '/root/fl_checkpoints/sgpa_PACS_c4_s2_R200_1776795770'

meta = json.load(open(os.path.join(ckpt_dir, 'meta.json')))
print(f'meta: {meta}')

gstate = torch.load(os.path.join(ckpt_dir, 'global_model.pt'), map_location='cpu', weights_only=False)
cstates = torch.load(os.path.join(ckpt_dir, 'client_models.pt'), map_location='cpu', weights_only=False)

print(f'\nglobal_state keys: {len(gstate)}')
print(f'client_states: {len(cstates)} clients')
print(f'client_states[0] keys: {len(cstates[0]) if cstates[0] else "None"}')

# 构造 model
model = FedDSASGPAModel(
    num_classes=7, feat_dim=1024, proj_dim=128,
    tau_etf=meta.get('tau_etf', 0.1), use_etf=bool(meta['use_etf']),
    ca=0, num_clients=4, backbone='alexnet',
)
print(f'\nmodel state_dict keys: {len(model.state_dict())}')

# 对比 keys
model_keys = set(model.state_dict().keys())
g_keys = set(gstate.keys())
c_keys = set(cstates[0].keys())

print(f'\nglobal - model: {g_keys - model_keys}')  # global 多的
print(f'model - global: {model_keys - g_keys}')    # model 缺的
print(f'\nclient0 - model: {c_keys - model_keys}')
print(f'model - client0: {model_keys - c_keys}')

# 看 head 的 key 值
print('\n=== head key samples ===')
for k in sorted(model_keys):
    if 'head' in k.lower() or 'M' == k:
        g = gstate.get(k)
        c = cstates[0].get(k)
        print(f'{k}: global_shape={g.shape if isinstance(g, torch.Tensor) else g} '
              f'client_shape={c.shape if isinstance(c, torch.Tensor) else c}')

# BN 对比
print('\n=== BN1 samples (encoder.features.bn1) ===')
for k in ['encoder.features.bn1.running_mean', 'encoder.features.bn1.running_var',
          'encoder.features.bn1.weight', 'encoder.features.bn1.bias']:
    if k in gstate and k in cstates[0]:
        g_v = gstate[k]
        c_v = cstates[0][k]
        print(f'{k}: global={g_v[:3].tolist()} client0={c_v[:3].tolist()}')

# Try load both, compare
print('\n=== Try load global vs client0, verify by 1 inference ===')
import torch.nn.functional as F
import flgo
from torchvision import transforms
from PIL import Image

data_root = os.path.join(flgo.benchmark.data_root, 'PACS')
art_dir = os.path.join(data_root, 'Homework3-PACS-master', 'PACS', 'art_painting')

# Find one image from guitar class
cls_dir = os.path.join(art_dir, 'guitar')
img_path = os.path.join(cls_dir, sorted(os.listdir(cls_dir))[0])
print(f'test image: {img_path}')

tr = transforms.Compose([transforms.Resize([256, 256]), transforms.PILToTensor()])
img = Image.open(img_path)
if len(img.split()) != 3:
    img = transforms.Grayscale(num_output_channels=3)(img)
x = tr(img).unsqueeze(0).float() / 255.0  # [1, 3, 256, 256]

for name, state in [('global', gstate), ('client0', cstates[0])]:
    m = FedDSASGPAModel(num_classes=7, feat_dim=1024, proj_dim=128,
                       tau_etf=meta.get('tau_etf', 0.1),
                       use_etf=bool(meta['use_etf']),
                       ca=0, num_clients=4, backbone='alexnet')
    missing, unexpected = m.load_state_dict(state, strict=False)
    m.eval()
    with torch.no_grad():
        logits = m(x)
        probs = F.softmax(logits, dim=1)
        print(f'\n{name}: missing={len(missing)} unexpected={len(unexpected)}')
        print(f'  logits (guitar img, true=3): {logits[0].tolist()}')
        print(f'  probs:                      {probs[0].tolist()}')
        print(f'  argmax pred: {logits.argmax(1).item()}  (3=guitar)')
