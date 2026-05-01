"""
analyze_mask_offline.py
=======================
离线分析 EXP-137 PG-DFC vanilla 模型 checkpoint 的 mask 真实形态。

不重训, 直接从 npz dump 加载 state_dict, forward 真 PACS test image,
hook DFD 输出的 mask, 算 7 个 mask diagnostic stats。

用法:
    python analyze_mask_offline.py
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone.ResNet_DC import compute_mask_stats
from backbone.ResNet_DC_PG import resnet10_dc_pg


PACS_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
DATA_BASE = '/Users/changdao/联邦学习/F2DC/rundata/dataset/PACS_7'


def hook_mask_collector(model):
    """hook DFD/DFC_PG forward 收集 layer4 mask。"""
    storage = {'mask': None}

    def dfc_hook(mod, inputs, output):
        # DFC_PG.forward(self, nr_feat, mask, r_feat=None) — args[1] is mask
        storage['mask'] = inputs[1].detach()

    h = model.dfc_module.register_forward_hook(dfc_hook)
    return storage, h


def load_pacs_test_loader(domain, batch_size=64):
    """加载 PACS 单个 domain 全部 image (作 forward sample 集合)。"""
    base = os.path.join(DATA_BASE, domain)
    ds = ImageFolder(base, transform=PACS_TEST_TRANSFORM)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def analyze_one_checkpoint(npz_path, domain='sketch', max_batches=10):
    """加载一个 checkpoint, forward sketch test, 算 mask 7-stat."""
    print(f'\n=== {os.path.basename(npz_path)} | domain={domain} ===')
    d = np.load(npz_path, allow_pickle=True)
    sd_np = d['state_dict'].item()
    round_id = int(d['round'])
    acc = float(d['current_acc'])

    # numpy → torch tensor 转换 (npz dump 把 tensor 存成 numpy array)
    sd = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
          for k, v in sd_np.items()}

    # 构造 model (PACS 7-class)
    model = resnet10_dc_pg(num_classes=7)
    # 删 register_buffer non-persistent (class_proto) — 它不在 state_dict
    msg = model.load_state_dict(sd, strict=False)
    if msg.missing_keys:
        print(f'  missing_keys: {[k for k in msg.missing_keys if "class_proto" not in k]}')
    if msg.unexpected_keys:
        print(f'  unexpected: {msg.unexpected_keys}')
    model.eval()

    # 加载 PACS sketch test set
    loader = load_pacs_test_loader(domain)
    storage, h = hook_mask_collector(model)

    # forward + 收集 mask 7-stat
    all_stats = []
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            _ = model(x, is_eval=True)
            mask = storage['mask']
            assert mask is not None, 'hook 没抓到 mask'
            stats = compute_mask_stats(mask)
            all_stats.append(stats)
    h.remove()

    # 平均 batch 间
    keys = list(all_stats[0].keys())
    agg = {k: float(np.mean([s[k] for s in all_stats])) for k in keys}
    print(f'  Round {round_id}, current_acc={acc:.2f}, n_batches={len(all_stats)}')
    print(f'  mask4_mean       = {agg["mean"]:.4f}')
    print(f'  mask4_unit_std   = {agg["unit_std"]:.4f}      ★ 关键: <0.05=没切 / >0.4=真二值化')
    print(f'  mask4_hard_ratio = {agg["hard_ratio"]:.4f}    ★ 关键: <0.1=没二值化 / >0.7=真二值化')
    print(f'  mask4_mid_ratio  = {agg["mid_ratio"]:.4f}     ★ 关键: >0.7=卡 0.5')
    print(f'  mask4_sample_std  = {agg["sample_std"]:.5f}')
    print(f'  mask4_channel_std = {agg["channel_std"]:.5f}')
    print(f'  mask4_spatial_std = {agg["spatial_std"]:.5f}')

    # 真 selective 判决
    if agg['unit_std'] > 0.4 and agg['hard_ratio'] > 0.7:
        verdict = '✅✅ 真二值化'
    elif agg['unit_std'] > 0.2 and agg['mid_ratio'] < 0.3:
        verdict = '✅ 强 selective'
    elif agg['unit_std'] > 0.05:
        verdict = '✓ 中度 selective'
    elif agg['unit_std'] > 0.01:
        verdict = '⚠️ 弱 spread'
    else:
        verdict = '❌ 没切分(常数)'
    print(f'  → 判决: {verdict}')
    return agg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='EXP-137_pg_rerun_after_fix')
    parser.add_argument('--variant', default='diag_pgdfc_pacs_s333')
    parser.add_argument('--domains', nargs='+', default=['sketch', 'photo', 'art', 'cartoon'])
    parser.add_argument('--max_batches', type=int, default=5)
    args = parser.parse_args()

    base = f'/Users/changdao/联邦学习/experiments/ablation/{args.exp}/{args.variant}'

    # 选关键 checkpoint: best_R057 (sketch peak) / best_R082 (final peak) / final_R100
    chosen_npz = ['best_R035', 'best_R057', 'best_R082', 'final_R100']
    print(f'分析 {args.variant}, checkpoints: {chosen_npz}')
    print(f'测试 domain: {args.domains}')

    results = {}
    for ckpt in chosen_npz:
        npz = f'{base}/{ckpt}.npz'
        if not os.path.exists(npz):
            print(f'  ⚠️ {npz} 不存在, 跳过')
            continue
        for dom in args.domains:
            try:
                key = f'{ckpt}_{dom}'
                results[key] = analyze_one_checkpoint(npz, domain=dom,
                                                     max_batches=args.max_batches)
            except Exception as e:
                print(f'  ❌ {ckpt}/{dom} failed: {e}')

    # 汇总表
    print('\n' + '=' * 80)
    print('汇总表 (按 checkpoint × domain)')
    print('=' * 80)
    print(f'{"ckpt_domain":35} {"mean":>8} {"unit_std":>10} {"hard%":>8} {"mid%":>8} {"channel_std":>13}')
    for k, v in results.items():
        print(f'{k:35} {v["mean"]:>8.3f} {v["unit_std"]:>10.4f} '
              f'{v["hard_ratio"]*100:>7.1f}% {v["mid_ratio"]*100:>7.1f}% '
              f'{v["channel_std"]:>13.5f}')
