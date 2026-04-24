# -*- coding: utf-8 -*-
"""EXP-125 OCSD Verify 1 v2 — 只测 Art client 的 test split (不含 train data).

Fix: 之前版本测了 full 2048 张 Art 图 (混测 train+test), acc 92% 虚高.
这版用 flgo 正常 pipeline 拿 `clients[0].test_data` = 和训练时一致的 test split.

Uses feddsa_sgpa_vib checkpoint (VIB architecture, EXP-113 variants).
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

FDSE_ROOT = Path(__file__).parent.parent / 'FDSE_CVPR25'
sys.path.insert(0, str(FDSE_ROOT))


PACS_CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='SGPA checkpoint dir')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--config', default=None,
                    help='config yml (if absent, use feddsa_sgpa_vib default)')
    ap.add_argument('--algorithm', default='feddsa_sgpa_vib')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    ck = Path(args.ckpt)
    meta = json.load(open(ck / 'meta.json'))
    gstate = torch.load(ck / 'global_model.pt', map_location='cpu', weights_only=False)
    cstates = torch.load(ck / 'client_models.pt', map_location='cpu', weights_only=False)

    print(f'[load] {args.ckpt}')
    print(f'  meta: {meta}')

    # flgo init setup (from run_sgpa_inference.py)
    import flgo
    import flgo.simulator

    os.chdir(FDSE_ROOT)
    algo_mod = importlib.import_module(f'algorithm.{args.algorithm}')

    # Build option
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            option = yaml.safe_load(f)
        option['num_rounds'] = 0
    else:
        # Default: feddsa_sgpa_vib 18 algo_para
        option = {
            'num_rounds': 0, 'num_epochs': 0,
            'batch_size': 50, 'learning_rate': 0.05,
            'proportion': 1.0, 'train_holdout': 0.2,
            'local_test': True, 'no_log_console': True, 'log_file': False,
            'algo_para': [
                1.0, 0.1, 128, 10, 1e-3, 2, 0,
                int(meta['use_etf']),
                int(meta.get('use_whitening', 1)),
                int(meta.get('use_centers', 1)),
                0,      # se
                0.0,    # lp
                0,      # ca
                1,      # vib (critical — match VIB checkpoint)
                0,      # us
                1.0,    # lib
                1.0,    # lsc
                20,     # vws
                50,     # vwe
                0.07,   # sct
            ],
        }
    option['seed'] = int(meta['seed'])
    option['dataseed'] = int(meta['seed'])
    option['gpu'] = args.gpu

    from flgo.experiment.logger import BasicLogger
    try:
        from logger import PerRunLogger
    except Exception:
        PerRunLogger = BasicLogger

    print(f'[flgo] initializing...')
    runner = flgo.init(
        os.path.join('task', meta['task']), algo_mod, option,
        Logger=PerRunLogger, Simulator=flgo.simulator.DefaultSimulator,
    )
    server = runner
    print(f'[flgo] server + {len(server.clients)} clients initialized')

    # Load checkpoint into server + client 0
    print(f'[ckpt] loading global_model.pt into server...')
    missing, unexpected = server.model.load_state_dict(gstate, strict=False)
    print(f'  server load: missing={len(missing)} unexpected={len(unexpected)}')

    # Client 0 (Art): set up local model with client 0 state (FedBN BN)
    client0 = server.clients[0]
    if client0.model is None:
        client0.model = copy.deepcopy(server.model)
    missing, unexpected = client0.model.load_state_dict(cstates[0], strict=False)
    print(f'  client 0 load: missing={len(missing)} unexpected={len(unexpected)}')

    # Inference on client 0 test_data
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    model = client0.model.to(device)
    model.eval()

    test_data = client0.test_data
    if test_data is None or len(test_data) == 0:
        print(f'[error] Client 0 test_data is None/empty!')
        return
    print(f'[data] Client 0 (Art) test split: {len(test_data)} samples')

    # To get image paths, we need the underlying PACSDomainDataset
    # client0.test_data is torch.utils.data.Subset — unwrap it
    paths_list = []
    labels_list = []
    imgs_list = []
    for i in range(len(test_data)):
        img, lab = test_data[i]
        # Try to fetch path from underlying dataset
        try:
            # client.test_data is typically Subset(dataset, indices)
            if hasattr(test_data, 'indices'):
                # Find the nested PACSDomainDataset
                ds_inner = test_data
                idx_path = i
                while hasattr(ds_inner, 'dataset'):
                    idx_path = ds_inner.indices[idx_path]
                    ds_inner = ds_inner.dataset
                # ds_inner should have images_path attr
                if hasattr(ds_inner, 'images_path'):
                    paths_list.append(ds_inner.images_path[idx_path])
                else:
                    paths_list.append(f'unknown_idx_{i}')
            else:
                paths_list.append(f'unknown_idx_{i}')
        except Exception as e:
            paths_list.append(f'err_{i}')
        labels_list.append(int(lab))
        imgs_list.append(img)

    # Batch forward
    results = []
    bs = 32
    with torch.no_grad():
        for start in range(0, len(imgs_list), bs):
            end = min(start + bs, len(imgs_list))
            batch = imgs_list[start:end]
            imgs = torch.stack(batch).to(device)
            if imgs.dtype in (torch.uint8, torch.int8):
                imgs = imgs / 255.0
            labels = torch.tensor(labels_list[start:end], device=device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for i in range(len(batch)):
                results.append({
                    'image_path': paths_list[start + i],
                    'true_label_idx': labels_list[start + i],
                    'true_label': PACS_CLASSES[labels_list[start + i]],
                    'pred_label_idx': int(pred[i].item()),
                    'pred_label': PACS_CLASSES[int(pred[i].item())],
                    'confidence': float(conf[i].item()),
                    'prob_true': float(probs[i, labels[i]].item()),
                })

    total = len(results)
    correct = sum(1 for r in results if r['pred_label_idx'] == r['true_label_idx'])
    ocw = [r for r in results
           if r['pred_label_idx'] != r['true_label_idx']
           and r['confidence'] > 0.8]
    occ = [r for r in results
           if r['pred_label_idx'] == r['true_label_idx']
           and r['confidence'] > 0.8]

    print(f'\n[inference done on Client 0 TEST ONLY]')
    print(f'  Total: {total}')
    print(f'  Acc: {correct}/{total} = {correct/total*100:.2f}%')
    print(f'  Over-conf correct: {len(occ)} ({len(occ)/total*100:.1f}%)')
    print(f'  Over-conf WRONG:   {len(ocw)} ({len(ocw)/total*100:.1f}%) ← OCSD target')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / 'all_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    ocw.sort(key=lambda r: -r['confidence'])
    with open(out / 'over_conf_wrong.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(ocw)

    # Copy top 50
    n = min(50, len(ocw))
    imgs_dir = out / 'top50_images'
    imgs_dir.mkdir(exist_ok=True)
    preview = [f'# Top {n} Over-Confident Wrong (Art test only, sgpa_vib s={meta["seed"]})\n\n',
               f'**Test split size**: {total}\n',
               f'**Accuracy**: {correct/total*100:.2f}%\n',
               f'**Over-conf wrong**: {len(ocw)} ({len(ocw)/total*100:.1f}%)\n\n',
               '| # | 缩略图 | 真 | **预测** | conf | orig |\n',
               '|---|---|---|---|:-:|---|\n']
    judge = [f'# Judgement Sheet ({n} over-conf wrong)\n\nA=风格极端(shortcut) / B=画得不像(本质难) / C=标注歧义\n\n',
             '| # | 文件 | 真 | 预测 | conf | A/B/C | 备注 |\n',
             '|---|---|---|---|:-:|:-:|---|\n']

    for i, r in enumerate(ocw[:n], 1):
        orig = os.path.basename(r['image_path']) if '/' in r['image_path'] else r['image_path']
        new = f"{i:03d}_{r['true_label']}_asPred_{r['pred_label']}_conf{r['confidence']:.2f}_{orig}"
        if os.path.exists(r['image_path']):
            try:
                shutil.copy2(r['image_path'], imgs_dir / new)
            except Exception as e:
                print(f'[warn] copy failed: {e}')
                continue
            preview.append(f'| {i} | ![](top50_images/{new}) | {r["true_label"]} | '
                           f'**{r["pred_label"]}** | {r["confidence"]:.3f} | {orig} |\n')
            judge.append(f'| {i} | {new} | {r["true_label"]} | {r["pred_label"]} '
                         f'| {r["confidence"]:.3f} |   |   |\n')

    (out / 'PREVIEW.md').write_text(''.join(preview), encoding='utf-8')
    (out / 'judgement_sheet.md').write_text(''.join(judge), encoding='utf-8')

    # Summary
    pair = Counter((r['true_label'], r['pred_label']) for r in ocw)
    sm = [
        f'# EXP-125 OCSD Verify 1 v2 (test only)\n\n',
        f'**Checkpoint**: {args.ckpt}\n',
        f'**Client 0 test split**: {total} samples (consistent with training split)\n\n',
        f'## Stats\n\n',
        f'- Acc: {correct}/{total} = **{correct/total*100:.2f}%**\n',
        f'- Over-conf correct: {len(occ)} ({len(occ)/total*100:.1f}%)\n',
        f'- **Over-conf WRONG**: **{len(ocw)}** ({len(ocw)/total*100:.1f}%)\n\n',
        f'## Confusion\n\n| True → Pred | Count |\n|---|:-:|\n',
    ]
    for (t, p), c in sorted(pair.items(), key=lambda x: -x[1]):
        sm.append(f'| {t} → **{p}** | {c} |\n')
    sm += [
        f'\n## Decision\n\n',
        f'人眼看 top 50 → 统计 A/B/C:\n',
        f'- A 主导 (>60%) → shortcut 假设成立, 继续 OCSD\n',
        f'- A ≈ B → 部分成立\n',
        f'- B/C 主导 → 放弃 OCSD, 切 RCA-GM\n',
    ]
    (out / 'SUMMARY.md').write_text(''.join(sm), encoding='utf-8')

    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
