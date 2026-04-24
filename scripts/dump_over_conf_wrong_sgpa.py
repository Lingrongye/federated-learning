# -*- coding: utf-8 -*-
"""EXP-125 OCSD Verification 1 — dump over-confident wrong samples on PACS Art.

Uses an existing sgpa (orth_uc1) checkpoint to run inference on PACS Art test set,
records per-sample (image_path, true_label, pred_label, confidence), filters
over-confident wrong samples, and produces a 50-image review package.

Why sgpa checkpoint is OK (not strict orth_only):
- The thing we diagnose is "which Art samples get over-confidently mis-classified
  by any competent model". This is a data property, stable across architectures
  (Stage B showed FedBN/orth/FDSE all have ~13-15% over_conf_wrong).
- Direction validity of OCSD depends on sample TYPE (style shortcut vs intrinsic
  hard), not specific model.

Usage:
    python dump_over_conf_wrong_sgpa.py \\
        --ckpt ~/fl_checkpoints/sgpa_PACS_c4_s2_R200_1776795770 \\
        --output_dir ./experiments/EXP-125_ocsd_verify/verify1/ \\
        --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add FDSE_CVPR25 root to path so algorithm.feddsa_sgpa imports correctly
FDSE_ROOT = Path(__file__).parent.parent / 'FDSE_CVPR25'
sys.path.insert(0, str(FDSE_ROOT))

from algorithm.feddsa_sgpa import FedDSASGPAModel  # noqa: E402


PACS_CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
PACS_DOMAINS = ('art_painting', 'cartoon', 'photo', 'sketch')


def load_ckpt(ckpt_dir: str, device: str):
    d = Path(ckpt_dir)
    meta = json.load(open(d / 'meta.json'))
    gstate = torch.load(d / 'global_model.pt', map_location=device, weights_only=False)
    cstates = torch.load(d / 'client_models.pt', map_location=device, weights_only=False)
    return meta, gstate, cstates


def build_pacs_art_dataset(transform, data_root: str):
    """Build PACS Art test split as (image, label, image_path)."""
    import torchvision
    from PIL import Image
    from torchvision import transforms

    # Replicate PACS_c4/config.py PACSDomainDataset but also expose image path
    class PACSArtDatasetWithPath(torchvision.datasets.VisionDataset):
        classes = PACS_CLASSES

        def __init__(self, root):
            super().__init__(root)
            self.domain_path = os.path.join(
                root, 'Homework3-PACS-master', 'PACS', 'art_painting'
            )
            self.images_path = []
            self.labels = []
            for i, c in enumerate(self.classes):
                cdir = os.path.join(self.domain_path, c)
                if not os.path.isdir(cdir):
                    continue
                for img in sorted(os.listdir(cdir)):
                    self.images_path.append(os.path.join(cdir, img))
                    self.labels.append(i)

        def __len__(self):
            return len(self.images_path)

        def __getitem__(self, item):
            img_path = self.images_path[item]
            label = self.labels[item]
            image = Image.open(img_path)
            if len(image.split()) != 3:
                image = transforms.Grayscale(num_output_channels=3)(image)
            image = transform(image)
            if image.dtype == torch.uint8 or image.dtype == torch.int8:
                image = image / 255.0
            return image, label, img_path

    return PACSArtDatasetWithPath(data_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint dir')
    ap.add_argument('--output_dir', required=True, help='dump output dir')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--use_client_bn', action='store_true', default=True,
                    help='use client 0 (Art) local BN (FedBN). Default True.')
    ap.add_argument('--data_root', default=None, help='flgo benchmark data root')
    args = ap.parse_args()

    # Setup
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f'[load] {args.ckpt}')
    meta, gstate, cstates = load_ckpt(args.ckpt, device='cpu')
    print(f'  task={meta["task"]}, seed={meta["seed"]}, '
          f'use_etf={meta["use_etf"]}, use_whitening={meta.get("use_whitening", 1)}, '
          f'use_centers={meta.get("use_centers", 1)}')

    # Build model
    model = FedDSASGPAModel(
        num_classes=7, feat_dim=1024, proj_dim=128,
        tau_etf=meta.get('tau_etf', 0.1), use_etf=bool(meta['use_etf']),
        ca=0, num_clients=4, backbone='alexnet',
    )

    # Load state: use client 0 (Art) BN for FedBN correctness
    if args.use_client_bn and cstates is not None and cstates[0] is not None:
        print('[load] client 0 (Art) local BN state')
        model.load_state_dict(cstates[0], strict=False)
    else:
        print('[load] global state (non-FedBN)')
        model.load_state_dict(gstate, strict=False)

    model.to(device).eval()

    # Data
    if args.data_root is None:
        import flgo
        args.data_root = os.path.join(flgo.benchmark.data_root, 'PACS')
    print(f'[data] PACS root: {args.data_root}')

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.PILToTensor(),
    ])
    ds = build_pacs_art_dataset(transform, args.data_root)
    print(f'[data] PACS Art test samples: {len(ds)}')

    # Inference loop
    results = []  # list of dicts
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    batch_size = 32
    idx_start = 0
    with torch.no_grad():
        while idx_start < len(ds):
            idx_end = min(idx_start + batch_size, len(ds))
            batch = [ds[i] for i in range(idx_start, idx_end)]
            imgs = torch.stack([b[0] for b in batch]).to(device)
            if imgs.dtype == torch.uint8 or imgs.dtype == torch.int8:
                imgs = imgs / 255.0
            labels = torch.tensor([b[1] for b in batch], device=device)
            paths = [b[2] for b in batch]
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for i in range(len(batch)):
                results.append({
                    'image_path': paths[i],
                    'true_label_idx': int(labels[i].item()),
                    'true_label': PACS_CLASSES[int(labels[i].item())],
                    'pred_label_idx': int(pred[i].item()),
                    'pred_label': PACS_CLASSES[int(pred[i].item())],
                    'confidence': float(conf[i].item()),
                    'prob_true': float(probs[i, labels[i]].item()),
                })
            idx_start = idx_end

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r['pred_label_idx'] == r['true_label_idx'])
    over_conf_wrong = [r for r in results
                       if r['pred_label_idx'] != r['true_label_idx']
                       and r['confidence'] > 0.8]
    over_conf_correct = [r for r in results
                         if r['pred_label_idx'] == r['true_label_idx']
                         and r['confidence'] > 0.8]

    print(f'\n[inference done]')
    print(f'  Total samples: {total}')
    print(f'  Correct: {correct}/{total} = {correct/total*100:.2f}%')
    print(f'  Over-conf (>0.8) correct: {len(over_conf_correct)} ({len(over_conf_correct)/total*100:.1f}%)')
    print(f'  Over-conf (>0.8) wrong: {len(over_conf_wrong)} ({len(over_conf_wrong)/total*100:.1f}%)')

    # Save full CSV
    import csv
    csv_path = out / 'all_predictions.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_path', 'true_label', 'pred_label', 'confidence',
            'prob_true', 'true_label_idx', 'pred_label_idx',
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f'[save] all_predictions.csv ({len(results)} rows)')

    # Sort over_conf_wrong by confidence desc
    over_conf_wrong.sort(key=lambda r: -r['confidence'])

    # Save over_conf_wrong CSV
    csv_wrong = out / 'over_conf_wrong.csv'
    with open(csv_wrong, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_path', 'true_label', 'pred_label', 'confidence',
            'prob_true', 'true_label_idx', 'pred_label_idx',
        ])
        writer.writeheader()
        writer.writerows(over_conf_wrong)

    # Copy top 50 images + generate PREVIEW + judgement sheet
    n_copy = min(50, len(over_conf_wrong))
    imgs_dir = out / 'top50_images'
    imgs_dir.mkdir(exist_ok=True)
    preview_lines = ['# Top 50 Over-Confident Wrong Samples — Art Domain\n',
                     f'\nTotal over_conf_wrong: **{len(over_conf_wrong)}** / {total} '
                     f'({len(over_conf_wrong)/total*100:.1f}%)\n',
                     f'Top 50 by confidence descending.\n\n',
                     '| # | 缩略图 | 真 label | 预测 | conf | 原图文件名 |\n',
                     '|---|---|---|---|:-:|---|\n']

    judge_lines = ['# Judgement Sheet — Over-Confident Wrong Samples\n\n',
                   '请对每张图打标 A/B/C:\n',
                   '- **A** = 风格极端 (painting/abstract 化, shortcut 嫌疑)\n',
                   '- **B** = 画得不像 (本质难样本, 标注可能歧义)\n',
                   '- **C** = 其他/标注错误\n\n',
                   '| # | 文件名 | 真 | 预测 | conf | 标签 (A/B/C) | 备注 |\n',
                   '|---|---|---|---|:-:|:-:|---|\n']

    for idx, r in enumerate(over_conf_wrong[:n_copy], start=1):
        orig_path = r['image_path']
        orig_name = os.path.basename(orig_path)
        new_name = f"{idx:03d}_{r['true_label']}_predAs_{r['pred_label']}_conf{r['confidence']:.2f}_{orig_name}"
        dst = imgs_dir / new_name
        try:
            shutil.copy2(orig_path, dst)
        except Exception as e:
            print(f'[warn] copy {orig_path} failed: {e}')
            continue
        preview_lines.append(
            f'| {idx} | ![](top50_images/{new_name}) | {r["true_label"]} | '
            f'{r["pred_label"]} | {r["confidence"]:.3f} | {orig_name} |\n'
        )
        judge_lines.append(
            f'| {idx} | {new_name} | {r["true_label"]} | {r["pred_label"]} '
            f'| {r["confidence"]:.3f} |   |   |\n'
        )

    (out / 'PREVIEW.md').write_text(''.join(preview_lines), encoding='utf-8')
    (out / 'judgement_sheet.md').write_text(''.join(judge_lines), encoding='utf-8')

    # Summary.md
    summary = [
        f'# EXP-125 OCSD Verification 1 — Summary\n\n',
        f'**Checkpoint**: `{args.ckpt}`\n',
        f'**Config**: seed={meta["seed"]}, use_etf={meta["use_etf"]}, '
        f'uw={meta.get("use_whitening", 1)}, uc={meta.get("use_centers", 1)}\n',
        f'**Client 0 (Art) BN**: {args.use_client_bn}\n\n',
        f'## Statistics (Art test, {total} samples)\n\n',
        f'- Overall accuracy: {correct}/{total} = **{correct/total*100:.2f}%**\n',
        f'- Over-conf (>0.8) correct: {len(over_conf_correct)} '
        f'({len(over_conf_correct)/total*100:.1f}%)\n',
        f'- Over-conf (>0.8) wrong: **{len(over_conf_wrong)}** '
        f'(**{len(over_conf_wrong)/total*100:.1f}%**) ← 这是 OCSD 要看的集合\n\n',
        f'## Predicted → True confusion (over_conf_wrong)\n\n',
    ]
    # Confusion matrix count
    from collections import Counter
    pair_cnt = Counter((r['true_label'], r['pred_label']) for r in over_conf_wrong)
    summary.append('| 真 label → 错 label | count |\n|---|:-:|\n')
    for (t, p), c in sorted(pair_cnt.items(), key=lambda x: -x[1]):
        summary.append(f'| {t} → **{p}** | {c} |\n')

    summary.append(f'\n## Next\n\n')
    summary.append(f'1. Open `top50_images/` and look at each image\n')
    summary.append(f'2. Fill out `judgement_sheet.md` with A/B/C label per sample\n')
    summary.append(f'3. Count A/B/C distribution:\n')
    summary.append(f'   - A 主导 (>60%) → shortcut 假设成立, 继续 OCSD\n')
    summary.append(f'   - A ≈ B (40-60 : 40-60) → 部分成立, OCSD 期望降低\n')
    summary.append(f'   - B/C 主导 → 放弃 OCSD, 转 RCA-GM\n')

    (out / 'SUMMARY.md').write_text(''.join(summary), encoding='utf-8')

    print(f'\n[done] Output dir: {out}')
    print(f'  top50_images/ - {n_copy} images')
    print(f'  PREVIEW.md - thumbnail grid')
    print(f'  judgement_sheet.md - A/B/C labeling template')
    print(f'  SUMMARY.md - stats + decision guide')
    print(f'  all_predictions.csv - {total} rows')
    print(f'  over_conf_wrong.csv - {len(over_conf_wrong)} rows')


if __name__ == '__main__':
    main()
