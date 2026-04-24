# -*- coding: utf-8 -*-
"""EXP-125 OCSD Verification 1 — dump over-confident wrong samples (orth_only version).

Uses a feddsa_scheduled (orth_only sm=0) checkpoint to run inference on PACS Art
test set. Simpler than sgpa version: FedDSAModel does pure forward
(encoder → semantic_head → head) without whitening/centers, so plain load + forward works.

Usage:
    python dump_over_conf_wrong_orth.py \\
        --save_dir /path/to/flgo_save_root \\
        --output_dir ./experiments/EXP-125_ocsd_verify/verify1_orth/ \\
        --seed 2 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

FDSE_ROOT = Path(__file__).parent.parent / 'FDSE_CVPR25'
sys.path.insert(0, str(FDSE_ROOT))

from algorithm.feddsa_scheduled import FedDSAModel  # noqa: E402


PACS_CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')


def find_saved_model(flgo_save_root: str, seed: int):
    """Find the .pth file saved by flgo's save_optimal mechanism.

    flgo saves to task/{task}/model/ or similar. Search recursively.
    """
    candidates = []
    for root, _, files in os.walk(flgo_save_root):
        for fn in files:
            if fn.endswith(('.pth', '.pt')) and f'_S{seed}_' in fn:
                fp = os.path.join(root, fn)
                candidates.append((os.path.getmtime(fp), fp))
    if not candidates:
        raise FileNotFoundError(f'No saved .pth/.pt found in {flgo_save_root} for seed={seed}')
    # Most recent
    candidates.sort(reverse=True)
    return candidates[0][1]


def build_pacs_art_dataset(transform, data_root: str):
    import torchvision
    from PIL import Image
    from torchvision import transforms

    class PACSArtWithPath(torchvision.datasets.VisionDataset):
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
            p = self.images_path[item]
            label = self.labels[item]
            image = Image.open(p)
            if len(image.split()) != 3:
                image = transforms.Grayscale(num_output_channels=3)(image)
            image = transform(image)
            if image.dtype == torch.uint8 or image.dtype == torch.int8:
                image = image / 255.0
            return image, label, p

    return PACSArtWithPath(data_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True,
                    help='path to saved model state dict (.pth)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--data_root', default=None)
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f'[load] {args.model_path}')
    state = torch.load(args.model_path, map_location='cpu', weights_only=False)

    # flgo saves either (model_state_dict, ...) or just state_dict, or wrapped
    if isinstance(state, dict) and 'model' in state and not isinstance(state['model'], dict):
        state = state['model'].state_dict() if hasattr(state['model'], 'state_dict') else state['model']
    elif isinstance(state, dict) and any(k in state for k in ('model', 'local_models')):
        # DALogger format: {'model': state_dict, 'local_models': [...]}
        if isinstance(state['model'], dict):
            # pick Client 0 (Art) for FedBN-style if available
            if 'local_models' in state and state['local_models'][0]:
                print('[load] using Client 0 (Art) local model (FedBN-style)')
                state = state['local_models'][0]
            else:
                state = state['model']
        else:
            state = state['model'].state_dict()
    # else assume state is plain state_dict

    model = FedDSAModel(num_classes=7, feat_dim=1024, proj_dim=128)

    # flgo's DecoratedModel wraps the real model as self.model — check keys
    sample_keys = list(state.keys())[:3] if isinstance(state, dict) else []
    if sample_keys and sample_keys[0].startswith('model.'):
        # strip 'model.' prefix
        state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
        print('[load] stripped "model." prefix from keys')

    # Load
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[load] missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'[load] unexpected keys ({len(unexpected)}): {unexpected[:5]}...')

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
    print(f'[data] PACS Art test: {len(ds)} samples')

    # Inference
    results = []
    bs = 32
    idx = 0
    with torch.no_grad():
        while idx < len(ds):
            end = min(idx + bs, len(ds))
            batch = [ds[i] for i in range(idx, end)]
            imgs = torch.stack([b[0] for b in batch]).to(device)
            if imgs.dtype in (torch.uint8, torch.int8):
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
            idx = end

    total = len(results)
    correct = sum(1 for r in results if r['pred_label_idx'] == r['true_label_idx'])
    over_conf_wrong = [r for r in results
                       if r['pred_label_idx'] != r['true_label_idx']
                       and r['confidence'] > 0.8]
    over_conf_correct = [r for r in results
                         if r['pred_label_idx'] == r['true_label_idx']
                         and r['confidence'] > 0.8]

    print(f'\n[inference done]')
    print(f'  Total: {total}, Acc: {correct/total*100:.2f}%')
    print(f'  Over-conf correct: {len(over_conf_correct)} ({len(over_conf_correct)/total*100:.1f}%)')
    print(f'  Over-conf wrong:   {len(over_conf_wrong)} ({len(over_conf_wrong)/total*100:.1f}%) ← OCSD target')

    # CSV
    import csv
    with open(out / 'all_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    over_conf_wrong.sort(key=lambda r: -r['confidence'])
    with open(out / 'over_conf_wrong.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(over_conf_wrong)

    # Copy top 50 images
    n = min(50, len(over_conf_wrong))
    imgs_dir = out / 'top50_images'
    imgs_dir.mkdir(exist_ok=True)
    preview = ['# Top 50 Over-Confident Wrong — PACS Art (orth_only)\n\n',
               f'Total over_conf_wrong: **{len(over_conf_wrong)}** / {total} '
               f'({len(over_conf_wrong)/total*100:.1f}%)\n\n',
               '| # | 缩略图 | 真 | 预测 | conf | orig |\n',
               '|---|---|---|---|:-:|---|\n']
    judge = ['# Judgement Sheet\n\nA=风格极端(shortcut) / B=画得不像(本质难) / C=标注歧义\n\n',
             '| # | 文件 | 真 | 预测 | conf | A/B/C | 备注 |\n',
             '|---|---|---|---|:-:|:-:|---|\n']
    for i, r in enumerate(over_conf_wrong[:n], 1):
        orig = os.path.basename(r['image_path'])
        new = f"{i:03d}_{r['true_label']}_asPred_{r['pred_label']}_conf{r['confidence']:.2f}_{orig}"
        try:
            shutil.copy2(r['image_path'], imgs_dir / new)
        except Exception as e:
            print(f'[warn] copy {orig} failed: {e}')
            continue
        preview.append(f'| {i} | ![](top50_images/{new}) | {r["true_label"]} | '
                       f'**{r["pred_label"]}** | {r["confidence"]:.3f} | {orig} |\n')
        judge.append(f'| {i} | {new} | {r["true_label"]} | {r["pred_label"]} '
                     f'| {r["confidence"]:.3f} |   |   |\n')

    (out / 'PREVIEW.md').write_text(''.join(preview), encoding='utf-8')
    (out / 'judgement_sheet.md').write_text(''.join(judge), encoding='utf-8')

    # Summary
    from collections import Counter
    pair_cnt = Counter((r['true_label'], r['pred_label']) for r in over_conf_wrong)
    summary = [
        f'# EXP-125 OCSD Verify 1 — Summary (orth_only)\n\n',
        f'**Model**: `{args.model_path}`\n',
        f'**Overall acc**: {correct}/{total} = **{correct/total*100:.2f}%**\n',
        f'**Over-conf (>0.8) wrong**: **{len(over_conf_wrong)}** '
        f'(**{len(over_conf_wrong)/total*100:.1f}%**)\n\n',
        f'## Confusion (over_conf_wrong)\n\n| True → Pred | Count |\n|---|:-:|\n',
    ]
    for (t, p), c in sorted(pair_cnt.items(), key=lambda x: -x[1]):
        summary.append(f'| {t} → **{p}** | {c} |\n')
    summary += [
        f'\n## Next\n\n',
        f'1. Open `top50_images/` — 人眼看 50 张\n',
        f'2. Fill `judgement_sheet.md` — A/B/C\n',
        f'3. Count 分布决定 OCSD 方向:\n',
        f'   - A 主导 (>60%) → shortcut 假设成立, 继续\n',
        f'   - A ≈ B → 部分成立\n',
        f'   - B/C 主导 → 放弃 OCSD, 切 RCA-GM\n',
    ]
    (out / 'SUMMARY.md').write_text(''.join(summary), encoding='utf-8')

    print(f'[done] {out}')
    print(f'  top50_images/ {n} imgs')
    print(f'  PREVIEW.md / judgement_sheet.md / SUMMARY.md')


if __name__ == '__main__':
    main()
