# -*- coding: utf-8 -*-
"""EXP-125 OCSD Verify 1 — sgpa_vib version (correct model class).

The checkpoints in ~/fl_checkpoints/sgpa_PACS_c4_* are actually sgpa_vib
(VIBSemanticHead with mu_head/log_var_head/prototype_ema), NOT plain sgpa.
Previous attempt loaded wrong model class → 18% acc (random).

Here we construct FedDSAVIBModel with vib=True to match checkpoint structure.
Inference in eval mode: z_sem = mu_head(h) (deterministic, no sampling).
"""
from __future__ import annotations

import argparse
import csv
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

from algorithm.feddsa_sgpa_vib import FedDSAVIBModel  # noqa: E402


PACS_CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')


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
            if image.dtype in (torch.uint8, torch.int8):
                image = image / 255.0
            return image, label, p

    return PACSArtWithPath(data_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--data_root', default=None)
    args = ap.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load ckpt
    ck = Path(args.ckpt)
    meta = json.load(open(ck / 'meta.json'))
    gstate = torch.load(ck / 'global_model.pt', map_location='cpu', weights_only=False)
    cstates = torch.load(ck / 'client_models.pt', map_location='cpu', weights_only=False)
    print(f'[load] {args.ckpt}')
    print(f'  meta: {meta}')

    # Build VIB model (vib=True to match checkpoint)
    model = FedDSAVIBModel(
        num_classes=7, feat_dim=1024, proj_dim=128,
        tau_etf=meta.get('tau_etf', 0.1),
        use_etf=bool(meta['use_etf']),
        ca=0, num_clients=4, backbone='alexnet',
        vib=True,  # <-- critical: use VIBSemanticHead
    )

    # Load client 0 state (Art, has local BN for FedBN + everything else)
    missing, unexpected = model.load_state_dict(cstates[0], strict=False)
    print(f'  missing keys: {len(missing)}')
    if missing:
        print(f'    {missing[:5]}')
    print(f'  unexpected keys: {len(unexpected)}')
    if unexpected:
        print(f'    {unexpected[:5]}')

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
    print(f'[data] Art test samples: {len(ds)}')

    # Inference
    # FedDSAVIBModel.forward in eval mode returns z_sem = mu (deterministic).
    # Then we need classify(z_sem) → logits.
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

            # Inference path matching train-time eval
            h = model.encode(imgs)
            # VIBSemanticHead.forward returns (z_sem, mu, log_var, kl) in training mode
            # But FedDSAVIBModel.get_semantic passes y=None, training={model.training}
            # In eval mode, training=False → z_sem = mu (deterministic)
            z_sem = model.get_semantic(h, y=None, training=False)
            if isinstance(z_sem, tuple):
                z_sem = z_sem[0]
            logits = model.classify(z_sem)
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
    print(f'  Over-conf WRONG:   {len(over_conf_wrong)} ({len(over_conf_wrong)/total*100:.1f}%) ← OCSD target')

    # Guard: if acc still low, model is broken
    if correct / total < 0.5:
        print(f'\n[ERROR] Accuracy {correct/total*100:.2f}% too low. Model load likely wrong.')
        print(f'Not generating images. Check missing/unexpected keys above.')
        return

    # Save
    with open(out / 'all_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    over_conf_wrong.sort(key=lambda r: -r['confidence'])
    with open(out / 'over_conf_wrong.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(over_conf_wrong)

    # Copy top 50 images + preview + judgement sheet
    n = min(50, len(over_conf_wrong))
    imgs_dir = out / 'top50_images'
    imgs_dir.mkdir(exist_ok=True)
    preview = [f'# Top {n} Over-Confident Wrong — PACS Art (sgpa_vib s={meta["seed"]})\n\n',
               f'**Total over_conf_wrong**: {len(over_conf_wrong)} / {total} '
               f'(**{len(over_conf_wrong)/total*100:.1f}%**)\n\n',
               f'Ordered by confidence descending (最自信的错在前).\n\n',
               '| # | 缩略图 | 真 label | **预测** | conf | orig filename |\n',
               '|---|---|---|---|:-:|---|\n']
    judge = [f'# Judgement Sheet — {n} over-conf wrong samples\n\n',
             '每张打标 A/B/C:\n',
             '- **A** = 风格极端 (油画化/抽象化, shortcut 嫌疑)\n',
             '- **B** = 画得不像 (本质难样本, 标注合理但画风不具备 class 特征)\n',
             '- **C** = 标注歧义/错误 (图和 label 不符)\n\n',
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
        preview.append(
            f'| {i} | ![](top50_images/{new}) | {r["true_label"]} | '
            f'**{r["pred_label"]}** | {r["confidence"]:.3f} | {orig} |\n'
        )
        judge.append(
            f'| {i} | {new} | {r["true_label"]} | {r["pred_label"]} '
            f'| {r["confidence"]:.3f} |   |   |\n'
        )

    (out / 'PREVIEW.md').write_text(''.join(preview), encoding='utf-8')
    (out / 'judgement_sheet.md').write_text(''.join(judge), encoding='utf-8')

    # Summary
    pair_cnt = Counter((r['true_label'], r['pred_label']) for r in over_conf_wrong)
    summary = [
        f'# EXP-125 OCSD Verify 1 — Summary\n\n',
        f'**Checkpoint**: `{args.ckpt}`\n',
        f'**Config**: seed={meta["seed"]}, VIB, use_whitening={meta["use_whitening"]}, '
        f'use_centers={meta["use_centers"]}\n',
        f'**Model**: FedDSAVIBModel (VIBSemanticHead, eval mode = mu deterministic)\n\n',
        f'## Statistics (PACS Art, {total} samples)\n\n',
        f'- Overall accuracy: {correct}/{total} = **{correct/total*100:.2f}%**\n',
        f'- Over-conf (>0.8) correct: {len(over_conf_correct)} '
        f'({len(over_conf_correct)/total*100:.1f}%)\n',
        f'- Over-conf (>0.8) wrong: **{len(over_conf_wrong)}** '
        f'(**{len(over_conf_wrong)/total*100:.1f}%**) ← OCSD target set\n\n',
        f'## Confusion (True → Predicted)\n\n',
        f'| True → Pred | Count |\n|---|:-:|\n',
    ]
    for (t, p), c in sorted(pair_cnt.items(), key=lambda x: -x[1]):
        summary.append(f'| {t} → **{p}** | {c} |\n')
    summary += [
        f'\n## Next — 人眼看 50 张图\n\n',
        f'1. Open `top50_images/` — 逐张看\n',
        f'2. Fill `judgement_sheet.md` — A/B/C 标注\n',
        f'3. Count 分布决定 OCSD 方向生死:\n',
        f'   - **A 主导 (>60%)** → style shortcut 假设成立, **继续 OCSD**\n',
        f'   - **A ≈ B (40-60 each)** → 部分成立, OCSD 期望降低\n',
        f'   - **B/C 主导** → shortcut 假设失败, **放弃 OCSD 切 RCA-GM**\n',
    ]
    (out / 'SUMMARY.md').write_text(''.join(summary), encoding='utf-8')

    print(f'\n[done] Output: {out}')
    print(f'  top50_images/ ({n} 张图)')
    print(f'  PREVIEW.md, judgement_sheet.md, SUMMARY.md')
    print(f'  all_predictions.csv ({total} 行), over_conf_wrong.csv ({len(over_conf_wrong)} 行)')


if __name__ == '__main__':
    main()
