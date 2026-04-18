"""
加载 FL checkpoint 生成 per-domain 误分类可视化

用法（在服务器上）：
  python eval_misclass_fl.py --task PACS_c4 --ckpt ~/fl_checkpoints/feddsa_s2_R200_best175_1776465533 --out ~/misclass_out/pacs_sas_s2
  python eval_misclass_fl.py --task office_caltech10_c4 --ckpt ~/fl_checkpoints/feddsa_s333_R200_best149_1776438612 --out ~/misclass_out/office_sas_s333

每个 client 用自己 personalized checkpoint 测对应 domain test set，
每个 domain 保存 top-16 "自信但错" 样本 + HTML 页面。
"""
import os, sys, argparse, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms as T

from algorithm.feddsa_scheduled import FedDSAModel


PACS_DOMAINS = ('art_painting', 'cartoon', 'photo', 'sketch')
OFFICE_DOMAINS = ('Caltech', 'amazon', 'dslr', 'webcam')


def load_task(task_name):
    """返回 (domains, classes, num_classes, datasets) — 与训练时 transform 一致。"""
    import flgo
    tf = T.Compose([T.Resize([256, 256]), T.PILToTensor()])
    if 'PACS' in task_name:
        sys.path.insert(0, os.path.join(ROOT, 'task', 'PACS_c4'))
        from config import PACSDomainDataset
        domains = PACS_DOMAINS
        classes = PACSDomainDataset.classes
        num_classes = len(classes)
        raw_root = os.path.join(flgo.benchmark.data_root, 'PACS')
        datasets = [PACSDomainDataset(raw_root, domain=d, transform=tf) for d in domains]
        return domains, classes, num_classes, datasets
    elif 'office' in task_name.lower():
        from benchmark.office_caltech10_classification.config import OCDomainDataset
        domains = OFFICE_DOMAINS
        classes = OCDomainDataset.classes
        num_classes = len(classes)
        raw_root = os.path.join(flgo.benchmark.data_root, 'office_caltech10')
        datasets = [OCDomainDataset(raw_root, domain=d, transform=tf) for d in domains]
        return domains, classes, num_classes, datasets
    else:
        raise ValueError(task_name)


def split_test_by_flgo(full_ds, holdout_ratio, seed):
    """复现 flgo 的 per-client train/test split。
    flgo 对每个 client 的数据按 holdout_ratio 切出 test；我们只取 test 部分。
    用固定种子保证复现。
    """
    import random
    n = len(full_ds)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(n * (1 - holdout_ratio))
    test_idx = indices[split:]
    return torch.utils.data.Subset(full_ds, test_idx)


def eval_client(model, loader, classes, device):
    model.eval()
    errors = []
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x_gpu = x.to(device)
            logits = model(x_gpu)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(1).cpu()
            conf = probs.max(1).values.cpu()
            for i in range(y.size(0)):
                total += 1
                if y[i].item() == pred[i].item():
                    correct += 1
                else:
                    errors.append({
                        'true': classes[y[i].item()],
                        'pred': classes[pred[i].item()],
                        'conf': conf[i].item(),
                        'img': x[i].clone(),
                    })
    acc = correct / total if total else 0.0
    return acc, errors, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['PACS_c4', 'office_caltech10_c4'])
    ap.add_argument('--ckpt', required=True, help='~/fl_checkpoints/xxx 目录')
    ap.add_argument('--out', required=True, help='输出目录')
    ap.add_argument('--topk', type=int, default=16)
    ap.add_argument('--holdout', type=float, default=0.2)
    ap.add_argument('--split_seed', type=int, default=0,
                    help='flgo 默认 train/test split 用种子 0')
    args = ap.parse_args()

    ckpt_dir = os.path.expanduser(args.ckpt)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(ckpt_dir, 'meta.json')))
    print(f'[Meta] seed={meta["seed"]} best@R{meta["best_round"]} avg={meta["best_avg_acc"]:.4f}')

    domains, classes, num_classes, full_datasets = load_task(args.task)
    print(f'[Task] {args.task} | domains={domains} | classes={num_classes}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    html = [
        '<html><head><meta charset="utf-8"><style>',
        'body{font-family:Arial;padding:20px}',
        'table{border-collapse:collapse;margin-bottom:20px}',
        'td{border:1px solid #ccc;padding:6px;vertical-align:top}',
        'h2{background:#eef;padding:8px}',
        '</style></head><body>',
        f'<h1>{args.task} 误分类可视化</h1>',
        f'<p><b>Checkpoint</b>: {ckpt_dir}<br>',
        f'<b>Meta</b>: seed={meta["seed"]}, best@R{meta["best_round"]}, best_avg_acc={meta["best_avg_acc"]:.4f}</p>',
        f'<p>每个 domain 展示 top-{args.topk} "自信但错"样本（softmax 置信度最高的错分样本）。</p>',
    ]

    stats = {}
    for cid, (dom, full_ds) in enumerate(zip(domains, full_datasets)):
        test_ds = split_test_by_flgo(full_ds, args.holdout, args.split_seed)
        loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

        ckpt_path = os.path.join(ckpt_dir, f'client_{cid}.pt')
        if not os.path.exists(ckpt_path):
            print(f'[Skip] client_{cid}.pt not found')
            continue

        model = FedDSAModel(num_classes=num_classes).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)

        acc, errors, total = eval_client(model, loader, classes, device)
        stats[dom] = {'acc': acc, 'errors': len(errors), 'total': total,
                      'client_id': cid}
        print(f'[{dom}] acc={acc:.4f} ({len(errors)}/{total} errors)')

        errors.sort(key=lambda e: -e['conf'])
        top = errors[:args.topk]
        dom_dir = os.path.join(out_dir, dom)
        os.makedirs(dom_dir, exist_ok=True)

        html.append(f'<h2>Client{cid} = {dom} | acc={acc:.2%} | '
                    f'errors={len(errors)}/{total}</h2>')
        html.append('<table><tr>')
        for i, e in enumerate(top):
            fn = f'{i:02d}_{e["true"]}_as_{e["pred"]}_conf{e["conf"]:.2f}.jpg'
            save_image(e['img'], os.path.join(dom_dir, fn), normalize=False)
            if i > 0 and i % 4 == 0:
                html.append('</tr><tr>')
            html.append(
                f'<td><img src="{dom}/{fn}" width=180><br>'
                f'<b>{e["true"]} &rarr; {e["pred"]}</b><br>'
                f'conf={e["conf"]:.2f}</td>')
        html.append('</tr></table>')

    html.append('</body></html>')

    with open(os.path.join(out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump({'meta': meta, 'task': args.task, 'stats': stats}, f, indent=2)

    print(f'\nDone. View: {out_dir}/index.html')
    for d, s in stats.items():
        print(f'  {d:20s}: acc={s["acc"]:.4f} ({s["errors"]}/{s["total"]})')


if __name__ == '__main__':
    main()
