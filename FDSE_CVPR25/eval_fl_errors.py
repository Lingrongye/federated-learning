"""加载 FL 训练保存的 best checkpoint，对每 domain test 做推理，收集误分类样本。

用法:
  python eval_fl_errors.py --ckpt_dir ~/fl_checkpoints/<tag> [--task PACS_c4]

输出:
  ~/misclass_fl/<tag>/{domain}/<true>_as_<pred>_conf<c>.jpg
  ~/misclass_fl/<tag>/index.html
  ~/misclass_fl/<tag>/per_domain_stats.json
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from torchvision import transforms
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--task', default='PACS_c4')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--top_k', type=int, default=16, help='top K most confident-but-wrong per domain')
    args = ap.parse_args()

    ckpt_dir = os.path.expanduser(args.ckpt_dir)
    assert os.path.exists(ckpt_dir), f'ckpt_dir not found: {ckpt_dir}'
    with open(os.path.join(ckpt_dir, 'meta.json')) as f:
        meta = json.load(f)
    print(f'Loaded meta: best@R{meta["best_round"]} avg={meta["best_avg_acc"]:.4f}')

    # 导入 task config
    if args.task == 'PACS_c4':
        from task.PACS_c4 import config as cfg
        CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
        DOMAINS = ('art_painting', 'cartoon', 'photo', 'sketch')
        ROOT = '/root/miniconda3/lib/python3.10/site-packages/flgo/benchmark/RAW_DATA/PACS'
        DatasetCls = cfg.PACSDomainDataset
    else:
        raise ValueError(f'task {args.task} not supported')

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 构建 per-domain test set (与 FL 训练相同的 seed split，保证 test set 一致)
    # NOTE: 这里假设 FL 训练用默认 seed 42 做 train/test split
    # 如果不一致会误差但大致趋势正确
    random.seed(42)
    test_by_domain = {}
    for d in DOMAINS:
        ds = DatasetCls(ROOT, domain=d, transform=tf)
        n = len(ds)
        idx = list(range(n)); random.shuffle(idx)
        test_idx = idx[int(n*0.8):]
        test_by_domain[d] = Subset(ds, test_idx)
        print(f'  {d}: test samples = {len(test_idx)}')

    # 加载每个 client 的模型（对应 domain）
    # FL task PACS_c4 client 顺序 = DOMAINS 顺序
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_base = os.path.expanduser(f'~/misclass_fl/{os.path.basename(ckpt_dir)}')
    os.makedirs(out_base, exist_ok=True)

    html = [f'<html><head><style>body{{font-family:Arial}}table{{border-collapse:collapse}}td{{border:1px solid #ccc;padding:4px;text-align:center}}</style></head><body>']
    html.append(f'<h1>FL Model Misclassifications — {os.path.basename(ckpt_dir)}</h1>')
    html.append(f'<p>Best @ R{meta["best_round"]}, AVG={meta["best_avg_acc"]:.2f}%</p>')
    html.append(f'<p>每个域展示 top-{args.top_k} 最"自信但错"的样本（模型预测错但置信度高）</p>')

    stats = {}
    for cid, d in enumerate(DOMAINS):
        ckpt_path = os.path.join(ckpt_dir, f'client_{cid}.pt')
        if not os.path.exists(ckpt_path):
            print(f'{d}: client_{cid}.pt not found, skip')
            continue
        net = cfg.get_model().to(device)
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state, strict=False)
        net.eval()

        loader = DataLoader(test_by_domain[d], batch_size=args.batch, shuffle=False, num_workers=2)
        errors = []
        total = correct = 0
        with torch.no_grad():
            for x, y in loader:
                x_gpu = x.to(device)
                logits = net(x_gpu)
                probs = torch.softmax(logits, 1)
                pred = logits.argmax(1).cpu()
                conf = probs.max(1).values.cpu()
                for i in range(y.size(0)):
                    total += 1
                    if y[i].item() == pred[i].item():
                        correct += 1
                    else:
                        errors.append({
                            'true': CLASSES[y[i].item()],
                            'pred': CLASSES[pred[i].item()],
                            'conf': conf[i].item(),
                            'img': x[i].clone(),
                        })
        acc = correct / total if total else 0
        stats[d] = {'acc': acc, 'errors': len(errors), 'total': total}
        print(f'  {d}: acc={acc:.4f} ({len(errors)}/{total})')

        errors.sort(key=lambda e: -e['conf'])
        top = errors[:args.top_k]
        domain_dir = os.path.join(out_base, d)
        os.makedirs(domain_dir, exist_ok=True)
        html.append(f'<h2>{d}: acc={acc:.2%} ({len(errors)}/{total} errors)</h2><table><tr>')
        for i, e in enumerate(top):
            fn = f"{e['true']}_as_{e['pred']}_conf{e['conf']:.2f}_{i:02d}.jpg"
            save_image(e['img'], os.path.join(domain_dir, fn))
            if i > 0 and i % 4 == 0:
                html.append('</tr><tr>')
            html.append(f'<td><img src="{d}/{fn}" width=180><br><b>{e["true"]} → {e["pred"]}</b><br>conf={e["conf"]:.2f}</td>')
        html.append('</tr></table>')

    html.append('</body></html>')
    with open(os.path.join(out_base, 'index.html'), 'w') as f:
        f.write('\n'.join(html))
    with open(os.path.join(out_base, 'per_domain_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nDone. View: {out_base}/index.html')

if __name__ == '__main__':
    main()
