"""
加载 FL checkpoint 生成 per-domain 误分类可视化（使用 flgo 真实 test set）

用法（在服务器上）：
  python eval_misclass_fl.py --task PACS_c4 --ckpt ~/fl_checkpoints/feddsa_s2_R200_best175_1776465533 --seed 2 --out ~/misclass_out/pacs_sas_s2
  python eval_misclass_fl.py --task office_caltech10_c4 --ckpt ~/fl_checkpoints/feddsa_s333_R200_best149_1776438612 --seed 333 --out ~/misclass_out/office_sas_s333

每个 client 用自己 personalized checkpoint 测对应 domain test set。
test set 通过 flgo.init() 复现训练时的真实划分 — seed + train_holdout 决定 split。
每个 domain 保存 top-K "自信但错" 样本 + HTML 页面。
"""
import os, sys, argparse, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# 切到 FDSE_CVPR25 目录，让 flgo.init('task/xxx') 的相对路径生效
os.chdir(ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

PACS_DOMAINS = ('art_painting', 'cartoon', 'photo', 'sketch')
OFFICE_DOMAINS = ('Caltech', 'amazon', 'dslr', 'webcam')


def init_runner(task_name, seed, num_rounds=200):
    """用 flgo.init 构造 runner，拿到真实的 train/test split。不训练。"""
    import flgo
    from algorithm import feddsa_scheduled
    option = {
        'gpu': [0],
        'seed': seed,
        'dataseed': seed,
        'num_rounds': num_rounds,
        'num_epochs': 1,
        'batch_size': 50,
        'learning_rate': 0.05,
        'weight_decay': 1e-3,
        'lr_scheduler': 0,
        'learning_rate_decay': 0.9998,
        'proportion': 1.0,
        'train_holdout': 0.2,
        'local_test': True,
        'no_log_console': True,
        'log_file': True,
        # FedDSA 特有参数，跟 EXP-084/086 对齐（sas=1, sas_tau=0.3）
        'algo_para': [1.0, 0.0, 1.0, 0.2, 5, 128, 0, 60, 30, 80, 10, 1.0, 0.25, 1, 1, 0.3],
    }
    task_path = os.path.join('task', task_name)
    runner = flgo.init(task_path, feddsa_scheduled, option=option)
    return runner


def eval_and_collect_errors(model, loader, classes, device):
    model.eval()
    errors = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[-1]
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
                        'img': x[i].clone().float() / 255.0 if x[i].dtype == torch.uint8 else x[i].clone(),
                    })
    acc = correct / total if total else 0.0
    return acc, errors, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['PACS_c4', 'office_caltech10_c4'])
    ap.add_argument('--ckpt', required=True, help='~/fl_checkpoints/xxx 目录')
    ap.add_argument('--seed', type=int, required=True, help='必须等于训练 seed 才能复现 test split')
    ap.add_argument('--out', required=True)
    ap.add_argument('--topk', type=int, default=16)
    args = ap.parse_args()

    ckpt_dir = os.path.expanduser(args.ckpt)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(ckpt_dir, 'meta.json')))
    print(f'[Meta] seed={meta["seed"]} best@R{meta["best_round"]} avg={meta["best_avg_acc"]:.4f}')
    assert int(meta['seed']) == args.seed, \
        f'seed mismatch: ckpt {meta["seed"]} vs --seed {args.seed}'

    if 'PACS' in args.task:
        domains = PACS_DOMAINS
    else:
        domains = OFFICE_DOMAINS

    print(f'[flgo.init] task={args.task} seed={args.seed} ...')
    runner = init_runner(args.task, args.seed, num_rounds=meta['num_rounds'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = None
    try:
        classes = runner.clients[0].train_data.datasets[0].classes
    except Exception:
        pass
    if classes is None:
        if 'PACS' in args.task:
            sys.path.insert(0, os.path.join(ROOT, 'task', 'PACS_c4'))
            from config import PACSDomainDataset
            classes = PACSDomainDataset.classes
        else:
            from benchmark.office_caltech10_classification.config import OCDomainDataset
            classes = OCDomainDataset.classes
    num_classes = len(classes)
    print(f'[Classes] {classes}')

    html = [
        '<html><head><meta charset="utf-8"><style>',
        'body{font-family:Arial;padding:20px}',
        'table{border-collapse:collapse;margin-bottom:20px}',
        'td{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:center}',
        'h2{background:#eef;padding:8px}',
        '</style></head><body>',
        f'<h1>{args.task} 方案A 误分类可视化（flgo 真实 test split）</h1>',
        f'<p><b>Checkpoint</b>: {ckpt_dir}<br>',
        f'<b>Meta</b>: seed={meta["seed"]}, best@R{meta["best_round"]}, '
        f'best_avg_acc={meta["best_avg_acc"]:.4f}</p>',
        f'<p>每个 domain 展示 top-{args.topk} "自信但错"样本（softmax 置信度最高的错分）。'
        f'test set 由 flgo 用 seed={args.seed} 固定划分，与训练时完全一致。</p>',
    ]

    stats = {}
    for cid, dom in enumerate(domains):
        client = runner.clients[cid]
        test_data = client.test_data
        if test_data is None or len(test_data) == 0:
            print(f'[Skip] client {cid} ({dom}) has no test data')
            continue

        loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

        ckpt_path = os.path.join(ckpt_dir, f'client_{cid}.pt')
        if not os.path.exists(ckpt_path):
            print(f'[Skip] client_{cid}.pt not found')
            continue

        # 先 deepcopy server.model 架构，然后 load 该 client personalized state
        import copy
        from algorithm.feddsa_scheduled import FedDSAModel
        model = FedDSAModel(num_classes=num_classes).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)

        acc, errors, total = eval_and_collect_errors(model, loader, classes, device)
        stats[dom] = {'acc': acc, 'errors': len(errors), 'total': total, 'client_id': cid}
        print(f'[{dom}] acc={acc:.4f} ({len(errors)}/{total} errors)')

        errors.sort(key=lambda e: -e['conf'])
        top = errors[:args.topk]
        dom_dir = os.path.join(out_dir, dom)
        os.makedirs(dom_dir, exist_ok=True)

        html.append(f'<h2>Client{cid} = {dom} | acc={acc:.2%} | errors={len(errors)}/{total}</h2>')
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
        json.dump({
            'meta': meta, 'task': args.task, 'seed': args.seed,
            'stats': stats,
        }, f, indent=2)

    print(f'\n[Done] View: {out_dir}/index.html')
    avg_acc = sum(s['acc'] for s in stats.values()) / len(stats) if stats else 0
    print(f'[AVG Last] {avg_acc*100:.2f}% (跨 {len(stats)} domains)')
    for d, s in stats.items():
        print(f'  {d:20s}: acc={s["acc"]:.4f} ({s["errors"]}/{s["total"]})')


if __name__ == '__main__':
    main()
