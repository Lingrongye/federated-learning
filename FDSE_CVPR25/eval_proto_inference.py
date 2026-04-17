"""方案 B 评估：加载 checkpoint，对比 FC head 推理 vs 原型最近邻推理。

用法:
  python eval_proto_inference.py --ckpt_dir ~/fl_checkpoints/<tag> [--task office_caltech10_c4]

输出: 打印 per-domain FC acc vs Prototype acc + 差值
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from collections import defaultdict
import random

def get_z_sem(model, x):
    """从 feddsa_scheduled 模型提取 semantic features z_sem"""
    # 模型结构: backbone -> pool -> (semantic_head, style_head) -> sem_classifier
    # 我们要的是 semantic_head 的输出
    with torch.no_grad():
        # 通用：用 encode() or get_semantic() 如果有
        if hasattr(model, 'get_semantic'):
            return model.get_semantic(x)
        # fallback：forward 到 semantic_head 但不到 classifier
        h = model.encode(x) if hasattr(model, 'encode') else None
        if h is not None and hasattr(model, 'semantic_head'):
            return model.semantic_head(h)
        # 实在不行就跑到 classifier 前的倒数第二层
        raise RuntimeError('Cannot extract z_sem from model')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--task', default='office_caltech10_c4')
    args = ap.parse_args()

    ckpt = os.path.expanduser(args.ckpt_dir)
    with open(os.path.join(ckpt, 'meta.json')) as f:
        meta = json.load(f)
    print(f'Checkpoint: {os.path.basename(ckpt)}')
    print(f'Best@R{meta["best_round"]}, avg={meta["best_avg_acc"]:.4f}')

    # Task config
    if args.task == 'office_caltech10_c4':
        from task.office_caltech10_c4 import config as cfg
        DOMAINS = ['Caltech', 'Amazon', 'DSLR', 'Webcam']
        # Office 的数据加载 — 看 config 里
    else:
        from task.PACS_c4 import config as cfg
        DOMAINS = ['Art', 'Cartoon', 'Photo', 'Sketch']

    # 加载每个 client 的 train + test
    # Office 每 client 1 个 domain，数据从 cfg.train_data / test_data 索引
    # 这里偷懒重新构建 per-client train/test（seed=42 split 80/20）
    if args.task == 'office_caltech10_c4':
        from task.office_caltech10_c4 import config as cfg
        # Office 没有 domain split 暴露，fallback 用 task/data.json
        # 这里只做 global test 上 FC vs Proto
        print('[WARN] Office per-domain split 需从 task data.json 还原，暂只做 global')
        return

    from task.PACS_c4 import config as cfg_pacs
    ROOT = '/root/miniconda3/lib/python3.10/site-packages/flgo/benchmark/RAW_DATA/PACS'
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    domain_ds = {}
    for d in DOMAINS:
        dname = {'Art': 'art_painting', 'Cartoon': 'cartoon', 'Photo': 'photo', 'Sketch': 'sketch'}[d]
        domain_ds[d] = cfg_pacs.PACSDomainDataset(ROOT, domain=dname, transform=tf)

    random.seed(42)
    train_by_c = []
    test_by_c = []
    for d in DOMAINS:
        ds = domain_ds[d]
        n = len(ds); idx = list(range(n)); random.shuffle(idx)
        split = int(n * 0.8)
        train_by_c.append(Subset(ds, idx[:split]))
        test_by_c.append(Subset(ds, idx[split:]))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_classes = len(cfg_pacs.PACSDomainDataset.classes)

    # Step 1: 对每 client 构造 local proto（用 train set 提 z_sem 按 class 均值）
    all_client_protos = []  # [ {class: avg_z_sem} per client ]
    client_models = []
    for cid, domain in enumerate(DOMAINS):
        state = torch.load(os.path.join(ckpt, f'client_{cid}.pt'), map_location=device)
        net = cfg_pacs.get_model().to(device)
        net.load_state_dict(state, strict=False)
        net.eval()
        client_models.append(net)

        class_feats = defaultdict(list)
        loader = DataLoader(train_by_c[cid], batch_size=64, shuffle=False, num_workers=2)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                try:
                    z = get_z_sem(net, x)
                except Exception as e:
                    print(f'  {domain}: z_sem extract failed: {e}')
                    z = None
                    break
                for i in range(y.size(0)):
                    class_feats[y[i].item()].append(z[i].cpu())
        if not class_feats:
            print(f'  {domain}: no features collected, abort')
            return
        proto = {c: torch.stack(feats).mean(0) for c, feats in class_feats.items()}
        all_client_protos.append(proto)
        print(f'  {domain}: collected protos for classes {sorted(proto.keys())}')

    # Step 2: 全局原型 = 各 client local proto 平均
    global_proto = {}
    for c in range(n_classes):
        feats = [cp[c] for cp in all_client_protos if c in cp]
        if feats:
            global_proto[c] = torch.stack(feats).mean(0)
    print(f'Global protos for classes: {sorted(global_proto.keys())}')

    # Step 3: 对每 client test，FC vs Proto 对比
    print(f'\n{"domain":10s} | {"FC acc":>7s} | {"Proto acc":>9s} | {"Δ":>6s}')
    print('-'*45)
    fc_accs = []
    proto_accs = []
    for cid, domain in enumerate(DOMAINS):
        net = client_models[cid]
        loader = DataLoader(test_by_c[cid], batch_size=64, shuffle=False, num_workers=2)
        fc_correct = proto_correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device); y = y.to(device)
                # FC predict
                logits = net(x)
                fc_pred = logits.argmax(1)
                # Proto predict: z_sem 对每类原型最小欧氏距
                z = get_z_sem(net, x)  # [B, d]
                # distance to each proto
                protos = torch.stack([global_proto[c].to(device) for c in sorted(global_proto.keys())], dim=0)  # [C, d]
                dists = torch.cdist(z, protos)  # [B, C]
                proto_pred = dists.argmin(1)
                fc_correct += (fc_pred == y).sum().item()
                proto_correct += (proto_pred == y).sum().item()
                total += y.size(0)
        fc_acc = fc_correct/total
        proto_acc = proto_correct/total
        fc_accs.append(fc_acc); proto_accs.append(proto_acc)
        print(f'{domain:10s} | {fc_acc*100:>6.2f}% | {proto_acc*100:>8.2f}% | {(proto_acc-fc_acc)*100:+6.2f}%')

    # 整体 mean
    fc_mean = sum(fc_accs)/len(fc_accs) * 100
    proto_mean = sum(proto_accs)/len(proto_accs) * 100
    print('-'*45)
    print(f'{"MEAN":10s} | {fc_mean:>6.2f}% | {proto_mean:>8.2f}% | {proto_mean-fc_mean:+6.2f}%')

if __name__ == '__main__':
    main()
