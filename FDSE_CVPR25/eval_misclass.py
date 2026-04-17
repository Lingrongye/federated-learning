"""生成误分类样本可视化：加载 PACS_c4 task，短训练后收集每域误分类图片。
用于人工检查：Art/Caltech 等难域的图片有什么共同特征？
"""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image

# 导入 FDSE 自己的 task config（使用本地 PACS dataset）
from task.PACS_c4 import config as pacs_cfg

PACS_CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
DOMAINS = ('art_painting', 'cartoon', 'photo', 'sketch')

# 使用 FDSE 的 AlexNet
model_fn = pacs_cfg.init_local_module if hasattr(pacs_cfg, 'init_local_module') else pacs_cfg.init_global_module
net = model_fn()
net = net.cuda()

# 加载 task data
import flgo.benchmark.toolkits.partition as fbp
train_data = pacs_cfg.train_data
test_data = pacs_cfg.test_data
print(f'train samples: {len(train_data)}, test samples: {len(test_data)}')

# 集中式训练几个 epoch 让模型有基础能力
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

print('Training 5 epochs (centralized)...')
net.train()
for epoch in range(5):
    total = 0; correct = 0; loss_sum = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        logits = net(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
    print(f'  epoch {epoch+1}: loss={loss_sum/len(train_loader):.3f} acc={correct/total:.3f}')

# 评估 + 收集误分类
print('Collecting misclassifications by domain...')
net.eval()

# PACS test_data 是 ConcatDataset，子 dataset 按 domain 顺序
errors_by_domain = {d: [] for d in DOMAINS}

# 手动遍历 test set
import numpy as np
with torch.no_grad():
    idx = 0
    for x, y in test_loader:
        x_gpu = x.cuda()
        logits = net(x_gpu)
        probs = torch.softmax(logits, 1)
        pred = logits.argmax(1).cpu()
        conf = probs.max(1).values.cpu()
        for i in range(y.size(0)):
            true_y = y[i].item()
            pred_y = pred[i].item()
            c = conf[i].item()
            if true_y != pred_y:
                # 找 domain — 需要在原始 test_data 里看
                # 简化：按 idx 推断 domain 边界（如果 test_data 是 ConcatDataset）
                domain = 'unknown'
                if hasattr(test_data, 'datasets'):
                    cum = 0
                    for di, sub in enumerate(test_data.datasets):
                        if idx + i - cum < len(sub):
                            domain = DOMAINS[di] if di < len(DOMAINS) else f'd{di}'
                            break
                        cum += len(sub)
                errors_by_domain.setdefault(domain, []).append({
                    'idx': idx + i,
                    'true': PACS_CLASSES[true_y],
                    'pred': PACS_CLASSES[pred_y],
                    'conf': c,
                    'img': x[i].cpu(),
                })
        idx += y.size(0)

# 按域保存 top-16 误分类图片
out_dir = os.path.expanduser('~/misclass_pacs')
os.makedirs(out_dir, exist_ok=True)

for d, errs in errors_by_domain.items():
    # 置信度高 = 模型确信但错 = 最难样本
    errs.sort(key=lambda e: -e['conf'])
    top = errs[:16]
    if not top:
        print(f'  {d}: 0 errors')
        continue
    domain_dir = os.path.join(out_dir, d)
    os.makedirs(domain_dir, exist_ok=True)
    for e in top:
        fn = f"{e['true']}_as_{e['pred']}_conf{e['conf']:.2f}_idx{e['idx']}.jpg"
        save_image(e['img'], os.path.join(domain_dir, fn), normalize=True)
    print(f'  {d}: {len(errs)} total errors, top 16 saved to {domain_dir}')

# 汇总 HTML
html = ['<html><body><h1>PACS Misclassifications (centralized 5-epoch baseline)</h1>']
for d in DOMAINS:
    html.append(f'<h2>{d} ({len(errors_by_domain[d])} total errors)</h2>')
    html.append('<table cellpadding="4"><tr>')
    for i, f in enumerate(sorted(os.listdir(os.path.join(out_dir, d)) if os.path.exists(os.path.join(out_dir, d)) else [])):
        if not f.endswith('.jpg'): continue
        if i > 0 and i % 4 == 0:
            html.append('</tr><tr>')
        html.append(f'<td><img src="{d}/{f}" width=180><br><tt>{f[:60]}</tt></td>')
    html.append('</tr></table>')
html.append('</body></html>')
with open(os.path.join(out_dir, 'index.html'), 'w') as f:
    f.write('\n'.join(html))
print(f'\nDone. View: {out_dir}/index.html')
