"""生成 PACS 误分类样本可视化"""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image
from torchvision import transforms

from task.PACS_c4 import config as pacs_cfg

PACS_CLASSES = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
DOMAINS = ('art_painting', 'cartoon', 'photo', 'sketch')
ROOT = '/root/miniconda3/lib/python3.10/site-packages/flgo/benchmark/RAW_DATA/PACS'

# 为每个 domain 构建 dataset
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

domain_datasets = {}
for d in DOMAINS:
    domain_datasets[d] = pacs_cfg.PACSDomainDataset(ROOT, domain=d, transform=tf)
    print(f'{d}: {len(domain_datasets[d])} images')

# train = 80% / test = 20%
import random
random.seed(42)
train_list = []
test_by_domain = {}
for d, ds in domain_datasets.items():
    n = len(ds)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]
    train_list.append(torch.utils.data.Subset(ds, train_idx))
    test_by_domain[d] = torch.utils.data.Subset(ds, test_idx)
    print(f'{d}: train={len(train_idx)} test={len(test_idx)}')

train_ds = ConcatDataset(train_list)

net = pacs_cfg.get_model().cuda()

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

print('\nTraining 5 epochs (centralized AlexNet)...')
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

# 误分类收集
print('\nCollecting misclassifications per domain...')
net.eval()
out_dir = os.path.expanduser('~/misclass_pacs')
os.makedirs(out_dir, exist_ok=True)

html = ['<html><head><style>body{font-family:Arial}table{border-collapse:collapse}td{border:1px solid #ccc;padding:4px}</style></head><body>']
html.append('<h1>PACS Misclassifications (centralized AlexNet 5-epoch baseline)</h1>')
html.append('<p>每个域展示 top-16 最"自信但错"的样本（模型置信度高 + 预测错）。'
            '这些是模型认为"像另一类"的样本，最能反映该域的难点。</p>')

stats = {}
for d in DOMAINS:
    loader = DataLoader(test_by_domain[d], batch_size=64, shuffle=False, num_workers=2)
    errors = []
    total = 0; correct = 0
    with torch.no_grad():
        for x, y in loader:
            x_gpu = x.cuda()
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
                        'true': PACS_CLASSES[y[i].item()],
                        'pred': PACS_CLASSES[pred[i].item()],
                        'conf': conf[i].item(),
                        'img': x[i].clone(),
                    })
    acc = correct / total if total else 0
    stats[d] = {'acc': acc, 'errors': len(errors), 'total': total}
    print(f'  {d}: acc={acc:.3f} ({len(errors)}/{total} errors)')

    errors.sort(key=lambda e: -e['conf'])
    top = errors[:16]
    domain_dir = os.path.join(out_dir, d)
    os.makedirs(domain_dir, exist_ok=True)
    html.append(f'<h2>{d}: acc={acc:.2%} ({len(errors)}/{total} errors)</h2>')
    html.append('<table><tr>')
    for i, e in enumerate(top):
        fn = f"{e['true']}_as_{e['pred']}_conf{e['conf']:.2f}_{i:02d}.jpg"
        save_image(e['img'], os.path.join(domain_dir, fn), normalize=False)
        if i > 0 and i % 4 == 0:
            html.append('</tr><tr>')
        html.append(f'<td><img src="{d}/{fn}" width=180><br><b>{e["true"]} → {e["pred"]}</b><br>conf={e["conf"]:.2f}</td>')
    html.append('</tr></table>')

html.append('</body></html>')
with open(os.path.join(out_dir, 'index.html'), 'w') as f:
    f.write('\n'.join(html))

print(f'\nDone.')
print(f'View: {out_dir}/index.html')
print(f'Stats: {json.dumps(stats, indent=2)}')
