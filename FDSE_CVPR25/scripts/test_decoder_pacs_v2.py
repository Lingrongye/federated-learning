"""PACS 单机重建 + swap 测试 v2 (适配 PACS task config 结构).

PACS train_data 是 list of 4 个 PACSDomainDataset (不是 ConcatDataset).
domains 顺序: ('art_painting', 'cartoon', 'photo', 'sketch')
classes: ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

PROJECT = '/home/lry/code/federated-learning'
sys.path.insert(0, os.path.join(PROJECT, 'FDSE_CVPR25'))

from algorithm.feddsa_dualenc import FedDSADualEncModel

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

# 加载 PACS - train_data 是 list[4]
sys.path.insert(0, os.path.join(PROJECT, 'FDSE_CVPR25/task/PACS_c4'))
from config import train_data
domains = ('art_painting', 'cartoon', 'photo', 'sketch')
classes = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
print(f'train_data type: {type(train_data)}, len: {len(train_data)}')
print(f'domains: {domains}')
print(f'classes: {classes}')
for i, ds in enumerate(train_data):
    print(f'  Domain {i} ({domains[i]}): {len(ds)} samples')

# 把 4 个 domain dataset concat 成一个用来训练
class WrappedConcat(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum = [0]
        for d in datasets:
            self.cum.append(self.cum[-1] + len(d))
    def __len__(self):
        return self.cum[-1]
    def __getitem__(self, idx):
        for i, c in enumerate(self.cum[1:]):
            if idx < c:
                local = idx - self.cum[i]
                img, lab = self.datasets[i][local]
                if img.dtype != torch.float32:
                    img = img.float() / 255.0
                return img, lab
        raise IndexError(idx)

ds = WrappedConcat(train_data)
print(f'\nTotal: {len(ds)} samples')
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

# 模型
STY_DIM = 32
model = FedDSADualEncModel(num_classes=7, sem_dim=512, sty_dim=STY_DIM, srm_hidden=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 50
print(f'\n=== Training (sty_dim={STY_DIM}, λ_rec=1.0, {EPOCHS} epochs) ===')
print(f'{"Epoch":>5} | {"L_rec":>8} | {"L_kl":>8} | {"Time":>6}')
print('-' * 45)
for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()
    sum_rec, sum_kl, n_b = 0.0, 0.0, 0
    kl_w = min(1.0, epoch / 10.0) * 0.001
    for x, _ in loader:
        x = x.to(device)
        x_target = x * 2 - 1
        optimizer.zero_grad()
        h = model.encode(x)
        z_sem = model.get_semantic(h)
        mu, logvar = model.get_style(h)
        z_sty = model.reparameterize(mu, logvar)
        x_hat = model.decode(z_sem, z_sty)
        l_rec = F.l1_loss(x_hat, x_target)
        l_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        loss = l_rec + kl_w * l_kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        sum_rec += l_rec.item()
        sum_kl += l_kl.item()
        n_b += 1
    el = time.time() - t0
    print(f'{epoch:>5} | {sum_rec/n_b:>8.4f} | {sum_kl/n_b:>8.3f} | {el:>5.1f}s', flush=True)

# === 采样: 每个 domain × 每个 class 找代表图 ===
print(f'\n=== 采样 ===')
model.eval()
domain_class_imgs = {}  # (d_idx, c_idx) -> img tensor

for d in range(4):
    dom_ds = train_data[d]
    found = {}
    for j in range(len(dom_ds)):
        img, lab = dom_ds[j]
        if lab not in found and lab < 7:
            if img.dtype != torch.float32:
                img = img.float() / 255.0
            found[lab] = img
            if len(found) >= 7:
                break
    for c, im in found.items():
        domain_class_imgs[(d, c)] = im
print(f'Got {len(domain_class_imgs)} (domain, class) pairs')

# === Grid 1: photo (域 2) 7 类自重建 ===
photo_idx = 2
samples_recon = []
for c in range(7):
    if (photo_idx, c) in domain_class_imgs:
        samples_recon.append((classes[c], domain_class_imgs[(photo_idx, c)]))
xs_class = torch.stack([s[1] for s in samples_recon]).to(device)
with torch.no_grad():
    h = model.encode(xs_class)
    z_sem = model.get_semantic(h)
    mu, _ = model.get_style(h)
    x_recon = model.decode(z_sem, mu)

# === Grid 2: dog 类 4×4 swap (跨域同类) ===
dog_imgs = []
for d in range(4):
    if (d, 0) in domain_class_imgs:
        dog_imgs.append(domain_class_imgs[(d, 0)])
xs_dog = torch.stack(dog_imgs).to(device)
with torch.no_grad():
    h_dog = model.encode(xs_dog)
    z_sem_dog = model.get_semantic(h_dog)
    mu_dog, _ = model.get_style(h_dog)
    grid_4x4 = torch.zeros(4, 4, 3, 256, 256, device=device)
    for i in range(4):
        for j in range(4):
            grid_4x4[i, j] = model.decode(z_sem_dog[i:i+1], mu_dog[j:j+1])[0]

# === Grid 3: photo dog + 4 风格 ===
photo_dog = xs_dog[photo_idx:photo_idx+1]
with torch.no_grad():
    h_pd = model.encode(photo_dog)
    z_sem_pd = model.get_semantic(h_pd)
    z_sem_4 = z_sem_pd.repeat(4, 1)
    x_4styles = model.decode(z_sem_4, mu_dog)

# === Grid 4: 4 类 photo × 4 域风格 ===
test_classes = [0, 2, 4, 6]  # dog, giraffe, horse, person
photo_imgs = []
for c in test_classes:
    if (photo_idx, c) in domain_class_imgs:
        photo_imgs.append((classes[c], domain_class_imgs[(photo_idx, c)]))
xs_photo = torch.stack([s[1] for s in photo_imgs]).to(device)
with torch.no_grad():
    h_p = model.encode(xs_photo)
    z_sem_p = model.get_semantic(h_p)
    n_classes = z_sem_p.size(0)
    grid_class_style = torch.zeros(n_classes, 4, 3, 256, 256, device=device)
    for i in range(n_classes):
        for j in range(4):
            grid_class_style[i, j] = model.decode(z_sem_p[i:i+1], mu_dog[j:j+1])[0]

# Save
SAVE_DIR = os.path.join(PROJECT, 'experiments/ablation/EXP-128_dualenc_full/viz/pacs_test')
os.makedirs(SAVE_DIR, exist_ok=True)

def to_uint8_tanh(t):
    t = (t.clamp(-1, 1) + 1.0) / 2.0
    return (t * 255).round().clamp(0, 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()
def to_uint8_01(t):
    return (t.clamp(0, 1) * 255).round().clamp(0, 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()

from PIL import Image
H, W = 256, 256
pad = 8

# Grid 1
big1 = np.full((7 * (H + pad) - pad, 2 * (W + pad) - pad, 3), 255, dtype=np.uint8)
for r in range(7):
    big1[r*(H+pad):r*(H+pad)+H, 0:W] = to_uint8_01(xs_class[r])
    big1[r*(H+pad):r*(H+pad)+H, (W+pad):(W+pad)+W] = to_uint8_tanh(x_recon[r])
out1 = os.path.join(SAVE_DIR, '01_7classes_photo_recon.png')
Image.fromarray(big1).save(out1)
print(f'\n✓ Grid 1: {out1}')

# Grid 2
big2 = np.full((4 * (H + pad) - pad, 4 * (W + pad) - pad, 3), 255, dtype=np.uint8)
for i in range(4):
    for j in range(4):
        big2[i*(H+pad):i*(H+pad)+H, j*(W+pad):j*(W+pad)+W] = to_uint8_tanh(grid_4x4[i, j])
out2 = os.path.join(SAVE_DIR, '02_dog_swap_4x4.png')
Image.fromarray(big2).save(out2)
print(f'✓ Grid 2 (dog 4×4 swap): {out2}')
print(f'   行 = 源域 dog (art/cartoon/photo/sketch)')
print(f'   列 = 注入风格 (art/cartoon/photo/sketch)')

# Grid 3
big3 = np.full((H, 5 * (W + pad) - pad, 3), 255, dtype=np.uint8)
big3[:, 0:W] = to_uint8_01(photo_dog[0])
for j in range(4):
    big3[:, (j+1)*(W+pad):(j+1)*(W+pad)+W] = to_uint8_tanh(x_4styles[j])
out3 = os.path.join(SAVE_DIR, '03_photo_dog_4styles.png')
Image.fromarray(big3).save(out3)
print(f'✓ Grid 3 (photo dog + 4 style): {out3}')

# Grid 4
big4 = np.full((n_classes * (H + pad) - pad, 4 * (W + pad) - pad, 3), 255, dtype=np.uint8)
for i in range(n_classes):
    for j in range(4):
        big4[i*(H+pad):i*(H+pad)+H, j*(W+pad):j*(W+pad)+W] = to_uint8_tanh(grid_class_style[i, j])
out4 = os.path.join(SAVE_DIR, '04_4classes_x_4styles.png')
Image.fromarray(big4).save(out4)
print(f'✓ Grid 4 (4 类 × 4 风格): {out4}')

# 量化
mse_recon = (x_recon - (xs_class * 2 - 1)).pow(2).mean().item()
psnr = 10 * np.log10(4.0 / max(mse_recon, 1e-9))
print(f'\n=== 量化 ===')
print(f'7 类 photo 自重建 MSE={mse_recon:.4f} PSNR={psnr:.2f} dB')
pairwise = []
for i in range(4):
    for j in range(i+1, 4):
        d = (x_4styles[i] - x_4styles[j]).pow(2).mean().item()
        pairwise.append(d)
print(f'Photo dog + 4 风格 pairwise L2: mean={np.mean(pairwise):.4f} max={np.max(pairwise):.4f}')
print(f'  (Office sty32 是 0.0113, 看 PACS 是不是明显大)')
