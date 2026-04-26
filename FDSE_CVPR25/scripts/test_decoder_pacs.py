"""PACS 单机重建 + swap 测试 (sty_dim=32, λ_rec=1.0).

PACS 4 域 (photo/art/cartoon/sketch) 风格差距极大, 用来验证 image-space cycle
是否在合适数据集上 work.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT = '/home/lry/code/federated-learning'
sys.path.insert(0, os.path.join(PROJECT, 'FDSE_CVPR25'))

from algorithm.feddsa_dualenc import FedDSADualEncModel

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

# 数据
sys.path.insert(0, os.path.join(PROJECT, 'FDSE_CVPR25/benchmark/PACS_classification'))
from config import train_data
print(f'Total samples: {len(train_data)}')
print(f'Domains: {train_data.domains}')
print(f'Cumulative sizes: {train_data.cumulative_sizes}')

# PACS 类别 (按照 PACS 标准顺序)
classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
print(f'Classes: {classes}')

class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, lab = self.base[idx]
        if img.dtype != torch.float32:
            img = img.float() / 255.0
        return img, lab

ds = WrappedDataset(train_data)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
print(f'Batches per epoch: {len(loader)}')

# 模型
STY_DIM = 32
model = FedDSADualEncModel(num_classes=7, sem_dim=512, sty_dim=STY_DIM, srm_hidden=256).to(device)
print(f'\nModel: sty_dim={STY_DIM}, num_classes=7')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练
EPOCHS = 50
print(f'\n=== Training 单机 PACS (λ_rec=1.0, sty_dim={STY_DIM}, {EPOCHS} epochs) ===')
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
    print(f'{epoch:>5} | {sum_rec/n_b:>8.4f} | {sum_kl/n_b:>8.3f} | {el:>5.1f}s')

# === 多类别采样 + 跨域同类 swap ===
print(f'\n=== 多类别采样 + swap ===')
model.eval()

# PACS 内每个域内, 按类顺序排列. 我们要找:
# - 每个域 × 每个类的第一张图
# 简化: 每个域 4 类 (dog/elephant/giraffe/guitar) 用于 swap demo
# 跨域采样: 找 photo/art/cartoon/sketch 各自的 dog 图

cum = [0] + list(train_data.cumulative_sizes)
domain_class_imgs = {}  # (domain_idx, class_idx) -> img tensor

# 遍历前 200 张/域 找各类
for d in range(4):
    start, end = cum[d], cum[d+1]
    found = {}
    for j in range(start, min(end, start + 500)):
        img, lab = train_data[j]
        if lab not in found and lab < 7:
            if img.dtype != torch.float32:
                img = img.float() / 255.0
            found[lab] = img
            if len(found) >= 7:
                break
    for c, im in found.items():
        domain_class_imgs[(d, c)] = im
print(f'Collected {len(domain_class_imgs)} (domain, class) image pairs')

# === 任务 1: 7 个类别自重建 (用 photo 域, 域 1) ===
samples_recon = []
for c in range(7):
    if (0, c) in domain_class_imgs:
        samples_recon.append((classes[c], domain_class_imgs[(0, c)]))

xs_class = torch.stack([s[1] for s in samples_recon]).to(device)
with torch.no_grad():
    h = model.encode(xs_class)
    z_sem = model.get_semantic(h)
    mu, _ = model.get_style(h)
    x_recon = model.decode(z_sem, mu)

# === 任务 2: dog 类 4×4 swap (跨域同类) ===
# 4 域的 dog 图
dog_imgs = []
for d in range(4):
    if (d, 0) in domain_class_imgs:
        dog_imgs.append(domain_class_imgs[(d, 0)])
xs_dog = torch.stack(dog_imgs).to(device)
with torch.no_grad():
    h_dog = model.encode(xs_dog)
    z_sem_dog = model.get_semantic(h_dog)
    mu_dog, _ = model.get_style(h_dog)

    # 4×4: 行 i 用 dog_d_i 的 z_sem, 列 j 用 dog_d_j 的 z_sty
    grid_4x4 = torch.zeros(4, 4, 3, 256, 256, device=device)
    for i in range(4):
        for j in range(4):
            grid_4x4[i, j] = model.decode(z_sem_dog[i:i+1], mu_dog[j:j+1])[0]

# === 任务 3: 同 photo dog + 4 域风格 ===
photo_dog = xs_dog[0:1]
with torch.no_grad():
    h_pd = model.encode(photo_dog)
    z_sem_pd = model.get_semantic(h_pd)
    z_sem_4 = z_sem_pd.repeat(4, 1)
    z_sty_4 = mu_dog
    x_4styles = model.decode(z_sem_4, z_sty_4)

# === 任务 4: dog 用每个域的风格 (网格 row=类别 col=风格) ===
# 取 4 个类 (dog/elephant/giraffe/horse) 各一张 photo, 用 4 域 sketch 风格 swap
test_classes = [0, 1, 2, 4]  # dog, elephant, giraffe, horse
photo_imgs = []
for c in test_classes:
    if (0, c) in domain_class_imgs:
        photo_imgs.append((classes[c], domain_class_imgs[(0, c)]))

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

# Grid 1: 7 类 photo 自重建
big1 = np.full((7 * (H + pad) - pad, 2 * (W + pad) - pad, 3), 255, dtype=np.uint8)
for r in range(7):
    big1[r*(H+pad):r*(H+pad)+H, 0:W] = to_uint8_01(xs_class[r])
    big1[r*(H+pad):r*(H+pad)+H, (W+pad):(W+pad)+W] = to_uint8_tanh(x_recon[r])
out1 = os.path.join(SAVE_DIR, '01_7classes_photo_recon.png')
Image.fromarray(big1).save(out1)
print(f'\n✓ Grid 1 (7类 photo 自重建): {out1}')

# Grid 2: dog 4×4 swap
big2 = np.full((4 * (H + pad) - pad, 4 * (W + pad) - pad, 3), 255, dtype=np.uint8)
for i in range(4):
    for j in range(4):
        big2[i*(H+pad):i*(H+pad)+H, j*(W+pad):j*(W+pad)+W] = to_uint8_tanh(grid_4x4[i, j])
out2 = os.path.join(SAVE_DIR, '02_dog_swap_4x4.png')
Image.fromarray(big2).save(out2)
print(f'✓ Grid 2 (dog 4×4 swap): {out2}')
print(f'   行 = 源域 dog 语义 [photo/art/cartoon/sketch]')
print(f'   列 = 注入风格 [photo/art/cartoon/sketch]')

# Grid 3: photo dog + 4 风格 (含原图)
big3 = np.full((H, 5 * (W + pad) - pad, 3), 255, dtype=np.uint8)
big3[:, 0:W] = to_uint8_01(photo_dog[0])
for j in range(4):
    big3[:, (j+1)*(W+pad):(j+1)*(W+pad)+W] = to_uint8_tanh(x_4styles[j])
out3 = os.path.join(SAVE_DIR, '03_photo_dog_4styles.png')
Image.fromarray(big3).save(out3)
print(f'✓ Grid 3 (photo dog + 4 风格): {out3}')

# Grid 4: 4 类 photo × 4 风格 swap
big4 = np.full((n_classes * (H + pad) - pad, 4 * (W + pad) - pad, 3), 255, dtype=np.uint8)
for i in range(n_classes):
    for j in range(4):
        big4[i*(H+pad):i*(H+pad)+H, j*(W+pad):j*(W+pad)+W] = to_uint8_tanh(grid_class_style[i, j])
out4 = os.path.join(SAVE_DIR, '04_4classes_x_4styles.png')
Image.fromarray(big4).save(out4)
print(f'✓ Grid 4 (4 类 photo × 4 风格 swap): {out4}')
print(f'   行 = {[s[0] for s in photo_imgs]}')
print(f'   列 = photo/art/cartoon/sketch 风格')

# 量化分析
print(f'\n=== 量化 ===')
mse_recon = (x_recon - (xs_class * 2 - 1)).pow(2).mean().item()
psnr = 10 * np.log10(4.0 / max(mse_recon, 1e-9))
print(f'7 类 photo 自重建 MSE={mse_recon:.4f} PSNR={psnr:.2f} dB')

# Swap 多样性: 比 office 应该高很多
pairwise = []
for i in range(4):
    for j in range(i+1, 4):
        d = (x_4styles[i] - x_4styles[j]).pow(2).mean().item()
        pairwise.append(d)
print(f'Photo dog + 4 风格 pairwise L2: mean={np.mean(pairwise):.4f} max={np.max(pairwise):.4f}')
print(f'  (Office 上是 mean={0.0113:.4f}, 看 PACS 是否明显更大)')
