"""DualEnc visualizations — 4x4 cross-domain style swap grid + cycle dump.

Hook 进 PerRunLogger 或离线脚本调用. 不需要重新训练.

输出:
- {save_dir}/round_{r:03d}_swap_grid.png    4 row (源域) x 4 col (目标风格), 含对角=自重建
- {save_dir}/round_{r:03d}_cycle.png         3 col: 原图 / swap 图 / cycle 还原图
- {save_dir}/round_{r:03d}_decoder_diversity.png  同图 4 风格的 pairwise L2 数值

Domain id 假设 0..D-1, 调用方需提供 per-domain 样本.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np


def _to_uint8(x):
    """Tanh output [-1,1] -> [0,255] uint8 HWC."""
    x = (x.clamp(-1, 1) + 1.0) / 2.0
    x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    if x.dim() == 4:
        x = x[0]
    # CHW -> HWC
    return x.permute(1, 2, 0).cpu().numpy()


def _save_png(np_array, path):
    """Save HWC uint8 numpy array as PNG. Falls back to PIL if matplotlib unavailable."""
    try:
        from PIL import Image
        if np_array.shape[-1] == 1:
            np_array = np_array[..., 0]
        Image.fromarray(np_array).save(path)
    except ImportError:
        # 兜底用 matplotlib
        import matplotlib.pyplot as plt
        plt.imsave(path, np_array)


def _make_grid(images_2d, row_labels=None, col_labels=None, pad=4):
    """images_2d: list of list of HWC uint8 numpy. Stitch to one big array."""
    R = len(images_2d)
    C = len(images_2d[0])
    H, W, _ = images_2d[0][0].shape
    grid = np.full((R * H + (R - 1) * pad, C * W + (C - 1) * pad, 3), 255, dtype=np.uint8)
    for i in range(R):
        for j in range(C):
            y0 = i * (H + pad)
            x0 = j * (W + pad)
            grid[y0: y0 + H, x0: x0 + W] = images_2d[i][j]
    return grid


@torch.no_grad()
def dump_4x4_style_swap_grid(
    model,
    samples,         # list of (x, domain_id), one per source domain
    style_bank,      # dict client_id -> tensor [N, sty_dim]
    save_path,
    device,
    num_domains=4,
    K_for_swap=4,
):
    """4x4 grid: row = source image domain, col = target style domain.
    Diagonal (col domain == source domain) = self-reconstruction (original z_sty).
    Off-diagonal = swap to target domain's style.

    samples: list of len num_domains, each item (x_tensor [3,H,W], domain_id).
    style_bank: dict client_id -> tensor of z_sty samples.
    NOTE: 调用方需保证 style_bank 的 client_id 跟 sample 的 domain_id 一致 (FedDSA-DualEnc
    在 4-client × 4-domain personalization 设定下天然成立). 否则可视化的对角线含义错位.

    Codex IMPORTANT 修正: 旧版用 i==j 判定对角, 但 samples[i].domain_id 不一定 == i
    (若调用方没有按 domain id 排序). 改用 samples[i] 的实际 domain_id 跟 col 的目标
    domain key (j 的实际值) 比较.
    """
    model.eval()
    R = len(samples)
    # 确定列对应的目标 domain ID 列表 (从 bank 或 0..num_domains-1)
    col_domains = sorted(style_bank.keys()) if style_bank else list(range(num_domains))
    if len(col_domains) < num_domains:
        # bank 不够时, 补默认 domain id
        for d in range(num_domains):
            if d not in col_domains:
                col_domains.append(d)
        col_domains = col_domains[:num_domains]
    C = len(col_domains)
    grid_imgs = []

    for i in range(R):
        x_i, dom_i = samples[i]
        # 转为 int 方便比较
        if torch.is_tensor(dom_i):
            dom_i = int(dom_i.item())
        x_i = x_i.unsqueeze(0).to(device)
        h_i = model.encode(x_i)
        z_sem_i = model.get_semantic(h_i)
        mu_i, _ = model.get_style(h_i)

        row = []
        for j_idx in range(C):
            target_dom = col_domains[j_idx]
            if dom_i == target_dom:
                z_sty_use = mu_i  # self-recon (源域 == 目标域)
            else:
                # 抽 target domain 的 z_sty (style_bank[target_dom])
                if target_dom in style_bank and style_bank[target_dom] is not None and style_bank[target_dom].numel() > 0:
                    pool = style_bank[target_dom].to(device)
                    n = pool.size(0)
                    K = max(1, min(K_for_swap, n))
                    idx = torch.randint(0, n, (K,))
                    chosen = pool[idx]
                    alpha = (torch.rand(K, device=device) * 2.0 - 1.0)
                    alpha = alpha / alpha.abs().sum().clamp(min=1e-6)
                    z_sty_use = (alpha.unsqueeze(-1) * chosen).sum(dim=0, keepdim=True)
                else:
                    # bank 缺这个 domain, 退回随机
                    z_sty_use = torch.randn(1, model.sty_dim, device=device)

            x_recon = model.decode(z_sem_i, z_sty_use)
            row.append(_to_uint8(x_recon))
        grid_imgs.append(row)

    grid = _make_grid(grid_imgs)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _save_png(grid, save_path)
    return save_path


@torch.no_grad()
def dump_cycle_verification(
    model,
    sample,         # (x_tensor, domain_id)
    style_bank,
    target_domain,
    save_path,
    device,
    K_for_swap=4,
):
    """3-col: 原图 / swap 到 target_domain / cycle 还原回 source domain."""
    model.eval()
    x, dom_src = sample
    x = x.unsqueeze(0).to(device)

    h = model.encode(x)
    z_sem = model.get_semantic(h)
    mu_orig, _ = model.get_style(h)

    # 抽目标域 z_sty
    if target_domain in style_bank and style_bank[target_domain] is not None and style_bank[target_domain].numel() > 0:
        pool = style_bank[target_domain].to(device)
        n = pool.size(0)
        K = max(1, min(K_for_swap, n))
        idx = torch.randint(0, n, (K,))
        chosen = pool[idx]
        alpha = (torch.rand(K, device=device) * 2.0 - 1.0)
        alpha = alpha / alpha.abs().sum().clamp(min=1e-6)
        z_sty_target = (alpha.unsqueeze(-1) * chosen).sum(dim=0, keepdim=True)
    else:
        z_sty_target = torch.randn(1, model.sty_dim, device=device)

    x_swap = model.decode(z_sem, z_sty_target)
    h_swap = model.encode(x_swap)
    z_sem_swap = model.get_semantic(h_swap)
    x_cycle = model.decode(z_sem_swap, mu_orig)  # 用回原 z_sty 还原

    row = [
        _to_uint8(x.float() * 2.0 - 1.0 if x.max() <= 1.5 else x),  # 原图标准化到 [-1,1]
        _to_uint8(x_swap),
        _to_uint8(x_cycle),
    ]
    grid = _make_grid([row])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _save_png(grid, save_path)
    return save_path


@torch.no_grad()
def measure_decoder_style_diversity(
    model,
    sample,
    style_bank,
    device,
    num_styles=4,
    K_for_swap=4,
):
    """同一图用 num_styles 个不同风格重建, 算 pairwise L2 平均.
    结果接近 0 = decoder 偷懒不用 z_sty.
    """
    model.eval()
    x = sample.unsqueeze(0).to(device) if sample.dim() == 3 else sample.to(device)
    h = model.encode(x)
    z_sem = model.get_semantic(h)

    recons = []
    domain_keys = list(style_bank.keys())[: num_styles]
    if len(domain_keys) < 2:
        # bank 太空, 用随机风格
        for _ in range(num_styles):
            z_sty = torch.randn(1, model.sty_dim, device=device)
            recons.append(model.decode(z_sem, z_sty))
    else:
        for d in domain_keys:
            pool = style_bank[d].to(device)
            n = pool.size(0)
            K = max(1, min(K_for_swap, n))
            idx = torch.randint(0, n, (K,))
            chosen = pool[idx]
            alpha = (torch.rand(K, device=device) * 2.0 - 1.0)
            alpha = alpha / alpha.abs().sum().clamp(min=1e-6)
            z_sty = (alpha.unsqueeze(-1) * chosen).sum(dim=0, keepdim=True)
            recons.append(model.decode(z_sem, z_sty))

    pairs = []
    for i in range(len(recons)):
        for j in range(i + 1, len(recons)):
            d = (recons[i] - recons[j]).pow(2).mean().item()
            pairs.append(d)
    return float(np.mean(pairs)) if pairs else 0.0


@torch.no_grad()
def cycle_psnr(model, sample, style_bank, target_domain, device, K_for_swap=4):
    """Cycle reconstruction PSNR: 原图 vs cycle 还原图."""
    model.eval()
    x = sample.unsqueeze(0).to(device) if sample.dim() == 3 else sample.to(device)
    # 标准化到 [-1, 1]
    if x.min().item() >= -0.5 and x.max().item() <= 1.5:
        x_norm = x * 2.0 - 1.0
    else:
        x_norm = x

    h = model.encode(x)
    z_sem = model.get_semantic(h)
    mu_orig, _ = model.get_style(h)

    if target_domain in style_bank and style_bank[target_domain] is not None and style_bank[target_domain].numel() > 0:
        pool = style_bank[target_domain].to(device)
        n = pool.size(0)
        K = max(1, min(K_for_swap, n))
        idx = torch.randint(0, n, (K,))
        chosen = pool[idx]
        alpha = (torch.rand(K, device=device) * 2.0 - 1.0)
        alpha = alpha / alpha.abs().sum().clamp(min=1e-6)
        z_sty_target = (alpha.unsqueeze(-1) * chosen).sum(dim=0, keepdim=True)
    else:
        z_sty_target = torch.randn(1, model.sty_dim, device=device)

    x_swap = model.decode(z_sem, z_sty_target)
    h_swap = model.encode(x_swap)
    z_sem_swap = model.get_semantic(h_swap)
    x_cycle = model.decode(z_sem_swap, mu_orig)

    mse = (x_norm - x_cycle).pow(2).mean().item()
    if mse < 1e-10:
        return float('inf')
    # PSNR for [-1, 1] range → max - min = 2
    return 10.0 * np.log10(4.0 / mse)
