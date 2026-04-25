"""Shape / Texture Sensitivity Probe — v2 (codex-reviewed).

假设: 训练后的 AlexNet 在 conv5 输出的每个 channel 对 shape-distortion 和
      texture-distortion 的敏感度分离 (不都落在 y=x).

判定:
  * 散点呈明显分离模式 → FedShapeGate 值得开发
  * 散点挤在对角线 → 假设失败, 方案作废

扰动选择 (codex review 后调整):
  * Shape-distortion: ElasticTransform (保纹理变形体)
  * Texture-distortion: FDA-style **低频窗口** amplitude swap (非 full-spectrum,
    保留 DC + 高频, 避免语义破坏 / clipping 主导效应)

数据源 (codex 指出的 BLOCKING #1 修复):
  * 从 client.train_data 的底层 dataset 的 images_path 拿真实文件, 不读硬编码路径

度量 (codex 指出的 BLOCKING #2 修复):
  * 排除低 energy channel (norm 低于 --low_norm_pct 分位数, 默认 10%)
  * 主度量: pooled sum(diff)/sum(norm), 不是 per-sample ratio 再平均
  * 同时 report 绝对 diff, 用于交叉验证
"""
import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FDSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FDSE_ROOT))
import flgo

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
IMG_SIZE = 256

# codex fix #6: 多 resample 避免 N=20 结论不稳
DEFAULT_N_STYLE_DONORS = 3  # 每张 test 图用 3 个不同 style donor 做平均


def detect_algo(ckpt_dir: Path):
    sd = torch.load(ckpt_dir / 'global_model.pt', map_location='cpu')
    if 'semantic_head.mu_head.0.weight' in sd:
        return 'vib'
    return 'sgpa'


class FeatureGrabber:
    """捕获 encoder.features (maxpool5 后) 的输出."""
    def __init__(self):
        self.feat = None
        self.handle = None

    def hook(self, module, inp, out):
        self.feat = out.detach()  # [B, 256, 7, 7]

    def attach(self, encoder):
        self.handle = encoder.features.register_forward_hook(self.hook)

    def detach(self):
        if self.handle:
            self.handle.remove()


def get_client_dataset_images(client) -> list:
    """codex fix #1 (r2): 从 client.train_data 严格采样, **保留 Subset.indices**,
    确保只拿 client 实际用的那一部分 (PACS_c4 在 train_holdout=0.2 下会 random_split).

    Returns: list of {'path': str, 'label': int, 'cls': str}
    """
    ds = client.train_data
    # 逐层 unwrap Subset, 累计 indices
    indices = None
    probe = ds
    while hasattr(probe, 'dataset'):
        if hasattr(probe, 'indices'):
            sub_idx = list(probe.indices)
            indices = sub_idx if indices is None else [sub_idx[i] for i in indices]
        probe = probe.dataset
    assert hasattr(probe, 'images_path') and hasattr(probe, 'labels'), (
        f'base dataset {type(probe).__name__} 缺 images_path/labels'
    )
    assert len(probe.images_path) == len(probe.labels), 'path/label 数量不一致'
    if indices is None:
        indices = list(range(len(probe.images_path)))
    items = []
    for idx in indices:
        items.append({
            'path': probe.images_path[idx],
            'label': int(probe.labels[idx]),
            'cls': PACS_CLASSES[int(probe.labels[idx])],
        })
    return items


def load_image_from_path(path: str):
    """加载一张图, 处理 RGBA/grayscale, 返回 3-channel PIL."""
    img = Image.open(path)
    if len(img.split()) != 3:
        img = T.Grayscale(num_output_channels=3)(img)
    return img.convert('RGB')


def preprocess_for_model(img_pil: Image.Image):
    """严格匹配 task/PACS_c4/config.py 的训练 transform."""
    tf = T.Compose([T.Resize([IMG_SIZE, IMG_SIZE]), T.PILToTensor()])
    x = tf(img_pil).float() / 255.0  # [3, 256, 256] ∈ [0, 1]
    return x


def shape_distort(x: torch.Tensor, alpha: float = 50.0, sigma: float = 5.0) -> torch.Tensor:
    """Elastic transform: 保纹理变形体."""
    et = T.ElasticTransform(alpha=alpha, sigma=sigma)
    return et(x)


def texture_distort_fda(
    x: torch.Tensor, x_style: torch.Tensor, beta: float = 0.1
) -> tuple:
    """FDA-style 低频窗口 amplitude swap (Yang 2020, CVPR).
    codex fix #4: 不用 full-spectrum swap, 只在以中心 (DC) 为中心的低频窗口交换 amp;
    窗口外的高频 amp 保持原样. 注意: 低频窗口**包含 DC**, DC 也被替换成 donor 的,
    这是 FDA 的标准做法 (平均亮度跟着 donor 走, 符合"变纹理"的预期).

    Args:
      x, x_style: [3, H, W] ∈ [0,1]
      beta: 低频窗口半径比例 (0.1 = 10% 低频, FDA 推荐 0.01~0.1)

    Returns:
      (x_distorted, clip_ratio)  clip_ratio 用于监控 clamp 是否主导
    """
    C, H, W = x.shape
    x_fft = torch.fft.fft2(x)
    s_fft = torch.fft.fft2(x_style)
    # shift 到中心
    x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
    s_fft_shift = torch.fft.fftshift(s_fft, dim=(-2, -1))
    # 低频窗口
    cy, cx = H // 2, W // 2
    half_h = max(1, int(H * beta / 2))
    half_w = max(1, int(W * beta / 2))
    # 仅在低频窗口内替换 amplitude
    x_amp = x_fft_shift.abs()
    x_phase = torch.angle(x_fft_shift)
    s_amp = s_fft_shift.abs()
    new_amp = x_amp.clone()
    new_amp[:, cy-half_h:cy+half_h, cx-half_w:cx+half_w] = \
        s_amp[:, cy-half_h:cy+half_h, cx-half_w:cx+half_w]
    # 重建
    new_fft_shift = new_amp * torch.exp(1j * x_phase)
    new_fft = torch.fft.ifftshift(new_fft_shift, dim=(-2, -1))
    out_raw = torch.fft.ifft2(new_fft).real
    # 统计 clipping fraction (监控用)
    clip_frac = float(((out_raw < 0) | (out_raw > 1)).float().mean().item())
    out = out_raw.clamp(0.0, 1.0)
    return out, clip_frac


def pooled_sensitivity(F_orig: torch.Tensor, F_pert: torch.Tensor, low_norm_mask: torch.Tensor = None):
    """codex fix #2: Pooled sum(diff)/sum(norm), 不是 per-sample ratio 再平均.
    避免低 norm channel 分母爆炸.

    Args:
      F_orig, F_pert: [N, C, H, W]
      low_norm_mask: [C] bool, True=该 channel norm 太低要排除

    Returns: [C] tensor, 低 norm channel 的位置填 NaN
    """
    diff = (F_orig - F_pert).pow(2).sum(dim=(2, 3)).sqrt()  # [N, C]
    norm = F_orig.pow(2).sum(dim=(2, 3)).sqrt()             # [N, C]
    # pooled: sum over N, 然后 sum(diff) / sum(norm)
    pooled_diff = diff.sum(dim=0)  # [C]
    pooled_norm = norm.sum(dim=0)  # [C]
    sens = pooled_diff / pooled_norm.clamp(min=1e-6)  # [C]
    if low_norm_mask is not None:
        sens = sens.clone()
        sens[low_norm_mask] = float('nan')
    return sens, pooled_diff, pooled_norm


# 反向逻辑: probe 只用 encoder.features (conv1..maxpool5), 这是唯一 critical 的部分.
# 其他所有 mismatch (head 架构差异 / VIB vs plain / classifier / EMA) 都 benign.
CRITICAL_PREFIX = 'encoder.features.'


def strict_load_check(model, sd, label: str):
    """probe 只读 encoder.features 的输出 (conv5 feature map). 这是 critical prefix.
    其他 mismatch (semantic_head 是 plain Linear 还是 VIB MuHead, classifier 维度等)
    对 probe 不影响, 视为 benign.
    """
    result = model.load_state_dict(sd, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    crit_missing = [k for k in missing if k.startswith(CRITICAL_PREFIX)]
    crit_unexpected = [k for k in unexpected if k.startswith(CRITICAL_PREFIX)]
    print(f'  [{label}] missing={len(missing)} (critical encoder.features.*={len(crit_missing)}), '
          f'unexpected={len(unexpected)} (critical={len(crit_unexpected)})')
    if crit_missing:
        raise RuntimeError(
            f'[{label}] CRITICAL encoder.features.* keys missing — ckpt incompatible:\n  ' +
            '\n  '.join(crit_missing)
        )
    if crit_unexpected:
        raise RuntimeError(
            f'[{label}] CRITICAL encoder.features.* keys unexpected:\n  ' +
            '\n  '.join(crit_unexpected)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--task', default='PACS_c4')
    ap.add_argument('--out', default='/tmp/shape_tex_probe_v2')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--n_per_domain', type=int, default=8, help='每 domain 采样数')
    ap.add_argument('--n_style_donors', type=int, default=DEFAULT_N_STYLE_DONORS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--elastic_alpha', type=float, default=50.0)
    ap.add_argument('--elastic_sigma', type=float, default=5.0)
    ap.add_argument('--fda_beta', type=float, default=0.1)
    ap.add_argument('--low_norm_pct', type=float, default=10.0,
                    help='排除 norm 低于该 percentile 的 channel')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = Path(args.ckpt)

    # 1. Init
    algo = detect_algo(ckpt_dir)
    print(f'[probe] algo={algo}')
    if algo == 'vib':
        import algorithm.feddsa_sgpa_vib as algo_mod
        algo_para = [1, 0]
    else:
        import algorithm.feddsa_sgpa as algo_mod
        algo_para = None
    option = {'num_rounds': 0, 'proportion': 1.0, 'seed': 2,
              'gpu': [args.gpu], 'load_mode': '', 'num_parallels': 1}
    if algo_para is not None:
        option['algo_para'] = algo_para
    runner = flgo.init(task=f'./task/{args.task}', algorithm=algo_mod, option=option)
    server = runner
    clients = server.clients

    # 2. Load ckpt - codex fix #3: strict 校验
    global_sd = torch.load(ckpt_dir / 'global_model.pt', map_location=device)
    strict_load_check(server.model, global_sd, 'server.global')
    client_states = torch.load(ckpt_dir / 'client_models.pt', map_location=device)
    # codex fix #5: hard assert 长度匹配
    assert isinstance(client_states, list), f'client_models.pt 必须是 list, got {type(client_states)}'
    assert len(client_states) == len(clients), \
        f'client_states 长度 {len(client_states)} != clients 数量 {len(clients)}'
    for cid, c in enumerate(clients):
        strict_load_check(c.model, client_states[cid], f'client{cid}')
        c.model = c.model.to(device).eval()

    # 3. 严格验证 client→domain 映射 (codex fix #5: hard-fail)
    print('[probe] verifying client→domain mapping...')
    client_items_map = {}  # cid -> list of items from this client's dataset
    for cid in range(len(DOMAINS)):
        items = get_client_dataset_images(clients[cid])
        expected = DOMAINS[cid]
        if expected not in items[0]['path']:
            raise RuntimeError(f'client {cid} domain mismatch! expected {expected}, '
                               f'got path={items[0]["path"]}')
        # 随机挑 n_per_domain 张, per class 平均分布
        rng = random.Random(args.seed + cid)
        by_cls = {}
        for it in items:
            by_cls.setdefault(it['cls'], []).append(it)
        picked = []
        classes_in_order = sorted(by_cls.keys())
        # 轮询每 class 挑 1 张, 直到达到 n_per_domain
        while len(picked) < args.n_per_domain:
            for cls in classes_in_order:
                if by_cls[cls]:
                    picked.append(by_cls[cls].pop(rng.randrange(len(by_cls[cls]))))
                if len(picked) >= args.n_per_domain:
                    break
        client_items_map[cid] = picked
        print(f'  ✓ client {cid} ({expected}): picked {len(picked)} samples, '
              f'classes={sorted(set(it["cls"] for it in picked))}')

    # 4. 把所有样本 flatten 成一个列表, 每条记 (cid, item)
    all_samples = []
    for cid, items in client_items_map.items():
        for it in items:
            all_samples.append({'cid': cid, **it})
    print(f'[probe] total {len(all_samples)} samples from real client datasets')

    # 5. Triple forward: original / shape-distorted / texture-distorted × n_style_donors
    all_F_orig, all_F_shape = [], []
    all_F_tex = [[] for _ in range(args.n_style_donors)]
    clip_frac_log = []

    with torch.no_grad():
        for i, s in enumerate(all_samples):
            cid = s['cid']
            client_model = clients[cid].model

            x_orig_pil = load_image_from_path(s['path'])
            x_orig = preprocess_for_model(x_orig_pil)              # [3, 256, 256]
            x_shape = shape_distort(
                x_orig, alpha=args.elastic_alpha, sigma=args.elastic_sigma
            )

            # codex fix #6: 多个 style donor, 后面平均
            # 限定不同 domain + 不同 sample
            other = [t for t in all_samples if t['cid'] != cid]
            donors = random.sample(other, min(args.n_style_donors, len(other)))
            x_tex_list = []
            for donor in donors:
                donor_pil = load_image_from_path(donor['path'])
                x_donor = preprocess_for_model(donor_pil)
                x_d, clip = texture_distort_fda(x_orig, x_donor, beta=args.fda_beta)
                x_tex_list.append(x_d)
                clip_frac_log.append(clip)

            # Forward
            grabber = FeatureGrabber()
            grabber.attach(client_model.encoder)

            # orig
            _ = client_model.encode(x_orig.unsqueeze(0).to(device))
            all_F_orig.append(grabber.feat.cpu())
            # shape
            _ = client_model.encode(x_shape.unsqueeze(0).to(device))
            all_F_shape.append(grabber.feat.cpu())
            # 每个 donor 一次 tex forward
            for j, x_d in enumerate(x_tex_list):
                _ = client_model.encode(x_d.unsqueeze(0).to(device))
                all_F_tex[j].append(grabber.feat.cpu())

            grabber.detach()
            if (i + 1) % 8 == 0:
                print(f'  [{i+1}/{len(all_samples)}] processed')

    # 6. 汇总 & check shape consistency
    F_orig = torch.cat(all_F_orig, dim=0)              # [N, 256, 7, 7]
    F_shape = torch.cat(all_F_shape, dim=0)
    F_tex_all = [torch.cat(lst, dim=0) for lst in all_F_tex]  # list of [N, 256, 7, 7]
    assert F_orig.shape == F_shape.shape
    for Ft in F_tex_all:
        assert Ft.shape == F_orig.shape
    print(f'[probe] feature shape: {tuple(F_orig.shape)}, '
          f'FDA clip frac mean={np.mean(clip_frac_log):.4f} '
          f'(>0.1 是警戒, 说明 texture 扰动过强)')

    # 7. codex fix #2: 计算低 norm channel mask
    # 用 F_orig 的 per-channel norm (over samples + space) 作判据
    channel_energy = F_orig.pow(2).sum(dim=(0, 2, 3)).sqrt()  # [256]
    threshold = torch.quantile(channel_energy, args.low_norm_pct / 100.0)
    low_norm_mask = channel_energy < threshold
    print(f'[probe] low-norm channels (bottom {args.low_norm_pct:.0f}% by energy): '
          f'{low_norm_mask.sum().item()}/256 excluded '
          f'(threshold={threshold.item():.4f})')

    # 8. 敏感度计算 (pooled, 多 donor 平均)
    shape_sens, _, _ = pooled_sensitivity(F_orig, F_shape, low_norm_mask)
    # texture: 多个 donor 的 pooled sensitivity 平均
    tex_sens_list = []
    for Ft in F_tex_all:
        ts, _, _ = pooled_sensitivity(F_orig, Ft, low_norm_mask)
        tex_sens_list.append(ts)
    tex_sens = torch.stack(tex_sens_list, dim=0).nanmean(dim=0)
    tex_sens_std = torch.stack(tex_sens_list, dim=0).std(dim=0)

    shape_sens_np = shape_sens.numpy()
    tex_sens_np = tex_sens.numpy()
    valid = ~(np.isnan(shape_sens_np) | np.isnan(tex_sens_np))

    # 9. 统计量
    pearson = np.corrcoef(shape_sens_np[valid], tex_sens_np[valid])[0, 1]
    # codex r6: 主 pearson NaN 几乎不可能 (需要 encoder 完全坍缩), 但兜底一下
    if not np.isfinite(pearson):
        raise RuntimeError(
            f'main pearson is NaN/inf — sensitivity vectors degenerate. '
            f'shape_sens std={shape_sens_np[valid].std():.4f}, '
            f'tex_sens std={tex_sens_np[valid].std():.4f}. '
            f'Likely encoder collapsed or all channels saturated.'
        )
    shape_dominant = ((shape_sens_np > tex_sens_np) & valid).sum()
    tex_dominant = ((tex_sens_np > shape_sens_np) & valid).sum()

    # codex fix r4: Hierarchical bootstrap 严格匹配采样 design
    # (1) Stratified by cid: 每 client domain 内部独立重采样, 保持 balanced design
    # (2) Nested by image: 每张图重采样自己的 donor multiplicities (不是全局 slot)
    # (3) Per-replicate low_norm_mask: 每个 bootstrap iteration 内重算 mask
    rng_np = np.random.default_rng(args.seed)
    N = F_orig.shape[0]
    n_donors = len(F_tex_all)
    sample_cids = np.array([s['cid'] for s in all_samples])  # 每个原始 sample 的 cid
    # 先建 per-cid 索引池
    cid_pools = {cid: np.where(sample_cids == cid)[0] for cid in np.unique(sample_cids)}

    # F_tex_stack: [n_donors, N, 256, 7, 7], 用于 nested donor lookup
    F_tex_stack = torch.stack(F_tex_all, dim=0)

    boot_pearsons = []
    for boot_i in range(500):
        # (1) Stratified image resampling: 每 cid 内部 resample, 然后 concat
        img_idx_parts = []
        for cid_val, pool in cid_pools.items():
            chosen = rng_np.choice(pool, size=len(pool), replace=True)
            img_idx_parts.append(chosen)
        img_idx = np.concatenate(img_idx_parts)
        n_b = len(img_idx)

        F_orig_b = F_orig[img_idx]
        F_shape_b = F_shape[img_idx]

        # (2) Nested donor: 每张图独立重抽 n_donors 个 donor index
        # codex fix r5: bootstrap estimator 必须 mirror point estimate
        # point estimate: tex_sens = mean_d pooled_sens(F_orig, F_tex_d)
        # 所以 bootstrap 也必须先 per-donor-slot 算 sensitivity, 再 mean over slots
        donor_idx_matrix = rng_np.choice(n_donors, size=(n_b, n_donors), replace=True)

        # (3) Per-replicate low_norm_mask (基于 F_orig_b)
        ch_energy_b = F_orig_b.pow(2).sum(dim=(0, 2, 3)).sqrt()
        thr_b = torch.quantile(ch_energy_b, args.low_norm_pct / 100.0)
        mask_b = ch_energy_b < thr_b

        # Shape sensitivity
        ss_b, _, _ = pooled_sensitivity(F_orig_b, F_shape_b, mask_b)

        # Texture sensitivity: per slot k, build F_tex_b_k by per-image nested donor index,
        # 然后 pooled_sens, 最后 mean over slots (mirror point estimator)
        ts_b_per_slot = []
        for k in range(n_donors):
            # 对 slot k, 每张图 j 取 donor donor_idx_matrix[j, k]
            F_tex_b_k = F_tex_stack[donor_idx_matrix[:, k], img_idx]  # [n_b, 256, 7, 7]
            ts_k, _, _ = pooled_sensitivity(F_orig_b, F_tex_b_k, mask_b)
            ts_b_per_slot.append(ts_k)
        ts_b = torch.stack(ts_b_per_slot, dim=0).nanmean(dim=0)

        ss_np = ss_b.numpy()
        ts_np = ts_b.numpy()
        v = ~(np.isnan(ss_np) | np.isnan(ts_np))
        if v.sum() < 10:
            continue
        p = np.corrcoef(ss_np[v], ts_np[v])[0, 1]
        boot_pearsons.append(p)

        if (boot_i + 1) % 100 == 0:
            print(f'  [bootstrap {boot_i+1}/500] pearson distrib so far: '
                  f'mean={np.mean(boot_pearsons):.3f}, '
                  f'std={np.std(boot_pearsons):.3f}')

    boot_pearsons = np.array(boot_pearsons)
    # codex r6: 过滤 NaN replicate, 用 nanpercentile + 验证有效样本量
    boot_finite = boot_pearsons[np.isfinite(boot_pearsons)]
    if len(boot_finite) < 100:
        raise RuntimeError(
            f'only {len(boot_finite)}/500 bootstrap replicates finite — '
            f'CI not trustworthy. Likely too many degenerate sensitivity vectors.'
        )
    pearson_ci_low = np.percentile(boot_finite, 2.5)
    pearson_ci_high = np.percentile(boot_finite, 97.5)
    print(f'\n=== sensitivity 统计 (N channel valid={valid.sum()}/256) ===')
    print(f'  shape_sens    range [{shape_sens_np[valid].min():.3f}, {shape_sens_np[valid].max():.3f}]')
    print(f'  tex_sens      range [{tex_sens_np[valid].min():.3f}, {tex_sens_np[valid].max():.3f}]')
    print(f'  pearson(shape, tex) = {pearson:.3f}  '
          f'[95% CI: {pearson_ci_low:.3f}, {pearson_ci_high:.3f}]')
    print(f'  #channels shape>tex = {shape_dominant}/{valid.sum()}')
    print(f'  #channels tex>shape = {tex_dominant}/{valid.sum()}')
    print(f'  tex_sens donor std (avg) = {tex_sens_std.nanmean().item():.4f} (越小越稳)')

    # 10. Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 主 scatter
    ax = axes[0]
    ax.scatter(shape_sens_np[valid], tex_sens_np[valid], s=15, alpha=0.6, c='tab:blue')
    mx = max(shape_sens_np[valid].max(), tex_sens_np[valid].max())
    ax.plot([0, mx], [0, mx], 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel('shape sensitivity (pooled rel L2)')
    ax.set_ylabel('texture sensitivity (pooled rel L2, mean over donors)')
    ax.set_title(
        f'Per-channel sensitivity (N_valid={valid.sum()}/256)\n'
        f'pearson={pearson:.3f}  shape>tex: {shape_dominant}, tex>shape: {tex_dominant}'
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # 绝对变化量 cross-validation (codex fix #2 建议同时 report abs)
    diff_shape = (F_orig - F_shape).pow(2).sum(dim=(0, 2, 3)).sqrt().numpy()
    diff_tex = torch.stack([
        (F_orig - Ft).pow(2).sum(dim=(0, 2, 3)).sqrt() for Ft in F_tex_all
    ], dim=0).mean(dim=0).numpy()
    ax = axes[1]
    ax.scatter(diff_shape[valid], diff_tex[valid], s=15, alpha=0.6, c='tab:green')
    mx2 = max(diff_shape[valid].max(), diff_tex[valid].max())
    ax.plot([0, mx2], [0, mx2], 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel('shape diff (pooled abs L2)')
    ax.set_ylabel('texture diff (pooled abs L2)')
    p_abs = np.corrcoef(diff_shape[valid], diff_tex[valid])[0, 1]
    ax.set_title(f'Absolute diff (cross-validation)\npearson={p_abs:.3f}')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    png = os.path.join(args.out, 'shape_tex_sens_scatter.png')
    plt.savefig(png, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[probe] saved {png}')

    # dump
    np.savez(
        os.path.join(args.out, 'shape_tex_sens_data.npz'),
        shape_sens=shape_sens_np, tex_sens=tex_sens_np,
        tex_sens_std=tex_sens_std.numpy(),
        diff_shape=diff_shape, diff_tex=diff_tex,
        low_norm_mask=low_norm_mask.numpy(),
        channel_energy=channel_energy.numpy(),
        clip_frac_log=np.array(clip_frac_log),
        pearson=pearson,
        pearson_ci=np.array([pearson_ci_low, pearson_ci_high]),
        boot_pearsons=boot_pearsons,
    )

    # verdict — 使用 95% CI 的上界做保守判决
    print('\n=== verdict ===')
    if pearson_ci_low > 0.85:
        print(f'  ❌ pearson 95% CI 下界 {pearson_ci_low:.3f} > 0.85, '
              f'channels 几乎同等敏感 → 假设失败, FedShapeGate 作废')
    elif pearson_ci_high < 0.5:
        print(f'  ✅ pearson 95% CI 上界 {pearson_ci_high:.3f} < 0.5, '
              f'channel 明显分离 → 假设成立, 可开发 FedShapeGate')
    else:
        print(f'  🟡 pearson 95% CI [{pearson_ci_low:.3f}, {pearson_ci_high:.3f}] 跨中间区, '
              f'证据不足 → 建议: 增 N_per_domain, 换 ckpt, 或 sweep elastic α/β')


if __name__ == '__main__':
    main()
