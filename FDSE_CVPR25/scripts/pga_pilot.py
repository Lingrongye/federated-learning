"""PGA Pilot: Prototype-Guided Attribution at feature map level.

验证关键问题: 跨 client 聚合的 class 原型能不能在 Sketch 上
把"线条 pixel"正确识别为 class-relevant (频率方法在这里失败).

流程:
1. 加载现有 sgpa ckpt + flgo 数据
2. forward hook 抓每个样本的 conv5 feature map [256, 6, 6]
3. per class 平均 pool 得到 [7, 256] class prototypes
4. 对 4 domain 各一张 guitar:
   - α[h, w] = max_c cos(feature_map[:, h, w], P_c)
   - upsample α 到原图尺寸, overlay 可视化
5. 同时做 freq baseline 对比

用法:
    python pga_pilot.py --ckpt /root/fl_checkpoints/xxx/ --task PACS_c4 \
        --out /tmp/pga_pilot --cls_idx 3  # guitar = class 3 in PACS
"""
import argparse, os, sys, glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FDSE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FDSE_ROOT))

import flgo

# PACS class index (按 flgo 构建时的字母序)
PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_RAW = '/root/autodl-tmp/federated-learning/PFLlib/dataset/PACS/rawdata/PACS'


def detect_algo(ckpt_dir: Path):
    gs = torch.load(ckpt_dir / 'global_model.pt', map_location='cpu')
    if 'semantic_head.mu_head.0.weight' in gs:
        return 'vib', 1
    return 'sgpa', 0


class FeatureGrabber:
    """Hook to grab conv5 feature map and resize to fixed 6x6 for consistency."""
    TARGET_HW = (6, 6)

    def __init__(self):
        self.feat = None
        self.handle = None

    def hook(self, module, inp, out):
        # Always resize to [B, 256, 6, 6] to be consistent across input sizes
        self.feat = F.adaptive_avg_pool2d(out.detach(), self.TARGET_HW)

    def attach(self, encoder):
        self.handle = encoder.features.register_forward_hook(self.hook)

    def detach(self):
        if self.handle:
            self.handle.remove()


def build_prototypes(clients, grabber, device, num_classes=7):
    """Iterate all clients' train data, collect per-class avg pooled features.

    Returns: P ∈ [num_classes, 256] — class prototypes in channel space.
    """
    sums = torch.zeros(num_classes, 256, device=device)
    cnts = torch.zeros(num_classes, device=device)

    for cid, c in enumerate(clients):
        data = c.train_data
        loader = c.calculator.get_dataloader(data, batch_size=c.batch_size)
        c.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = c.calculator.to_device(batch)
                x, y = batch[0], batch[-1]
                _ = c.model.encode(x)            # triggers hook
                feat = grabber.feat              # [B, 256, H, W]
                pooled = feat.mean(dim=(2, 3))   # [B, 256] global avg pool
                for cls in range(num_classes):
                    mask = (y == cls)
                    if mask.any():
                        sums[cls] += pooled[mask].sum(0)
                        cnts[cls] += mask.sum()
        print(f'[proto] client {cid} done')

    P = sums / cnts.unsqueeze(1).clamp(min=1)    # [num_classes, 256]
    print(f'[proto] counts per class: {cnts.cpu().numpy()}')
    return P


def compute_attribution(feat_map, P):
    """DEPRECATED: global-pooled prototype version.
    Kept for compatibility. Use compute_attribution_spatial instead.
    """
    C, H, W = feat_map.shape
    f_flat = feat_map.reshape(C, -1).T
    f_norm = F.normalize(f_flat, dim=1)
    P_norm = F.normalize(P, dim=1)
    sim = f_norm @ P_norm.T
    alpha_flat, _ = sim.max(dim=1)
    alpha = alpha_flat.reshape(H, W)
    a_min = alpha.min(); a_max = alpha.max()
    alpha_norm = (alpha - a_min) / (a_max - a_min + 1e-8)
    return alpha_norm.cpu().numpy(), alpha.cpu().numpy()


def compute_attribution_spatial(feat_map, P):
    """Spatial-aware attribution.
    feat_map: [C, H, W], P: [num_classes, C, H, W]
    At each (h, w), compute cos between feat_map[:, h, w] and each P[c, :, h, w].
    Returns: α ∈ [H, W] with values in [0, 1].
    """
    C, H, W = feat_map.shape
    num_cls = P.shape[0]

    # Normalize per-location vectors along channel dim
    # feat_map: [C, H, W] → [H, W, C]
    f = feat_map.permute(1, 2, 0)                      # [H, W, C]
    f_norm = F.normalize(f, dim=-1)

    # P: [num_cls, C, H, W] → [num_cls, H, W, C]
    P_r = P.permute(0, 2, 3, 1)                        # [num_cls, H, W, C]
    P_norm = F.normalize(P_r, dim=-1)

    # For each (h, w) and class c: cos = f_norm[h,w] · P_norm[c,h,w]
    # Broadcast: [H, W, C] vs [num_cls, H, W, C] → sum over C
    sim = (f_norm.unsqueeze(0) * P_norm).sum(dim=-1)   # [num_cls, H, W]
    alpha, argmax_cls = sim.max(dim=0)                 # [H, W]

    a_min = alpha.min(); a_max = alpha.max()
    alpha_norm = (alpha - a_min) / (a_max - a_min + 1e-8)
    return alpha_norm.cpu().numpy(), alpha.cpu().numpy()


def load_pacs_image_matching_train(domain, cls, pick_idx=0):
    """加载图像并用与训练 **完全一致** 的 preprocessing:
       Resize([256, 256]) → PILToTensor() → float / 255.0 → [0, 1] range。
       NO ImageNet normalize. NO 224×224 resize.
    """
    from torchvision import transforms as T
    folder = os.path.join(PACS_RAW, domain, cls)
    files = sorted(glob.glob(os.path.join(folder, '*.jpg')) +
                   glob.glob(os.path.join(folder, '*.png')))
    path = files[pick_idx % len(files)]
    img_pil = Image.open(path)
    # 跟 PACS_c4/config.py __getitem__ 里完全一致
    if len(img_pil.split()) != 3:
        img_pil = T.Grayscale(num_output_channels=3)(img_pil)
    # 先存一份用于可视化的 256×256 numpy (uint8)
    img_pil_256 = img_pil.resize((256, 256), Image.BILINEAR)
    img_np = np.array(img_pil_256.convert('RGB'))
    # 构造模型输入: 完全匹配训练
    train_tf = T.Compose([
        T.Resize([256, 256]),
        T.PILToTensor(),                 # uint8
    ])
    x = train_tf(img_pil).float() / 255.0   # → [0, 1]
    x = x.unsqueeze(0)                        # [1, 3, 256, 256]
    return img_np, x, path


def freq_attribution(img_np, sigma_ratio=0.1):
    """Low-freq mask baseline, returns α ∈ [H, W] in [0, 1]."""
    gray = img_np.mean(axis=2)
    H, W = gray.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    sigma = sigma_ratio * min(H, W)
    mask = np.exp(-r2 / (2 * sigma**2))     # low-freq Gaussian
    # α interpretation: high-α = class-relevant.
    # 按频率方法: low-freq energy = class signal. 但这里 mask 是 freq-domain,
    # 要得到 pixel-domain attribution, 用 ifft of low-pass img:
    F_img = np.fft.fftshift(np.fft.fft2(gray))
    low = np.fft.ifft2(np.fft.ifftshift(F_img * mask)).real
    # α = 低频能量归一化
    low_abs = np.abs(low)
    low_norm = (low_abs - low_abs.min()) / (low_abs.max() - low_abs.min() + 1e-8)
    return low_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--task', default='PACS_c4')
    ap.add_argument('--out', default='/tmp/pga_pilot')
    ap.add_argument('--cls_idx', type=int, default=3, help='guitar = 3')
    ap.add_argument('--pick_idx', type=int, default=0,
                    help='which image to pick per domain folder')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = Path(args.ckpt)

    # 1. Algo detection + flgo init
    algo_name, vib_flag = detect_algo(ckpt_dir)
    print(f'[pga] algo={algo_name}, vib={vib_flag}')
    if algo_name == 'vib':
        import algorithm.feddsa_sgpa_vib as algo_mod
    else:
        import algorithm.feddsa_sgpa as algo_mod

    task_path = f'./task/{args.task}'
    option = {'num_rounds': 0, 'proportion': 1.0, 'seed': 2,
              'gpu': [args.gpu], 'load_mode': '', 'num_parallels': 1}
    if vib_flag:
        option['algo_para'] = [1, 0]
    runner = flgo.init(task=task_path, algorithm=algo_mod, option=option)
    server = runner
    clients = server.clients

    # 2. Load ckpt
    global_state = torch.load(ckpt_dir / 'global_model.pt', map_location=device)
    server.model.load_state_dict(global_state, strict=False)
    client_states = torch.load(ckpt_dir / 'client_models.pt', map_location=device)
    for cid, c in enumerate(clients):
        if isinstance(client_states, list) and cid < len(client_states):
            c.model.load_state_dict(client_states[cid], strict=False)
        c.model = c.model.to(device)
        c.model.eval()

    # 3. Build PER-LOCATION class prototypes: P ∈ [num_classes, 256, H, W]
    # Fixes Bug 2: spatial-aware prototype, not globally pooled.
    print('[pga] building PER-LOCATION class prototypes from all clients train data...')
    num_classes = server.model.num_classes
    # Determine feature map spatial size with one forward
    test_batch = next(iter(clients[0].calculator.get_dataloader(
        clients[0].train_data, batch_size=2)))
    test_batch = clients[0].calculator.to_device(test_batch)
    grabber_probe = FeatureGrabber()
    grabber_probe.attach(clients[0].model.encoder)
    with torch.no_grad():
        _ = clients[0].model.encode(test_batch[0])
    _, C_feat, H_feat, W_feat = grabber_probe.feat.shape
    grabber_probe.detach()
    print(f'[pga] feature map shape: [{C_feat}, {H_feat}, {W_feat}]')

    sums = torch.zeros(num_classes, C_feat, H_feat, W_feat, device=device)
    cnts = torch.zeros(num_classes, device=device)
    for cid, c in enumerate(clients):
        grabber = FeatureGrabber()
        grabber.attach(c.model.encoder)
        data = c.train_data
        loader = c.calculator.get_dataloader(data, batch_size=c.batch_size)
        with torch.no_grad():
            for batch in loader:
                batch = c.calculator.to_device(batch)
                x, y = batch[0], batch[-1]
                _ = c.model.encode(x)
                feat = grabber.feat                        # [B, C, H, W]
                for cls in range(num_classes):
                    mask = (y == cls)
                    if mask.any():
                        sums[cls] += feat[mask].sum(0)     # [C, H, W]
                        cnts[cls] += mask.sum()
        grabber.detach()
        print(f'[pga] client {cid} done')

    P = sums / cnts.view(-1, 1, 1, 1).clamp(min=1)         # [num_classes, C, H, W]
    print(f'[pga] per-class sample counts: {cnts.cpu().tolist()}')

    # 4. For each domain, use that domain's OWN client encoder (fixes Bug 1: FedBN)
    # PACS_c4 client order: [art_painting=0, cartoon=1, photo=2, sketch=3]
    # 验证 client→domain 映射: 看 client k 的 train_data 里第一张图的路径是否含 domain 名
    print('[pga] verifying client→domain mapping...')
    for cid in range(len(DOMAINS)):
        c = clients[cid]
        # 拿第一个 batch 看 dataset 的 images_path
        ds = c.train_data
        # flgo 的 train_data 可能是 Subset, 往下钻找原始 dataset
        probe = ds
        while hasattr(probe, 'dataset'):
            probe = probe.dataset
        if hasattr(probe, 'images_path') and probe.images_path:
            first_path = probe.images_path[0]
            expected_domain = DOMAINS[cid]
            if expected_domain not in first_path:
                print(f'  ⚠️ client {cid}: expected {expected_domain} but path={first_path}')
            else:
                print(f'  ✓ client {cid}: {expected_domain} confirmed ({os.path.basename(os.path.dirname(first_path))})')
        else:
            print(f'  ? client {cid}: cannot verify (no images_path attr)')

    viz_data = {}
    for cid, domain in enumerate(DOMAINS):
        domain_client = clients[cid]                        # use matching client
        grabber = FeatureGrabber()
        grabber.attach(domain_client.model.encoder)

        img_np, x, path = load_pacs_image_matching_train(
            domain, PACS_CLASSES[args.cls_idx], pick_idx=args.pick_idx)
        x = x.to(device)
        with torch.no_grad():
            _ = domain_client.model.encode(x)
        feat_map = grabber.feat[0]                          # [C, H, W]
        alpha_norm, alpha_raw = compute_attribution_spatial(feat_map, P)
        # Freq baseline
        alpha_freq = freq_attribution(img_np, sigma_ratio=0.10)
        viz_data[domain] = {
            'img': img_np, 'alpha_pga': alpha_norm,
            'alpha_raw': alpha_raw, 'alpha_freq': alpha_freq,
            'path': path, 'feat_shape': tuple(feat_map.shape),
        }
        print(f'[pga] {domain} (client {cid}): feat_shape={feat_map.shape}, '
              f'alpha raw range [{alpha_raw.min():.3f}, {alpha_raw.max():.3f}]')
        grabber.detach()

    # 5. Visualize 4 × 4 grid: orig | PGA heatmap overlay | PGA binary top-25% | freq baseline
    fig, axes = plt.subplots(len(DOMAINS), 4, figsize=(16, 4*len(DOMAINS)))
    cls_name = PACS_CLASSES[args.cls_idx]
    for i, d in enumerate(DOMAINS):
        vd = viz_data[d]
        img = vd['img']
        H_img, W_img = img.shape[:2]

        # Upsample α to image size
        alpha = torch.from_numpy(vd['alpha_pga']).unsqueeze(0).unsqueeze(0).float()
        alpha_up = F.interpolate(alpha, size=(H_img, W_img),
                                  mode='bilinear', align_corners=False)
        alpha_up = alpha_up.squeeze().numpy()

        # Col 0: original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'{d} (orig)')
        axes[i, 0].axis('off')

        # Col 1: PGA heatmap overlay
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(alpha_up, cmap='jet', alpha=0.5)
        axes[i, 1].set_title(f'{d} PGA attribution\n'
                             f'(raw range {vd["alpha_raw"].min():.2f}-{vd["alpha_raw"].max():.2f})')
        axes[i, 1].axis('off')

        # Col 2: binary mask top-25%
        thresh = np.percentile(alpha_up, 75)
        mask_bin = (alpha_up > thresh).astype(np.float32)
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(mask_bin, cmap='Reds', alpha=0.6)
        axes[i, 2].set_title(f'{d} PGA top-25% regions')
        axes[i, 2].axis('off')

        # Col 3: freq baseline (low-freq energy)
        axes[i, 3].imshow(img)
        axes[i, 3].imshow(vd['alpha_freq'], cmap='jet', alpha=0.5)
        axes[i, 3].set_title(f'{d} FREQ low-pass baseline')
        axes[i, 3].axis('off')

    plt.suptitle(f'PGA vs FREQ attribution — class={cls_name}', fontsize=14)
    plt.tight_layout()
    png = os.path.join(args.out, f'pga_vs_freq_{cls_name}.png')
    plt.savefig(png, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'[pga] saved {png}')

    # Dump prototypes and alpha for later inspection
    np.savez(os.path.join(args.out, f'pga_data_{cls_name}.npz'),
             P=P.cpu().numpy(),
             **{f'alpha_{d}': viz_data[d]['alpha_raw'] for d in DOMAINS})


if __name__ == '__main__':
    main()
