"""验证"低频=结构/形状, 高频=纹理/笔触"的频率分解直觉.

输入: PACS 4 domain 的同 class 图片 (guitar)
做法: 对每张图的灰度版做 2D FFT → 低通/高通 mask → IFFT → 可视化
输出: 3 列 (原图 / 低频 / 高频) × 4 行 (4 domain) 的对比图
"""
import argparse, os, glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gaussian_mask(shape, sigma_ratio):
    """中心 Gaussian 低通 mask. sigma_ratio 是 sigma/min(H,W) 比例."""
    H, W = shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy)**2 + (xx - cx)**2
    sigma = sigma_ratio * min(H, W)
    return np.exp(-r2 / (2 * sigma**2))


def fft_filter(img_gray, mask):
    """对灰度图做 FFT → mask → IFFT, 返回实数重建图."""
    F = np.fft.fftshift(np.fft.fft2(img_gray))
    F_filtered = F * mask
    recon = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real
    return recon


def load_image(path, size=256):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size))
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/autodl-tmp/federated-learning/PFLlib/dataset/PACS/rawdata/PACS')
    ap.add_argument('--cls', default='guitar', help='class name')
    ap.add_argument('--out', default='/tmp/freq_check')
    ap.add_argument('--sigma_ratios', default='0.05,0.10,0.20',
                    help='list of sigma ratios to try')
    ap.add_argument('--size', type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    sigma_list = [float(s) for s in args.sigma_ratios.split(',')]

    # 每个 domain 随机挑 1 张 guitar
    imgs = {}
    for d in domains:
        folder = os.path.join(args.root, d, args.cls)
        files = sorted(glob.glob(os.path.join(folder, '*.jpg')) +
                       glob.glob(os.path.join(folder, '*.png')))
        if not files:
            print(f'WARN: no images in {folder}')
            continue
        # 取第一张
        imgs[d] = load_image(files[0], size=args.size)
        print(f'loaded {d}/{args.cls}/{os.path.basename(files[0])}')

    # 对每个 sigma_ratio 画一张图
    for sigma_r in sigma_list:
        # rows = 4 domains, cols = 原图 + 低频 + 高频
        fig, axes = plt.subplots(len(domains), 3, figsize=(9, 3*len(domains)))
        if len(domains) == 1:
            axes = axes.reshape(1, -1)

        for i, d in enumerate(domains):
            if d not in imgs:
                for j in range(3):
                    axes[i, j].axis('off')
                continue
            img_rgb = imgs[d]
            img_gray = img_rgb.mean(axis=2)           # HxW

            mask = gaussian_mask(img_gray.shape, sigma_r)

            low  = fft_filter(img_gray, mask)          # 低通保留
            high = fft_filter(img_gray, 1 - mask)      # 高通保留

            # 归一化到 [0, 255] 显示
            def norm(x):
                x = x - x.min()
                return x / (x.max() + 1e-8)

            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f'{d} (orig)'); axes[i, 0].axis('off')
            axes[i, 1].imshow(norm(low), cmap='gray')
            axes[i, 1].set_title(f'{d} LOW-freq (shape?)'); axes[i, 1].axis('off')
            axes[i, 2].imshow(norm(high), cmap='gray')
            axes[i, 2].set_title(f'{d} HIGH-freq (texture?)'); axes[i, 2].axis('off')

        plt.suptitle(f'{args.cls} · sigma_ratio={sigma_r}', fontsize=14)
        plt.tight_layout()
        png = os.path.join(args.out, f'freq_{args.cls}_sigma{sigma_r:.2f}.png')
        plt.savefig(png, dpi=110, bbox_inches='tight')
        plt.close()
        print(f'saved {png}')


if __name__ == '__main__':
    main()
