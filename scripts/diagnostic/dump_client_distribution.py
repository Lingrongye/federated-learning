#!/usr/bin/env python3
"""
F2DC client 分布诊断脚本
=========================
不需要训练任何模型,只用 F2DC 的真实 dataloader 复现 EXP-130 的 client 分布.
seed 固定 → 分布完全确定.

目的:
1. 验证 review M1 (batch-mean EMA 在小 class count 时方差爆炸)
   → 看每 batch per-class 样本数分布
2. 验证 review M2 (sample-weighted aggregation 不消除 domain skew)
   → 看各 client 训练集大小差异 + 各 client 跨 class 不均衡

使用 (在 sc3 上跑):
    cd /root/autodl-tmp/federated-learning/F2DC
    python ../scripts/diagnostic/dump_client_distribution.py \
        --datasets pacs office digits \
        --seeds 2 15 333 \
        --out client_distribution.json

输出: JSON 报告 + 关键决策标准
"""
import os, sys, json, argparse
from argparse import Namespace
from collections import Counter
import numpy as np

# F2DC 仓库根目录(脚本应放在 F2DC 同级 scripts/diagnostic/ 下)
F2DC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../F2DC'))
if not os.path.exists(F2DC_DIR):
    # fallback: 假设脚本被 cd 到 F2DC/ 后调用
    F2DC_DIR = os.getcwd()

# 注意:F2DC/datasets/utils 也叫 utils,如果把 F2DC/datasets 加到 sys.path 头部,
# 会让 `from utils.conf` 解析成 datasets/utils(错!)。所以只加 F2DC_DIR 就够了,
# datasets/backbone/models 都通过 package 路径访问。
sys.path.insert(0, F2DC_DIR)

from datasets import get_prive_dataset
from utils.conf import set_random_seed


# ============================================================================
# 跟 EXP-130 完全一致的配置
# ============================================================================
DATASET_CONFIGS = {
    'pacs': dict(
        dataset_arg='fl_pacs',
        parti_num=10,
        local_batch_size=46,   # F2DC best_args 默认
        domains=['photo', 'art', 'cartoon', 'sketch'],
        num_classes=7,
    ),
    'office': dict(
        dataset_arg='fl_officecaltech',
        parti_num=10,
        local_batch_size=64,
        domains=['caltech', 'amazon', 'webcam', 'dslr'],
        num_classes=10,
    ),
    'digits': dict(
        dataset_arg='fl_digits',
        parti_num=20,
        local_batch_size=64,
        domains=['mnist', 'usps', 'svhn', 'syn'],
        num_classes=10,
    ),
}


def reproduce_selected_domain_list(args):
    """
    完全复制 utils/training.py:train() 里的 selected_domain_list 生成逻辑.
    必须在 set_random_seed 之后调用.
    """
    domains_list = DATASET_CONFIGS[args._ds_short]['domains']
    domains_len = len(domains_list)
    max_num = 10
    is_ok = False
    while not is_ok:
        if args.dataset == "fl_officecaltech":
            selected = np.random.choice(domains_list,
                                         size=args.parti_num - domains_len,
                                         replace=True, p=None)
            selected_domain_list = list(selected) + domains_list
        elif args.dataset == "fl_digits":
            selected_domain_list = list(np.random.choice(
                domains_list, size=args.parti_num, replace=True, p=None))
        elif args.dataset == "fl_pacs":
            selected_domain_list = list(np.random.choice(
                domains_list, size=args.parti_num, replace=True, p=None))

        result = dict(Counter(selected_domain_list))
        for k in result:
            if result[k] > max_num:
                is_ok = False
                break
        else:
            is_ok = True
    return selected_domain_list


def analyze_client_distribution(loader, n_classes):
    """
    遍历 loader,统计:
    - n_total: client 总样本数
    - class_counts: 每类样本数
    - per_batch_class_stats: 每 batch 内各类出现次数(单 batch dict list)
    - single_sample_freq: 单 batch 单 class 只 0-1 个样本的比例
    """
    class_counts = Counter()
    n_total = 0
    per_batch_stats = []

    for batch_idx, (data, target) in enumerate(loader):
        n_total += len(target)
        target_list = target.tolist()
        batch_count = Counter(target_list)
        per_batch_stats.append(dict(batch_count))
        for c in target_list:
            class_counts[c] += 1

    # 单 batch 单 class 样本数统计
    n_single_or_zero_pairs = 0       # 单 batch 内某 class 只 0-1 个样本的次数
    n_total_class_pairs = 0
    for batch in per_batch_stats:
        for c in range(n_classes):
            count = batch.get(c, 0)
            n_total_class_pairs += 1
            if count <= 1:
                n_single_or_zero_pairs += 1

    # per-class 在多少 batch 出现 ≥ 2 次
    class_appears_2plus = {c: 0 for c in range(n_classes)}
    for batch in per_batch_stats:
        for c in range(n_classes):
            if batch.get(c, 0) >= 2:
                class_appears_2plus[c] += 1

    return {
        'n_total': n_total,
        'class_counts': dict(class_counts),
        'n_classes_present': len([c for c, v in class_counts.items() if v > 0]),
        'class_distribution_ratios': {
            c: class_counts.get(c, 0) / max(n_total, 1) for c in range(n_classes)
        },
        'n_batches': len(per_batch_stats),
        'avg_batch_size': n_total / max(len(per_batch_stats), 1),
        'single_sample_pair_ratio': n_single_or_zero_pairs / max(n_total_class_pairs, 1),
        'class_appears_2plus_ratio': {
            c: class_appears_2plus[c] / max(len(per_batch_stats), 1)
            for c in range(n_classes)
        },
        'min_class_count': min(class_counts.values()) if class_counts else 0,
        'max_class_count': max(class_counts.values()) if class_counts else 0,
    }


def make_args(ds_name, seed, model='f2dc'):
    """构造 namespace,跟 EXP-130 一致"""
    cfg = DATASET_CONFIGS[ds_name]
    args = Namespace(
        dataset=cfg['dataset_arg'],
        parti_num=cfg['parti_num'],
        local_batch_size=cfg['local_batch_size'],
        local_epoch=10,
        local_lr=0.01,
        seed=seed,
        rand_dataset=True,
        model=model,
        device_id=0,
        communication_epoch=100,
        online_ratio=1.0,
        averaing='weight',
        learning_decay=False,
        pri_aug='weak',
        save=False,
        save_name='No',
        structure='heterogeneity',
        ma_select='resnet',
        # PG-DFC 关心的方法超参,这里随便填(只为构造 dataset)
        gum_tau=0.1,
        tem=0.06,
        agg_a=1.0,
        agg_b=0.4,
        lambda1=0.8,
        lambda2=1.0,
    )
    args._ds_short = ds_name
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['pacs', 'office', 'digits'],
                        help='哪些数据集')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2, 15, 333])
    parser.add_argument('--out', type=str, default='client_distribution.json')
    args_cli = parser.parse_args()

    all_results = {}
    print("=" * 80)
    print("F2DC client distribution dump")
    print(f"Datasets: {args_cli.datasets}")
    print(f"Seeds:    {args_cli.seeds}")
    print("=" * 80)

    for ds_name in args_cli.datasets:
        for seed in args_cli.seeds:
            key = f"{ds_name}_seed{seed}"
            print(f"\n[{key}] starting...")

            # 1. 设置 seed
            set_random_seed(seed)

            # 2. 构造 args + 加载 dataset
            args = make_args(ds_name, seed)
            try:
                priv_dataset = get_prive_dataset(args)
            except Exception as e:
                print(f"  ❌ get_prive_dataset failed: {e}")
                all_results[key] = {'error': str(e)}
                continue

            # 3. 复现 selected_domain_list (set_random_seed 后立即调,跟 train() 同 seed 状态)
            selected_domain_list = reproduce_selected_domain_list(args)
            print(f"  selected_domain_list: {selected_domain_list}")

            # 4. 拿 loaders
            try:
                train_loaders, _ = priv_dataset.get_data_loaders(selected_domain_list)
            except Exception as e:
                print(f"  ❌ get_data_loaders failed: {e}")
                all_results[key] = {'error': str(e)}
                continue

            # 5. 分析每 client 分布
            cfg = DATASET_CONFIGS[ds_name]
            client_stats = []
            for client_id, loader in enumerate(train_loaders):
                domain = selected_domain_list[client_id]
                stats = analyze_client_distribution(loader, cfg['num_classes'])
                stats['client_id'] = client_id
                stats['domain'] = domain
                client_stats.append(stats)
                print(f"    Client {client_id} ({domain}): "
                      f"n_total={stats['n_total']}, "
                      f"single_sample_freq={stats['single_sample_pair_ratio']:.3f}, "
                      f"min/max_class={stats['min_class_count']}/{stats['max_class_count']}")

            # 6. 跨 client 汇总
            n_total_per_client = [s['n_total'] for s in client_stats]
            single_freq_per_client = [s['single_sample_pair_ratio'] for s in client_stats]
            summary = {
                'mean_client_size': float(np.mean(n_total_per_client)),
                'std_client_size': float(np.std(n_total_per_client)),
                'min_client_size': int(min(n_total_per_client)),
                'max_client_size': int(max(n_total_per_client)),
                'client_size_skew_ratio': float(max(n_total_per_client) / max(min(n_total_per_client), 1)),
                'mean_single_sample_freq': float(np.mean(single_freq_per_client)),
                'max_single_sample_freq': float(max(single_freq_per_client)),
            }

            all_results[key] = {
                'config': dict(
                    dataset=ds_name, seed=seed,
                    parti_num=cfg['parti_num'],
                    batch_size=cfg['local_batch_size'],
                    n_classes=cfg['num_classes'],
                ),
                'selected_domain_list': selected_domain_list,
                'client_stats': client_stats,
                'summary': summary,
            }

            print(f"  Summary:")
            print(f"    Client size: mean={summary['mean_client_size']:.0f}, "
                  f"std={summary['std_client_size']:.0f}, "
                  f"skew_ratio={summary['client_size_skew_ratio']:.2f}x")
            print(f"    Single-sample-freq: mean={summary['mean_single_sample_freq']:.3f}, "
                  f"max={summary['max_single_sample_freq']:.3f}")

    # 7. 输出 JSON
    with open(args_cli.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Dumped to {args_cli.out}")

    # 8. 决策报告 — per-seed 详表
    print("\n" + "=" * 80)
    print("Per-seed 详表 (每个 (dataset, seed) 一行)")
    print("=" * 80)
    print(f"{'Dataset_Seed':<20} {'M1 (single_sample)':<25} {'M2 (size_skew)':<20} {'M1 决策':<25} {'M2 决策'}")
    print("-" * 130)
    for key, data in all_results.items():
        if 'error' in data:
            print(f"{key:<20} ERROR: {data['error']}")
            continue
        s = data['summary']
        m1_freq = s['mean_single_sample_freq']
        m2_skew = s['client_size_skew_ratio']

        if m1_freq < 0.05:
            m1_decide = "EMA OK"
        elif m1_freq > 0.20:
            m1_decide = "★ 必须改 sample 累加"
        else:
            m1_decide = "EMA + n>=2 保护"

        if m2_skew < 1.5:
            m2_decide = "sample-weighted OK"
        elif m2_skew > 3.0:
            m2_decide = "★ 必须 L2-norm 等权"
        else:
            m2_decide = "可加 cap"

        print(f"{key:<20} {m1_freq:<25.3f} {m2_skew:<20.2f} {m1_decide:<25} {m2_decide}")

    # 9. 决策报告 — per-dataset 3-seed mean (最终决策依据)
    print("\n" + "=" * 80)
    print("★ Per-dataset 3-seed mean 决策报告 (最终 M1/M2 fix 决策依据)")
    print("=" * 80)
    print(f"{'Dataset':<10} {'M1 mean±std':<20} {'M2 mean±std':<20} {'M1 最终决策':<25} {'M2 最终决策'}")
    print("-" * 120)

    # 按 dataset 聚合 3 seed
    per_ds_agg = {}
    for key, data in all_results.items():
        if 'error' in data:
            continue
        ds = data['config']['dataset']
        if ds not in per_ds_agg:
            per_ds_agg[ds] = {'m1': [], 'm2': []}
        per_ds_agg[ds]['m1'].append(data['summary']['mean_single_sample_freq'])
        per_ds_agg[ds]['m2'].append(data['summary']['client_size_skew_ratio'])

    for ds, vals in per_ds_agg.items():
        m1_arr = np.array(vals['m1'])
        m2_arr = np.array(vals['m2'])
        m1_mean, m1_std = m1_arr.mean(), m1_arr.std()
        m2_mean, m2_std = m2_arr.mean(), m2_arr.std()

        # 最终决策(看 mean,但任一 seed > 阈值也升级警报)
        if m1_mean < 0.05 and m1_arr.max() < 0.10:
            m1_final = "EMA OK"
        elif m1_mean > 0.20 or m1_arr.max() > 0.30:
            m1_final = "★ 必须改 sample 累加"
        else:
            m1_final = "EMA + n>=2 保护"

        if m2_mean < 1.5 and m2_arr.max() < 2.0:
            m2_final = "sample-weighted OK"
        elif m2_mean > 3.0 or m2_arr.max() > 5.0:
            m2_final = "★ 必须 L2-norm 等权"
        else:
            m2_final = "可加 cap"

        print(f"{ds:<10} "
              f"{m1_mean:.3f}±{m1_std:.3f}      "
              f"{m2_mean:.2f}±{m2_std:.2f}        "
              f"{m1_final:<25} {m2_final}")
    print("=" * 80)
    print("\n★ 决策准则:")
    print("  M1 — 看 single_sample_pair_ratio (单 batch 内某类只 0-1 样本的比例)")
    print("       mean<0.05 且 max<0.10 → EMA 可用")
    print("       mean>0.20 或 max>0.30 → 必须改 sample 累加")
    print("       中间 → EMA + n>=2 保护")
    print("  M2 — 看 client_size_skew_ratio (max/min client 样本数比)")
    print("       mean<1.5 且 max<2.0 → sample-weighted 可用")
    print("       mean>3.0 或 max>5.0 → 必须 L2-normalize + 等权")
    print("       中间 → sample-weighted 加 cap")
    print("  注: 任一 seed 突破上限即升级警报 (max 检查),防止某 seed 偶然均匀掩盖问题")


if __name__ == '__main__':
    main()
