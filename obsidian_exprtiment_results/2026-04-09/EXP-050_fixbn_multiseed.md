# EXP-050 | FixBN 多seed验证

## 基本信息
- **目的**:验证 EXP-048 的 BN 修复效果是否在多 seed 下稳定成立
- **算法**:feddsa_fixbn (server 端聚合 BN running stats)
- **配置**:feddsa_exp048.yml
- **seeds**:15, 333
- **状态**:✅ 已完成

## 背景
EXP-048 (seed=2) 结果:
- Best 80.73 (vs 原版 82.24, -1.51)
- Last 76.27 (vs 原版 75.46, +0.81)
- Gap 4.46 (vs 原版 6.78, **-2.32 改善 34%**)

问题:Best 降了 1.5%,gap 缩了 2.3%。这个 trade-off 在多 seed 下是否稳定?
- 如果多 seed 均值 > 原版 80.74 → FixBN 应作为默认版本
- 如果多 seed 均值 < 原版 80.74 → FixBN 仅提供稳定性但牺牲精度

## 对比
| 方法 | seed=2 | seed=15 | seed=333 | 5-seed mean |
|---|---|---|---|---|
| FedDSA 原版 | 82.24 | 80.59 | 81.05 | 80.74 ± 1.37 |
| FedDSA FixBN | 80.73 | **待测** | **待测** | ? |

## 运行命令
```bash
nohup python run_single.py --task PACS_c4 --algorithm feddsa_fixbn --gpu 0 \
  --config ./config/pacs/feddsa_exp048.yml --logger PerRunLogger --seed 15 \
  > /tmp/exp050_s15.out 2>&1 &
nohup python run_single.py --task PACS_c4 --algorithm feddsa_fixbn --gpu 0 \
  --config ./config/pacs/feddsa_exp048.yml --logger PerRunLogger --seed 333 \
  > /tmp/exp050_s333.out 2>&1 &
```

## 结果
| seed | Best | Last | Gap |
|---|---|---|---|
| 2 (EXP-048) | 80.73 | 76.27 | 4.46 |
| **15** | **79.70** | **76.05** | **3.65** |
| **333** | **80.43** | **74.94** | **5.49** |
| **3-seed mean** | **80.29 ± 0.53** | **75.75** | **4.53** |

## 对比(3-seed: s2/15/333)
| 方法 | Best mean ± std | Last mean | Gap mean |
|---|---|---|---|
| FedDSA 原版 | 81.29 ± 0.86 | 75.46 | 5.83 |
| **FedDSA FixBN** | **80.29 ± 0.53** | **75.75** | **4.53** |
| Delta | **-1.00** | **+0.29** | **-1.30** |

## 结论
- **Best 降 1.00%**:聚合 BN running stats 导致 server model 偏向域均值,削弱单 seed 峰值
- **Last 提升 0.29%**:train/test BN 分布匹配改善了后期稳定性
- **Gap 改善 1.30** (从 5.83 → 4.53, -22%):稳定性确实提升
- **方差更小** (0.53 vs 0.86):FixBN 跨 seed 更稳定
- **trade-off 不划算**:牺牲 1% Best 只换来 0.3% Last 提升
- **结论**:FixBN 不作为默认版本;broken BN 充当了隐性正则化,意外对峰值有利
- **paper 价值**:可作为 ablation 表中"BN 聚合策略"的一行
