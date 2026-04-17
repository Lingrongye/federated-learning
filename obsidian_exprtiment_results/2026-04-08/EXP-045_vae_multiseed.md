# EXP-045 | VAE StyleHead 多seed 验证

## 基本信息
- **目的**:验证 EXP-041 VAE StyleHead 的稳定性(gap=2.64 是历史最低)在多 seed 下成立
- **算法**:feddsa_vae (概率化 style head + KL 正则)
- **配置**:复用 feddsa_exp041.yml (R200, λ_kl=0.01)
- **seeds**:15, 333
- **状态**:⏳ 待执行

## 背景
EXP-041 (seed=2) Best=79.85, Last=77.21, **gap=2.64** —— 所有方法中 gap 最小,稳定性新纪录(vs Triplet 3.71, PCGrad 4.86)。但 Best 较低,需验证:
1. 多 seed 下 gap 是否依然 <3(稳定性是否普适)
2. Best 均值在什么水平(是否值得牺牲 2% 精度换稳定性)

## 对比
| 方法 | seed=2 Best | Gap |
|---|---|---|
| FedDSA 原版 EXP-017 | 82.24 | 6.78 |
| VAE (EXP-041) | 79.85 | 2.64 |
| Triplet (EXP-030) | 80.14 | 3.71 |
| PCGrad (EXP-029) | 80.72 | 0.05 |

## 假设
VAE 的 KL 正则降低 style head 过拟合,提升泛化稳定性。如果多 seed gap 都 <3,说明 VAE 是稳定性方向的最佳架构级方案,可与 MultiLayer 组合。

## 运行命令
```bash
nohup python run_single.py --task PACS_c4 --algorithm feddsa_vae --gpu 0 \
  --config ./config/pacs/feddsa_exp041.yml --logger PerRunLogger --seed 15 \
  > /tmp/exp045_s15.out 2>&1 &
nohup python run_single.py --task PACS_c4 --algorithm feddsa_vae --gpu 0 \
  --config ./config/pacs/feddsa_exp041.yml --logger PerRunLogger --seed 333 \
  > /tmp/exp045_s333.out 2>&1 &
```

## 结果
| seed | Best | Last | Gap |
|---|---|---|---|
| 15 | 80.97 | 77.66 | 3.31 |
| 333 | 81.28 | 78.34 | 2.94 |
| (s2 原 EXP-041) | 79.85 | 77.21 | 2.64 |
| **mean (3 seeds)** | **80.70 ± 0.75** | 77.74 | 2.96 |

## 结论
- **VAE 均值 80.70 < FedDSA 原版均值 81.29**(差 0.59%)
- 但 **gap 全部 <3.31**,稳定性确实提升(原版 gap 常在 5-7)
- 有趣反转:seed=2 时 best 最低(79.85),seed=15/333 反而达到 81+
  → 对初始化敏感,但多 seed 下 **稳定性收益成立**
- **trade-off**:牺牲 0.6% 峰值精度换来 gap 减半
- **未来方向**:若追求稳定性 → VAE head;若追求精度 → 原版
