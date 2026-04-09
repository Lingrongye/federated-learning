# EXP-050 | FixBN 多seed验证

## 基本信息
- **目的**:验证 EXP-048 的 BN 修复效果是否在多 seed 下稳定成立
- **算法**:feddsa_fixbn (server 端聚合 BN running stats)
- **配置**:feddsa_exp048.yml
- **seeds**:15, 333
- **状态**:⏳ 待执行

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
| 15 | | | |
| 333 | | | |
| **3-seed mean** | | | |

## 结论
