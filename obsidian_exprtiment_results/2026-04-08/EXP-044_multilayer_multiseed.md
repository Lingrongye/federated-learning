# EXP-044 | MultiLayer Style 多seed 验证

## 基本信息
- **目的**:验证 EXP-040 MultiLayer Style 的 81.96% 不是运气,确定真实均值
- **算法**:feddsa_multilayer (fc1+fc2 双层风格注入)
- **配置**:复用 feddsa_exp040.yml (R200, orth=1.0, hsic=0, sem=1.0)
- **seeds**:15, 333
- **状态**:⏳ 待执行

## 背景
EXP-040 (seed=2) 跑到 185/200 挂掉时 Best=81.96 —— 是所有架构级改动中唯一接近 SOTA (82.24) 的方案。但只有单 seed,必须补多 seed 验证真实均值。

## 对比
| 方法 | seed=2 | seed=15 | seed=333 | 均值 |
|---|---|---|---|---|
| FedDSA 原版 (EXP-017) | 82.24 | 80.59 | 81.05 | 81.29 |
| MultiLayer (EXP-040) | 81.96 | **待测** | **待测** | **?** |

## 假设
如果 MultiLayer 均值 > 81.29,说明多层注入确实是正向架构改动,这是架构级最大突破。

## 运行命令
```bash
nohup python run_single.py --task PACS_c4 --algorithm feddsa_multilayer --gpu 0 \
  --config ./config/pacs/feddsa_exp040.yml --logger PerRunLogger --seed 15 \
  > /tmp/exp044_s15.out 2>&1 &
nohup python run_single.py --task PACS_c4 --algorithm feddsa_multilayer --gpu 0 \
  --config ./config/pacs/feddsa_exp040.yml --logger PerRunLogger --seed 333 \
  > /tmp/exp044_s333.out 2>&1 &
```

## 结果
| seed | Best | Last | Gap |
|---|---|---|---|
| 15 | 80.09 | 78.13 | 1.96 |
| 333 | 80.18 | 74.16 | 6.02 |
| (s2 原 EXP-040) | 81.96 | 75.51 | 6.45 |
| **mean (3 seeds)** | **80.74 ± 0.85** | 75.93 | 4.81 |

## 结论
- **MultiLayer 均值 80.74 < FedDSA 原版均值 81.29**(差 0.55%)
- EXP-040 的 81.96 是单 seed 运气值,多 seed 下回归到基线以下
- **架构级"多层风格注入"无正向收益**,建议回归原架构
- 副作用:seed=333 last 崩塌到 74.16(gap 6.02),多层注入反而增加不稳定性
