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
| 15 | | | |
| 333 | | | |
| mean | | | |

## 结论
