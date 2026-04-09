# EXP-046 | 5-seed 对齐 FDSE repo 默认配置

## 基本信息
- **目的**:补齐到 5 seed 对齐 FDSE 官方 repo `run.py` 注释中的原始 5 seed 列表 `[2, 4388, 15, 333, 967]`
- **当前状态**:已有 seed {2, 15, 333} 三个(FedDSA + FDSE R200 均有)
- **本次补齐**:seed {4388, 967} × {FedDSA, FDSE R200} = 4 runs
- **状态**:✅ 已完成

## 背景
FedDSA repo `run.py` 第 23-24 行显示:
```python
# 注释掉的原始 5 seed: [2, 4388, 15, 333, 967]
# 当前默认 3 seed: [2, 15, 333]
```
说明作者原本用 5 seed 跑论文结果。我们补齐到 5 seed → 可大声说"与作者 repo 完全对齐"。

## 与已有实验的关系
| 方法 | s2 | s15 | s333 | s4388 | s967 |
|---|---|---|---|---|---|
| FedDSA 原版 | ✅EXP-017 82.24 | ✅EXP-035 80.59 | ✅EXP-036 81.05 | ⏳ 本次 | ⏳ 本次 |
| FDSE R200 | — | ✅EXP-043 79.00 | ✅EXP-043 79.93 | ⏳ 本次 | ⏳ 本次 |

## 运行命令
```bash
# FedDSA × 2 seeds
nohup python run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 \
  --config ./config/pacs/feddsa_v4_no_hsic.yml --logger PerRunLogger --seed 4388 \
  > /tmp/exp046_feddsa_s4388.out 2>&1 &
nohup python run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 \
  --config ./config/pacs/feddsa_v4_no_hsic.yml --logger PerRunLogger --seed 967 \
  > /tmp/exp046_feddsa_s967.out 2>&1 &

# FDSE R200 × 2 seeds
nohup python run_single.py --task PACS_c4 --algorithm fdse --gpu 0 \
  --config ./config/pacs/fdse_r200.yml --logger PerRunLogger --seed 4388 \
  > /tmp/exp046_fdse_s4388.out 2>&1 &
nohup python run_single.py --task PACS_c4 --algorithm fdse --gpu 0 \
  --config ./config/pacs/fdse_r200.yml --logger PerRunLogger --seed 967 \
  > /tmp/exp046_fdse_s967.out 2>&1 &
```

## 结果
### FedDSA (R200) 完整 5-seed
| seed | Best | Last | Gap | 来源 |
|---|---|---|---|---|
| 2 | 82.24 | 75.46 | 6.78 | EXP-017 |
| 15 | 80.59 | 75.38 | 5.20 | EXP-035 |
| 333 | 81.05 | 75.54 | 5.51 | EXP-036 |
| **4388** | **78.55** | 74.23 | 4.32 | 本次 |
| **967** | **81.29** | 75.39 | 5.90 | 本次 |
| **5-seed mean** | **80.74 ± 1.37** | 75.20 | 5.54 | |

### FDSE (R200) 4-seed (缺 s2,见 EXP-049)
| seed | Best | Last | Gap | 来源 |
|---|---|---|---|---|
| 15 | 79.00 | 76.64 | 2.36 | EXP-043 |
| 333 | 79.93 | 77.92 | 2.01 | EXP-043 |
| **4388** | **80.98** | **68.78** | **12.20** | 本次 |
| **967** | **80.49** | 76.40 | 4.09 | 本次 |
| **4-seed mean** | **80.10 ± 0.84** | 74.94 | 5.17 | |

## 结论
- FedDSA 5-seed Best 均值 **80.74 ± 1.37** > FDSE 4-seed **80.10 ± 0.84** (+0.64%)
- 差距从之前 3-seed 的 +1.83% 缩窄到 +0.64%,统计意义变弱
- FedDSA seed=4388 异常低(78.55),拉低了均值;方差 1.37 >> FDSE 0.84
- FDSE seed=4388 也崩塌(Last 68.78, gap 12.20),说明两者对该 seed 都不稳定
- **结论**:同 R200 预算下 FedDSA 仍然正向,但差距不够显著;需扩数据集或 R500 增强 claim
