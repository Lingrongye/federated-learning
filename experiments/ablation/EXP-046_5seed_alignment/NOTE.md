# EXP-046 | 5-seed 对齐 FDSE repo 默认配置

## 基本信息
- **目的**:补齐到 5 seed 对齐 FDSE 官方 repo `run.py` 注释中的原始 5 seed 列表 `[2, 4388, 15, 333, 967]`
- **当前状态**:已有 seed {2, 15, 333} 三个(FedDSA + FDSE R200 均有)
- **本次补齐**:seed {4388, 967} × {FedDSA, FDSE R200} = 4 runs
- **状态**:⏳ 待执行

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
### FedDSA (R200)
| seed | Best | Last | Gap |
|---|---|---|---|
| 4388 | | | |
| 967 | | | |
| **5-seed mean** | | | |

### FDSE (R200)
| seed | Best | Last | Gap |
|---|---|---|---|
| 4388 | | | |
| 967 | | | |
| **4-seed mean** (缺 s2) | | | |

## 结论
