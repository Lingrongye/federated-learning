# EXP-049 | FDSE R200 seed=2 补齐 5-seed

## 基本信息
- **目的**:补齐 FDSE R200 的第 5 个 seed(s2),对齐 FedDSA 的 5-seed 集合
- **算法**:fdse (FDSE 原版基线)
- **配置**:fdse_r200.yml
- **seed**:2
- **状态**:⏳ 待执行

## 背景
FedDSA 已完成 5-seed (2,15,333,4388,967),FDSE 只有 4-seed (15,333,4388,967)。
补齐 s2 后两者可做完全对等的 5-seed 比较。

## 已有 FDSE R200 数据
| seed | Best | Last | Gap |
|---|---|---|---|
| 15 | 79.00 | 76.64 | 2.36 |
| 333 | 79.93 | 77.92 | 2.01 |
| 4388 | 80.98 | 68.78 | 12.20 |
| 967 | 80.49 | 76.40 | 4.09 |
| **2** | **待测** | | |
| 4-seed mean | 80.10 ± 0.84 | 74.94 | 5.17 |

## 运行命令
```bash
nohup python run_single.py --task PACS_c4 --algorithm fdse --gpu 0 \
  --config ./config/pacs/fdse_r200.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp049.out 2>&1 &
```

## 结果
| seed | Best | Last | Gap |
|---|---|---|---|
| 2 | | | |
| **5-seed mean** | | | |

## 结论
