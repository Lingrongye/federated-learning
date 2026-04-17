# EXP-043 | FDSE 基线多seed 验证 (R200)

## 基本信息
- **目的**：验证 FDSE 单seed 82.16% 是否是运气值,补齐多seed 均值
- **算法**：fdse (官方 FDSE_CVPR25)
- **配置**：R200, LR 0.05, 其他同 fdse.yml
- **seeds**:15, 333 (补齐 seed=2 的 82.16)
- **状态**:🔄 重启(首次启动时服务器 01:12 挂掉,只跑到 13/200 轮)

## 背景
EXP-017 (FedDSA, seed=2) 达到 82.24% 曾被认为破 FDSE 82.16%,但:
- EXP-035 (FedDSA, seed=15) = 80.59%
- EXP-036 (FedDSA, seed=333) = 81.05%
- FedDSA 3-seed 均值 = 81.29 ± 0.86

问题:FDSE 的 82.16 也可能含运气。必须补齐多seed 确认真实均值,才能得到可靠对比。

## 对比组
| 方法 | seed=2 | seed=15 | seed=333 | 均值 |
|---|---|---|---|---|
| FedDSA | 82.24 | 80.59 | 81.05 | **81.29** |
| **FDSE** | 82.16 | **待测** | **待测** | **?** |

## 配置差异说明
- 原 fdse.yml R500, 本实验用 R200 匹配 FedDSA 预算
- 这是公平对比:同 R200 下 FedDSA vs FDSE 谁赢
- R500 下 82.16 仅作参考点

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm fdse --gpu 0 \
    --config ./config/pacs/fdse_r200.yml \
    --logger PerRunLogger --seed 15 > /tmp/exp043_s15.out 2>&1 &

nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm fdse --gpu 0 \
    --config ./config/pacs/fdse_r200.yml \
    --logger PerRunLogger --seed 333 > /tmp/exp043_s333.out 2>&1 &
```

## 结果 (R200)
| seed | Best | Last | Gap |
|---|---|---|---|
| 15 | 79.00 | 76.64 | 2.36 |
| 333 | 79.93 | 77.92 | 2.01 |
| **mean** | **79.46 ± 0.47** | 77.28 | 2.18 |

> 注:首次启动时服务器 01:12 挂掉,只跑了 13 轮。本次为重启后完整结果。

## 结论
- **FDSE R200 均值 79.46** << FedDSA R200 均值 81.29(差 **1.83%**)
- 与 FDSE 原论文 R500 5-seed 均值 82.17 ± 1.49 相比,R200 预算下精度下降约 2.7%
- **关键发现**:同训练预算下 FedDSA 稳定击败 FDSE
- **尚需**:补跑 FDSE R500 多seed 匹配论文配置,作为完整对比

## ⚠️ 与 FDSE 论文对比方法论差异
| 维度 | FDSE 论文 | 本实验 |
|---|---|---|
| Rounds | **500** | 200 |
| Seeds | **5** (mean±std) | 2 (+s2 原 R500) |
| LR 调优 | 各方法独立 grid | 固定 0.05 |
| 度量 | AVG + ALL | AVG only |
| Backbone | AlexNet ✅ | AlexNet ✅ |
| 数据分区 | 4 clients = 4 domains | 4 clients = 4 domains ✅ |
