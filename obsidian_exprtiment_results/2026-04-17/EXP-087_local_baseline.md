# EXP-087 | Local (Standalone) Baseline — 每 client 独立训练

## 基本信息
- **日期**: 2026-04-17 / 2026-04-18
- **算法**: `standalone` (FDSE 自带，每 client 独立训练无聚合)
- **服务器**: SC2 GPU 0 (Office) + Lab-lry 试跑 PACS (挂)
- **状态**: ⚠️ Office 跑中（Webcam 可能 early stop 失败）, PACS 放弃复现

## ⚠️ PACS 放弃复现，引用 FDSE 论文原值

**原因**：FDSE 自带 `standalone.py` 在 PACS task 下有 **flgo 框架集成 bug**：
- `log_once()` 访问 `self.server.model` 为 `None` → `get_device()` 崩
- Office 能跑（server.model 被正确初始化），PACS 不能
- 修复需侵入 flgo init 流程，改动面大、风险高

**行业惯例**：引用原论文数字（FDSE CVPR 2025 Table 1）：
- PACS: Local ALL **61.29 ± 2.47** / AVG **57.16 ± 2.85**
- Office-Caltech10: Local ALL **64.47 ± 2.52** / AVG **62.72 ± 7.81**

我们用这些作为"下限基线"，足以证明：FL 机制相比 Local **净增益 +25%+**（方案 A 约 89 vs Local 63）。

## 动机

FDSE 论文 Table 1 第一行就是 "Local" 基线（每 client 独立训练不联邦）。我们还没复现过：
- 没有 Local，无法证明 "联邦真的帮到我们的方案"
- 审稿人必问："相比每 client 自己训练，你的方案提升多少？"

补这个基线填空白。

## 算法说明

`FDSE_CVPR25/algorithm/standalone.py`：
- 每 client 独立训练 `num_epochs` 个 epoch
- **完全不聚合**（Server.run 只调用 client.finetune）
- 每 client 在自己的 val 上做 early stop

## 配置

| 参数 | Office | PACS | 说明 |
|------|--------|------|------|
| num_epochs | **200** | **1000** | 对齐 FL 总训练预算（R200×E=1/5） |
| learning_rate | 0.05 | 0.05 | 对齐 FL 公平 |
| batch_size | 50 | 50 | 同 |
| weight_decay | 1e-3 | 1e-3 | 同 |
| early_stop | 20 | 20 | val loss 20 epoch 不降就停 |

## 实验矩阵 (6 runs)

| # | 数据集 | Seed |
|---|--------|------|
| 1 | Office | 2 |
| 2 | Office | 15 |
| 3 | Office | 333 |
| 4 | PACS | 2 |
| 5 | PACS | 15 |
| 6 | PACS | 333 |

## 参考值（FDSE 论文 Table 1）

| 数据集 | ALL | AVG |
|--------|-----|-----|
| Office-Caltech10 | 64.47 ± 2.52 | 62.72 ± 7.81 |
| PACS | 61.29 ± 2.47 | 57.16 ± 2.85 |

## 结果（待填）

### Office R?（early stop 可能提前）

| seed | ALL Best | ALL Last | AVG Best | AVG Last |
|------|---------|---------|---------|---------|
| 2 | - | - | - | - |
| 15 | - | - | - | - |
| 333 | - | - | - | - |
| **mean** | - | - | - | - |

### PACS R?

| seed | ALL Best | ALL Last | AVG Best | AVG Last |
|------|---------|---------|---------|---------|
| 2 | - | - | - | - |
| 15 | - | - | - | - |
| 333 | - | - | - | - |
| **mean** | - | - | - | - |

## 成功标准

1. Office Local 3-seed mean AVG Best **接近 FDSE 论文 62.72**（±5%内），证明复现合理
2. PACS Local 接近 57.16
3. 我们方案 A（EXP-084 Office 89.82）**显著超 Local**（期望 +20% 以上）→ 说明 FL 机制实际有效

## 下一步

训练完成后填入结果表，然后更新 MASTER_RESULTS.md 加一行 Local baseline。
