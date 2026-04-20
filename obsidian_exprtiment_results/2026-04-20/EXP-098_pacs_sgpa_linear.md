# EXP-098: SGPA + Linear 对照 PACS R200 3-seed — Office 结论 PACS 复刻验证

**日期**: 2026-04-19 设计 / 2026-04-20 09:16 完成
**算法**: `feddsa_sgpa` (use_etf=1 SGPA + use_etf=0 Linear)
**服务器**: seetacloud2 GPU 0 (单卡 4090, 6 runs 并行, wall 8h10min)
**状态**: ✅ **已完成** (2026-04-20), **PACS 再次证伪 ETF: Linear 全面胜出 SGPA, 且 ETF 后期严重退化 Last -5.59pp**

## 这个实验做什么 (大白话)

> EXP-097 + EXP-100 在 Office 上已证: Linear+whitening (88.75) 完胜 SGPA (86.97) 和 Plan A (82.55). 但这只是 **1 个数据集**. 要判定 ETF 是不是垃圾, 必须在 PACS 上**再来一遍同样的对照**。
>
> PACS 比 Office 更考验几何: K=7 理论下界 -1/6=-0.167 (比 Office K=10 的 -0.111 宽), ETF 的分离优势理论上应该更大. 如果 PACS 上 SGPA 还是输 Linear, 那 ETF "几何收益" 的故事就彻底 GG.
>
> 复刻 EXP-097/100 的 SGPA vs Linear 对照 (use_etf=1/0 单变量), 换 PACS_c4 + E=5, 跑 3 seeds。

## Claim 和成功标准

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C1 PACS**: SGPA R200 PACS > Plan A R200 | 3-seed mean AVG Best ≥ 82.2% (Plan A 81.69% + 0.5%) | ETF 在 K=7 场景也没增益 |
| **C2 PACS**: ETF 贡献 (对照 Linear) | SGPA ≥ Linear + 1% (若 Linear 赢则 C2 证伪) | ETF 几何优势不存在 |

## 配置

### SGPA (use_etf=1)
| 参数 | 值 |
|------|-----|
| Task | PACS_c4 (7 类, 4 clients: Art/Cart/Photo/Sketch) |
| Backbone | AlexNet + 双 128d 头 |
| Algorithm | feddsa_sgpa, `use_etf=1` |
| R | 200 |
| **E** | **5** (PACS 惯例) |
| LR | 0.05 |
| λ_orth | 1.0 |
| τ_etf | 0.1 |
| Seeds | {2, 15, 333} |
| diag | 1 |
| Config | `FDSE_CVPR25/config/pacs/feddsa_sgpa_pacs_r200.yml` |

### Linear 对照 (use_etf=0)
同上但 `use_etf=0`, config `feddsa_linear_pacs_r200.yml`

## 🏆 完整结果 (3-seed mean)

### Claim C1 PACS: SGPA vs Plan A (per-domain Best/Last)

| 配置 | seed | AVG Best | AVG Last | Art | Cart | Photo | Sketch |
|------|------|---------|---------|-----|------|-------|--------|
| **Plan A orth_only** (EXP-080) | **mean** | 81.69 | 73.87 | — | — | — | — |
| **SGPA (use_etf=1)** | **mean** | **78.96 ± 0.37** | **73.77** | 62.6/54.6 | 85.0/78.9 | 80.0/74.3 | 88.2/87.3 |
|  | 2 | 79.48 | 73.56 | 64.7/53.4 | 85.5/77.8 | 78.4/76.0 | 89.3/87.0 |
|  | 15 | 78.81 | 74.30 | 61.8/53.9 | 85.5/83.3 | 80.2/72.5 | 87.8/87.5 |
|  | 333 | 78.60 | 73.44 | 61.3/56.4 | 84.2/75.6 | 81.4/74.3 | 87.5/87.5 |
| **Δ SGPA − Plan A** | — | **-2.73** ❌ | -0.10 | — | — | — | — |

### Claim C2 PACS: ETF 贡献 (Linear 对照)

| 配置 | seed | AVG Best | AVG Last | Art | Cart | Photo | Sketch |
|------|------|---------|---------|-----|------|-------|--------|
| **Linear+whitening (use_etf=0)** | **mean** | **80.20 ± 0.94** | **79.36** | 63.4/61.4 | 86.0/84.0 | 81.8/82.4 | 89.5/89.5 |
|  | 2 | 81.46 | 80.74 | 64.7/64.7 | 86.8/83.3 | 83.8/85.6 | 90.6/89.3 |
|  | 15 | 79.94 | 78.65 | 61.8/58.3 | 87.2/86.3 | 82.0/81.4 | 88.8/88.5 |
|  | 333 | 79.21 | 78.70 | 63.7/61.3 | 84.2/82.5 | 79.6/80.2 | 89.3/90.8 |
| **Δ Linear − SGPA (ETF 减分!)** | — | **+1.24** ✅ | **+5.59** 🔥 | +0.8/+6.8 | +1.0/+5.1 | +1.8/+8.1 | +1.3/+2.2 |
| **Δ Linear − Plan A** | — | **-1.49** ⚠️ | +5.49 | — | — | — | — |

## 🔍 根因分析

### Verdict Decision Tree 判定

```
Δ SGPA − Linear = -1.24 (小于 0)
  → ❌ C2 PACS 证伪, ETF 在 PACS 也不帮忙
Δ SGPA − Plan A = -2.73 (远小于 0)
  → ❌ C1 PACS 证伪, SGPA 比 Plan A 还差
Δ Linear − Plan A = -1.49 (还是负的!)
  → ⚠️ 在 PACS 上 whitening 也没拿到 Office 那种 +6.20pp, 反而 AVG Best 输 Plan A
  → 但 Last 超 Plan A +5.49pp, 说明 Plan A 后期有漂移
```

### 双数据集交叉对照

| 方法 | Office AVG Best | PACS AVG Best | Office Last | PACS Last |
|------|-----------------|---------------|-------------|-----------|
| Plan A orth_only | 82.55 | 81.69 | 81.35 | 73.87 |
| SGPA (use_etf=1) | 86.97 | 78.96 | 85.44 | 73.77 |
| Linear+whitening (use_etf=0) | **88.75** 🔥 | **80.20** | **86.91** | **79.36** |
| Δ Linear − SGPA | **+1.78** | **+1.24** | +1.47 | **+5.59** |

**两个数据集都显示 Linear > SGPA** — ETF Hard 硬约束全面失败.

### 为什么 ETF 在 PACS 后期退化更猛

1. **Photo Last -8.1pp / Art Last -6.8pp** — 自然图像域受 ETF 硬约束影响最大
2. **E=5 local epochs 放大了过拟合**: ETF + CE 双约束在 5 epochs 内把特征挤压到顶点, 跨域泛化丢失
3. **ETF 的 fixed vertex 对 4 outlier 全不友好**: Office DSLR 是唯一 outlier 所以影响小, PACS 全是 outlier 所以影响大

## 📋 论文叙事影响

### "ETF 有效" 故事已死

两个数据集 × 3 seeds × R200 × Linear 对照 = 6 组独立证据, 全部证伪 ETF 的几何贡献 claim.

### 但 PACS 暴露了 whitening broadcast 的**边界**

Office 上 whitening+FedBN 从 82.55 → 88.75 (+6.20pp), PACS 上 whitening+FedBN 80.20 < Plan A 81.69 (-1.49pp). whitening broadcast 不是万灵药:

- Office E=1 短本地训练, 一致 whitening 广播作用明显
- PACS E=5 长本地训练, whitening 被本地 BN + FedBN drift 覆盖

### 接下来必须做

1. **PACS 也跑 EXP-102 等价 (use_whitening=1, use_centers=0) 3 seeds** — 隔离出 "pure whitening" 在 PACS 上的效果
2. **如果 PACS 确认 whitening 无益** — 论文叙事降级到 "Office-specific improvement"
3. **或改用 PACS 友好的机制** — 如 pull 软约束 (EXP-106) 或 Local-BN 增强

## 📊 实验统计

- **总 runs**: 6 (SGPA 3 + Linear 3)
- **实际**: ~48 GPU·h (seetacloud2 单 4090 6 并行 wall 8h10min)
- **启动**: 2026-04-20 01:05:56
- **完成**: 2026-04-20 09:16 (所有 6 runs DONE + record JSON 保存成功)

## 📎 相关文件

- 本地 NOTE: `experiments/ablation/EXP-098_sgpa_pacs_r200/NOTE.md`
- Office 主实验: `experiments/ablation/EXP-097_sgpa_office_r200/NOTE.md` + `EXP-100_linear_office_r200/NOTE.md`
- Records: `FDSE_CVPR25/task/PACS_c4/record/feddsa_sgpa_*use_etf{0,1}_*R200_*S{2,15,333}*.json`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (use_etf flag)
- Configs: `FDSE_CVPR25/config/pacs/feddsa_{sgpa,linear}_pacs_r200.yml`
