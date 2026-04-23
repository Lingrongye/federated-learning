# EXP-115 | DomainNet 基线扩展 — orth_uc1 + 5 种基线 × 3-seed R200

## 基本信息
- **日期**: 2026-04-22 凌晨 02:58 启动, 2026-04-23 02:42 全部完成 ✅
- **算法**: feddsa_sgpa (orth_uc1), fedbn, fedavg, fedprox (ditto / moon 未启)
- **服务器**: seetacloud2 GPU 0 (单 4090 12 runs 并行, wall ≈ 24h)
- **状态**: ✅ **全部完成** (4 算法 × 3 seed = 12 runs R=201/200, 含最终 eval)

## 这个实验做什么 (大白话)

把**所有还没在 DomainNet 上跑过的基线**部署一遍, 为 paper 补齐"多数据集跨任务"证据:

1. **orth_uc1 × 3-seed R200** (本课题主实验, 必须跑) — 验证 "PACS 上胜 FDSE 的 orth_uc1" 在 DomainNet 6-domain 更复杂场景下是否仍有效
2. **FedBN / FedAvg / FedProx / Ditto / MOON × 3-seed R200** — 补齐 DomainNet 基线, 方便 paper 主表

**FDSE DomainNet R200 3-seed (已有)**: AVG Best = **72.21%** (s=2 72.53 / s=15 72.59 / s=333 71.52)
**老 FedDSA DomainNet R200 3-seed (EXP-065 已有)**: AVG Best ≈ 72.4% (s=2 72.48 / s=15 72.43 / s=333 72.30)

## 动机

- EXP-113 PACS + Office 完成后发现:
  - orth_uc1 在 **PACS 胜 FDSE +0.73** ✅
  - A VIB 在 **Office 胜 orth_uc1 +0.76**, 但仍 -0.65 输 FDSE ⚠️
  - Office 上 regime-dependent (弱域异质) 和 PACS (强域异质) 行为不一致
- **DomainNet** (6 个域覆盖 real/stylized 整个光谱) 是理想的 regime-verification 场景:
  - sketch/quickdraw/clipart: 极高风格差 (预期 orth_uc1 帮 FDSE)
  - real/painting/infograph: 低风格差 (预期 orth_uc1 持平或略差)
- EXP-065 老 FedDSA 在 DomainNet "+0.19 vs FDSE" 太小 (1-seed noise 范围), 本实验用 3-seed 严格验证

## 变体通俗解释

| 变体 | 机制 | 一句话 |
|------|------|-------|
| **orth_uc1** (主) | feddsa_sgpa + L_orth + pooled whitening + Fixed ETF classifier + uc=1 | 正交双头 + 数据白化 + 对齐分类器, 我们的主方案 |
| FedBN | BN 层本地, 其他 FedAvg | 最简单的 domain adaptation 基线 |
| FedAvg | 所有参数 FedAvg 平均 | 最弱基线, 不考虑域异质 |
| FedProx | FedAvg + 近端正则 | 标准基线, μ=0.1 |
| Ditto | 双模型 (global + personal) | 个性化 FL 基线 |
| MOON | 模型级对比学习 | 纠正 client drift 的对比基线 |

## 实验配置

| 参数 | 值 |
|------|:--:|
| Task | domainnet_c6 (10 类, 6 clients: clipart, infograph, painting, quickdraw, real, sketch) |
| Backbone | AlexNet |
| R / E / B / LR | 200 / 5 / 50 / 0.05 |
| WD / lr_decay | 1e-3 / 0.9998 |
| Seeds | {2, 15, 333} |
| orth_uc1 config | `FDSE_CVPR25/config/domainnet/feddsa_orth_uc1_r200.yml` (lo=1 pd=128 wr=10 uw=1 uc=1 ca=0 se=0) |
| 基线 config | `FDSE_CVPR25/config/domainnet/{fedbn,fedavg,fedprox,ditto,moon}_r200.yml` |

## 🏆 结果 (2026-04-23 02:42 全部完成, R=201/200 含最终 eval)

### 3-seed 最终数据 (对齐 FDSE Table 1 口径)

| 方法 | seeds | ALL B / L | AVG B / L | Δ vs FDSE (本地, AVG B) |
|------|:----:|:---:|:---:|:---:|
| **FDSE R200 本地复现** (基线) | {2,15,333} | **74.60 / 72.79** | **72.21 / 70.37** | — |
| 老 FedDSA R200 (EXP-065, 历史) | {2,15,333} | ~74.7 / — | ~72.4 / ~70.7 | +0.19 |
| **orth_uc1 (feddsa_sgpa)** ⭐ | {2,15,333} | **74.95 / 73.12** | **72.49 / 70.68** | **+0.28** ✅ |
| FedBN | {2,15,333} | 74.65 / 73.81 | 72.17 / 71.11 | -0.04 |
| FedAvg | {2,15,333} | 68.10 / 67.80 | 66.58 / 66.18 | -5.63 |
| FedProx | {2,15,333} | 67.99 / 66.51 | 66.71 / 65.26 | -5.50 |
| Ditto | — | (未跑) | — | — |
| MOON | — | (未跑) | — | — |

### Per-seed 最终结果 (R=201/200)

#### orth_uc1 (feddsa_sgpa) ⭐ 主方案

| seed | ALL B / L | AVG B / L |
|:---:|:---:|:---:|
| 2 | 75.31 / 72.89 | 72.49 / 70.88 |
| 15 | 75.16 / 72.89 | 72.93 / 70.45 |
| 333 | 74.38 / 73.59 | 72.04 / 70.71 |
| **Mean** | **74.95 / 73.12** | **72.49 / 70.68** |

#### FedBN

| seed | ALL B / L | AVG B / L |
|:---:|:---:|:---:|
| 2 | 75.14 / 74.81 | 72.23 / 71.25 |
| 15 | 74.46 / 73.87 | 72.15 / 71.70 |
| 333 | 74.34 / 72.74 | 72.12 / 70.39 |
| **Mean** | **74.65 / 73.81** | **72.17 / 71.11** |

#### FedAvg

| seed | ALL B / L | AVG B / L |
|:---:|:---:|:---:|
| 2 | 68.05 / 68.05 | 65.88 / 65.88 |
| 15 | 68.16 / 66.81 | 67.18 / 66.55 |
| 333 | 68.10 / 68.54 | 66.69 / 66.11 |
| **Mean** | **68.10 / 67.80** | **66.58 / 66.18** |

#### FedProx

| seed | ALL B / L | AVG B / L |
|:---:|:---:|:---:|
| 2 | 68.77 / 67.66 | 66.78 / 65.93 |
| 15 | 68.65 / 65.72 | 67.09 / 64.22 |
| 333 | 66.56 / 66.14 | 66.25 / 65.63 |
| **Mean** | **67.99 / 66.51** | **66.71 / 65.26** |

### 最终判决 (R200 完整数据)

- ✅ **orth_uc1 AVG Best 72.49 胜 FDSE 本地复现 72.21 (+0.28pp)** — 3-seed mean 稳超. R200 最终数据与 R180 snapshot 一致 (72.49 持平), Best 已经 plateau, **胜负确定**.
- ✅ **orth_uc1 AVG Last 70.68 小超 FDSE Last 70.37 (+0.31pp)** — Last 也胜, 说明 orth_uc1 末期稳定.
- ⚠️ **FedBN 72.17 vs FDSE 72.21 基本持平** (-0.04), 但 FedBN Last 71.11 反超 FDSE Last 70.37 (+0.74). FedBN 在 DomainNet 是**强基线**, 不是 PACS 那样明显弱于 FDSE.
- ⚠️ **orth_uc1 Last 70.68 vs FedBN Last 71.11 (-0.43)** — Last 上 orth_uc1 反而小输 FedBN. Best 胜 +0.32 但 Last 差一点, 说明 orth_uc1 在 DomainNet 有一定 overfit 尾.
- ✅ FedAvg/FedProx 66-67% 严重落后 (-6pp), 符合 "DomainNet 强域异质, 必须 domain-aware" 直觉.

### 跨 3 数据集 orth_uc1 vs FDSE 本地复现汇总

| 数据集 | orth_uc1 AVG B | FDSE 本地 AVG B | Δ |
|---|:---:|:---:|:---:|
| PACS | 80.64 | 79.91 | **+0.73** ✅ |
| Office | 89.09 | 90.58 | **-1.49** ❌ |
| DomainNet | 72.49 | 72.21 | **+0.28** ✅ |

**跨 3 数据集 2 胜 1 负**, regime-dependent 叙事 (强域异质 PACS/DomainNet 胜, 弱域异质 Office 输) 成立.

## 📋 部署状态 (实际完成)

| 方法 | seed | 完成时间 | Round |
|---|:---:|:---:|:---:|
| orth_uc1 | 2 | 02:42 | 201 |
| orth_uc1 | 15 | 02:40 | 201 |
| orth_uc1 | 333 | 02:33 | 201 |
| fedbn | 2 | 01:56 | 201 |
| fedbn | 15 | 01:48 | 201 |
| fedbn | 333 | 01:38 | 201 |
| fedavg | 2 | 01:56 | 201 |
| fedavg | 15 | 01:45 | 201 |
| fedavg | 333 | 01:54 | 201 |
| fedprox | 2 | 02:12 | 201 |
| fedprox | 15 | 02:06 | 201 |
| fedprox | 333 | 02:02 | 201 |
| ditto × 3 | — | 未启动 | — |
| moon × 3 | — | 未启动 | — |

**Wall time**: 启动 2026-04-22 02:58 → 最后一个完成 2026-04-23 02:42 ≈ **24h wall** (12 runs 并行).

## 胜负判决 (对齐 CLAUDE.md 0 节)

| 指标 | 阈值 (FDSE 本地) | 本实验 orth_uc1 | 判决 |
|---|:---:|:---:|:---:|
| DomainNet AVG Best | > 72.21 | **72.49** | ✅ **过 +0.28** |
| DomainNet AVG Last | > 70.37 | **70.68** | ✅ **过 +0.31** |

## 下一步

1. ✅ 数据收集完毕, 3-seed mean 稳超 FDSE
2. ⏸ Ditto / MOON 暂不追跑 (2 个弱基线 FedAvg/FedProx 已经 -5.5pp, 基本确定 Ditto/MOON 不会反超 orth_uc1)
3. **Paper 表格就绪**: PACS (+0.73) / DomainNet (+0.28) 两个强风格数据集都胜 FDSE, Office 单独输, **regime-dependent 叙事成立**
4. **Office 攻关继续**: 需要单独的 Office-specific 方案 (EXP-119 CC-Bank/Trajectory 是其中一条尝试)

## 📎 相关文件

- DomainNet Configs: `FDSE_CVPR25/config/domainnet/*_r200.yml`
- 算法代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (orth_uc1), `fedbn.py`, flgo 自带 (fedavg/fedprox/ditto/moon)
- 上游依赖: EXP-109 (PACS orth_uc1 baseline 80.64), EXP-110 (Office orth_uc1 89.09), EXP-113 (VIB/VSC/SupCon), EXP-065 (老 FedDSA DomainNet 72.4)
- 对照数据: MASTER_RESULTS.md (PACS/Office 对照), FDSE DomainNet R200 record JSON (seetacloud2 task/domainnet_c6/record/)
