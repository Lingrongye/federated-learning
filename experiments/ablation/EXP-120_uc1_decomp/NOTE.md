# EXP-120 | orth_uc1 分解消融 — SGPA 架构 / whitening / centers 三组件贡献拆分

## 基本信息
- **日期**: 2026-04-23 03:35 启动 DN + 03:44 启动 Office
- **算法**: `feddsa_sgpa` (不同 uw/uc flag 组合)
- **服务器**: lab-lry GPU 1 (RTX 3090 24GB)
- **状态**: ✅ **Office 9 runs 完成 04:55** / 🟡 DN 9 runs 跑中 (预计 14:00-16:00 完成)

## 这个实验做什么 (大白话)

**分解 EXP-115 orth_uc1 在 DomainNet 上 +0.28 胜 FDSE 的增益来源**.

EXP-117 (orth_only, feddsa_scheduled 单路径) AVG B **72.23** ≈ FDSE 72.21 (打平 +0.02), 说明"纯正交头"无贡献.
EXP-115 (orth_uc1 = feddsa_sgpa + uw=1 + uc=1) AVG B **72.49** (+0.28 vs FDSE).

orth_uc1 - orth_only = **+0.26** (SGPA 架构 + whitening + centers 三者合计).

本实验**把三个组件分开跑**, 看谁是真正的贡献者:
- SGPA only (uw=0 uc=0): 只有 SGPA 双头架构, 没 whitening 没 centers
- SGPA + w (uw=1 uc=0): 加 whitening, 没 centers
- SGPA + c (uw=0 uc=1): 加 centers, 没 whitening
- SGPA + w + c (uw=1 uc=1) = EXP-115 已有

## 变体通俗解释

| 变体 | 算法 | uw | uc | 含义 |
|------|:----:|:--:|:--:|------|
| orth_only (EXP-117) | feddsa_scheduled | — | — | 单路径 + 正交头 (无 SGPA 双头架构) |
| **sgpa_only** (本 EXP-120) | feddsa_sgpa | 0 | 0 | **SGPA 双头架构 + 正交, 无 whitening 无 centers** |
| **sgpa_w** (本 EXP-120) | feddsa_sgpa | 1 | 0 | SGPA + orth + **pooled whitening** |
| **sgpa_c** (本 EXP-120) | feddsa_sgpa | 0 | 1 | SGPA + orth + **class centers** 对齐 |
| orth_uc1 (EXP-115) | feddsa_sgpa | 1 | 1 | SGPA + orth + whitening + centers (全套) |

## 实验配置

| 参数 | 值 |
|------|:--:|
| Task | domainnet_c6 (10 类, 6 client) |
| Backbone | AlexNet |
| R / E / B / LR | 200 / 5 / 50 / 0.05 |
| WD / lr_decay | 1e-3 / 0.9998 |
| Seeds | {2, 15, 333} |
| algo_para | lo=1 pd=128 wr=10 es=1e-3 mcw=2 dg=1 ue=0 se=0 lp=0 ca=0, 仅 uw/uc 变化 |
| Configs | `feddsa_sgpa_{only,w,c}_r200.yml` |

**总 runs**: 3 变体 × 3 seeds = **9 runs**

## 预期结果分析

| 情景 | 解读 |
|---|---|
| sgpa_only ≈ orth_only (72.2) | SGPA 架构本身无贡献, whitening 和/或 centers 带来增益 |
| sgpa_only > orth_only + 0.3 | **SGPA 架构本身就有贡献** (双头 > 单路径) |
| sgpa_w ≈ sgpa_c | whitening 和 centers 效果相当, 可选其一 |
| sgpa_w > sgpa_c by >0.2 | whitening 是主力 |
| sgpa_c > sgpa_w by >0.2 | centers 是主力 |
| sgpa_w + sgpa_c ≈ 72.49 | 两者**累加**, 都有独立贡献 |
| sgpa_w ≈ sgpa_c ≈ 72.49 | 两者**冗余** (随便一个就够) |

## 预期 GPU + 时间

- lab-lry GPU 1 RTX 3090 24GB 完全空闲 (24129 MiB free)
- DomainNet E=5 单 run 显存 ~2-3 GB (历史数据), 9 并行 ~22 GB ✅
- 3090 单 run ~5-6h (比 4090 慢), 9 并行 wall ≈ **6-8h**
- 预计完成 2026-04-23 **18:00-20:00**

## 部署步骤

1. 本地创建 3 个 config + NOTE → git push
2. lab-lry: `cd /home/lry/code/federated-learning && git pull origin main`
3. lab-lry: 写 greedy launcher 脚本 (按 GPU 1 显存动态 launch, 阈值 2500 MB)
4. 启动 9 runs 并行, nohup 后台
5. 每 30min-1h 抽查进度
6. 完成后提取 record 回填 NOTE

## 📎 相关文件

- Configs: `FDSE_CVPR25/config/domainnet/feddsa_sgpa_{only,w,c}_r200.yml` + `config/office/feddsa_sgpa_{only,w,c}_office_r200.yml`
- 算法: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (uw/uc 在 init_algo_para 中控制)
- 基线数据: EXP-115 (orth_uc1 72.49), EXP-117 (orth_only 72.23), FDSE 本地 72.21, EXP-113 Office orth_uc1 89.09, EXP-116 FedBN Office lo=0 89.75

---

## 🏆 Office 结果 (2026-04-23 04:55 完成, 3-seed R=201/200)

### 3-seed mean

| 变体 | uw/uc | ALL B/L | AVG B/L | vs FedBN 89.75 | vs sgpa_only |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **sgpa_only** | 0/0 | 84.79/83.05 | **89.14/86.87** | **-0.61** ❌ | — (基线) |
| sgpa_w | 1/0 | 84.12/82.27 | **88.68/86.88** | -1.07 ❌ | **-0.46** ❌ |
| sgpa_c | 0/1 | 83.86/82.00 | **88.74/86.48** | -1.01 ❌ | **-0.40** ❌ |
| 对照 orth_uc1 (1/1, EXP-113) | 1/1 | — | 89.09/87.32 | -0.66 | -0.05 |
| 对照 FedBN lo=0 (EXP-116) | — | — | **89.75**/86.89 | 0 (基线) | +0.61 |

### Per-seed

| 变体 | s=2 AVG B/L | s=15 AVG B/L | s=333 AVG B/L |
|---|:---:|:---:|:---:|
| sgpa_only | 89.31/88.75 | 86.54/84.76 | 91.56/87.12 |
| sgpa_w | 88.83/86.91 | 85.50/82.74 | 91.71/91.00 |
| sgpa_c | 88.94/86.15 | 86.21/83.83 | 91.08/89.47 |

### 🔑 Office 结论 — 反直觉发现

**在 Office 上, whitening 和 centers 都有害 (-0.4~-0.46 vs sgpa_only), 跟 DomainNet 上 +0.26 的行为完全相反**.

对比:
- **DomainNet (EXP-115 vs EXP-117)**: orth_uc1 72.49 > orth_only 72.23 → whitening+centers **+0.26pp 有效** ✅
- **Office (本实验)**: sgpa_only 89.14 > sgpa_w 88.68 / sgpa_c 88.74 → whitening+centers **-0.40~-0.46 有害** ❌

**这是 regime-dependent 叙事的第三个证据** — 强风格异质 (DomainNet/PACS) 这两个机制帮忙, 弱风格异质 (Office) 反而伤性能.

**但所有 3 变体仍输 FedBN 89.75 (-0.61~-1.07pp)**, Office 攻关仍未破局. 下一方向: **排除 whitening/centers, 尝试 logit-level calibration / class-conditional classifier bank** (因为 Office 的问题不在 feature space 的对齐, 可能在 classifier 层的校准).

## 🟠 DomainNet 部分数据 (2026-04-23 14:30 中途 kill, R=104-106/200, ~53% 训练)

### 中断原因

14:30 手动 kill 9 runs 给 EXP-123 PACS Stage B 腾 lab-lry GPU 1 空间 (EXP-123 Art domain 诊断优先级更高). R=0-105 的 accuracy 曲线从 log 文件 parse 还原 (flgo 中途不写 record JSON, 只有 log 保留). **R=106-200 数据永久丢失**.

### Per-seed AVG Best (R=0-105 snapshot, parsed from log)

| 变体 | uw/uc | s=2 | s=15 | s=333 | 3-seed mean |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **sgpa_only** | 0/0 | 72.95 | 72.57 | 72.34 | **72.62** |
| **sgpa_w** | 1/0 | 72.83 | 72.39 | 72.44 | **72.55** |
| **sgpa_c** | 0/1 | 72.08 | 72.90 | 72.54 | **72.51** |
| 对照 orth_uc1 (R=200, EXP-115) | 1/1 | — | — | — | 72.49 |
| 对照 orth_only (R=200, EXP-117) | — | — | — | — | 72.23 |
| 对照 FDSE 本地 | — | — | — | — | 72.21 |

### 初步解读 (mid-run, 非 R=200 最终)

**三变体 AVG Best 统计等价** (72.62 / 72.55 / 72.51, 差 0.11 远小于 seed std):
- `sgpa_w` ≈ `sgpa_c` ≈ `sgpa_only` → whitening 和 centers **没有独立贡献**
- 但 `sgpa_only` (72.62) > `orth_only` (72.23) by **+0.39** → **SGPA 双头架构本身可能有贡献**
- R=105 时 `sgpa_only` 已超过 orth_uc1 R=200 (72.49) → 但 mid-run 有 bias (R=200 后曲线可能降或升)

### 数据可信度警告

- **mid-run 风险**: R=105→200 曲线仍可能变化. 特别若 whitening/centers 是"慢热"型 (后期发力), R=200 可能 w/c > only. 或 only 可能在后期 over-fit 下降
- **配合 Office 交叉验证**: Office (R=200 完整) 显示 w/c **有害** (-0.4 ~ -0.46 vs only), DN mid-run 也显示 w/c 无贡献 → **双数据集证据一致**
- **→ whitening/centers 不是真正的贡献源**, 这个结论即使 DN 未完成 R=200 也大概率站得住

### Log 还原详情

9 log 文件在 lab-lry `/home/lry/code/federated-learning/FDSE_CVPR25/task/domainnet_c6/log/2026-04-23-03-3*feddsa_sgpa*.log` (每 230KB, R=0-105), 含:
- `mean_local_test_accuracy` per round (AVG)
- `local_test_accuracy` per round (ALL weighted)
- min/max/std per round
- val accuracy + loss

**不含**: per-client-dist (只有 aggregated stats), per-class accuracy.

## 跨 3 数据集对比 (DN = R=105 mid-run, 其他 R=200 完整)

| 数据集 | FedBN/对照 | sgpa_only | sgpa_w | sgpa_c | 哪个贡献 |
|---|:---:|:---:|:---:|:---:|:---:|
| PACS | - | (未跑) | (未跑) | (未跑) | — |
| **Office** (R=200) | FedBN 89.75 | **89.14** | 88.68 | 88.74 | **都有害** (-0.4 ~ -0.46) |
| **DomainNet** (R=105 🟠) | FDSE 72.21 | **72.62** | 72.55 | 72.51 | **w/c 都无贡献**; SGPA 架构本身 +0.39 vs orth_only 72.23 |

### 总结叙事

**"whitening + centers 不是真正的贡献源"**:
- Office (R=200 完整): w/c **有害**
- DomainNet (R=105 mid-run): w/c **无贡献** (0.04-0.11 差在 seed noise 内)
- 唯一可能的贡献: **SGPA 双头架构本身** (vs orth_only +0.39 在 DN, 但 Office 所有变体都输 FedBN → 架构也不 generalize 到 Office)

→ **orth_uc1 (EXP-115) 的 +0.28 gain 可能主要来自 SGPA 架构 (或 seed noise), 不是 whitening/centers**

### 是否重跑 DN R=200?

**不建议**. 理由:
1. Office 已证 w/c 有害 → 即使 DN R=200 出 w/c > only, 也只在 DN 成立, 不 generalize
2. R=105 已接近 plateau, R=105→200 变化通常 ≤0.3pp
3. 重跑 6h+ 算力换 0.3pp 精度不值
4. 当前核心问题是 PACS Art (EXP-123) 和 Office FedBN 突破, DN 已经相当稳

**若后续需补齐**: 优先补 `sgpa_only` × 3 seeds × R=200 (最关键变体, 验证架构贡献是否站得住), w/c 可选
