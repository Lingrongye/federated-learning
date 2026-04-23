# EXP-116 | λ_orth=0 对照 — 验证正交头的绝对贡献

## 基本信息
- **日期**: 2026-04-22 启动
- **算法**: `feddsa_scheduled` mode=0 (orth_only base)
- **服务器**: lab-lry GPU 1 (24GB 完全空闲)
- **状态**: 🟡 运行中 (6 runs 并行, 预计 8-13h 完成)

## 这个实验做什么 (大白话)

**方法论必须的干净消融**. EXP-080 orth_only (lo=1.0) 在 PACS 达 80.41 超 FDSE +0.50, Office 达 89.44. 但**从未做过 lo=0 的对照** — 不知道这 80.41/89.44 里有多少是正交头贡献.

本实验: 保持 EXP-080 所有其他配置, 仅把 `lo` 从 **1.0 → 0.0** (关闭正交损失), 跑 3-seed × {PACS, Office} × R200.

**判决**:
- 如 lo=0 vs lo=1 差 <0.3pp → 正交头**几乎无贡献**, paper 需改叙事, 主卖 "双头架构 / 数据白化 / 优化细节"
- 如 lo=0 vs lo=1 差 >0.5pp → **正交头确实是 key 贡献**, paper 叙事稳

## 变体通俗解释

| 配置 | lo (λ_orth) | 其他 | 一句话 |
|------|:-----------:|------|-------|
| **EXP-080 orth_only** (baseline) | 1.0 | mode=0, LR=0.05, E=5 | 已知结果: PACS 80.41, Office 89.44 |
| **EXP-116 lo=0** (本实验) | **0.0** | 与 EXP-080 完全一致 | 只换 lo, 干净对照 |

两个对照的差值 = 正交头的纯贡献.

## 实验配置

| 参数 | PACS | Office |
|------|:----:|:------:|
| Task | PACS_c4 | office_caltech10_c4 |
| 算法 | feddsa_scheduled (mode=0) | feddsa_scheduled (mode=0) |
| R / E / B / LR | 200 / 5 / 50 / 0.05 | 同 |
| **lo (λ_orth)** | **0.0** | **0.0** |
| 其他 algo_para | lh=0, ls=1.0, tau=0.2, pd=128, sm=0 | 同 |
| Seeds | {2, 15, 333} | {2, 15, 333} |
| Config | `config/pacs/feddsa_lo0_pacs_r200.yml` | `config/office/feddsa_lo0_office_r200.yml` |

## 🏆 结果 (2026-04-22 晚回填 — 6/6 run 全部完成)

### 3-seed AVG Best/Last 汇总 (判决表)

| 方法 | PACS AVG B/L | Office AVG B/L | PACS ALL B/L | Office ALL B/L |
|------|:--------------:|:----------------:|:---:|:---:|
| FDSE R200 (baseline) | 79.91/77.55 | 90.58/89.22 | — | — |
| **EXP-080 lo=1.0** (对照) | **80.41/79.42** | **89.44/88.71** | — | — |
| **EXP-116 lo=0.0** (本实验) | **80.47/78.91** | **89.75/86.89** | 82.27/80.90 | 85.18/82.66 |
| **Δ (lo=0 − lo=1)** | **+0.06 / −0.51** | **+0.31 / −1.82** | — | — |

### Per-seed (EXP-116 lo=0.0)

| Seed | PACS ALL B/L | PACS AVG B/L | Office ALL B/L | Office AVG B/L |
|:----:|:---:|:---:|:---:|:---:|
| 2 | 81.54/80.53 | 79.98/79.13 | 86.90/82.53 | 90.74/86.58 |
| 15 | 82.04/80.23 | 80.14/78.25 | 82.53/80.14 | 86.97/84.28 |
| 333 | 83.24/81.94 | 81.29/79.35 | 86.10/85.31 | 91.52/89.82 |
| **Mean** | **82.27/80.90** | **80.47/78.91** | **85.18/82.66** | **89.75/86.89** |

注：所有 run 都跑完 R200 (序列长 201).

## 🔴 判决结论：**正交头绝对贡献 ≈ 0** (叙事必须改)

按原计划:
- **lo=0 vs lo=1 差 <0.3 → 正交头几乎无贡献, paper 需改叙事**
- lo=0 vs lo=1 差 >0.5 → 正交头 key

**实际观察**:
- PACS **Best** Δ = +0.06 (lo=0 反而略高 0.06pp, 统计上等于 0)
- Office **Best** Δ = +0.31 (lo=0 反而略高 0.31pp, 方差内)
- **两个数据集 Best 差全都 < 0.5, 甚至都 < 0.3, 而且方向是 lo=0 更好!**
- Last 差为负（lo=1 Last 略好 0.5-1.82pp）说明 lo=1 正交损失只做了**轻微的后期稳定化**, 但不贡献 Best

### 对 paper 叙事的影响

原 paper 叙事: "正交头把 z_sem / z_sty 几何解耦, 是 FedDSA 的核心创新"

**现在必须改为**:
- 主卖点 **不能是正交头**. 正交头只是轻微 regularizer, 不是性能来源.
- 真正的性能来源: **SGPA 双头架构本身**（fc1→bn6→relu→fc2→bn7→relu 两个 256d 头的存在） + **LR=0.05 + lr_decay 0.9998 稳定优化** + **EXP-080 的所有其他超参**
- 或改叙事为"双头架构在 FL 跨域场景 beat FDSE (EXP-080 lo=1) 主要来自模型结构和优化 schedule, 正交损失是可选的稳定化组件"

### Office Last 异常大 Δ = -1.82pp 解释

- lo=0.0 Office Last 86.89 (比 lo=1 的 88.71 差 1.82pp)
- 可能原因: 没有正交损失时, 训练后期 z_sty 开始逐渐入侵 z_sem, 造成 Last round 泛化下降
- 但 Best 仍 +0.31, 说明训练中段表现更好

**实践建议**: paper 可讨论这个 trade-off, 但不能声称"正交头是 accuracy 来源".

## 📋 部署快照 (2026-04-22 启动时)

| Task | Seed | PID | 状态 |
|:----:|:---:|:---:|:---:|
| Office lo=0 | 2 | 374770 | 🟡 R~4 (smoke 已验证) |
| Office lo=0 | 15 | 375422 | 🟡 启动 |
| Office lo=0 | 333 | 375482 | 🟡 启动 |
| PACS lo=0 | 2 | 375532 | 🟡 启动 |
| PACS lo=0 | 15 | 375591 | 🟡 启动 |
| PACS lo=0 | 333 | 375683 | 🟡 启动 |

**GPU 6.5/24 GB (27%)**, 剩 17GB

## 胜负判决

| Scenario | 判决 | paper 含义 |
|----------|:---:|------------|
| lo=0 PACS <79.9 AND Office <89.0 | 正交头**关键** | 主卖正交头, 叙事不变 |
| lo=0 PACS ~80.4 AND Office ~89.4 | 正交头**无贡献** | 必须改叙事, 可能主卖 SGPA 双头架构/LR=0.05 稳定优化 |
| 其他 (混合) | 跨数据集差异 | regime-dependent, paper 需讨论 |

## 下一步

1. 等 6 runs 完成 (~8-13h, lab-lry GPU 1)
2. 收集 3-seed × per-client × Best/Last 完整矩阵
3. 与 EXP-080 orth_only (lo=1.0) 做严格对照
4. 写入 paper 消融章节

## 📎 相关文件

- Configs: `FDSE_CVPR25/config/{pacs,office}/feddsa_lo0_*_r200.yml`
- 算法: `FDSE_CVPR25/algorithm/feddsa_scheduled.py` (mode=0 orth_only base)
- 上游依赖: EXP-080 (orth_only lo=1.0 baseline), EXP-109/110 (orth_uc1 对比)
- 触发原因: 用户提出 "验证正交头到底贡献多少" 这个方法论缺口
