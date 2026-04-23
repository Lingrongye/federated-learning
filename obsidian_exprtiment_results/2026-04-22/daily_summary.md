# 2026-04-22 实验日志总览

> 今日核心: EXP-119 FedPTR Sanity 方案完成 design + 代码 + review + 部署 (24 runs chained). 同时 EXP-115/116/117 在其他机器跑中.

## 今日部署的实验 (4 个 in-flight)

| EXP ID | 名称 | 机器 | 状态 | 结果概况 |
|:------:|---|:----:|:-----:|:---:|
| **EXP-115** | DomainNet 基线扩展 (feddsa_sgpa + 4 baseline) | seetacloud2 | ✅ **全部完成 02:42** | feddsa_sgpa AVG B **72.49** > FDSE 本地 72.21 **+0.28** ✅ |
| **EXP-116** | λ_orth=0 对照 (6 runs: PACS+Office × 3seed) | lab-lry GPU 1 | ✅ **已全部完成 (6/6)** | PACS 80.47 / Office 89.75, 正交头贡献 ≈ 0 |
| **EXP-117** | orth_only × DomainNet R200 3-seed | lab-lry GPU 1 | ✅ **全部完成 01:11-01:19** | AVG B **72.23** vs FDSE 72.21 **+0.02 打平** (orth_only 对 DomainNet Best 无显著贡献) |
| **EXP-118** | 完整 FedBN vs 半 FedBN 对照 (Office × 3seed) | lab-lry GPU 1 | ✅ **Office 3/3 完成** | 半/完整 FedBN Δ=+0.24 无实质差异 |
| **EXP-119** | FedPTR Sanity (centralized + CC-Bank + Trajectory) | **westb:13399 (RTX 4090 24GB)** | ✅ **Wave 1/2/3 全部完成 11:01** | C1 INVALID, C2 FAILED, C3 MARGINAL |

## 🔴 今日重大发现 (回填后出来的)

### 发现 6 (EXP-116 完成): **正交头对 Best accuracy 贡献 ≈ 0**

| 指标 | lo=1 | **lo=0** | Δ |
|---|:---:|:---:|:---:|
| PACS AVG Best | 80.41 | **80.47** | **+0.06** (lo=0 反而略好) |
| Office AVG Best | 89.44 | **89.75** | **+0.31** (lo=0 反而略好) |

**叙事必须改**: paper 不能再卖正交头, 真正 accuracy 来源是 SGPA 双头架构 + 优化 schedule.

### 发现 7 (EXP-115 R200 最终确认): **DomainNet orth_uc1 +0.28 胜 FDSE**

| 数据集 | orth_uc1 AVG B | FDSE 本地 AVG B | Δ | 胜负 |
|---|:---:|:---:|:---:|:---:|
| PACS | 80.64 | 79.91 | **+0.73** | ✅ |
| Office | 89.09 | 90.58 | **-1.49** | ❌ |
| DomainNet | **72.49** | **72.21** | **+0.28** | ✅ 🎯 (R200 最终确认, snapshot 一致) |

跨 3 数据集 **2 胜 1 负**, regime-dependent 主叙事成立 (强风格异质胜, 弱异质输).

**完整 EXP-115 R200 3-seed 数据**:

| 方法 | ALL B/L | AVG B/L |
|---|:---:|:---:|
| FDSE 本地 | 74.60/72.79 | 72.21/70.37 |
| **orth_uc1** | **74.95/73.12** | **72.49/70.68** |
| FedBN | 74.65/73.81 | 72.17/71.11 |
| FedAvg | 68.10/67.80 | 66.58/66.18 |
| FedProx | 67.99/66.51 | 66.71/65.26 |

详见 `关键实验发现备忘.md` 发现 6 + 7.

### 发现 9 (EXP-119 W1): Centralized 81.81 — "method space 上限" **伪命题**

Centralized + global BN 在多域数据 81.81 < FedBN 88.68, 因为 global BN 学到的是"4 domain 混合均值"不匹配单 domain 分布. **C1 判决 invalid**, 不能用来否决 backbone.

### 发现 10 (EXP-119 W1): CC-Bank α=0.5 Office 89.35 没打过 FedBN 89.75

AdaIN 历史第 5 次"部分失败" — 没崩但没赚. 等 Wave 2 (α=0.3/0.7) + Wave 3 PACS 再判.

### 发现 11 (EXP-119 W2): CC-Bank Office 全 α 扫都没赢 FedBN

| α | Office AVG Best | vs FedBN 89.75 |
|:-:|:---:|:---:|
| 0.3 | 89.11 | -0.64 |
| 0.5 | 89.35 | -0.40 |
| **0.7** | **89.61** | -0.14 (最接近但仍输) |

**CC-Bank Office 最终判决: AdaIN 第 5 次部分失败** (没崩没赢). 命运决定权交给 Wave 3 PACS (强异质).

### 发现 12 (W3 部署教训): chained dispatcher 浪费 14h wall

原 dispatcher 设计 "Wave 1→wait→Wave 2→wait→Wave 3a→wait→3b→wait→3c" 把每批之间都串行，实际 GPU 13GB/24GB 完全可以并行下一批。**改 greedy launcher 按显存动态 launch**, 12 PACS 全并行 = 节省 14h wall (21h → 7h). 规则已写入 CLAUDE.md 17.8 GPU 并行原则 (强制).

### 发现 13 (EXP-119 Wave 3 PACS 最终): **FedPTR 3 组件 Sanity Phase 判决**

**Sanity A (Centralized)**: PACS R=101 AVG B 66.96, Office R=200 AVG B 81.81, **INVALID** (global BN 多域 mismatch 不是 capacity 上限, 不作判据)

**Sanity B (CC-Bank)**:
| 数据集 | α scan | Best | 对比 FedBN | Δ |
|---|---|:---:|:---:|:---:|
| Office | 0.3/0.5/0.7 = 89.11/89.35/89.61 | 89.61 | 89.75 | **-0.14** ❌ |
| PACS | α=0.5 = 78.88 | 78.88 | 80.41 | **-1.53** ❌ |

→ **CC-Bank 整个砍**, AdaIN 家族第 5 次失败定案

**Sanity C (Trajectory PACS R=200)**:
| Seed | η=0 (对照) | η=0.5 (预测) | Δ |
|:---:|:---:|:---:|:---:|
| 2 | 81.76/80.07 | 81.38/79.51 | -0.38 B |
| 15 | 79.77/78.48 | 80.39/79.13 | **+0.62 B** |
| 333 | 79.65/77.33 | 80.53/77.94 | **+0.88 B** |
| **Mean** | **80.39/78.63** | **80.77/78.86** | **+0.38 B / +0.23 L** |

→ **Trajectory 小赢 PACS FedBN +0.36pp, 2 赢 1 负**, 未过 0.5pp 阈值, **保留但降权**, 不作主卖点

**最终决定**: **不走 FedPTR 全量代码**, 回到 orth_uc1 / VIB / VSC 线 pivot 做 Office 攻关.

---

## 各机器实时状态

### 1. westb:13399 (新 RTX 4090 24GB, EXP-119 专用)

- **硬件**: RTX 4090, 24564 MiB 显存, 驱动 560.35.03
- **当前显存**: 13.5 / 24 GB
- **当前运行**: 6 个 python (EXP-119 Wave 1)
  - 3 × `centralized office_caltech10_c4` seeds {2, 15, 333}
  - 3 × `fedbn_ccbank α=0.5 office_caltech10_c4` seeds {2, 15, 333}
- **后续自动接续**: Wave 2 (Office α=0.3/0.7) → Wave 3a/3b/3c (PACS)
- **CC-Bank AdaIN 公式修复已生效** (commit 2ed462e, batch-per-feature self-norm)

### 2. seetacloud2 (RTX 4090 24GB, EXP-115 专用)

- **当前显存**: 22 / 24 GB (几乎满)
- **当前运行**: 12 个 python (EXP-115 DomainNet), 已跑 **20h35m+**
  - 3 × feddsa_sgpa (orth_uc1) × 3 seeds
  - 3 × fedbn × 3 seeds
  - 3 × fedavg × 3 seeds
  - 3 × fedprox × 3 seeds
- **快完成**: 正在逼近 R=200, 早前估 6-10 点结束
- **完成后可用**: 24h 内可空出整卡 (如果 EXP-115 收尾)

### 3. lab-lry GPU 0 (RTX 3090 24GB, 他人占用)

- **当前显存**: 20.5 / 24 GB (wjc 独占)
- **wjc 任务**: SFLGenPerL TinyImagenet (14h) + SplitGP × 2 (其中一个 **2 天 21 小时**)
- **我们不能碰这张卡**

### 4. lab-lry GPU 1 (RTX 3090 24GB, 部分共享)

- **当前显存**: 20.6 / 24 GB
- **lry (我们)**: 3 个 feddsa_scheduled_orth_only DomainNet (EXP-116+117), 已跑 9h 到 round 80/200
- **lty (他人)**: HotpotQA counterfactual labels 脚本, 17 GB 占用
- **不能加新 run** (即使 kill lry 的也只腾 3.4GB, lty 仍占 17GB)

---

## 今日完成的工作非实验部分

| 类别 | 内容 |
|:---:|:---|
| 🧠 Research Pipeline L4 | 完成 FedPTR → FedPTR refined → FedDSA-Swap 3 轮提案 + review + novelty check |
| 📝 代码实现 | 写 `fedbn_ccbank.py` (~350 行) + `fedbn_trajectory.py` (~270 行) + `centralized.py` (~100 行) + 8 configs |
| 🧪 单测 | 38 个 pytest 单测 (18 CC-Bank + 20 Trajectory) 全部 pass |
| 🔍 Self-review | 发现 CC-Bank AdaIN hybrid 公式 bug, 修为 batch-per-feature |
| 🐛 flgo 兼容性修复 | `_unwrap_alexnet` helper (FModule wrapper) + 删除误导性空 init hooks |
| ✅ flgo smoke | R=3 smoke test 3/3 exit=0 on seetacloud2 |
| 🚀 部署 | 5-wave chained dispatcher, 24 runs 全部排队 on westb:13399 |
| 📚 Obsidian | EXP-119 NOTE.md 部署状态更新, 关键实验发现备忘.md (5 条发现) |

---

## 明天 (2026-04-23) 预期完成的数据

假设所有实验无故障:

### EXP-115 (2026-04-22 早上 6-10 点):
- DomainNet feddsa_sgpa (orth_uc1) 3-seed AVG Best / Last
- DomainNet FedBN / FedAvg / FedProx 3-seed AVG Best / Last
- Ditto / MOON 还需另起 (EXP-115 只起了前 4 个)

### EXP-116 (2026-04-22 中午前):
- PACS lo=0 × 3 seeds (与 EXP-080 lo=1 对照, 判决正交头绝对贡献)
- Office lo=0 × 3 seeds

### EXP-117 (2026-04-22 白天):
- DomainNet orth_only × 3 seeds (对 EXP-115 orth_uc1 提供"纯正交"参照)

### EXP-119 (2026-04-23 夜晚 22:30):
- **Wave 1-2 Office 12 runs** 在 2026-04-23 01:30 前完成
  - Go/No-Go 判定: Centralized Office ≥ 93? CC-Bank Office ≥ 89.2?
- **Wave 3 PACS 12 runs** 在 2026-04-23 22:30 前完成
  - Go/No-Go 判定: Trajectory η=0.5 vs η=0 Δ > 0.5pp?

---

## 当前最关键的 3 个 Open Question (等数据回来判)

1. **EXP-115**: feddsa_sgpa (orth_uc1) 在 DomainNet 能否 > FDSE 72.21? 老 FedDSA 只 +0.19 (1-seed noise 范围), 我们要 3-seed mean 稳超。
2. **EXP-116**: 正交头绝对贡献有多大? lo=0 vs lo=1 差 < 0.3 → 叙事改; 差 > 0.5 → paper 叙事稳。
3. **EXP-119**: AlexNet 在 Office centralized 能否 ≥ 93 (method-space 还有 +2pp 空间)? 如果 < 90 → 换 backbone 或降 venue 到 WACV。

---

## TODO 下次对话 (2026-04-23 早)

1. Check EXP-115 是否完成 (seetacloud2) + 收结果
2. Check EXP-119 Wave 1/2 (Office) 完成情况, 回填 NOTE.md, 判 C1 & C2 Go/No-Go
3. 如果 C1 (centralized Office) 过 93 & C2 (CC-Bank Office) 过 89.2, 等 Wave 3 PACS 继续
4. 如果任一判死, 立即按 Fallback 路径换方案 (logit-level calibration 或 ResNet-18)
