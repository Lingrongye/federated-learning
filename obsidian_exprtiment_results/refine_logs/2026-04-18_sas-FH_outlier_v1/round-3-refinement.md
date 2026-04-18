# 第 3 轮细化（→ 目标在 R4 达成 READY）

## 问题锚点（原文照录）
[与 R0/R1/R2 相同；锚点已保留。]

## 锚点核查
✅ R3 reviewer："Problem Anchor is preserved... stays on that bottleneck and does not drift."。无 drift 警告。

## 简洁性核查
- 主导贡献不变：style-conditioned classifier routing，作为 meta-framing。
- 所有 R3 改动都是**最终 paper 层面的证据承诺 + 措辞微调**，方法/实验结构没有变化。

## 已做的修改

### 1. C2 无条件升到 3-seed（唯一 blocking fix）
- **R3 reviewer 意见**：strong-margin 分支里 C2 单 seed 不足以出 top-venue paper。C2 必须与 A2 同等 statistical weight。
- **动作**：重写决策规则：
  > **Final-paper protocol（强制）**：
  > - Triage（single-seed s=2）：A2、C1、C2 一次过筛 → Go/No-Go
  > - **若 A2 通过 triage，A2 和 C2 全部升到 3-seed {2, 15, 333}**（无 margin 豁免）
  > - C1 保持 1-seed 作为 secondary ablation
  > - 若 A2 triage 未过（margin < 0.5 over max(B1, C2)）：halt
- **理由**：核心 claim A2 > C2 的 causal status 要求 statistically-matched baseline。不允许 strong margin 变成跳过 C2 扩展的借口。
- **影响**：Compute plan 从"strong-margin path 6.5h / statistical path 14.5h"统一为 **"only pass triage 时固定 14.5h"**。单一路径，无歧义。

### 2. 统计测试 framing 调整
- **R3 reviewer 意见**：3-seed paired t-test 较弱，应 "let effect size + per-seed consistency do most of the work"。
- **动作**：Validation 报告格式：
  - Primary：3-seed mean ± std、per-seed delta、effect size
  - Secondary：paired t-test 作为 supplementary（标注 "n=3 caveat"），**不作为主 significance claim**
  - Consistency check：3 seeds 全部显示 A2 > C2（方向一致）才算 confirmed
- **理由**：诚实处理小样本统计，避免过度 claim 假显著。

### 3. `S` 集合定义统一
- **R3 reviewer 意见**："Clarify whether S in C2 is all clients or participating clients in that round, and keep identical to A2's participation set."
- **动作**：显式加入：
  > 对 A2 和 C2，都以**同一轮参与聚合的客户端集合 S_t** 为聚合对象。在 Office-Caltech10 c4 全量参与（proportion=1.0）设定下，S_t = {0,1,2,3} 每轮。这使 A2 与 C2 的参与集合完全一致，聚合对象相同，仅权重策略不同。
- **理由**：消除可能的实现歧义。
- **影响**：代码实现需要共用 participation mask。

### 4. Claim 3 heatmap 定性化
- **R3 reviewer 意见**：不要量化承诺 "Caltech self > 0.7"，改为定性。
- **动作**：Expected signature 改为：
  > Predicted: the Caltech row of W shows higher self-concentration (max entry on diagonal) than other rows; the other 3 rows are more evenly distributed. Exact magnitude depends on `sas_tau` and final style_proto geometry, reported as-observed.
- **理由**：不越过 over-claim 红线。

### 5. Claim 3 折叠进 evidence section
- **R3 reviewer 意见**："If paper gets crowded, fold Claim 3 into Claim 1/2 mechanism evidence rather than separate claim."
- **动作**：Claim 3 改名为 **"Mechanism Visualization (supplementary to Claim 2)"** —— 不再是独立 claim，只作为 Section "Mechanism Evidence" 的一部分。
- **理由**：单篇 paper 一个 dominant claim + 2 个支撑证据 artefact（Claim 2 swap + Claim 3 matrix）比 3 个 claim 更聚焦。

### 6. C1 明确为 secondary status
- **R3 reviewer 意见**："Keep C1 explicitly secondary. If budget tight, preserve A2 and C2 first."
- **动作**：C1 在 Main Table 里标注 "(secondary, 1-seed)"，预算紧时先保 A2/C2 完整。
- **理由**：工程实用。

---

# 修订后的提案（Round 3 Final → 目标 R4 READY）

# Research Proposal: Style-Conditioned Classifier Routing for Outlier-Domain Recovery in Federated Domain Generalization

## Problem Anchor
[不变]

## Technical Gap
[与 R2 相同]

## Method Thesis

- **一句话 thesis**：Under style heterogeneity, the key federated-learning question at the classifier layer is not *whether* to personalize but *how to route* the sharing — and style similarity is the natural routing signal.
- **具体实现**：把 FedDSA Plan A 的 sas 从 `semantic_head` 扩到 full head chain（projection + classifier），零新可训练组件。
- **为何 timely**：FedROD/FedAlign/DualFed 在 pFL 中确立了 per-client classifier；我们首次把个性化条件化到 style similarity 上。

## Contribution Focus

- **主导贡献**：Style-conditioned classifier routing (sas-FH)，将其 framing 为 classifier 层的 routing problem。
- **支撑贡献**：NONE。

## Proposed Method

### Config switch（sas flag 0..4）
```
sas=0: baseline no sas (B0 = EXP-083)
sas=1: sem_head sas only, classifier FedAvg (B1 = Plan A = EXP-084)
sas=2: sem_head + classifier, both style-conditioned (OURS, A2)
sas=3: sem_head sas + classifier uniform-avg (C2)
sas=4: sem_head sas + classifier fully local (C1)
```

### Aggregation rule (for k ∈ sas_keys，**每轮 identical participation set S_t**)
- **A2**：`per_param_i[k] = Σ_{j ∈ S_t} w_{ij} · client_state[j][k]`, `w_{ij} = softmax(cos(sp_i, sp_j)/sas_tau)`
- **C2**：`per_param_i[k] = (1/|S_t|) · Σ_{j ∈ S_t} client_state[j][k]`（所有 i 共享同一个 head）
- **C1**：`per_param_i[k] = client_state[i][k]`（仅本地）

Client-side 训练不变：CE + λ_orth · L_orth。

### Integration
`FDSE_CVPR25/algorithm/feddsa_scheduled.py` pack() 方法。Config：sas 扩到 0..4。代码：约 40 行（含 3 个分支 + unit tests）。

### Training / Hyperparameters
R200 E=1 B=50 LR=0.05 decay=0.9998。Seeds {2,15,333}。Office-Caltech10 c4 主路径；PACS c4 appendix-only。

### Failure Modes
- Caltech 无增益 → Claim 2 swap diagnostic；若 swap 信号为零 → halt。
- 训练不稳定 → sas_tau=0.5 重试。

## Claim-Driven Validation

### Claim 1 (Main): A2 > C2 隔离出 style routing causality

**Triage（single-seed s=2，6h）**：A2、C1、C2 pilot。

**决策规则**：
- `AVG_Best(A2) − max(AVG_Best(B1), AVG_Best(C2)) ≥ 0.5`：**把 A2 和 C2 都升到 3-seed {2,15,333}**（**强制，无 margin 豁免**）
- `< 0.5`：halt，thesis 被伪证

**最终 paper 主表**（两者都升级）：
| ID | sem_head | classifier | seeds | source |
|----|----------|-----------|-------|--------|
| B0 | FedAvg | FedAvg | 3 | EXP-083 已有 |
| B1 | sas | FedAvg sample-weighted | 3 | EXP-084 已有 |
| **A2 (ours)** | **sas** | **sas style-conditioned** | **3**（新） | main |
| **C2** | **sas** | **uniform-avg** | **3**（新） | counterfactual |
| C1 (secondary) | sas | local-only | 1 | ablation |

**统计报告**：
- Primary：3-seed mean ± std + per-seed Δ(A2, C2)
- Secondary（supplementary）：paired t-test，带 n=3 caveat
- Consistency：要求 A2 > C2 在 **全部 3 个 seed** 都成立才 claim confirmed

预期：A2 Caltech ≥ 77 (+2)，A2 AVG ≥ 90.5（超过 FDSE 90.58），A2 − C2 ≥ 1%。

### Claim 2 (Mechanism Diagnostic, same-round swap, zero training)

加载 A2 R149 best-round checkpoint。保留 (i) personalized `(encoder_i, sem_head_i, head_i)` 和 (ii) same-round pack-time FedAvg snapshot `global_head`（在 R149 聚合时作为标准中间量计算得到）。

Per-client swap：
- (a) `(encoder_i, sem_head_i, head_i)` on client i test
- (b) `(encoder_i, sem_head_i, global_head)` on same test

**Same-round guarantee**：两个 head 都来自 R149 —— 无训练动态 confound。

预期：Caltech (a) > (b) 至少 2%；其他 Δ < 0.5%。

### Mechanism Evidence (Claim 2 的 supplementary): Style similarity 矩阵 + routing weights

在 A2 R149，提取 4×4 矩阵 `S_{ij} = cos(sp_i, sp_j)` 和 `W_{ij} = softmax(S/sas_tau)`。Heatmap。

**定性预期**：Caltech 行的 self-concentration 比其他行更高；其他行更分散。精确数值按实测报告。

### Claim 4 (Appendix, OPTIONAL): PACS scope boundary

仅当 Office A2 成功时才跑 PACS A2 × 3-seed。预期 AVG ≤ EXP-086 Plan A 79.76。解释：PACS 中无 majority reference → routing 失去意义。Appendix only。

## Experiment Handoff

- Must-prove：Claim 1 A2 > C2 ≥ 1% on 3 seeds + per-seed consistency；Claim 2 swap 信号。
- Critical：Office Caltech Best、AVG Best。

## Compute & Timeline

| Experiment | Runs | Per run | Total |
|-----|-----|--------|------|
| Triage: Office s=2 × {A2, C1, C2} | 3 | 2h | 6h |
| **强制升级（triage 通过后）：A2 + C2 × (s=15, s=333)** | 4 | 2h | 8h |
| Claim 2 swap + Mechanism viz | — | 0.5h | 0.5h |
| Claim 4 PACS A2 × 3-seed (optional appendix) | 3 | 3.5h | 10.5h |
| **Must-run（若通过）** | 7 | — | **~14.5h** |
| **Full（含 PACS）** | 10 | — | ~25h |

关键路径：Triage 6h → 决策 gate → Escalate 8h → paper draft。
