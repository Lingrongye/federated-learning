# 第 2 轮细化

## 问题锚点（原文照录）

- **核心问题**：FedDSA Plan A 在 outlier 域严重失败。PACS art 65.69%, Office Caltech 77.68%。70 个 art 错例全部 conf ≥ 0.85，集中在有机体类互串。
- **必须解决的瓶颈**：共享 classifier 稀释 outlier 域决策边界。
- **非目标**：不换骨干、不加新可训练模块、不改推理、不做 per-sample sas。
- **约束**：3-seed {2,15,333} × R200 × E=1；SC2 单卡 2-4h；≤50 行代码；与 EXP-083/084/086 公平对比。
- **成功条件**：art/Caltech outlier +2% AVG Best，其他域 drop < 1%，Office AVG Best ≥ 90.58 (FDSE)。

## 锚点核查

- ✅ 锚点完全保留。Reviewer R2 明确："Problem Anchor is preserved. The revision does not drift into a broader personalization paper."
- ✅ Reviewer 无 drift 警告。

## 简洁性核查

- **主导贡献**：仍是唯一一个 —— **"style similarity as routing signal for classifier-boundary sharing"**（R2 reviewer 已确认 "sharper than R1"）。
- **所有 R2 建议都是操作性细化，而非结构性增加**：
  - C2 update rule 精确定义 —— **澄清现有 config 的参数，不增代码**
  - Claim 2 "same-round" 说明 —— **澄清 eval 协议，不加实验**
  - Pre-commit C2 升 3-seed —— **条件式资源分配，不改方法**
  - Thesis 再紧 —— **文字调整**
  - Style matrix reporting —— **分析 artifact 已有数据即可**
- **拒绝的 reviewer 建议（视为不必要复杂度）**：无（R2 reviewer 没有提新复杂度）。
- **为何仍是最小方案**：方法本体仍是"把 classifier.* 加入 sas_keys"，0 行新代码；新增证据（swap、style matrix）都是事后 artifact。

## 已做的修改

### 1. C2 update rule 精确定义（MS=9→10）
- **Reviewer 意见**："C2 needs exact interface definition. 'FedROD-style uniform' is conceptually clear, but server update must be unambiguous."
- **动作**：明确 C2 为：
  > **C2 (uniform-avg per-client classifier)**：每轮 server 将 `head.*` 做**无 style conditioning 的均匀聚合** ——
  > `per_head_i = (1/|S|) · Σ_{j ∈ S} head_state[j]`  for all clients i
  >
  > 等价于"每个 client 收到同一个未加权平均 head"。
  > 与 Plan A（B1，即 sample-count weighted FedAvg）的区别：B1 按 `n_j` 加权，C2 等权（不让 Caltech 的样本数优势干扰）。
  > 与 A2 的区别：A2 用 `softmax(cos(sp_i, sp_j)/τ)` 加权（每 client 得到 unique head）。
- **代码层面**：复用现有 sas 聚合函数，只把 `w_ij = 1/N` (C2 硬编码)。新增 config flag `sas=3` 作为 "uniform-avg per-client classifier" 的 ablation 开关。代码约 10 行（只是 sas 分支多一个 case）。
- **理由**：把含糊的 "FedROD-style" 替换成可执行规则；同时澄清与 B1 的边界。
- **影响**：Claim 1 一天内即可由工程师实现全部 configs。

### 2. 预承诺 C2 3-seed 扩展（VF=8→9）
- **Reviewer 意见**："If A2 − C2 is within modest band, 1-seed is insufficient for top-venue. Pre-commit to 3-seed if needed."
- **动作**：明确决策规则：
  - Office s=2 单 seed 跑完 A2 + B1 + C1 + C2
  - 若 `AVG_Best(A2, s=2) − max(AVG_Best(B1, s=2), AVG_Best(C2, s=2)) ≥ 2.0`：**跳过 C2 3-seed**，单 seed 已足够（大边界）
  - 若 margin `∈ [0.5, 2.0)`：**扩 A2 + C2 到 3-seed**（s=2,15,333），必要时做 paired t-test
  - 若 margin `< 0.5`：**halt**，thesis 被伪证
- **理由**：把 reviewer 的 "if promising" 条件具体化为数值 threshold。
- **影响**：资源预算从"可能 21h"变成"2h triage + 10h 或 6h 追加"。

### 3. Claim 2 "same-round" 澄清（MS）
- **Reviewer 意见**：Global FedAvg head 要从**同一 round/checkpoint** 的 A2 取，避免 confound。
- **动作**：Claim 2 重写为：
  > 加载 A2 R149（best round）checkpoint。Checkpoint **保存了两件东西**：
  > (i) 每个 client 的 personalized `(encoder_i, sem_head_i, head_i)`；
  > (ii) server 端在该轮 pack 前计算的 **global FedAvg head**（即所有 `head_j` 的 sample-weighted mean，这是我们现有 pipeline 已经算过的中间变量）。
  >
  > 交换实验：
  > - (a) 用 `(encoder_i, sem_head_i, head_i)` eval client i 的 test set
  > - (b) 用 `(encoder_i, sem_head_i, global_head)` eval 同一 test set
  >
  > 两条件共享 encoder + sem_head（都是 personalized），**只交换 head**。`global_head` 来自**相同 round 的 pack 前快照**，避免训练动态漂移造成的 confound。
- **影响**：诊断逻辑严谨，可直接写进 paper 的 Section "Mechanism Evidence"。

### 4. Thesis 再紧（VR=8→9）
- **Reviewer 意见**："The key question is not whether to personalize the classifier, but how to route classifier sharing."
- **动作**：Paper 主论点从：
  > "Style similarity is the routing signal for sharing classification decision boundaries."

  改为：
  > **"Under style heterogeneity, the key federated-learning question at the classifier layer is not *whether* to personalize but *how to route* the sharing — and style similarity is the natural routing signal."**
- **理由**：把一个"加 sas to head"升级为一个 **meta-framing**："classifier-layer aggregation is a routing problem."
- **影响**：Intro/Abstract 的 positioning 把 contribution 放到更高抽象层。

### 5. 增加 style similarity matrix 报告（mechanism evidence）
- **Reviewer 意见**："Report style-similarity matrix and resulting routing weights for Office once."
- **动作**：在 Claim 2 之后新增 mini-analysis：
  - 4×4 的 cos-sim 矩阵 `{cos(sp_i, sp_j)}` for Office 4 clients at R149
  - 4×4 的 routing weight 矩阵 `w_ij = softmax(sim / τ)`
  - 用 heatmap 可视化：Caltech 行预期对自己权重最高，对其他 3 个低
- **理由**：提供 **mechanism 的图示证据**，而不是只有 accuracy 数字。top-venue paper 常规做法。
- **影响**：Paper 的 visualization budget + 1 张 figure，但不加实验负担（checkpoint eval 已有数据）。

### 6. PACS 彻底降级 + art_painting 从 success criteria 移除
- **Reviewer 意见**："Remove art-painting from success criteria unless PACS is actually run."
- **动作**：
  - Success condition 从 "art **和** Caltech 各 +2%" 改为 **"Caltech +2%；PACS 作为 scope boundary，仅当 Office 成功时才跑"**
  - PACS 3-seed 运行在 Compute Plan 里标注 "(optional appendix)"，不进入主路径
- **理由**：把 Office 作为唯一 main path，PACS 只是 appendix 的 negative control。
- **影响**：关键路径缩短。

---

# 修订后的提案（Round 2 Final）

# Research Proposal: Style-Conditioned Classifier Routing for Outlier-Domain Recovery in Federated Domain Generalization

## Problem Anchor
[不变 —— 见文件顶部]

## Technical Gap

Plan A sas 仅覆盖 `semantic_head`，classifier `head` 仍 FedAvg。outlier client 获得 personalized 特征映射，但所有 client 共享同一 classification plane → feature right, boundary wrong。

**Missing mechanism**：一个决定 *which clients should share classification boundaries* 的信号。FedROD 用 uniform per-client classifier（无 conditioning）；FedAlign 用 classifier personalization + feature alignment（不用 style）。**核心 claim：style similarity 是 classifier-boundary sharing 的 natural routing signal。**

最小充分方案：把 `classifier.*` 加入 sas_keys。零新参数。

## Method Thesis

- **一句话 thesis**：**Under style heterogeneity, the key federated-learning question at the classifier layer is not *whether* to personalize but *how to route* the sharing — and style similarity is the natural routing signal.**
- **具体实现**：把 FedDSA Plan A 的 sas 覆盖范围从 `semantic_head` 扩到整个 head 链（projection + classifier），零新可训练模块。
- **为何 timely**：FedROD/FedAlign/DualFed 已建立 per-client classifier 为 pFL 范式；我们首次把个性化条件化到 style similarity 上。

## Contribution Focus

- **主导贡献**：Style-conditioned classifier routing (sas-FH)，以 "routing question" framing 作为 paper 的 meta-thesis。
- **可选支撑贡献**：NONE。
- **明确的非贡献**：classifier personalization 本身（FedROD 已做），MixStyle/SupCon/focal loss，BN 策略变更，per-sample / multi-centroid。

## Proposed Method

### Complexity Budget
- **冻结**：Plan A 全架构 / style_proto / FedBN / L_orth / sas softmax / 所有超参。
- **改变**：server pack() `classifier.*` 聚合路径，FedAvg → sas 个性化。
- **新增可训练参数**：0。

### System Overview

Client i: encoder(x) → h; h → sem_head → head → logits; h → style_head → style_proto. Local SGD: CE + λ_orth · L_orth.

Server: 维护 `client_sem_states[j]`, `client_head_states[j]` [NEW], `client_style_protos[j]`. For each target client i:
- `w_{ij} = softmax(cos(sp_i, sp_j) / sas_tau)`
- `per_sem_i = Σ w_{ij} · sem_state[j]`
- `per_head_i = Σ w_{ij} · head_state[j]`  [NEW]
- `encoder, style_head, BN` → standard FedAvg
- 把 personalized `(encoder, sem_head_i, head_i)` 下发给 client i。

### Core Mechanism

**Config switch**:
```python
# sas=0: no personalization (baseline B0 = EXP-083)
# sas=1: sem_head only (B1 = Plan A = EXP-084)
# sas=2: sem_head + classifier, style-conditioned  <- OURS (A2)
# sas=3: sem_head + classifier, uniform-avg        <- C2 counterfactual
# sas=4: sem_head sas + classifier fully local     <- C1 counterfactual
```

**Aggregation rule (for k ∈ sas_keys)**:
```
A2:  per_param_i[k] = Σ_j w_ij · client_state[j][k], w_ij = softmax(cos(sp_i, sp_j)/τ)
C2:  per_param_i[k] = (1/|S|) · Σ_{j ∈ S} client_state[j][k]          # uniform, same for all i
C1:  per_param_i[k] = client_state[i][k]                               # local, no sharing
```

### Integration
File: `FDSE_CVPR25/algorithm/feddsa_scheduled.py` `pack()`. Config: `sas=0..4`. 代码：约 40 行（含 C1/C2 分支 + unit tests）。Client-side 不变。

### Training Plan
R200, E=1, B=50, LR=0.05, decay=0.9998。Seeds {2,15,333}。Office-Caltech10 c4 主路径；PACS c4 可选 appendix。

### Failure Modes
- Caltech 无增益 → Claim 2 swap diagnostic；若 swap 也无效 → classifier 不是 bottleneck → halt。
- 训练不稳定 → 重试 sas_tau=0.5。

### Novelty Argument
vs FedROD: style conditioning，非 uniform。vs FedAlign: 不加 feature alignment。vs Plan A: sas 覆盖整个 head 链。

**Paper story**: *"Classifier aggregation in federated DG is a routing problem, not a scope problem. Style similarity provides the routing."*

## Claim-Driven Validation Sketch

### Claim 1 (Main): **A2 > C2** 隔离出 routing signal

Configurations:
| ID | sem_head | classifier | Interpretation |
|----|----------|------------|---------------|
| B0 | FedAvg | FedAvg | Baseline no sas (EXP-083) |
| B1 | sas | FedAvg (sample-weighted) | Plan A (EXP-084) |
| **A2** (ours) | **sas** | **sas (style-conditioned)** | **Target** |
| **C2** | **sas** | **uniform mean** | **Key counterfactual: is style conditioning the causal factor?** |
| C1 | sas | local-only (no sync) | Secondary: classifier 到底该不该共享? |

实验协议：
1. **Triage (2h)**：Office s=2 单 seed × {A2, C1, C2}；B0/B1 用已有数据
2. **决策规则**：
   - 若 `AVG_Best(A2) − max(AVG_Best(B1), AVG_Best(C2)) ≥ 2.0`：single-seed strong，作为 main result 发表 + 3-seed 追认
   - 若 margin ∈ [0.5, 2.0)：**把 A2 + C2 升到 3-seed**（s=15、s=333），paired t-test
   - 若 margin < 0.5：halt，thesis 被伪证
3. **最终主表**（假设走 promotion path）：
   - A2 3-seed、C2 3-seed、B0 3-seed（已有）、B1 3-seed（已有）
   - C1 1-seed 作为 secondary ablation

Metric: AVG Best / AVG Last / per-domain Caltech Best/Last。

预期：A2 Caltech ≥ 77（EXP-084 75.0 + 2）；A2 AVG ≥ 90.5（超过 FDSE 90.58）；A2 − C2 ≥ 1%。

### Claim 2 (Mechanism Diagnostic, zero-training): feature right, boundary wrong (same-round swap)

加载 A2 R149 best-round checkpoint。Checkpoint 保存 (i) personalized `(encoder_i, sem_head_i, head_i)` 以及 (ii) **same-round** pack-time FedAvg snapshot `global_head`（这是标准 FedAvg 对 `head.*` 参数聚合时 server 端一直会计算的中间量）。

Per client i：
- Condition (a)：`(encoder_i, sem_head_i, head_i)` → eval client i 的 test set
- Condition (b)：`(encoder_i, sem_head_i, global_head)` → eval 同一 test set

**Same-round guarantee**：`global_head` 是 R149 pack 阶段 **没下发** 的 FedAvg snapshot，与 `head_i` 同 round，避免训练动态漂移 confound。

预期信号：
- Caltech：(a) > (b) 至少 2%
- Amazon/DSLR/Webcam：(a) ~ (b)，Δ < 0.5%

若信号成立：直接证明 classifier sharing 专门伤害 outlier，问题定位在 head 层。

### Claim 3 (Mechanism Visualization): style similarity matrix + routing weights

在 A2 R149：提取 Office 的 4×4 矩阵 `{cos(sp_i, sp_j)}` 和 `{w_ij = softmax(sim / sas_tau)}`。以 heatmap 可视化。

预期：Caltech 行对自己权重非常集中（> 0.7）、对他人很低；Amazon/DSLR/Webcam 行则更分散。

支撑 thesis："style similarity is the routing signal" 不是空话，而是可直接看到的权重分布。

### Claim 4 (Negative Control / Scope Boundary, OPTIONAL): PACS 展示 mechanism limits

仅当 Claim 1 成功时才跑。PACS c4 3-seed A2。预期：AVG ≤ EXP-086 Plan A 79.76。解释：PACS 4 个域互为 outlier，无 majority reference → routing 无意义。Paper appendix。

## Experiment Handoff Inputs

- **Must-prove**：Claim 1 A2 > C2（核心 novelty）；Claim 2 swap 信号成立。
- **Must-isolate**：A2 > C2 ≥ 1%（style conditioning 是 causal factor）
- **关键数据集 / 指标**：Office-Caltech10 c4 AVG Best、Caltech Best/Last。
- **最高风险假设**：
  1. style_proto 能区分 Caltech ✓（EXP-084 已验证）
  2. classifier 层是 bottleneck（Claim 2 直接测试）
  3. style conditioning > uniform（Claim 1 A2 vs C2 直接测试）

## Compute & Timeline

| Experiment | Runs | Per run | Serial |
|-----|-----|--------|------|
| Triage: Office A2 + C1 + C2 × 1-seed s=2 | 3 | 2h | 6h |
| If A2−C2 ∈ [0.5, 2): extend A2 + C2 to 3-seed (s=15, s=333) | 4 | 2h | 8h |
| Claim 2 swap diagnostic (checkpoint eval) | — | 0.5h | 0.5h |
| Claim 3 style matrix viz | — | 0.2h | 0.2h |
| Claim 4 PACS A2 × 3-seed (optional) | 3 | 3.5h | 10.5h |
| **Must-run (strong-margin path)** | 3 | — | **~6.5h** |
| **Must-run (statistical path)** | 7 | — | **~14.5h** |
| **Full (incl PACS)** | 10 | — | ~25h |

关键路径：Triage 2h → 决策 gate。要么 single-seed 可发表，要么 3-seed 追认。
