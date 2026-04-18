# Round 1 Refinement

## 问题锚点（与 round 0 一致）

- **底线问题**：FedDSA Plan A（sas 仅个性化 `semantic_head`，`classifier head` 与 backbone 通过 FedAvg 聚合）在 outlier 域严重失败。3-seed 证据：PACS art_painting 65.69%（其他 83-89%），Office Caltech 77.68%（其他 89-100%）。70 个 art 错例全部 conf ≥ 0.85，错分集中在有机体类互串（person↔dog↔horse↔guitar）。
- **必须解决的瓶颈**：当同一模型在多域共用分类器时，outlier 域的分类决策边界被其他域的语义偏好稀释。
- **Non-goals**：不换骨干、不加新可训练模块、不改推理方式、不做 per-sample sas。
- **约束条件**：3-seed {2,15,333} × R200 × E=1；SC2 单卡 2-4h；代码改动 ≤ 50 行；与 EXP-083/084/086 完全可比。
- **成功条件**：art/Caltech outlier 各 +2% AVG Best，其他域 drop < 1%，整体 PACS ≥ 80.41 / Office ≥ 90.58。

## Anchor Check

- **原始瓶颈**：classifier 共享拖累 outlier 域的类决策边界。
- **修订后方法仍解决此瓶颈**：是。Reviewer 明确 "This stays on the anchored bottleneck" (PF 9/10)。核心机制 sas-FH 不变。
- **被拒绝为 drift 的 reviewer 建议**：无。Reviewer 反而警告我 A3 (encoder last block) 和 Top-2 multi-centroid 若升级到主 story **会** drift，这与我的 non-goals 一致。

## Simplicity Check

- **修订后的主贡献**：仍是 sas-FH（仅一个，无新增），但 **thesis 从"scope extension"收紧为"style similarity is the right routing signal for classifier-boundary sharing"**。
- **移除或合并的组件**：
  - ❌ A3 ablation（encoder last block）：删掉，reviewer 点名它模糊"smallest adequate"故事
  - ❌ Top-2 multi-centroid fallback：从 Failure 2 的 fallback 里移除，另起 future-work
  - ❌ PACS co-equal claim：降级为 negative-control，只当作 "scope boundary" 证据
- **被拒绝为不必要复杂度的 reviewer 建议**：无。reviewer 本身建议的两个新增（counterfactual baseline + swap diagnostic）都是**替换**我原有实验的，不增复杂度。
- **为什么剩下的机制仍是最小的**：整个方法仍是 "把 `classifier.*` 加入 sas_keys"，30 行代码。新增的 counterfactual 是 ablation（不改方法本身），swap 是事后诊断（不改训练）。

## 已做的修改（Changes Made）

### 1. Validation：删 A3，加 matched counterfactual（CRITICAL fix）
- **Reviewer 说**："sem-only vs full-head 不足以证明 novelty"，因为可能只是 "classifier 个性化"本身的红利，未必是 "style conditioning"。
- **行动**：
  - 删除 Claim 2 里的 A3 (sas=3, encoder last block)
  - 新增 **Counterfactual baseline C1**：`sem_head sas + local classifier`（classifier 完全本地化，等同 FedROD local-personalized head）
  - 新增 **Counterfactual baseline C2**：`sem_head sas + uniformly per-client classifier`（per-client classifier 但不按 style conditioning，uniform average of all client classifiers 作为基准）
  - 如果 C1 ≈ sas-FH 或 C1 > sas-FH → style conditioning 无效
  - 如果 C2 < sas-FH 明显 → style conditioning 确实是关键
- **理由**：这是 reviewer 指出的唯一 CRITICAL 问题。没有这个对照，我们的 claim 等价于 FedROD 的重新包装。
- **影响**：主 validation 从"granularity sweep"变成"isolate the conditioning signal"，故事收紧一个层级。

### 2. Validation：加 classifier-swap mechanism diagnostic
- **Reviewer 说**：需要直接测试 "feature right, boundary wrong" thesis。
- **行动**：新增 **Claim 3 (Mechanism diagnostic)**：在 R150 freeze sas-FH checkpoint，对每个 client i：
  - 保持 personalized `encoder + sem_head`
  - 分别挂 (a) personalized `head_i` (b) global FedAvg `head_global`
  - 对比两组在 client i 的 test set 上的准确率
  - 若 Caltech 上 (a) > (b) 显著，而其他域 (a) ≈ (b)，则直接证明"outlier 域的 boundary 确是病根"
- **理由**：零额外训练（只是 checkpoint eval），但证据直击 thesis。
- **影响**：增 1 个小实验（~30 min），换来 mechanism 可视化证据。

### 3. Thesis 收紧（IMPORTANT fix）
- **Reviewer 说**：故事还像 "scope tweak"。
- **行动**：主 thesis 从
  > "Extend Plan A's sas to the full head chain for outlier recovery"

  改为
  > **"Style similarity is the natural routing signal for sharing classification decision boundaries, and classifier-level personalization is the minimal sufficient granularity under style-driven domain outliers."**
- **理由**：把 "把 key 加进 sas_keys" 这种实现细节抽象成 "routing signal for boundary sharing" 这种概念贡献。
- **影响**：Abstract/intro 的故事更干净，reviewer 不会问 "why not just an ablation"。

### 4. PACS → negative control 定位
- **Reviewer 说**：PACS 应当作 "scope boundary"，不做 co-claim。
- **行动**：
  - 从 main contributions 移除 PACS
  - PACS 仅在 Claim 3 的 boundary-condition 部分出现（"mechanism only works under majority-similar + few-outliers distribution"）
  - PACS 3-seed 运行降级为**可选**（仅当 Office 成功才跑）
- **理由**：EXP-086 已经证明 PACS 上 Plan A 失败（-0.65/-1.03），sas-FH 没理由扭转。硬上反而削弱故事。
- **影响**：减 10.5h compute 风险（PACS 3-seed），总实验从 20.5h 缩到 ~10h。

### 5. 删除 Top-2 multi-centroid fallback
- **Reviewer 说**：不同 method direction，稀释 elegance。
- **行动**：从 Failure Mode 2 的 fallback 里删除"switch to multi-centroid"，改为"halt and diagnose style_proto direction"。
- **理由**：Top-2 是另一条路（multi-proto），不是 sas-FH 的 fallback。
- **影响**：proposal body 更干净，不引用 Future work。

---

# 修订后的提案（Revised Proposal）

# 研究提案：Style-Conditioned Classifier Routing 用于联邦域泛化中的 outlier 域恢复

## 问题锚点

[与 round 0 一致，原文保留]
- 底线问题、必须解决的瓶颈、non-goals、约束条件、成功条件 — 见上。

## Technical Gap

Plan A sas 仅覆盖投影头 `semantic_head`，最后的线性分类器 `head` 仍 FedAvg。
- sas = 服务器按 `softmax(cos(sp_i, sp_j)/τ)` 为每个 client 生成 personalized `sem_head` 权重
- 但 `head (1024→num_classes FC)` 是**唯一决定类决策边界**的层
- Outlier client 收到的 personalized `sem_head` 让特征向自己风格靠，**但所有 client 共享同一 classification plane**，plane 向 majority 偏好倾斜 → feature right, boundary wrong

**缺失的机制**：一个把 style similarity 用作"该不该共享 class boundary"的 routing 信号。FedROD 做了 uniform per-client classifier（无 conditioning），FedAlign 做了 classifier personalization + feature alignment（不用 style）。**我们首次把 style similarity 作为 classifier-sharing 的 routing signal**。

**最小充分干预**：把 `classifier.*` keys 移入 sas_keys，零新参数。

**核心 claim**：*Style similarity is the natural routing signal for sharing classification decision boundaries; classifier-level personalization is the minimal sufficient granularity under style-driven outlier domains.*

## Method Thesis

- **一句话 thesis**：**Style similarity is the right routing signal for deciding which clients should share classification decision boundaries**；应用于 FedDSA Plan A，即把 style-aware aggregation 从单独的 semantic projection head 扩展到整个 head 链（projection + classifier），不引入任何新的可训练组件。
- **为什么是最小充分干预**：服务器 pack() 阶段把 `classifier.*` 加入 sas 聚合 key 集合，30 行代码。
- **为什么 timely**：2023-25 pFL 领域（FedROD / FedAlign / DualFed）已证 per-client classifier 是 outlier-friendly 范式。我们的 contribution 是**把这个 personalization 条件化到 style similarity 上**，让 outlier 获得比 uniform/random 更好的起点。

## Contribution Focus

- **主贡献**：**Style-conditioned classifier routing (sas-FH)** — 把 style-aware aggregation 的覆盖范围从半头扩到整个 head 链，**但核心 novelty 是 "style similarity 作为 classifier-sharing 的条件信号"**，不是 scope extension 本身。
- **可选支撑贡献**：NONE（原 Granularity Ablation 被 reviewer 指出不够 isolate，已改为 counterfactual baseline，是 Claim 1 的一部分而非 parallel contribution）。
- **明确的非贡献**：
  - 不 claim classifier personalization 本身的 novelty（FedROD 已做）
  - 不加 MixStyle / SupCon / focal loss
  - 不改 BN 策略
  - 不做 per-sample / multi-centroid aggregation（不同方向）

## 提出的方法

### 复杂度预算
- **冻结 / 复用**：Plan A 全架构（AlexNet + sem_head + style_head + classifier）、style_proto 提取、FedBN BN-local、L_orth、sas softmax、所有超参
- **改动**：server pack() 中 `classifier.*` 的聚合路径，从 FedAvg 改到 sas 个性化
- **新增可训练组件**：0
- **刻意排除**：per-client sas_tau、style_head 个性化、backbone sas、encoder last-block sas、multi-centroid

### 系统概览

```
Client i:                                    Server:
  encoder(x) → h                              Maintain client_sem_states[j]
  h → sem_head → z_sem → head → logits       Maintain client_head_states[j]  [NEW in sas-FH]
  h → style_head → z_sty → style_proto       Maintain client_style_protos[j]
  
  Local SGD: CE + λ_orth · L_orth             For each target client i at pack time:
                                                w_{ij} = softmax(cos(sp_i, sp_j)/τ)
  Upload: encoder, sem_head, style_head,         per_sem_i  = Σ w_{ij} · sem_state[j]
          head, style_proto                      per_head_i = Σ w_{ij} · head_state[j]  [NEW]
                                                encoder, style_head, BN → FedAvg
                                              Send personalized (encoder, sem_head_i, head_i) to i
```

### 核心机制

- **输入**：各 client 最新上传的 `classifier.state_dict()` 和 `style_proto`
- **输出**：每个 client 的 personalized classifier 权重，下一轮下发
- **改动**：
  ```python
  # Before (Plan A)
  sas_keys = [k for k in global_dict if k.startswith('semantic_head.')]
  # After (sas-FH)
  sas_keys = [k for k in global_dict if k.startswith('semantic_head.') or k.startswith('head.')]
  ```
- **聚合规则不变**：`w_{ij} = softmax(cos(sp_i, sp_j) / sas_tau); per_param_i[k] = Σ_j w_{ij} · client_state[j][k]` for `k ∈ sas_keys`
- **客户端训练不变**：CE + L_orth，无新 loss/增强

### 现代原语使用

**无**。Reviewer 确认："FM-era machinery would change the paper, not solve the stated bottleneck."

### 与 pipeline 的集成

- 改动文件：`FDSE_CVPR25/algorithm/feddsa_scheduled.py` `pack()` 方法
- 代码量：~30 行（含 config flag `sas=2`）
- 与 baseline 对比：仅 sas 聚合范围不同，保证公平

### 训练计划
- 单阶段联合训练（同 Plan A）
- Data：Office-Caltech10 c4（主 benchmark），PACS c4（negative control only）
- Loss：CE + λ_orth · L_orth
- Schedule：R200, E=1, B=50, LR=0.05, decay=0.9998
- Seeds：{2, 15, 333}
- 无 warmup / curriculum / pseudo-labels

### 失败模式与诊断

- **Failure 1**：Office Caltech 无提升
  - 检测：R100 Caltech AVG < 75.5
  - Fallback：用 Claim 3 的 swap diagnostic 定位 —— 若 swap 后 Caltech 无差 → classifier 不是 bottleneck，说明 Plan A 的 sem_head 已接近 ceiling，应转研究方向（not this paper）
- **Failure 2**：训练不稳定（drop > 3%）
  - 检测：AVG Last < AVG Best - 3%
  - Fallback：sas τ=0.3 在 head 级过 sharp，改 τ=0.5 重跑
- **诊断信号**：
  - 追踪 cos_sim(personalized_head_i, global_head) per client per round
  - 追踪 ‖personalized_head_i − old_personalized_head_i‖ 确认收敛

### Novelty 与 Elegance 论证

**最近工作**：
- FedROD (ICLR 23)：per-client classifier + shared backbone，**uniform** aggregation
- FedAlign (CVPR Workshop 25)：classifier personalization + cross-client feature alignment
- FedDSA Plan A (EXP-084)：sas only on sem_head

**精确差别**：
- vs FedROD：style similarity 决定 classifier sharing 方向（not uniform）
- vs FedAlign：不加 feature alignment，纯 aggregation-level change
- vs Plan A：把 sas 的 conditioning 范围从 sem_head 扩到 classifier

**Paper story（一句话）**：*Style similarity is the natural conditioning signal for sharing classification decision boundaries under domain heterogeneity.*

## Claim 驱动的验证草图

### Claim 1（主）：sas-FH 在 outlier-heavy Office 上显著提升 outlier 域，且**style conditioning 本身是关键**（通过 counterfactual baselines 隔离）

- **主实验**：Office-Caltech10 c4 × 3-seed × R200 × E=1
- **配置**：
  | Config | sem_head | classifier | 预期 |
  |--------|----------|-----------|---------|
  | B0 = EXP-083 | FedAvg | FedAvg | baseline (sas=0) |
  | B1 = EXP-084 | **sas** | FedAvg | Plan A (sas=1) |
  | **A2 = sas-FH (ours)** | **sas** | **sas** | **target** (sas=2) |
  | **C1 counterfactual** | **sas** | **local only**（per-client，不共享） | 测试：classifier 到底需不需要共享？ |
  | **C2 counterfactual** | **sas** | **uniform per-client**（FedROD-style） | 测试：style conditioning 是否关键？ |
- **决定性对比**：
  - A2 > B1：full-head sas 优于 half-head sas（主效应）
  - A2 > C1：classifier 应该共享，而不是完全本地
  - A2 > C2：style conditioning > uniform conditioning
- **指标**：AVG Best / AVG Last / per-domain Caltech Best/Last
- **预期**：A2 Caltech 77+，AVG 90+，C1/C2 < A2 by 0.5-1.5%

### Claim 2（Mechanism Diagnostic）："feature right, boundary wrong" 直接验证

- **实验**：载入 sas-FH R150 best checkpoint，**零训练**，per client swap：
  - (a) personalized encoder + personalized sem_head + **personalized head_i**
  - (b) personalized encoder + personalized sem_head + **global FedAvg head**
- **预测 signature**：
  - Caltech：(a) > (b) by > 2%
  - Amazon/DSLR/Webcam：(a) ≈ (b)，Δ < 0.5%
- **若 signature 成立**：直接证明 classifier sharing 专门伤害 outlier
- **若 (a) ≈ (b) 在 Caltech 上也成立**：thesis 被伪证，转回 Plan A 不再追 full-head

### Claim 3（Negative Control / Scope Boundary）：PACS 展示机制的适用边界

- **实验**：PACS c4 × 3-seed × R200，**可选**（仅当 Office A2 成功才跑）
- **预测**：AVG Best ≤ EXP-086 Plan A 79.76
- **为什么仍有用**：验证 EXP-086 的 mechanism 假设 — style-conditioning 仅在 "majority-similar + few-outliers" 分布下有效。PACS 4 域彼此都是 outliers → 共享 classifier 的 reference 就不存在。
- **在论文中的定位**："Our mechanism's scope boundary, validated on PACS."

## 实验交接输入

- **必证 claims**：Claim 1（Office 3-seed + 2 counterfactual single-seed），Claim 2（Mechanism swap diagnostic）
- **必跑 ablations**：Claim 1 B1 vs A2 vs C1 vs C2（固定 seed ≥ 1 的 4-way 对比，仅 A2 扩到 3-seed）
- **关键数据集 / 指标**：Office-Caltech10 c4 AVG Best, Caltech Best/Last
- **最高风险假设**：
  1. style_proto 足以区分 Caltech（✅ 已由 EXP-084 验证）
  2. classifier 层是 bottleneck（Claim 2 swap diagnostic 直接测试）
  3. style conditioning > uniform（Claim 1 C2 直接测试）

## 计算与时间估计

| 实验 | Runs | 单 run | 串行总时 |
|-----|-----|--------|------|
| Claim 1 A2 × 3-seed (Office) | 3 | 2h | 6h |
| Claim 1 C1, C2 × 1-seed (Office) | 2 | 2h | 4h |
| Claim 2 swap diagnostic（checkpoint eval，不训练） | 1 | 30min | 0.5h |
| Claim 3 PACS A2 × 3-seed（**可选**） | 3 | 3.5h | 10.5h（若 Claim 1 失败则跳过） |
| **必做小计** | 6 | — | **~10.5h** |
| **含 PACS negative control** | 9 | — | ~21h |

- GPU-hours（必做）：~10
- 代码实现：1h（含单元测试 + counterfactual config）
- Critical path：Office A2 s=2 单 seed 2h → 若 Caltech > 77 扩 3-seed + 2 counterfactuals
