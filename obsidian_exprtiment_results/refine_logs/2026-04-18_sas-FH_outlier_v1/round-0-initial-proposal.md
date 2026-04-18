# 研究提案：Full-Head Style-Aware Aggregation (sas-FH) 用于联邦域泛化中的 outlier 域恢复

## 问题锚点（Problem Anchor）

- **底线问题**：FedDSA Plan A（sas 仅个性化 `semantic_head`，`classifier head` 与 backbone 通过 FedAvg 聚合）在 outlier 域严重失败。3-seed 证据：PACS art_painting Client 0 AVG 仅 65.69%（其他 3 域 83-89%），Office Caltech 77.68%（其他 3 域 89-100%）。错分模式显示 70 个 art 错例全部 conf ≥ 0.85（自信地错），错分集中在有机体类互串（person↔dog↔horse↔guitar）。
- **必须解决的瓶颈**：当同一模型在多域共用分类器时，**outlier 域的分类决策边界被其他域的语义偏好稀释**。验证：EXP-086 PACS Plan A 对所有 4 域都 -0.65/-1.03 AVG，因为 PACS 4 域都是彼此 outlier，share classifier 双向拖累；EXP-084 Office Plan A Caltech +2.4/+3.6 但仍距 FDSE -3.9%（Caltech Best），说明 sem_head 个性化已抓到一部分红利但**未抓完**。
- **Non-goals**：
  - 不换骨干（保持 AlexNet 对齐 FDSE baseline）
  - 不引入新的可训练模块或新损失项
  - 不改推理方式（TTA / temperature scaling 不算）
  - 不做 per-sample sas（保持 client-level sas 的通信效率）
- **约束条件**：
  - 3-seed {2, 15, 333} × R200 × E=1 对齐 FDSE 实验协议
  - SC2 单卡串行，单 run 2-4h（PACS ~3.5h, Office ~2h）
  - 代码改动 ≤ 50 行，可在 1h 内完成 + 单元测试
  - 与已跑 baseline（EXP-083 / EXP-084 / EXP-086）完全可比（同 config 仅 sas 聚合范围不同）
- **成功条件**：在不破坏其他域的前提下：
  1. **主指标**：art_painting Client 0 AVG Best **+2% 以上**（即 ≥ 67.7%），Caltech Client 0 AVG Best **+2% 以上**（即 ≥ 77.0% → 逼近 FDSE 78.9%）
  2. **副指标**：其他域 drop < 1%
  3. **整体**：PACS AVG Best 回到 ≥ baseline orth_only 80.41，Office AVG Best ≥ FDSE 90.58

## Technical Gap

**当前 Plan A 的 sas 只覆盖 projection head (`semantic_head`)，最后的线性分类器仍然 FedAvg**。
- sas = Style-Aware Semantic head aggregation，服务器按 `cos_sim(style_proto_i, style_proto_j) / τ` 的 softmax 为每个目标 client 计算个性化的 sem_head 聚合权重
- classifier `head` 只有 1024→num_classes 的 FC，**却是唯一决定类间决策边界的层**
- art 作为 style outlier，收到的 personalized `sem_head` 让特征朝自己的 style 方向走，**但所有 client 共享同一条分类平面** → 即便特征对了，决策平面向 photo/sketch 偏好倾斜（因为它们占 sem majority）

**为什么"加大模型 / 加更多 epoch / 加更多数据增强"不行**：
- 所有这些都不解决"分类器共享"这个结构问题
- 加数据增强（如 MixStyle）可能反让 art 更混乱
- 加 epoch 会让共享分类器过拟合 majority domains，outlier 更差

**最小充分干预**：将 sas 的聚合粒度从 "仅 sem_head" 扩展到 "sem_head + classifier head"。
- 零新增可训练参数（只改 server 聚合时的 key 分组）
- 理论 motivation 与 FedROD (ICLR 23) "per-client classifier" 一致
- 与 FedDSA 原架构 100% 兼容，不影响 backbone/style_head/BN 的现有处理

**核心技术 claim**：*当风格异质性导致不同客户端需要不同的分类器决策边界时，将 style-aware aggregation 从单一投影头扩展到整条 head 链（sem_head + classifier）是恢复 outlier 域准确率的最小充分机制，无需新增可训练模块或损失项。*

**所需证据**：
1. 在 PACS/Office 上 sas-FH 对比 sas-半头（EXP-084 Plan A）显示 outlier 域显著提升
2. granularity ablation 证明"全头 sas 比 半头 sas 更好，比 全模型 sas 不差"（寻找 sweet spot）

## Method Thesis

- **一句话 thesis**：将 FedDSA Plan A 的 style-aware aggregation 从单独的 semantic projection head 扩展到整个 head 链（semantic projection + final classifier），使 outlier 客户端同时获得个性化的特征映射**和**个性化的类决策边界——无需任何新的可训练组件。
- **为什么是最小充分干预**：只修改 server 聚合时的参数分组（把 `classifier.weight/bias` 从 shared_keys 移入 sas_keys），不加任何新模块、新损失、新数据增强。
- **为什么这条路线 timely**：FedROD (ICLR 23) 已证实 per-client classifier + shared backbone 是 pFL 的有效范式；FedAlign (CVPR Workshop 25) 在 FedDG 场景也采用 classifier personalization。我们的贡献是 **把 classifier personalization 条件化到 style similarity 上**（而非均匀 per-client），让 outlier 客户端获得比随机初始化更好的起点。

## Contribution Focus

- **主贡献**：*Full-head sas (sas-FH)* — 把 style-aware personalized aggregation 的覆盖范围从半头扩展到整个 head 链，作为 Plan A 的唯一结构延伸。
- **可选支撑贡献**：*Granularity sweet-spot analysis* — 做 4 档 ablation（no-sas / sem-head-only / full-head / full-head + encoder-sas）证明 full-head 是性能拐点，也回答 reviewer 常问的"为什么不 sas 一切"。
- **明确的非贡献**：
  - 不提 classifier personalization 的 novelty（FedROD 已做），只讲"style-conditional classifier personalization"
  - 不加 MixStyle / SupCon / focal loss 等独立增强
  - 不改 BN 策略（保持 FedBN 本地化，与 baseline 对齐）
  - 不做 per-sample aggregation

## 提出的方法（Proposed Method）

### 复杂度预算（Complexity Budget）
- **冻结 / 复用**：FedDSA Plan A 整个架构（encoder AlexNet + semantic_head + style_head + classifier head）、style_proto 提取流程、FedBN BN 本地化、orthogonal loss L_orth、sas softmax 公式、所有超参（sas_tau=0.3, lambda_orth=1.0 等）
- **改动**：仅 server `pack()` 时对 `classifier.weight` / `classifier.bias` 的聚合路径 — 从 shared（FedAvg）改到 sas-personalized（与 sem_head 同一套权重）
- **新增可训练组件**：**0**
- **刻意不采用的诱人添加**：
  - per-client sas_tau（绕过，先看 global τ 是否已够）
  - style_head 个性化（不做，因为 style_head 正是用来提取 style_proto 的，个性化会破坏 style similarity 的可比性）
  - backbone sas（不做，会引入过多通信/计算，且 EXP-051 FDSE 的差异化 backbone 聚合复杂度高）
  - classifier bias-only sas（不做，weight 和 bias 一起更自然）

### 系统概览（System Overview）
```
Client i:                                    Server:
  encoder(x) → h                              Maintain latest client_sem_states[j]
  h → sem_head → z_sem → head → logits       Maintain latest client_head_states[j]  ← NEW
  h → style_head → z_sty → style_proto       Maintain latest client_style_protos[j]
                                              
  Local SGD: CE(logits, y) + λ_orth · L_orth  For each target client i at pack time:
                                                 w_{ij} = softmax(cos(sp_i, sp_j)/τ)  
  Upload: encoder, sem_head, style_head,           per_sem_i  = Σ w_{ij} · sem_state[j]
          head(NEW: sent to sas branch),           per_head_i = Σ w_{ij} · head_state[j]  ← NEW
          style_proto                              encoder, style_head, BN → standard FedAvg
                                                Send personalized (encoder, sem_head_i, head_i)
```

### 核心机制（Core Mechanism）

- **输入**：Client i 最近上传的 `classifier.state_dict()` 和 `style_proto_i`；Server 维护所有 client 的 classifier states 和 style protos。
- **输出**：Personalized classifier head 权重 for client i，下发作为 client i 下一轮训练初始状态。
- **架构或策略**：无新模块。改动仅在 server 的 pack 函数中把 `classifier.*` 的 key 和 `sem_head.*` 合并处理：

  ```python
  # Before (EXP-084 Plan A):
  sas_keys = [k for k in global_dict if k.startswith('semantic_head.')]
  
  # After (sas-FH):
  sas_keys = [k for k in global_dict if k.startswith('semantic_head.') or k.startswith('head.')]
  ```

  聚合公式不变：
  ```
  w_{ij} = softmax(cos(sp_i, sp_j) / sas_tau), j = 1..N
  per_param_i[k] = Σ_j w_{ij} · client_state[j][k], for k in sas_keys
  ```

- **训练信号 / 损失**：**完全不变**。Client 端 forward / backward / CE / L_orth 都不动。
- **为什么这是主 novelty**：
  - FedROD 给每个 client 一个独立 classifier（per-client personalization，均匀）
  - FedAlign 把 classifier 个性化与 cross-client feature alignment 结合
  - **我们首次把 style-conditional aggregation 应用到 classifier 级**：利用 style similarity 决定 "两个 client 是否应该共享分类边界"
  - 对"多数相似 + 少数 outlier"分布（Office Caltech 式）特别有效，对"全部 outlier"分布（PACS 4 域彼此异）的退化行为与 EXP-086 一致（预期仍 neutral-to-slightly-positive，待验证）

### 现代原语使用（Modern Primitive Usage）

**无**（无 LLM/VLM/Diffusion/RL 元素）。这是故意的：
- 本课题的 compute 约束（AlexNet R200 3-seed）不适合上 foundation models
- 核心 gap 是"classifier 共享"，foundation models 解决不了这个结构问题
- 走 "minimal structural extension" 路线比 "trendy add-on" 更诚实

### 与现有 pipeline 的集成

- **改动位置**：`FDSE_CVPR25/algorithm/feddsa_scheduled.py` 的 `pack()` 方法（约 30 行）
- **改动前**：`sas_keys = [k for k in global_dict if k.startswith('semantic_head.')]`（Plan A 现状）
- **改动后**：`sas_keys = [k for k in global_dict if k.startswith('semantic_head.') or k.startswith('head.')]`
- **完全冻结**：client 端代码、style_proto 提取、L_orth、所有超参默认值、单元测试（除新增 head-sas 覆盖测试外）
- **Config 新增**：`sas=2`（0=off, 1=sem-only=Plan A, 2=sem+head=Plan A-FH）— **开关而已，不改默认数据流**

### 训练计划（Training Plan）

- **Stage / joint**：单阶段联合训练（与 Plan A 一致），不需要 warmup 或 stagewise 策略
- **Data**：PACS 4 域 c4 / Office-Caltech10 c4（与 EXP-084/086 相同）
- **Loss**：CE + λ_orth · L_orth（与 Plan A 相同）
- **Schedule**：R200, E=1, B=50, LR=0.05, decay=0.9998（对齐 FDSE）
- **Seeds**：{2, 15, 333} 对齐 FDSE 官方
- **无 warmup / 无 curriculum / 无 pseudo-labels / 无新数据增强**

### 失败模式与诊断（Failure Modes and Diagnostics）

- **Failure 1**：PACS 仍整体 -0.5% 或更差（同 EXP-086 无效）
  - **如何检测**：R50 时 PACS AVG Best 仍 < 79.5
  - **Fallback**：回到 Plan A sem-head-only；在论文里把 sas-FH 作为 "Office-applicable, PACS-neutral" 定位（与 EXP-086 的 mechanism analysis 一致）

- **Failure 2**：Office Caltech 持平不涨
  - **如何检测**：R100 时 Caltech AVG < 75.5
  - **Fallback**：检查是否 style_proto 方向变了（EXP-084 的 Caltech outlier 位置应在 sas-FH 下更凸显）；若 style_proto 未变，说明 classifier 本就不是 bottleneck → 转向 Top-2 (multi-centroid)

- **Failure 3**：训练不稳定（drop > 3%）
  - **如何检测**：任一 seed 的 AVG Last < AVG Best - 3%
  - **Fallback**：这不该发生（纯聚合变化，不引入新梯度流），若发生说明 sas τ=0.3 在 head 级太极端，改 τ=0.5

- **诊断信号**（训练时 log）：
  - 追踪 `cos_sim(personalized_head, global_head)` per client per round，看 outlier 域是否真的偏离
  - 追踪 `‖personalized_head - old_personalized_head‖` per client，看收敛是否稳定

### Novelty 与 Elegance 论证

- **最近工作**：
  - FedROD (ICLR 23)：per-client classifier, uniform aggregation — 我们是 style-conditional aggregation
  - FedAlign (CVPR Workshop 25)：classifier personalization + feature alignment — 我们不加 feature alignment（保持 Plan A 的 orth 就够）
  - FedDSA Plan A (EXP-084)：sas only on sem_head — 我们扩到 full head
- **精确差别**：三步归因
  1. 保持 Plan A 的 style-aware aggregation 机制
  2. 把聚合范围从 "feature mapping head" 扩到 "feature mapping + class boundary"
  3. 保证 classifier personalization 的方向由 style similarity 而非随机/uniform 决定
- **为什么 focused**：整个论文故事 = "1 篇 paper 1 句话"：*Style similarity is the natural conditioning signal for deciding which clients should share classification decision boundaries.*

## Claim 驱动的验证草图

### Claim 1（主）：sas-FH 在 outlier-heavy Office 上显著改善 outlier 域（Caltech），整体 AVG Best 提升 ≥ 1%

- **最小实验**：Office-Caltech10 c4 × 3-seed × R200 × E=1
  - EXP-093-A: FedDSA + sas-FH (sas=2, sas_tau=0.3, others 同 EXP-084)
- **Baselines**：
  - EXP-083 (Plan A sas=0, orth_only) — 已有 3-seed data
  - EXP-084 (Plan A sas=1, sem-head-only) — 已有 3-seed data
  - FDSE (EXP-051) — 已有 3-seed data
- **指标**：AVG Best, AVG Last, per-domain Caltech Best/Last
- **预期证据**：
  - Caltech AVG Best 75.0 (EXP-084) → ≥ 77.0 (sas-FH)，逼近 FDSE 78.9
  - AVG Best 89.82 (EXP-084) → ≥ 90.5, 超 FDSE 90.58
  - 其他 3 域 drop < 1%

### Claim 2（支撑）：sas granularity sweet-spot analysis 证明 full-head 是拐点

- **最小实验**：Office-Caltech10 × 1 seed × R200 (just s=2)
  - A0: sas=0 (no sas) ← EXP-083
  - A1: sas=1 (sem-head only) ← EXP-084
  - A2: sas=2 (sem-head + classifier) ← NEW
  - A3: sas=3 (sem-head + classifier + encoder last block) ← NEW as ablation
- **Baselines**：same 4 configs above
- **指标**：AVG Best / Caltech Best
- **预期证据**：
  - Monotone increase A0 → A1 → A2
  - A3 ≤ A2 或 A3 ~= A2（full-model sas 不再带来收益，甚至退化）
  - ⇒ "full-head" 是最优 granularity

### Claim 3（诊断，可选）：PACS 的 mechanism failure 的统一解释

- 重跑 PACS sas-FH × 3-seed (与 EXP-086 相同 config 仅 sas=2)
- 若 PACS 整体仍 ≤ baseline：说明 PACS 4 域均为 outlier，classifier share 本就没有 majority 可"分享"
- 这验证了 EXP-086 的 mechanism 假设："sas 仅在 多数相似+少数 outlier 分布下有效"

## 实验交接输入（Experiment Handoff Inputs）

- **必证 claims**：Claim 1 (Office outlier recovery)
- **必跑 ablations**：Claim 2 (granularity sweet-spot), Claim 3 (PACS mechanism consistency)
- **关键数据集 / 指标**：Office-Caltech10 c4 (AVG Best, Caltech Best/Last), PACS c4 (AVG Best)
- **最高风险假设**：
  1. style_proto 足够区分 Caltech vs 其他 3 个 Office 域（EXP-084 已验证：Caltech +3.6% 说明 style similarity 信号成立）
  2. classifier layer 是 bottleneck（未直接验证，是本实验要验证的）
  3. 零新训练参数 ⇒ 不会引入不稳定性（基于 Plan A 已验证的稳定性）

## 计算与时间估计（Compute & Timeline Estimate）

| 实验 | Runs | 单 run | 串行总时 | 并行 (SC2+Lab-lry) |
|-----|-----|--------|---------|-------------------|
| Claim 1: Office sas-FH × 3-seed | 3 | 2h | 6h | 3h (2 台) |
| Claim 2: Office granularity ablation × 1 seed × 4 configs (2 个新) | 2 | 2h | 4h | 2h |
| Claim 3: PACS sas-FH × 3-seed | 3 | 3.5h | 10.5h | 5.5h |
| **总计** | 8 | — | **20.5h** | **~10.5h** |

- **GPU-hours**：~20 GPU-hours (单卡)
- **代码实现**：1h (含单元测试)
- **监控 + 回填**：每 30min ssh check，总 ~1h 人工
- **论文 critical path**：Office 数据 3h 出信号 → 若通过扩展 PACS

快速止损路径：先跑 Office s=2 单 seed（2h），看 Caltech 是否 +2%。若不到 +1% 直接停。
