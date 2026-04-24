# Round 1 Refinement — FedDSA-BiProto v2

**Date**: 2026-04-24
**Based on**: Round 1 review (6.5/10 REVISE)

## Problem Anchor (逐字复刻自 round-0)

- **Bottom-line problem**: 在 cross-domain 联邦学习 (PACS / Office-Caltech10) 下, 用 AlexNet from scratch, 3-seed {2, 15, 333} × R200 mean AVG Best **必须同时严格超过 FDSE 本地复现 baseline** (PACS > 79.91 / Office > 90.58). 当前 orth_only 战况: PACS 80.64 ✅ (+0.73), Office 89.09 ❌ (−1.49).
- **Must-solve bottleneck**: Office-Caltech10 必须补回 −1.49 pp 且至少再涨 +0.5 pp, 同时 PACS 不得退 (≥ 80.91).
- **Non-goals**: 不换数据集; 不做诊断论文; 不堆模块凑 novelty; 不预训练; 不换骨干.
- **Constraints**: AutoDL 4090 / lab 3090; 预算 ≤ 50 GPU-h (hard) for pilot, ≤ 120 GPU-h total incl. ablations; AlexNet + FedBN; PACS E=5 / Office E=1; 1 周内 refine + pilot.
- **Success condition**: 3-seed {2,15,333} × R200 mean AVG Best PACS ≥ 80.91 且 Office ≥ 91.08; AVG Last 不退; 3 套可视化 evidence (t-SNE / probe ladder / prototype+feature metrics) 齐备.

## Anchor Check

- **原始 bottleneck 是什么**: Office 必补 −1.49 且至少再涨 +0.5, PACS 保住 ≥ 80.91
- **修订后方法是否依然对这个 bottleneck?**: 是. Pd + proto-level exclusion 依然目标是让 z_sem 表征在 cross-domain 一致性上更稳, 直接指向 Office gap. Stage 0 gate 依然保护 anchor — 若 gate 不过则 kill
- **Reviewer 建议中被 reject 作 drift 的?**:
  - 无. Reviewer 没要求换数据集 / 换 backbone / 引入 LLM/VLM / 改 success metric. 所有建议都是紧方法, 不是 drift.

## Simplicity Check

- **修订后的 dominant contribution**: **Federated Domain Prototype (Pd) as a first-class shared object, enforced orthogonal to class prototypes at prototype level**. 整篇 paper 一句话 thesis: "**在联邦 cross-domain 学习中, domain 应该被建模为与 class 对等的一阶联邦共享原型**".
- **被删除/合并的组件**:
  - ❌ **L_sem_proto 删除** — Pc 改为 EMA running class centroid (from z_sem, no backward), 只作为 L_proto_excl 的 **target buffer**, 不独立做 InfoNCE
  - ❌ **α-sparsity 默认关闭** — 只在 pilot 真崩时补第三层安全阀
  - ✅ **Asymmetric statistic encoder 降级为 enabling infrastructure**: 不写成第二 contribution, 只作为 "为 Pd 提供 proper inductive bias 的 necessary design choice"
  - ✅ **Pd 语义重命名**: 从 "per-client concat" 改为 "domain-indexed bank Pd ∈ ℝ^{D×d_z}" (PACS: D=4, Office: D=4; 在 one-client-one-domain setup 下实现等价, 但语义更 clean)
  - ✅ **Visual 套件从 5 压缩到 3**: (i) t-SNE 双面板 (ii) probe ladder (iii) prototype+feature metrics 合并
- **Reviewer 建议中被 reject 作不必要复杂性的?**:
  - 无. Reviewer 的所有 simplification 建议全部接受, 没有一条是 "reject due to unnecessary complexity introduction"
- **修订后机制为何依然是 smallest adequate**:
  - 新增组件: encoder_sty (1 个 MLP + LayerNorm) + Pd (domain-indexed buffer) + 1 条 InfoNCE loss + 1 条 cosine exclusion loss = **2 个 buffer + 2 条 loss**
  - 相比原方案: 删 1 个 loss (L_sem_proto), 删 1 个 buffer 独立性 (Pc 变 derived target), Vis 从 5 压到 3
  - 相比 CDANN (3 个对抗 loss) / FedDP (MI 最小化) / FedSTAR (FiLM 模块 + Transformer 聚合), 结构更简

## Changes Made

### 1. 叙事重构: "Trunk decontamination" → "Proto-space geometric exclusion"
- **Reviewer said**: `taps.detach()` 让 "修 trunk" 叙事站不住, 实际方法是 prototype-space regularizer
- **Action**: 全篇重写 thesis + mechanism description, 诚实 reframe 为 **"z_sem/z_sty 在 proto 空间被几何互斥, 从而迫使 encoder_sem 学到与 domain 正交的 class representation"**. 不再 claim 修复 trunk 里 channel 级别的 class 弥散
- **Reasoning**: reviewer 正确指出 mechanism ≠ stated story, 继续原叙事是 dishonest. 新叙事 "proto-space exclusion 迫使 encoder_sem 表征几何上远离 domain" 是 accurate 且依然有效 — 其实这就是 Neural Collapse 文献里 ETF prototype 的同类机制, 只不过针对 domain 而非 class
- **Impact**: Claim C1 更 defensible, novelty 更集中

### 2. 删除 L_sem_proto
- **Reviewer said**: Pc + L_sem_proto 让 paper 看起来是 dual-prototype system, 稀释 Pd 的核心
- **Action**: Pc 只作为 EMA running class centroid 使用, 来源为 **no-backward** EMA over z_sem (类似 FedProto 的 centroid 但不反传). Loss 清单从 6 条降到 5 条
- **Reasoning**: Pc + L_sem_proto 本质上等价于 FedProto-style 的 sem prototype alignment, 这个 setup 下 EXP-076 已证"无安全阀会崩", 且它不是本文 novelty. 保留作为 L_proto_excl 的 target 已经提供 "class 方向的参照 axis", 独立 InfoNCE 冗余
- **Impact**: Contribution 更 focused, Loss 冲突面减小, Feasibility 上升

### 3. C0 改为 Matched Intervention Test
- **Reviewer said**: head-only retrain 无法干净区分 "CE 污染 trunk" vs "聚合/outlier/capacity"; 应该做更接近真方案的小 pilot
- **Action**: Claim 0 改为 **"frozen encoder_sem + 加 encoder_sty + Pd + proto_excl 做 Office R20-30 rounds"** 的 micro-pilot. 若此 branch 无法提升 Office (vs orth_only head-only retrain baseline), 则 kill 整个 BiProto 方向
- **Reasoning**: Matched intervention 直接测 "方案的实际增量是否真帮得了 Office", 比抽象的 "trunk capacity" 诊断更 actionable. 如果 frozen trunk 上 Pd + proto_excl 都没用, 加 encoder 全量训练也不会有
- **Impact**: Validation Focus 从 6 → 预期 8

### 4. Pd 语义: domain-indexed, 不 depend on "one-client-one-domain"
- **Reviewer said**: per-client concat Pd 太 benchmark-specific, 应该 domain-indexed
- **Action**: Pd ∈ ℝ^{D × d_z}, 每个 entry 是 **所有见到该 domain 的样本 z_sty 的 EMA centroid**. 在 PACS/Office 下 "client_id == domain_id" 所以实现和原方案等价, 但语义可以推广到多 client-per-domain 或 client-with-subdomain 场景
- **Reasoning**: Reviewer 正确指出 "per-client concat" 是实现细节不是 method, paper 叙事要 domain-indexed 才 robust. 同时避开 reviewer 质疑 "inference 时怎么处理看过的 domain 之外"
- **Impact**: Method Specificity + Venue Readiness 上升

### 5. Feasibility: Stage-Gate Budget
- **Reviewer said**: 90 GPU-h 违反 50 GPU-h 预算; 一次上太多旋钮
- **Action**: 分阶段 budget, 前 2 阶段 hard ≤ 20 GPU-h:
  - S0: C0 matched-gate (2 GPU-h, Office seed=2)
  - S1: BiProto smoke (4 GPU-h, Office seed=2 R200)
  - S2: BiProto 3-seed Office (20 GPU-h), **通过才上 PACS**
  - S3: BiProto 3-seed PACS (30 GPU-h)
  - S4: ablation 按需 (40 GPU-h)
  - Total ≤ 100 GPU-h, Pilot 只需 ≤ 30 GPU-h 即可出第一判决
- **Reasoning**: Reviewer 对的, 必须 stage-gate, Office-first
- **Impact**: Feasibility 从 5 → 预期 8

### 6. Visual Evidence 从 5 压缩到 3 套
- **Reviewer said**: Vis-1~5 超过 paper 所需
- **Action**: 合并为 3 套:
  - **Vis-A: t-SNE 双面板** (z_sem by class / z_sty by domain) × 4 methods, silhouette quantification
  - **Vis-B: Probe ladder** (linear/MLP-64/MLP-256 × 4 directions)
  - **Vis-C: Prototype + Feature Health matrix**: 合并 Pd/Pc 分离度 + Pc⊥Pd cosine + feature norm/rank/orth trajectory
- **Impact**: Validation Focus 上升, 可执行性强

### 7. Interface Specification 新增
- **Reviewer said**: Pd/Pc update, normalization, 缺类处理, partial participation, gradient flow, inference usage 均不够紧
- **Action**: 新增 "Interface Specification" 小节, 给出精确数学定义 + gradient-flow 表
- **Impact**: Method Specificity 从 6 → 预期 8

---

## Revised Proposal (v2)

### Title
**FedDSA-BiProto: Federated Domain Prototype as a First-Class Shared Object for Cross-Domain Learning**

### One-sentence thesis
**Domain should be modeled as a first-class federated prototype object Pd, dual to class prototype Pc, with their mutual exclusion enforced in prototype space — this yields cross-domain semantic consistency without feature-level adversarial training.**

### Technical Gap (refined)

当前 FL cross-domain 解耦方法对 domain 信息只有 3 种处置:
1. **擦除** (FDSE, FedDP, FedSeProto): domain 是噪声, feature-level MI min / layer decomposition. Office 上有效但 PACS regime-dependent.
2. **私有** (FedSTAR, FedBN, FedSDAF): domain 本地保留, 不跨 client 共享. Style 降级为 nuisance, 无法 leverage cross-domain 结构.
3. **Feature-level 对抗** (CDANN 变体): GRL 强制 z_sem 去 domain. EXP-108 已 3-seed 证伪 (probe 0.96, 0 accuracy).

**Missing**: 没人把 domain 建模为与 class **对等**的联邦共享原型. I2PFL / MP-FedCL 有 domain-aware prototype 但都是 class prototype 的 variance 源, 不是独立一阶对象.

### Contribution Focus (refined)

- **Dominant (C1, the only headline contribution)**:  
  **Federated Domain Prototype Pd ∈ ℝ^{D×d_z} with Prototype-Level Class-Domain Exclusion**.
  Pd 是服务器端维护的 domain-indexed 原型 bank, 跨 client 共享. 通过 (i) InfoNCE over Pd 让 encoder_sty 学到 domain discriminative features, (ii) L_proto_excl = cos²(Pc, Pd) 强制 Pd 和 Pc (class centroid) 在低维 proto 空间几何正交.
  **Novelty**: 首次在 FL cross-domain 学习中将 domain 作为与 class 对等的一阶联邦共享对象建模, 并用低维 proto-level geometric exclusion 替代 feature-level MI/GRL.
- **Supporting infrastructure (enabling)**: 非对称统计 encoder_sty (~1M params) 为 Pd 提供 structural inductive bias — 它只看 (μ, σ) 统计量而非空间 feature, 物理上减少承载 class 的能力. **不 claim 为独立 contribution**, 只是为 Pd 提供 proper input representation 的必要选择.
- **Explicit non-contributions**: 不 claim MI-optimal / 不 claim 通信效率 / 不做收敛证明 / 不跨 backbone / 不做 DP privacy / 不 claim "修复 trunk".

### Proposed Method (v2)

#### Complexity Budget

| 项目 | 处理 | 说明 |
|---|---|---|
| AlexNet encoder_sem (~60M) | 继承 orth_only | 只被 L_CE / L_CE_aug / L_orth 训, **不被 sty 侧任何 loss 反传** |
| semantic_head, sem_classifier | 继承 orth_only | 不改 |
| BN running stats | FedBN 原则, 本地 | 不聚合 |
| style bank (μ, σ) AdaIN | 继承 FedDSA 原方案 | 用于 L_CE_aug, 不变 |
| **encoder_sty (新增)** | 统计 encoder, ~1M params MLP + LayerNorm | 只被 L_orth + L_sty_proto 训 |
| **Pc (新增, derived)** | domain-invariant class centroid bank ℝ^{C×d_z} | **EMA over z_sem, no backward**; 不是 free parameter |
| **Pd (新增, core)** | domain-indexed prototype bank ℝ^{D×d_z} | **EMA over z_sty, no backward from exclusion loss**; server aggregate |
| **L_sty_proto (新增)** | InfoNCE(z_sty, Pd, domain_label) + 0.5·MSE(z_sty, stopgrad(Pd[d])) | 推 encoder_sty 学 domain discriminative |
| **L_proto_excl (新增)** | cos²(Pc_row_i, Pd_row_j) over all (i,j) | 低维 proto-space 几何互斥, 全程 on |
| ~~L_sem_proto~~ | **删除** (根据 reviewer 建议) | Pc 只作为 L_proto_excl 的 target, 不独立 InfoNCE |
| α-sparsity | **默认关闭** | 只有 pilot 真崩时补 |

**总净增 vs orth_only**:
- 参数: +1M (~1.7%)
- Buffer: +2 (Pc, Pd, 都是 derived EMA)
- Loss: +2 (L_sty_proto, L_proto_excl)
- 安全阀: Bell schedule + MSE anchor (α-sparsity 按需)

#### Interface Specification (新增, 回应 CRITICAL 1)

**Notation**:
- `K` = num clients (PACS/Office = 4)
- `D` = num domains (PACS/Office = 4, 在 one-client-one-domain setup 下 D = K)
- `C` = num classes (PACS 7 / Office 10)
- `d_z` = proto dim = 128
- `B` = batch size
- `m` = EMA decay (0.9 default)
- `τ` = InfoNCE temperature (0.1 default)

**Domain index lookup**: 每个 sample i 有 `domain_id(i) ∈ {0, ..., D-1}`. 在 one-client-one-domain 下 `domain_id(i) = client_id` 恒成立. 若未来扩展到 multi-client-per-domain, 可从 metadata 读.

**Statistic encoder_sty**:
```
Input: taps = [f_1, f_2, f_3] where f_l ∈ ℝ^{B × C_l × H_l × W_l} is **pre-activation** (我们取 conv-l 输出的 BN 后 / ReLU 前)
Pre-processing: 对每个 f_l detach (stop-grad), 然后
  μ_l = mean_{H,W} f_l  ∈ ℝ^{B × C_l}
  σ_l = std_{H,W}(f_l) + 1e-5  ∈ ℝ^{B × C_l}
Concat: s = concat([μ_1, σ_1, μ_2, σ_2, μ_3, σ_3])  ∈ ℝ^{B × (2·Σ_l C_l)}
Forward: z_sty = MLP(s) ∈ ℝ^{B × d_z}
  MLP = Linear(in, 512) → LayerNorm → ReLU → Linear(512, d_z)
Normalization: z_sty ← F.normalize(z_sty, dim=-1)  # L2 unit norm for cosine
```

**Pd update (每 round 结束 server 聚合)**:
```
For each client k:
  client_mean_k = mean_{i in client_k} z_sty_i  (on client, after local epochs)
  client_count_k = |{i : client k's samples}|
  Upload (client_mean_k, client_count_k, domain_id_k)

Server side (domain-indexed, handles partial participation):
For each domain d ∈ {0, ..., D-1}:
  participating_clients = {k : k participated this round AND domain_id_k == d}
  If participating_clients is empty:
    Pd[d] ← Pd[d]  # no update, freeze previous value
  Else:
    aggregated_d = Σ_{k ∈ participating} count_k · client_mean_k / Σ count_k
    Pd[d] ← m · Pd[d] + (1-m) · F.normalize(aggregated_d, dim=-1)
  
Broadcast Pd to all clients.
```

**Pc update (同样 EMA, no-backward)**:
```
On each client at end of local epoch:
  For each class c in local data:
    class_mean_c = mean_{i : y_i == c} z_sem_i  (detached)
    client_class_count_c = count
  Upload (class_mean_c, count_c) for all present classes

Server side:
For each class c ∈ {0, ..., C-1}:
  aggregated_c = Σ_k count_k_c · class_mean_k_c / Σ count (over participating clients with class c)
  If any client has class c:
    Pc[c] ← m · Pc[c] + (1-m) · F.normalize(aggregated_c, dim=-1)
  Else:
    Pc[c] ← Pc[c]  # no update

Broadcast Pc.
```

**Initialization**: Pc and Pd 初始化为 random unit vector on sphere (seed-fixed). 前 R50 warmup 期间不使用 L_sty_proto 和 L_proto_excl, 但 Pc/Pd 保持 EMA 累积, 保证 R50 后首次启用时 Pc/Pd 已经 warmed-up.

**Gradient-flow table**:

| Module | L_CE | L_CE_aug | L_orth | L_sty_proto | L_proto_excl |
|---|:-:|:-:|:-:|:-:|:-:|
| encoder_sem | ✅ | ✅ | ✅ (via z_sem) | ❌ (taps detach) | ❌ (Pc no-grad) |
| semantic_head | ✅ | ✅ | ✅ | ❌ | ❌ |
| sem_classifier | ✅ | ✅ | ❌ | ❌ | ❌ |
| encoder_sty | ❌ | ❌ | ✅ (via z_sty) | ✅ | ❌ (Pd no-grad) |
| Pc | ❌ | ❌ | ❌ | ❌ | ❌ (EMA only) |
| Pd | ❌ | ❌ | ❌ | ❌ (MSE uses stopgrad(Pd)) | ❌ (EMA only) |

**关键说明**:
- encoder_sem 的梯度路径和 orth_only **完全一致** (L_CE + L_CE_aug + L_orth via z_sem). Sty 侧所有新增 loss 都不反传到 encoder_sem
- Pc / Pd 都是 derived EMA buffer, 不是 free parameter, **no gradient**
- L_proto_excl 只通过 L2-normalized Pc, Pd 计算 cos², gradient 只反传到 Pc/Pd (但 Pc/Pd 不可训), 所以这个 loss **实际是 zero-gradient to model**, 其作用体现在 **EMA buffer 本身**: 如果 encoder_sem 学到的 class centroid Pc 和 encoder_sty 学到的 domain centroid Pd 角度不正交, cos² 值高, 作为 **evaluation indicator** 来指引后续 lr / weight schedule 调整
- **重要澄清**: 若要 L_proto_excl 直接作用于模型, 需要让 Pc/Pd 通过 *on-the-fly* mean (有 gradient) 而非 EMA. 我们选项:
  - **Option (a)**: L_proto_excl use on-the-fly centroid (batch-local mean of z_sem by class, z_sty by domain) — 有 gradient, 有实际推力
  - **Option (b)**: L_proto_excl use EMA Pc/Pd — 无 gradient, 仅作为 diagnostic, 不是 loss
  - **We choose (a)**: `L_proto_excl = cos²(batch_class_centroid(z_sem), batch_domain_centroid(z_sty))` over 所有 class-domain 对. 这样梯度直接传到 encoder_sem 和 encoder_sty, 同时 EMA Pc/Pd 仍保留为 long-term stable reference
- 这个细化后的 **on-the-fly** 设计解决了 reviewer 对 "L_proto_excl 怎么作用于模型" 的 specificity 质疑

**修正后 Gradient-flow table**:

| Module | L_CE | L_CE_aug | L_orth | L_sty_proto | L_proto_excl (on-the-fly) |
|---|:-:|:-:|:-:|:-:|:-:|
| encoder_sem | ✅ | ✅ | ✅ | ❌ | ✅ (via batch class centroid) |
| semantic_head | ✅ | ✅ | ✅ | ❌ | ✅ |
| sem_classifier | ✅ | ✅ | ❌ | ❌ | ❌ |
| encoder_sty | ❌ | ❌ | ✅ | ✅ | ✅ (via batch domain centroid) |

#### Loss 组合 + 安全阀 Schedule (精简版)

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{CE,aug}} + \lambda_1 \mathcal{L}_{\text{orth}} + \lambda_2 \mathcal{L}_{\text{sty\_proto}} + \lambda_3 \mathcal{L}_{\text{proto\_excl}}
$$

| Round | λ_orth | λ_sty_proto | λ_proto_excl | MSE coef (in L_sty_proto) | 说明 |
|:-:|:-:|:-:|:-:|:-:|---|
| 0-49 | 1.0 | 0 | 0 | 0 | Warmup: 只 orth_only + AdaIN, Pc/Pd EMA 预热 |
| 50-80 | 1.0 | 0→0.5 | 0→0.3 | 0.5 | Ramp-up: 线性升起 |
| 80-150 | 1.0 | 0.5 | 0.3 | 0.5 | Peak phase |
| 150-200 | 1.0 | 0.5→0 | 0.3 | 0.5 | L_sty_proto ramp-down, L_proto_excl 全程 on (低维小约束不崩) |

**已删**: α-sparsity (默认关闭, 只在 pilot smoke test 若 R100 崩 > 2pp 再补).

**已删**: L_sem_proto (整条 loss 不存在).

**Rationale**: Bell schedule (R0-50-150-200) 复刻 EXP-076 bell_60_30 最稳形态; MSE anchor 复刻 EXP-077 mode 4 (FPL-style); proto_excl 全程 on 因为在低维 proto 空间梯度半径极小, 不会像 feature-level HSIC 那样全局污染.

#### FL 聚合协议

| 参数组 | 聚合 | 通信 cost |
|---|---|---|
| encoder_sem | FedAvg | ~240 MB/client/round |
| semantic_head | FedAvg | <1 MB |
| sem_classifier | FedAvg | <1 KB |
| **encoder_sty** (~1M) | FedAvg | ~4 MB/client/round (+1.7%) |
| BN running stats | Local (FedBN) | 0 |
| LayerNorm (sty side) | FedAvg | <1 KB |
| **Pc** (C × 128) | EMA aggregate server-side | <1 KB broadcast |
| **Pd** (D × 128) | EMA aggregate server-side | <1 KB broadcast |
| style bank (μ, σ) | 继承 FedDSA 原方案 | 已有 |

**Total comm overhead ≤ +2% vs baseline**, 满足 FL 原则.

#### Failure Modes and Diagnostics

| Failure Mode | Detection | Fallback |
|---|---|---|
| **C0 matched-gate 失败** | S0 frozen-trunk Office R20-30 提升 < +0.5 vs head-only baseline | **Kill 整个 BiProto**, 改投资聚合/outlier 侧 (SAS τ tune / Caltech 权重) |
| **z_sty 坍缩** (norm → 0) | 每 10 round 监测 z_sty_norm, < 0.3 警告 | 增 L_sty_proto 权重 or 加 L_sty_norm_reg |
| **InfoNCE R100 后崩盘** | AVG Last 掉 > 2pp | Bell ramp-down 已内置; 若仍崩, 退到 α-sparsity 补第三层 |
| **Pd 跨 domain 区分度不足** | pairwise cos(Pd[i], Pd[j]) > 0.9 | 增 L_sty_proto 权重 or 加 hardness curriculum |
| **L_proto_excl 与 L_CE 冲突** | R50 后 L_CE 异常上涨 > 20% | λ_proto_excl 从 0.3 → 0.1 |

### Novelty and Elegance Argument (refined)

**Closest works differentiation (4 维矩阵, 聚焦于 Pd 维)**:

| 方法 | Disent? | Proto? | **Domain-as-1st-class-shared?** | Exclusion level | Params |
|---|:-:|:-:|:-:|:-:|:-:|
| FedProto / FPL / MP-FedCL / FedPLVM | ❌ | ✅ class only | ❌ | — | 1× |
| I2PFL | ❌ | ✅ intra+inter domain | ❌ (domain = class variance) | — | 1× |
| FedSTAR | ✅ FiLM | ✅ content | ❌ (style 本地私有) | feature | 1.1× |
| FDSE / FedDP / FedSeProto | ✅ erasure | ❌ | ❌ (erase) | feature | ~1× |
| FedFSL-CFRD | ✅ reconstruction | ❌ | ❌ (common/personal, 非 class/domain) | feature | 1.2× |
| FedPall | ❌ | ✅ class | ❌ | feature (adversarial) | 1.1× |
| FISC/PARDON | ❌ | ❌ | ⚠️ style stats shared, 无 proto | — | 1× |
| CDANN 变体 | ✅ GRL | ❌ | ❌ | feature | ~1× |
| **FedDSA-BiProto (v2)** | ✅ stat-encoder | ✅ **class + domain 对偶 bank** | ✅ **首次** (Pd as 1st-class federated object) | **proto (low-dim)** | **~1.02×** |

**Canonical sentence**: *"We are the first to model domain as a first-class federated prototype object Pd — dual to class prototype Pc — and enforce their mutual exclusion in low-dimensional prototype space. This replaces feature-level adversarial or MI-minimization routes (CDANN, FedDP, FedSeProto), which were falsified in our setup (EXP-108 probe 0.96 but zero accuracy gain)."*

### Claim-Driven Validation Sketch (refined, 3 claims)

#### Claim 0 (PRE-REQUISITE): Matched Intervention Gate (S0)

- **Hypothesis**: 在 orth_only Office checkpoint 基础上, 仅加 encoder_sty + Pd + L_proto_excl (frozen encoder_sem), 若 20-30 round 内 Office AVG 提升 < +0.5, 则完整 BiProto 无机会
- **Setup**:
  - 加载 EXP-105 orth_only Office R200 seed=2 checkpoint
  - Freeze encoder_sem + semantic_head + sem_classifier
  - **只训** encoder_sty + Pd bank + proto_excl 支路
  - Office R20-R30, seed=2
- **Baseline**: Head-only fine-tune (同 checkpoint, freeze encoder, 只训 head; reviewer 建议的对照)
- **Decision rule**:
  - S0 提升 ≥ +1.0 pp → 强 signal, 进 S1 smoke 全方案
  - S0 提升 +0.3 ~ +1.0 pp → 弱 signal, 降档继续但预期增益降档
  - S0 提升 < +0.3 pp → **Kill BiProto, 改投聚合/outlier 侧**
- **Cost**: 2 GPU-h

#### Claim 1 (DOMINANT): Accuracy Win vs FDSE

- **Setup**: BiProto v2 full × 3-seed × R200 × {Office, PACS}
- **Pilot order** (stage-gated):
  - **S1**: Office seed=2 R200 smoke (4 GPU-h) — 必须 ≥ 90.0 AVG Best 才 promote
  - **S2**: Office 3-seed {2, 15, 333} R200 (20 GPU-h) — 必须 3-seed mean ≥ 91.08 才 promote
  - **S3**: PACS 3-seed R200 (30 GPU-h) — 必须 3-seed mean ≥ 80.91 (PACS 不得退)
- **Baselines**: FDSE 本地复现 (已有), orth_only (已有)
- **Ablations (S4, 仅 S2/S3 通过才跑)**:
  - BiProto − encoder_sty (换回 orth_only 的 style_head, 验证 asymmetric encoder 必要)
  - BiProto − Pd (只保留 encoder_sty + proto_excl with Pc only, 验证 Pd 必要)
  - BiProto − L_proto_excl (保留 Pd but 不做 exclusion, 验证 proto-level exclusion 必要)
- **Decisive metric**: 3-seed mean AVG Best
- **Expected**:
  - Office ≥ 91.1 (超 FDSE 90.58 + 0.5 buffer)
  - PACS ≥ 80.9 (保住 orth_only, 不得退)
  - Ablation − Pd 预期退到 orth_only + 0.3 以内 (证明 Pd 是贡献来源)
- **Kill**: R50 任一 seed AVG < 80 (PACS) / < 86 (Office) or R150 崩 > 3pp
- **Cost**: S1=4 + S2=20 + S3=30 + S4=40 = 94 GPU-h (但若 S0/S1 kill 则总预算 ≤ 10)

#### Claim 2 (SUPPORTING): Visual Evidence (3 suites only)

**Vis-A: t-SNE 双面板** (paper Fig 2)
- 4 methods × 2 dataset × 2 feature (z_sem / z_sty) = 16 subplots
- Color: z_sem by class (预期 class cluster), z_sty by domain (预期 domain cluster)
- Quantification: silhouette score by target label (class for z_sem, domain for z_sty)
- **Target**: BiProto silhouette(z_sem, class) > orth_only, silhouette(z_sty, domain) > orth_only > CDANN (CDANN 把 class 挤进 z_sty 所以 score 低)

**Vis-B: Probe ladder** (paper Fig 3 / Table 2)
- Linear / MLP-64 / MLP-256 × 4 directions (z_sem→class, z_sem→domain, z_sty→class, z_sty→domain) = 12 probe values × 4 methods = 48 numbers (编为 4×12 表)
- **Target**: BiProto z_sty→class MLP-256 < 0.50 (vs orth_only 0.81, EXP-109 实测), 其他 3 方向符合预期矩阵

**Vis-C: Prototype + Feature Health Matrix** (合并, paper Fig 4 single plot)
- Pc 分离度 (C×C cosine off-diagonal mean) trajectory
- Pd 分离度 (D×D) trajectory
- Pc⊥Pd cosine matrix final value (heatmap)
- z_sem/z_sty norm + effective rank trajectory
- cos(z_sem, z_sty) + HSIC trajectory (orth quality)

**删除**: 原 Vis-4, Vis-5 合并入 Vis-C; Vis-3 降为 Vis-C 一部分.

### Experiment Handoff Inputs (refined)

- **Must-prove**: C0 (gate) → C1 Office S1 smoke → C1 Office S2 3-seed → C1 PACS S3 3-seed → C2 evidence
- **Must-run ablations** (仅 S2/S3 pass 才跑):
  1. BiProto − Pd (核心)
  2. BiProto − encoder_sty (换对称)
  3. BiProto − L_proto_excl
  4. τ sweep {0.05, 0.1, 0.2} on L_sty_proto
- **Critical datasets**: PACS + Office-Caltech10, metric AVG Best + AVG Last + per-domain
- **Highest-risk**: C0 gate 假设 (matched intervention 能提升 Office)

### Compute & Timeline (v2, stage-gated)

| Stage | 内容 | GPU-h | Wall | 晋级条件 |
|---|---|:-:|:-:|---|
| S0 | C0 matched-gate | 2 | 2h | Office +0.3 才进 S1 |
| Impl | 写代码 + unit test + codex review | 0 | 0.5d | AST/unit test 全过 |
| S1 | Office seed=2 R200 smoke | 4 | 4h | AVG Best ≥ 90.0 才进 S2 |
| S2 | Office 3-seed R200 | 20 | 12h (6 并行) | 3-seed mean ≥ 91.08 才进 S3 |
| S3 | PACS 3-seed R200 | 30 | 20h | 3-seed mean ≥ 80.91 才进 S4 |
| S4 | Ablation × 4 | 40 | 24h | — |
| Vis | 出图 | 2 | 0.5d | — |
| **Total** | — | **≤ 98** | ~4-5d wall | — |
| **To-first-kill** | S0+Impl+S1 | **≤ 6** | ~8h | — |
| **To-decision** | S0+S1+S2 | **≤ 26** | ~18h | Office 判决 |

**预算对齐 anchor**: Pilot 阶段 (S0+S1+S2) ≤ 26 GPU-h 严格在 50 预算内. Ablation (S4) 只在方案判决正面才跑, 可接受超预算到 ≤ 100 total.

## Intentionally Excluded (不变)

DomainNet (留 future work), ResNet-18, DP privacy, learnable τ/α, FINCH multi-cluster, GRL, HSIC, Kendall weighting, **L_sem_proto** (新删).

## Summary of v1 → v2 Deltas

| 项 | v1 | v2 | 原因 |
|---|---|---|---|
| Loss 条数 | 6 (L_CE, L_CE_aug, L_orth, L_sem_proto, L_sty_proto, L_proto_excl) | **5** (删 L_sem_proto) | Reviewer: Pc + L_sem_proto 稀释 dominant |
| Contribution 数 | 2 (Pd + asymmetric encoder) | **1 + 1 enabling** (Pd only, asym = support) | Reviewer: 单一 contribution 更 sharp |
| Pd 定义 | per-client concat | **domain-indexed bank** | Reviewer: benchmark-agnostic |
| C0 诊断 | head-only retrain | **matched intervention** (frozen encoder + Pd 支路) | Reviewer: cleaner causal test |
| Safety valve | Bell + MSE + α-sparsity 三件套 | **Bell + MSE** (α 按需) | Reviewer: 太多旋钮 |
| Visual | Vis-1~5 (5 套) | **3 套** (A: t-SNE, B: probe, C: 合并) | Reviewer: paper 过载 |
| 叙事 | "修复 trunk 污染" | **"proto-space geometric exclusion"** | Reviewer: mechanism ≠ story |
| Budget | ≤ 90 GPU-h flat | **stage-gated, pilot ≤ 26** | Reviewer: 违反 50 anchor |
| Interface Spec | 缺 | **新增 gradient-flow 表** + Pd/Pc update 公式 | Reviewer: implementation 不够紧 |
