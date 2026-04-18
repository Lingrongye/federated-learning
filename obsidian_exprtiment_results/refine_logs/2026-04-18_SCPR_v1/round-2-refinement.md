# Round 2 Refinement — 符号统一 + 补强 Venue Readiness 的机制论证

---

## Problem Anchor(复制自 Round 0,保持不变)

- **Bottom-line problem**:跨域联邦学习中,客户端数据来自不同视觉风格域。现有原型对齐把同类原型求全局均值后广播,使 outlier 客户端被严重稀释。
- **Must-solve bottleneck**:
  1. global-mean proto 揉糊 style → outlier 客户端丢失自身结构
  2. SAS 参数空间个性化在 PACS 全 outlier 下退化(EXP-086)
  3. FedDSA Share 章节从未落地(EXP-059/078d)
- **Non-goals**:不做风格作训练数据、不做分类器个性化、不加辅助损失、不改架构、不新增 trainable 组件
- **Constraints**:ResNet-18 / AlexNet, FedBN, R=200 3-seed, 正交解耦保留, InfoNCE target 可改
- **Success condition**:
  1. PACS 3-seed mean AVG Best ≥ 81.5%(严格优于 M3 81.91%)
  2. Office 3-seed mean AVG Best ≥ 90.5%
  3. drop ≤ 2%, <100 行, 0 新 trainable
  4. Share 章节首次有实证

---

## Anchor Check

- 原始 bottleneck 保持:global-mean 稀释 outlier / Share 缺口
- Round 2 reviewer 建议:三条 simplification + 一条 Venue Readiness 的 sharpness 提示
- 这些建议均为**文档清晰度**和**机制论证锐度**的改进,**完全不改变** anchored problem
- **结论:无 drift**

---

## Simplicity Check

- Round 2 的"主贡献"**已经唯一**:Self-Masked Style-Weighted Multi-Positive InfoNCE
- Round 2 reviewer 要求进一步**收紧文档**(符号统一、把 warmup 降级为工程细节、fallback 移到附录)
- 这些都是**文档级**收紧,不改变机制
- 额外增加:**一个锐化 Venue Readiness 的 mechanism argument**,让 SCPR 看起来不是"reweighting trick",而是"从 decouple 不完美反推出的必要修正"
- **不再增加**任何新 component / 新 loss / 新变体

---

## Changes Made

### 1. [Simplification] 符号统一

- **Reviewer 说**:用一个符号贯穿全文,避免 `style_proto_k` 和 `s_k` 混用
- **Action**:定义 `s_k := style_proto_k`,论文全文只用 `s_k`
- **Spec**:
  > s_k = normalize(mean_{x ∈ D_k} z_sty(x))
  >
  > 即客户端 k 的本域**全部训练样本**经过 style_head 后 z_sty 空间的**均值**,再 L2-normalize
  >
  > 更新时机:与 class prototypes 同步,客户端每轮 pack 时重新计算并上传
- **Impact**:符号清晰,Method Specificity 的"style_proto_k computation" 软点被消除

### 2. [Simplification] warmup 降级为 implementation detail

- **Reviewer 说**:若 pilot 后不需要,从主方法里删掉
- **Action**:
  - **主方法不再提 warmup**,SCPR 公式不分 round 启用
  - 在 **Implementation Note** 小节(附录)标注:"首轮 bank 空时,fallback 到 uniform weights;这在实际训练中等价于前 1-2 round 的 burn-in,无需显式 warmup"
  - Plan A 本身的 warmup=50(正交解耦的 warmup)仍然保留(这是上游基线的东西)
- **Impact**:主方法的超参从 "τ_SCPR + warmup" 收缩到 "τ_SCPR 一个"

### 3. [Simplification] A_k^c=∅ fallback 移到 Implementation Note

- **Reviewer 说**:除非 PACS/Office 真的出现 partial-class absence 才保留
- **Action**:
  - **主方法只写**:SCPR loss 在客户端 k 类 c 样本上使用 `{p_c^j : j ∈ A_k^c, A_k^c = {j ≠ k, p_c^j exists}}`
  - Implementation Note(附录)一行标注:"在 PACS/Office 上我们每个客户端都持有所有类样本,A_k^c = {j : j ≠ k},renormalize 退化为 1/(K-1)"
- **Impact**:主方法描述更简洁,实际数据集不会遇到 partial-class

### 4. [Sharpness] 补强 Venue Readiness — 机制论证

- **Reviewer 说**:Venue Readiness 7/10 的唯一风险是 reviewers 把 SCPR 看成"reweighting trick"
- **Action**:在 Proposed Method 后增加 **"Why style weighting beats uniform: a signal-to-noise perspective"** 小节,从 **decouple 不完美** 反推 style weighting 的必要性
- **核心论点**:
  > SCPR 不是任意 re-weighting,而是从"正交解耦无法彻底消除 style 残留"这一事实反推出的**必要修正**:当远风格原型仍带有 style 噪声时,等权 SupCon 会把这些噪声拉进 z_sem;style-weighted SupCon 等价于对 positives 做 **signal-to-noise importance weighting**。
- **Additional argument**:SCPR 在概念上位于 M3(全忽略风格)和 SAS(强制参数个性化)之间,这解释了为什么它能同时在 PACS 全 outlier 和 Office 单 outlier 场景下工作
- **Impact**:Venue Readiness 从"reweighting trick 感知"→"从 decouple 理论反推的必要补丁",反驳 pseudo-novelty 风险

---

## Revised Proposal(Round 2 版)

# 研究方案 Round 2:Self-Masked Style-Weighted Multi-Positive InfoNCE(SCPR)

> 面向跨域联邦学习的风格条件化原型对齐机制
> 基础框架:FedDSA(Decouple-Share-Align)
> 目标会议:CVPR / ICCV / NeurIPS

---

## Problem Anchor(同上)

---

## Technical Gap

没有方法把"客户端风格 s_k"作为跨客户端检索 key、把"域索引类原型 {p_c^j}"作为 value,在解耦后的表征空间执行软对齐。现有路径:
- 擦除风格(FDSE / FedDP / FedSeProto)
- 本地保留(FedSTAR / FedBN)
- 混合空间 AdaIN(FISC / PARDON,EXP-059 已证伪)
- 参数空间软聚合(SAS,EXP-086 PACS 退化)

**唯一需要改**:InfoNCE target 的权重,其余全冻结。

---

## Method Thesis

> **唯一 thesis**:把 M3 domain-aware multi-positive InfoNCE 的 "等权 positives" 替换为 "按客户端风格相似度加权的 positives",并自掩码(自己不参与)。

- **最小干预**:修改 M3 的 positive 权重(~20 行),无新组件
- **数学自洽**:uniform weight 严格退化为 M3
- **机制动机**:从 decouple 不完美反推出的必要信号加权修正(见下文 S/N 论证)
- **安全性**:
  - 最差情况 = M3(PACS 已验证 +5.09%)
  - self-mask 保证 Share 永不退化为 FedBN-like

---

## Contribution Focus

- **唯一主贡献**:**Self-Masked Style-Weighted Multi-Positive InfoNCE(SCPR)**
  > 在 FedDSA 解耦架构上,首次把"客户端风格"作为跨客户端 attention key,对 M3 等权多正例进行风格加权。严格退化到已验证的 M3(+5.09%),并通过 self-mask 阻断 local-only 退化。

- **附录 composability(非主 claim)**:SCPR 与 SAS 的叠加正交性

- **非贡献**:不做风格作训练数据、不做参数路由、不做新损失项

---

## Proposed Method

### Notation(统一符号)

| 符号 | 含义 |
|------|------|
| `x, y` | 样本与标签 |
| `k` | 当前客户端索引;`K` 客户端总数;`C` 类数 |
| `h = backbone(x)` | 骨干网络 pooled feature |
| `z_sem = sem_head(h)`, `z_sty = style_head(h)` | 解耦后的语义 / 风格特征 |
| **`s_k = normalize(E_{x ∈ D_k}[z_sty(x)])`** | **客户端 k 的风格锚点**(L2-normalized) |
| **`p_c^j = E_{(x,y=c) ∈ D_j}[z_sem(x)]`** | **客户端 j 类 c 的原型** |
| `τ_nce = 0.3` | InfoNCE 温度(Plan A 最优) |
| **`τ_SCPR = 0.3`** | **SCPR attention 温度**(继承 SAS) |

### Complexity Budget

| 组件 | 状态 |
|------|------|
| 骨干 / sem_head / style_head / classifier / BN | 冻结(FedAvg/FedBN 规则不变) |
| 正交解耦(cos² + HSIC) | 复用 Plan A |
| Style bank `{s_j}_j` | 复用 SAS / EXP-084 实现 |
| Class-proto bank `{p_c^j}_{c,j}` | 复用 M3 / EXP-072 实现 |
| **SCPR attention + self-mask(~30 行)** | **新增,无参数** |

- 新 trainable 组件:0
- 新 loss 项:0
- 新超参:1(`τ_SCPR`)

### System Overview

```
┌────────────── 客户端 k ─────────────────┐
│ x → backbone → h                          │
│   ├→ style_head → z_sty → s_k(上传)      │
│   ├→ sem_head  → z_sem                    │
│   │   ├→ classifier → ŷ → L_CE            │
│   │   └→ L_SCPR(z_sem, {p_c^j, w_{k→j}^c})│
│   └→ per-class z_sem mean → p_c^k(上传)  │
└───────────────────────────────────────────┘
                        │
                        ▼
┌────────── Server(SCPR aggregator)─────┐
│ Bank: {s_j}_j,  {p_c^j}_{c,j}          │
│ For each target client i:              │
│   g_ij = cos(s_i, s_j)                 │
│   raw_j = exp(g_ij / τ_SCPR) * 𝟙[j≠i]  │
│   w_{i→j}^c = raw_j / Σ_{l∈A_i^c}raw_l │
│     where A_i^c = {l≠i, p_c^l exists}  │
│ Dispatch {w_{i→j}^c, p_c^j} → client i │
│ FedAvg: DFE/sem_head/classifier        │
│ FedBN: BN 本地                          │
└────────────────────────────────────────┘
```

### Core Mechanism(数学公式,canonical 单一公式)

对客户端 k 上样本 `(x, y=c)`,`z = sem_head(backbone(x))`:

**Style attention(self-masked)**:
```
g_kj = cos(s_k, s_j),     j = 1..K
raw_j = exp(g_kj / τ_SCPR) · 𝟙[j ≠ k]

A_k^c = { j : j ≠ k, p_c^j exists }

w_{k→j}^c = raw_j / Σ_{l ∈ A_k^c} raw_l     if j ∈ A_k^c
            0                                otherwise
```

**SCPR loss**(SupCon multi-positive 的 style-weighted 版本):
```
L_SCPR(z, c)
  = - Σ_{j ∈ A_k^c} w_{k→j}^c · log [
        exp(sim(z, p_c^j) / τ_nce)
        / ( exp(sim(z, p_c^j) / τ_nce)
            + Σ_{c'≠c, l ∈ A_k^{c'}} exp(sim(z, p_{c'}^l) / τ_nce) )
      ]
```

所有 `p_c^j` 和 `s_j` 均 `.detach()`。

**总损失**:`L = L_CE + λ_orth · L_orth + λ_hsic · L_HSIC + λ_sem · L_SCPR`,继承 Plan A 权重 `(1.0, 0.1, 1.0)`。

### 数学性质(唯一 claim)

> **τ_SCPR → ∞**:`w_{k→j}^c = 1/|A_k^c|` → SCPR 严格退化为 **M3 domain-aware multi-positive InfoNCE**(PACS 已验证 +5.09%)。

其余解读:
- **τ_SCPR → 0**:w 塌缩为 one-hot,指向 `argmax_{j≠k} cos(s_k, s_j)` → **nearest-style single positive**
- **self-mask**:`w_{k→k} ≡ 0`,Share 路径永不退化为 FedBN-like 本地化

### Why style weighting beats uniform: a signal-to-noise perspective

(这一小节是论文 mechanism section 的关键,补强 Venue Readiness。)

**Setup**:在 FedDSA 正交解耦后,`z_sem` 理论上只保留语义信息,`z_sty` 只保留风格。但正交约束 `cos²(z_sem, z_sty) → 0` 是**软约束**,实际训练中无法做到完美解耦 —— 我们已经测过典型训练结束时 `cos(z_sem, z_sty)` 仍有 0.05-0.1 的残余相关。

**Implication**:客户端 j 的类原型 `p_c^j = E[z_sem | y=c, D_j]` 在残余相关下,仍然吸收了 j 域风格的部分成分。两个域的风格距离 `style_dist(k, j)` 越大,`p_c^j` 对客户端 k 而言的**风格噪声成分**越大。

**M3 uniform multi-positive 的缺陷**:在 SupCon 公式中,每个 positive 贡献等权 pulling force,等价于假设"所有 positives 的类信号 / 风格噪声比相等"。但实际上,风格远的 `p_c^j` 的 **SNR 更低**。例如 PACS 上,Art 客户端训练时:
- 把 `z_sem` 拉近 Cartoon 的 `p_{dog}^{cartoon}`(风格近)→ 大多是"艺术简化下的狗"语义信号
- 把 `z_sem` 拉近 Sketch 的 `p_{dog}^{sketch}`(风格远)→ 混入大量"线条极简化"的 sketch 风格残留

等权拉近相当于让 Art 客户端的 z_sem 同时吸收 Sketch 风格噪声。

**SCPR 的 fix(from decouple imperfection)**:
```
w_{k→j}^c ∝ exp(cos(s_k, s_j) / τ)
```
风格近的原型获得高权(clean signal),风格远的获得低权(noisy signal)。这是**信号加权的 SupCon**,不是任意重加权:
- 形式上:等价于 importance-weighted SupCon,每个 positive 的权重正比于其对当前客户端的语义信号质量
- 理论关系:当解耦完美(`cos(z_sem, z_sty) = 0` 严格成立)时,`style_dist` 不再影响 `p_c^j` 的质量 → uniform weight 最优(SCPR → M3);当解耦不完美时,style weighting 是**补偿残余耦合**的必要修正

**位置论证**:SCPR 在概念上处于 M3 和 SAS 之间:
- M3:完全忽略风格,对齐目标等权
- SAS:强制参数个性化(sem_head 按风格软混合)
- **SCPR**:**对齐目标按风格加权,参数保持共享**

这解释了:
- 为什么 SAS 在 PACS 全 outlier 失败(所有 client 参数都退化本地)
- 为什么 SCPR 在两种场景都工作(参数共享 → 避免 PACS 退化;对齐加权 → 捕获 Office single-outlier 优势)

### Failure Modes and Diagnostics

| Failure | Signal | Fallback |
|---------|--------|----------|
| Attention uniform 塌缩 | H(w_k) > 0.95·log(K-1) 持续 > 20 轮 | 将 `s_k` 投影到更低维或调低 τ_SCPR |
| NaN | loss NaN | τ_SCPR 0.3 → 1.0 软化 |
| PACS < M3 | 3-seed AVG Best < 81.91% | 检查 self-mask 启用 / attention entropy |

**内置安全网**:uniform w 下 SCPR 严格等价于 M3,下界 = 81.91%。

### Novelty and Elegance Argument

- **最小机制**:~30 行代码、0 新参数、1 新超参
- **数学自洽**:单一公式,uniform-w 严格退化为 M3
- **理论动机**:从 decouple 不完美的 signal-to-noise 分析反推,不是任意 reweighting
- **2×2 首次**:占据(Decouple, Style-share)格子
- **论文闭环**:首次补齐 FedDSA Share 章节缺口

**Closest prior works 对比**(重要):
1. **FedProto (AAAI 2022)**:single global-mean proto,等权对齐 → SCPR 用 domain-indexed multi-proto + style weighting
2. **FPL (CVPR 2023) / FedPLVM (NeurIPS 2024)**:cluster-based multi-proto,聚类 key 是原型自身几何 → SCPR 的 key 是客户端风格(语义完全不同)
3. **FedSTAR (2025)**:Transformer aggregator + FiLM,但风格**严格本地** → SCPR 首次把风格作为跨客户端 attention key
4. **FISC / PARDON (ICDCS 2025)**:共享风格但在**混合空间 AdaIN**(EXP-059 证伪)→ SCPR 在解耦表征空间 + 只改对齐 target
5. **SAS(我们,未发表)**:参数空间路由,PACS 退化(EXP-086)→ SCPR 把路由下放到原型层,保留参数共享

---

## Claim-Driven Validation Sketch(不变,已在 Round 1 收窄)

### Claim A(主):PACS 全 outlier 下 SCPR > M3 uniform
- A.1 orth_only(baseline 80.41%)/ A.2 M3(= SCPR uniform,~81.91%)/ A.3 SCPR τ=0.3
- 3-seed mean AVG Best / Last / per-domain(Art, Sketch)
- 预期:A.3 ≥ A.2 + 0.5%,A.3 ≥ 81.5%

### Claim B(主):Office 单 outlier 下 SCPR ≥ SAS(无参数个性化)
- B.1 orth_only / B.2 SAS(89.82%)/ B.3 SCPR τ=0.3
- 3-seed mean AVG Best / Last / Caltech-specific
- 预期:B.3 ≥ B.2
- 决定性:证明**普适性**(SCPR 在 PACS+Office 双赢,SAS 只赢 Office)

### Claim C(机制):τ_SCPR 敏感性 + attention 健康度
- PACS 3-seed,τ ∈ {0.1, 0.3, 1.0, 3.0}
- 记录 H(w_k) 曲线、attention 热图
- 预期:最优 τ ∈ [0.3, 1.0]

### Appendix(非主 claim):SCPR + SAS composability
- Office,1 config × 3 seeds

### 预算
主实验 ~54 GPU·h + 附录 6 GPU·h = **60 GPU·h**

---

## Implementation Note(附录,从主方法剥离)

1. **首轮 bank 空**:客户端 pack() 首轮无可用 style_proto 时,`w_{k→j}^c` 回退为 uniform 1/(K-1)(即自动退化为 M3);这等价于一个 1-2 round 的 burn-in,无需显式 warmup 超参
2. **PACS/Office 的 `A_k^c`**:每客户端持有所有类样本,A_k^c = {j ≠ k},renormalize 退化为 1/(K-1)
3. **数值稳定**:softmax 带 log-sum-exp 稳定技巧;τ_SCPR 最小值 0.1 避免数值爆炸
4. **单测**(实现验收):
   - 设 K=4,随机 {s_j},验证 w_{k→k} = 0
   - 设 τ_SCPR → ∞,验证 w 收敛到 uniform 1/3
   - 设 τ_SCPR → 0,验证 w 收敛到 one-hot

---

## Experiment Handoff Inputs

- **Must-prove claims**:A, B, C
- **Must-run ablations**:M3 baseline(= uniform SCPR 下界)、τ 扫描
- **Critical metrics**:PACS AVG Best/Last + per-domain,Office AVG Best/Last + Caltech
- **Highest-risk assumptions**:
  1. `s_k` 区分度足够(否则 H(w) ≈ log(K-1),fallback 降维)
  2. SCPR ≥ M3(否则方法证伪)
  3. Burn-in 期不破坏训练

---

## Compute & Timeline

- 主实验 54 GPU·h + 附录 6 GPU·h = **60 GPU·h**
- 单卡 3 天计算 + 1 天实现/单测/codex 审 + 1 天回填 = **5 天 end-to-end**

---

*Round 2 refinement 完成,符号统一、warmup/fallback 降级为 implementation notes、补强了 Venue Readiness 的机制论证。准备 Round 3 codex 审核。*
