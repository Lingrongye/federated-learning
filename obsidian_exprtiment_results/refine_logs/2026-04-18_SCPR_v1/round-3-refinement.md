# Round 3 Refinement — 加入 Minimal Derivation + 弱化 SNR Claim + 补诊断实验

---

## Problem Anchor(保持不变)

(复制自 Round 0,略)

---

## Anchor Check

- 原 bottleneck 保持:global-mean 稀释 outlier / Share 章节缺口
- Round 3 reviewer 要求:弱化 "equivalent to SNR importance weighting" 表述,加入 minimal derivation 让机制论证更 defensible
- 这两项都是**机制论证锐度**的调整,不改变 anchored problem
- **结论:无 drift**

---

## Simplicity Check

- 主 contribution 仍然唯一:Self-Masked Style-Weighted Multi-Positive InfoNCE
- Round 3 不增加 contributions
- 增加的**不是方法**,是一个**理论论证小节**(~5 句 minimal derivation)
- 在 Claim C 里增加**一个诊断指标**(attention weight vs style distance 相关性),不是新实验,是已有训练过程中的 log
- **无新 component / 新 loss / 新 variant**

---

## Changes Made(针对 Round 3 feedback)

### 1. [IMPORTANT for Venue Readiness] 从"claimed"到"derived":加入 Minimal Derivation block

- **Reviewer 说**:"SNR importance weighting equivalent" 是 **claimed 而非 derived**
- **Action**:在 mechanism 小节增加一个 **Minimal Derivation** block,从 decouple residue 反推 style weighting 的必要性
- **新文字**(直接嵌入论文 mechanism section):

> **Minimal derivation under imperfect decoupling.**
> 
> Decompose each prototype as `p_c^j = p_c^* + eps_j`,其中:
> - `p_c^*` 是"理想风格无关的类 c 原型"
> - `eps_j` 是客户端 j 的风格残留(源于 orthogonal decouple `cos²(z_sem, z_sty) → 0` 不能严格达到 0;实测约 0.05-0.1)
> 
> 在客户端 k 上对样本 z 做 multi-positive SupCon 时,每个 positive 的 pulling 贡献可分解:
> ```
> sim(z, p_c^j) = sim(z, p_c^*) + sim(z, eps_j)
>                   ──────────      ──────────
>                   语义信号          风格噪声拉力
> ```
> 
> 噪声项 `E[sim(z, eps_j) | client k]` 的幅度正比于 `style_dist(k, j) = 1 - cos(s_k, s_j)`(风格越远,`eps_j` 对 k 的语义结构扰动越大)。
> 
> **SCPR 的权重 `w_{k→j}^c ∝ exp(cos(s_k, s_j) / τ)`** 对风格远的 positive 做指数衰减,直接压低噪声拉力项:
> ```
> E[ Σ_j w_{k→j}^c · sim(z, eps_j) ]
>   = Σ_j w_{k→j}^c · c_j · style_dist(k, j)     (c_j > 0)
> ```
> 该期望在 `w_{k→j}^c` 按 cos(s_k, s_j) 加权时**最小化**(这是 softmax attention 的标准性质)。
> 
> **两极限**:
> - Decouple 完美(`eps_j = 0` ∀j):噪声项为 0,style weighting 无意义 → uniform w 最优 → **SCPR 严格退化为 M3**
> - Decouple 不完美(实际情形):噪声项非零,style weighting 是**最小单参数修正**,压低噪声拉力
> 
> **SCPR 不是任意重加权**,而是 decouple 不完美条件下的**必要噪声抑制机制**。

### 2. [IMPORTANT for Contribution Quality] 弱化 SNR "equivalent" 到 "interpreted as"

- **Reviewer 说**:"equivalent" 过强
- **Action**:把原文
  > This is **importance weighting by SNR over positives**
  
  改为:
  > This **can be interpreted as** a bias-control / SNR-aware weighting under imperfect decoupling, as derived in the minimal derivation above.
- **Impact**:论证口径从"定理"弱化到"解读",配合 minimal derivation 的"机制描述"形成闭环

### 3. [Sharpness] 在 Claim C 里加入 attention vs style distance 相关性诊断

- **Reviewer 说**:可以加一个"marginal usefulness of p_c^j decays with style distance"的诊断
- **Action**:在 Claim C(τ_SCPR sensitivity + attention health)加一个子指标:
  > 在训练过程中,对每轮每个客户端对 `(k, j)`:
  > 记录 `w_{k→j}` 和 `style_dist(k, j) = 1 - cos(s_k, s_j)`
  > 报告 **Spearman correlation ρ(w, -style_dist)** 应为**正**(attention 主动压低远风格 positive)
  > 预期:ρ > 0.7(强负相关),证明 SNR mechanism 实际在被激活
- **Impact**:给 SNR 论证提供实证支持,不是纯理论
- **代价**:0 GPU-hours 额外(只是 log 现有量),不改实验预算

### 4. [Minor 清理] 删除 "importance weighting by SNR over positives" 原文,用 minimal derivation 替代

已在 #1, #2 涵盖

---

## Revised Proposal(Round 3 最终版)

(以下为完整 revised proposal,embedded for reviewer)

---

# 研究方案 Round 3:Self-Masked Style-Weighted Multi-Positive InfoNCE(SCPR)

> 面向跨域联邦学习的风格条件化原型对齐机制
> 基础框架:FedDSA (Decouple-Share-Align)
> 目标会议:CVPR / ICCV / NeurIPS

---

## Problem Anchor(同上)

---

## Technical Gap

没有方法把"客户端风格 s_k"作为跨客户端检索 key、"域索引类原型 {p_c^j}"作为 value,在解耦表征空间执行软对齐。现有路径要么擦除风格、本地保留、要么混合空间 AdaIN,要么参数空间路由。SCPR 占据(Decouple, Style-share)象限的首个位置。

**唯一需要改**:InfoNCE target 的权重,其余全冻结。

---

## Method Thesis

> 把 M3 domain-aware multi-positive InfoNCE 的 "等权 positives" 替换为 "按客户端风格相似度加权的 positives",并自掩码(自己不参与)。

**机制动机**:**从 decouple 不完美反推出的必要噪声抑制机制**(下面 minimal derivation)。

---

## Contribution Focus

- **唯一主贡献**:**Self-Masked Style-Weighted Multi-Positive InfoNCE(SCPR)**,在 FedDSA 解耦架构上首次把客户端风格作为跨客户端 attention key,对 M3 等权多正例做风格加权。严格退化到已验证的 M3(+5.09%);self-mask 阻断 local-only 退化;理论上是 decouple imperfection 的最小单参数修正。
- **附录 composability(非主 claim)**:SCPR + SAS 在 Office 上的叠加

---

## Proposed Method

### Notation(统一符号)

| 符号 | 含义 |
|------|------|
| `s_k := normalize(E_{x ∈ D_k}[z_sty(x)])` | 客户端 k 的 L2-normalized 风格锚点 |
| `p_c^j := E_{(x,y=c) ∈ D_j}[z_sem(x)]` | 客户端 j 类 c 的原型 |
| `τ_nce = 0.3` | InfoNCE 温度(Plan A) |
| `τ_SCPR = 0.3` | SCPR attention 温度(继承 SAS) |

### Complexity Budget

- 骨干 / sem_head / style_head / classifier / BN:冻结复用
- 正交解耦(cos² + HSIC):复用 Plan A
- Style bank & Class-proto bank:复用 SAS/M3 实现
- **新增**:SCPR attention + self-mask + renormalize,~30 行
- 0 新 trainable 组件,0 新 loss 项,1 新超参(`τ_SCPR`)

### Core Mechanism(单一 canonical 公式)

**Self-masked style attention**:
```
g_kj = cos(s_k, s_j),   j = 1..K
raw_j = exp(g_kj / τ_SCPR) · 𝟙[j ≠ k]
```

**Per-class renormalize**:
```
A_k^c = { j : j ≠ k, p_c^j exists }
w_{k→j}^c = raw_j / Σ_{l ∈ A_k^c} raw_l    if j ∈ A_k^c
            0                                otherwise
```

**SCPR loss**:
```
L_SCPR(z, c)
  = - Σ_{j ∈ A_k^c} w_{k→j}^c · log [
        exp(sim(z, p_c^j)/τ_nce)
        / ( exp(sim(z, p_c^j)/τ_nce)
            + Σ_{c'≠c, l ∈ A_k^{c'}} exp(sim(z, p_{c'}^l)/τ_nce) )
      ]
```

所有 `p_c^j` 和 `s_j` 均 detach。

**总损失**:`L = L_CE + λ_orth · L_orth + λ_hsic · L_HSIC + λ_sem · L_SCPR`(权重继承 Plan A)。

### 数学性质(唯一主 claim)

> `τ_SCPR → ∞`:`w_{k→j}^c = 1/|A_k^c|` → SCPR **严格退化为 M3 domain-aware multi-positive InfoNCE**(PACS 已验证 +5.09%)。

其余解读(非 claim):`τ_SCPR → 0` 退化为 nearest-style single positive;`self-mask` 保证 Share 永不退化为 FedBN。

### Why style weighting is a necessary correction(**补强版**)

**Minimal derivation under imperfect decoupling.**

将每个原型分解为:
```
p_c^j = p_c^* + eps_j
```
其中 `p_c^*` 是"理想风格无关类 c 原型",`eps_j` 是客户端 j 的风格残留。`eps_j` 源于 orthogonal decouple 的 `cos²(z_sem, z_sty) → 0` 软约束不能严格达到 0;实测约 0.05-0.1。

客户端 k 上对样本 z 做 multi-positive SupCon 时,每个 positive 的 pulling 可分解:
```
sim(z, p_c^j) = sim(z, p_c^*) + sim(z, eps_j)
                 ──────────       ──────────
                 语义信号(desired) 风格噪声拉力(undesired)
```

噪声项 `E[sim(z, eps_j) | client k]` 的幅度正比于 `style_dist(k, j) = 1 - cos(s_k, s_j)`。

**SCPR 的权重** `w_{k→j}^c ∝ exp(cos(s_k, s_j) / τ)` 对风格远的 positive 做指数衰减,直接压低噪声拉力项的期望:
```
E[ Σ_j w_{k→j}^c · sim(z, eps_j) ]
  随 w 按 cos(s_k, s_j) 加权而减小(softmax attention 标准性质)
```

**两极限论证**:
- Decouple 完美(`eps_j = 0`):噪声为 0,uniform w 最优 → **SCPR 严格退化为 M3**(无缝 fallback)
- Decouple 不完美(实际情形):噪声非零,style weighting 是**最小单参数修正**

**解释**:SCPR 因此**不是任意重加权**,而是 decouple 不完美条件下的**必要噪声抑制**。可以被解读为 `bias-control / SNR-aware weighting under imperfect decoupling`。

### Conceptual position(帮助 reviewer 理解)

- **M3**:忽略风格,所有 positive 等权 → PACS 良好但无法 exploit style 邻域先验
- **SAS**:参数空间风格路由 → Office single-outlier 胜,PACS 全 outlier 退化
- **SCPR**:对齐目标风格加权 + **参数保持共享** → **两种 regime 均适用**

这是 SCPR 同时解决 PACS 和 Office 的根因:风格路由下放到原型层,**避免**参数层的全局共识被破坏。

### Failure Modes and Diagnostics

| Failure | Signal | Fallback |
|---------|--------|----------|
| Attention uniform 塌缩 | H(w_k) > 0.95·log(K-1) 持续 > 20 轮 | 降维 s_k 或调低 τ_SCPR |
| NaN | loss NaN | τ_SCPR 0.3 → 1.0 |
| PACS < M3 | 3-seed AVG Best < 81.91% | 检查 self-mask / entropy |

**内置安全网**:uniform w 下 SCPR = M3,下界 81.91%。

### Novelty and Elegance

- 最小机制:~30 行,0 新参数,1 新超参
- 数学自洽:单一公式,uniform → M3
- **理论动机**:decouple imperfection SNR derivation(见上)
- **2×2 首次**:(Decouple, Style-share)
- 论文 Share 章节闭环

---

## Claim-Driven Validation Sketch

### Claim A(主):PACS 全 outlier,SCPR > M3 uniform
- A.1 orth_only(80.41%),A.2 M3(~81.91%),A.3 SCPR τ=0.3
- 3-seed mean AVG Best/Last + per-domain(Art, Sketch)
- 预期 A.3 ≥ A.2 + 0.5%, A.3 ≥ 81.5%

### Claim B(主):Office 单 outlier,SCPR ≥ SAS(不做参数个性化)
- B.1 orth_only, B.2 SAS(89.82%), B.3 SCPR τ=0.3
- 3-seed mean AVG Best/Last + Caltech
- 预期 B.3 ≥ B.2
- 决定性:证明普适性

### Claim C(机制):τ_SCPR 敏感性 + attention 健康 + **SNR 机制激活诊断(新增)**
- PACS 3-seed,τ ∈ {0.1, 0.3, 1.0, 3.0}
- 记录 H(w_k) 曲线、attention heatmap
- **新增 SNR 诊断**:训练过程中记录每对 (k, j) 的 `w_{k→j}` 和 `style_dist(k, j) = 1 - cos(s_k, s_j)`
  - 报告 **Spearman 相关 ρ(w_{k→j}, -style_dist)**
  - 预期 ρ ≥ 0.7(强正相关,即 attention 确实在压低远风格 positive)
  - 若 ρ 接近 0,说明 style key 区分度不够,机制未激活
- **预算**:0 额外 GPU-hours(log 现有 attention 权重即可)

### Appendix(非主 claim):SCPR + SAS composability on Office
1 config × 3 seeds

### 预算
主实验 54 GPU·h + 附录 6 GPU·h = **60 GPU·h**

---

## Implementation Note(附录)

1. 首轮 bank 空:`w` fallback 到 uniform 1/(K-1)(= M3),1-2 round burn-in,无需显式 warmup 超参
2. PACS/Office 每客户端持所有类,`A_k^c = {j ≠ k}`,renormalize 为 1/(K-1)
3. 数值:log-sum-exp 稳定,τ_SCPR 最小 0.1
4. 单测:(a) `w_{k→k} ≡ 0`;(b) `τ → ∞` 时 `w` 趋近 uniform 1/3;(c) `τ → 0` 时 `w` 趋近 one-hot
5. SNR 诊断:每轮记录 (style_dist, w) pair 到 `attention_log.jsonl`,离线计算相关性

---

## Experiment Handoff Inputs

- **Must-prove claims**:A, B, C(含 SNR 诊断)
- **Must-run ablations**:M3 lower bound, τ sweep
- **Critical metrics**:PACS AVG + per-domain, Office AVG + Caltech, **ρ(w, -style_dist)**
- **Highest-risk assumptions**:
  1. s_k 区分度充足
  2. SCPR ≥ M3
  3. SNR diagnostic 确实显示 ρ > 0.7(若否,机制没激活,方法证伪)

## Compute
60 GPU-h, 5 天

---

*Round 3 refinement:加入 minimal derivation + 弱化 SNR claim + 新增 SNR 机制激活诊断。目标冲 READY(≥9)。*
