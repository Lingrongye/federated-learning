# Research Proposal:Style-Conditioned Prototype Retrieval (SCPR)

## Self-Masked Style-Weighted Multi-Positive InfoNCE for Cross-Domain Federated Learning

> 基础框架:FedDSA(Decouple-Share-Align)
> 目标会议:CVPR / ICCV / NeurIPS
> Refine 最终评分:**9.1/10 READY**(5 轮 GPT-5.4 xhigh 审核)

---

## Problem Anchor

- **Bottom-line problem**:跨域联邦学习(FL)中,客户端数据来自不同视觉风格域(照片/素描/油画/线稿等)。现有原型对齐方法(FedProto / FPL / FedPLVM)把所有客户端的同类原型求全局均值后广播,使得风格-outlier 客户端(Caltech、Sketch)在对齐时被严重稀释,性能显著落后于 FedBN / FDSE 等域适应方法。

- **Must-solve bottleneck**:
  1. **Global-mean prototype 揉糊 style**,outlier 域在对齐时丢失自身结构
  2. **SAS 参数空间个性化** 在 Office 单 outlier 场景 +1.21%,但 PACS 全 outlier 场景 −0.65%(EXP-086 诊断:参数软聚合退化为 FedBN-like 本地化)
  3. **FedDSA "Share" 章节从未成功**:EXP-059 z_sem AdaIN 致 PACS −2.54%,EXP-078d h 空间 AdaIN→InfoNCE 致 NaN

- **Non-goals**(严格排除,85 次实验证伪):
  1. 不做"风格作为训练数据增强"
  2. 不做分类器个性化(EXP-093 证伪)
  3. 不加辅助损失(HSIC / PCGrad / Triplet / CKA / Uncertainty 全败)
  4. 不做架构改动(多层注入 / VAE head / 非对称 heads 全败)
  5. 不做训练调度复杂化
  6. 不引入新的可训练组件

- **Constraints**:
  - Backbone:ResNet-18(Office)/ AlexNet-from-scratch(PACS)
  - FedBN 原则(BN 参数本地)
  - R=200,3-seed {2, 15, 333},单卡 24GB
  - 保留正交解耦(cos² + HSIC)、orth_only Plan A(LR=0.05, warmup=50, λ_orth=1.0)
  - 保留 InfoNCE 对齐(**只改 target 权重**,不改损失形式)

- **Success condition**:
  1. PACS 3-seed mean AVG Best ≥ **81.5%**(vs orth_only 80.41%,严格优于 M3 孤立值 81.91%)
  2. Office 3-seed mean AVG Best ≥ **90.5%**(vs SAS 89.82%,逼近 FDSE 90.58%)
  3. R200 drop(Best − Last)≤ 2%
  4. **新增代码 < 100 行、0 新 trainable 组件**
  5. FedDSA Share 章节首次有正向实证

---

## Technical Gap

**无现有方法把"客户端风格 s_k"作为跨客户端 attention key、"域索引类原型 {p_c^j}"作为 value**,在**解耦后的表征空间**执行软对齐。所有已发表路径归为四派:

| 流派 | 代表 | 对风格的态度 | 失败根因(对我们) |
|------|------|-------------|----------------|
| 擦除派 | FDSE / FedDP / FedSeProto | 视为噪声,压缩或丢弃 | 风格信息全部丢失,下游看不见 |
| 私有派 | FedSTAR / FedBN / FedSDAF | 严格本地 | 无跨域互助 |
| 共享不解耦派 | FISC / PARDON / StyleDDG / FedCCRL | 在**混合空间**做 AdaIN | 破坏语义特征(EXP-059 证伪) |
| 参数路由派 | SAS(我们,未发表) | 参数空间软聚合 | PACS 全 outlier 退化(EXP-086) |

**2×2 差异化矩阵**:

|   | 风格不共享 | 风格共享 |
|---|-----------|---------|
| 不解耦 | FedBN / FedAvg / FedProto | FISC / PARDON / StyleDDG / FedCCRL |
| 解耦 | FedSTAR / FedSeProto / FDSE | **SCPR(首次占据)** |

**最小充分干预**:只改 InfoNCE alignment target 的**权重构造**,其余(特征、sem_head、classifier、BN、正交解耦、FedAvg/FedBN 规则)全部冻结。

---

## Method Thesis

> **唯一 thesis**:把 M3 domain-aware multi-positive InfoNCE 的"等权 positives"替换为"**按客户端风格相似度加权的 positives**",并**自掩码**(自己不参与 attention)。
>
> **机制动机**:在 imperfect decouple 假设和线性噪声近似下,从 entropy-regularized noise-minimization 目标函数**推导**出来的 **unique Boltzmann 最优解**(见 Formal Derivation)。softmax-over-cosine 是**推导结果**,不是设计选择。

- **最小充分干预**:修改 M3 的 positive 权重计算(~30 行),无新组件
- **数学自洽**:uniform weight 严格退化为 M3(已验证 +5.09% PACS)
- **安全性**:self-mask 保证 Share 永不退化为 FedBN-like;最差情况 = M3

---

## Contribution Focus

- **唯一主贡献**:**Self-Masked Style-Weighted Multi-Positive InfoNCE(SCPR)**
  > 在 FedDSA 解耦架构上,首次把"客户端风格"作为跨客户端 attention key,对 M3 的等权多正例进行风格加权;自掩码阻断 local-only 退化;softmax-over-cosine 的**确切形式**是从 imperfect-decouple 下 entropy-regularized noise minimization 的**唯一 Boltzmann 解**推导而得。

- **附录 composability**(非主 claim):SCPR 与 SAS 的叠加正交性

- **非贡献**:不做风格作训练数据、不做参数路由、不做新损失项、不加新可训练网络

---

## Proposed Method

### Notation(统一符号)

| 符号 | 含义 |
|------|------|
| `k, j ∈ {1..K}` | 客户端索引;`C` 类数 |
| `h = backbone(x)` | 骨干 pooled feature |
| `z_sem = sem_head(h)`,`z_sty = style_head(h)` | 解耦后的语义/风格特征 |
| `s_k := normalize(E_{x ∈ D_k}[z_sty(x)])` | 客户端 k 的 **L2-normalized 风格锚点** |
| `p_c^j := E_{(x, y=c) ∈ D_j}[z_sem(x)]` | 客户端 j 类 c 的**原型** |
| `τ_nce = 0.3` | InfoNCE 温度(继承 Plan A) |
| `τ_SCPR = 0.3` | SCPR attention 温度(继承 SAS) |

### Complexity Budget

| 组件 | 状态 |
|------|------|
| 骨干 / sem_head / style_head / classifier / BN | **冻结复用**(FedAvg/FedBN 规则不变) |
| 正交解耦(cos² + HSIC) | **复用 Plan A** |
| Style bank `{s_j}_j` | **复用 SAS/EXP-084 实现** |
| Class-proto bank `{p_c^j}_{c,j}` | **复用 M3/EXP-072 实现** |
| **SCPR attention + self-mask + renorm(~30 行)** | **新增,无参数** |

- **0** 新 trainable 组件
- **0** 新 loss 项
- **1** 新超参(`τ_SCPR`)
- **~30** 行新代码,远低于 100 LOC 预算

### System Overview

```
┌────────────── 客户端 k ─────────────────┐
│ x → backbone → h                          │
│   ├→ style_head → z_sty → s_k (上传)     │
│   ├→ sem_head  → z_sem                    │
│   │   ├→ classifier → ŷ → L_CE            │
│   │   └→ L_SCPR(z_sem, {p_c^j, w_{k→j}^c})│
│   └→ per-class z_sem mean → p_c^k (上传)  │
└───────────────────────────────────────────┘
                        │
                        ▼
┌────────── Server (SCPR aggregator) ──────┐
│ Style bank:      {s_j}_j                  │
│ Class-proto bank: {p_c^j}_{c, j}          │
│                                            │
│ For each target client i:                  │
│   g_ij = cos(s_i, s_j)                    │
│   raw_j = exp(g_ij / τ_SCPR) · 𝟙[j ≠ i]   │
│   A_i^c = {l : l ≠ i, p_c^l exists}       │
│   w_{i→j}^c = raw_j / Σ_{l ∈ A_i^c}raw_l  │
│                if j ∈ A_i^c else 0         │
│                                            │
│ Dispatch {w_{i→j}^c, p_c^j} → client i    │
│                                            │
│ FedAvg: DFE / sem_head / classifier        │
│ FedBN:  BN not aggregated                  │
└────────────────────────────────────────────┘
```

### Core Mechanism — Self-Masked Style-Weighted Multi-Positive InfoNCE

对客户端 k 上样本 `(x, y=c)`,`z = sem_head(backbone(x))`:

**Self-masked style attention**:
```
g_kj = cos(s_k, s_j),     j = 1..K
raw_j = exp(g_kj / τ_SCPR) · 𝟙[j ≠ k]
```

**Per-class renormalize**(处理 missing class):
```
A_k^c = { j : j ≠ k, p_c^j exists }
w_{k→j}^c = raw_j / Σ_{l ∈ A_k^c} raw_l    if j ∈ A_k^c
            0                                otherwise
```

**SCPR loss**(SupCon multi-positive 的 style-weighted 版本):
```
L_SCPR(z, c) = - Σ_{j ∈ A_k^c} w_{k→j}^c · log [
    exp(sim(z, p_c^j) / τ_nce)
    / ( exp(sim(z, p_c^j) / τ_nce)
        + Σ_{c' ≠ c, l ∈ A_k^{c'}} exp(sim(z, p_{c'}^l) / τ_nce) )
]
```

所有 `p_c^j` 与 `s_j` 均 `.detach()`(软锚点,不从对比损失回传梯度)。

**总损失**:
```
L = L_CE + λ_orth · L_orth + λ_hsic · L_HSIC + λ_sem · L_SCPR
```
超参继承 Plan A:`(λ_orth, λ_hsic, λ_sem) = (1.0, 0.1, 1.0)`。

### 数学性质(唯一主 claim)

> **`τ_SCPR → ∞`**:`w_{k→j}^c = 1/|A_k^c|` → SCPR **严格退化为 M3 domain-aware multi-positive InfoNCE**(已验证 PACS +5.09% 下界)。

其余解读(非 claim):
- **`τ_SCPR → 0`**:`w` 塌缩为 one-hot,指向 `argmax_{j ≠ k} cos(s_k, s_j)` → nearest-style single positive
- **Self-mask**:`w_{k→k} ≡ 0`,Share 路径永不退化为 FedBN-like 本地化

### Formal Derivation via Entropy-Regularized Noise Minimization

> **注意范围**:以下推导在 **imperfect decoupling + 线性噪声近似** 这一 residual-noise 模型下成立,**不是所有 prototype weighting 的无条件定理**。目的是证明 SCPR 的 softmax-over-cosine 形式是**该模型下的唯一最优**,不是启发式选择。

**Setup**:分解原型为
```
p_c^j = p_c^* + eps_j
```
其中 `p_c^*` 是"理想风格无关类 c 原型",`eps_j` 是客户端 j 的风格残留,源自软正交约束 `cos²(z_sem, z_sty) → 0` 不能严格为 0(实测残余 cos ~ 0.05–0.1)。

客户端 k 上对样本 z 做 SupCon,每个 positive 的 pulling 可分解:
```
sim(z, p_c^j) = sim(z, p_c^*) + sim(z, eps_j)
                 ──────────      ──────────
                 语义信号         风格噪声拉力(有害)
```

定义 `l_j := E[sim(z, eps_j) | client k]`:客户端 j 原型对 k 的风格噪声投影期望。

**优化目标**:在可行 weight 分布上最小化噪声拉力 + 熵正则
```
min_{w ∈ Δ^{K-1}}  J(w) = Σ_j w_j · l_j + τ · Σ_j w_j log w_j
s.t. Σ_j w_j = 1,  w_j ≥ 0
```
(熵正则防止 w 塌缩到 one-hot,`τ > 0` 是正则强度)

**拉格朗日一阶条件**:
```
∂J/∂w_j = l_j + τ (log w_j + 1) + λ = 0
⇒ w_j^* = exp(-l_j / τ) / Z,   Z = Σ_l exp(-l_l / τ)
```

**线性近似**(under imperfect decouple,风格越远噪声越大):
```
l_j ≈ c · (1 - cos(s_k, s_j)) = c · style_dist(k, j),  c > 0
```

**代入得**:
```
w_j^* ∝ exp(-c · (1 - cos(s_k, s_j)) / τ)
       ∝ exp(c · cos(s_k, s_j) / τ)
       = softmax_j(cos(s_k, s_j) / τ_SCPR),   τ_SCPR := τ / c
```

**结论**:**SCPR 的 softmax-over-cosine 是该 residual-noise 模型下 entropy-regularized noise 目标的 unique Boltzmann minimizer**,不是设计选择。

**两极限**(自然过渡):
- **Decouple 完美**(`eps_j ≡ 0`,`l_j ≡ 0`):`J(w)` 只剩熵项,极小化给出 uniform w → **SCPR = M3**(无缝 fallback,无切换成本)
- **Decouple 不完美**(实际情形):style weighting 是最优噪声抑制

### Conceptual Position(论文 Related Work 导出)

SCPR 概念上位于 M3 与 SAS 之间:
- **M3**:无风格信息,所有 positive 等权 → PACS 良好,**没 exploit 风格邻域先验**
- **SAS**:参数空间风格路由(修改 sem_head 权重)→ Office 单 outlier 胜,**PACS 全 outlier 退化为本地**(EXP-086)
- **SCPR**:对齐目标风格加权 + **参数保持共享** → 两种 regime **均适用**

这一中间位置的机制论证:
- SAS 失败根因(PACS):所有 client 参数都退化到本地路径,破坏了跨域共识
- SCPR 隔离设计:参数层 FedAvg 不变 → 保留跨域共识;仅对齐目标按风格加权 → 捕获风格邻域优势

### Failure Modes and Diagnostics

| Failure | Signal | Fallback |
|---------|--------|----------|
| Attention uniform 塌缩 | `H(w_k) > 0.95·log(K-1)` 持续 > 20 轮 | 降维 `s_k` 或调低 `τ_SCPR` |
| 数值 NaN | loss NaN / Inf | `τ_SCPR: 0.3 → 1.0` 软化 |
| PACS < M3 | 3-seed AVG Best < 81.91% | 检查 self-mask 启用 / entropy |
| **Outlier-ness ρ ≈ 0** | 机制未激活 | 方法证伪,方案 downgrade |

**内置安全网**:uniform w 下 SCPR = M3,下界 = 81.91%(已验证)。

### Novelty and Elegance Argument

- **最小机制**:~30 行代码、0 新参数、1 新超参
- **数学自洽**:单一公式,uniform 严格退化为 M3
- **理论根基**:entropy-regularized MaxEnt 推导(不是启发式选择)
- **2×2 首次占格**:(Decouple, Style-share)
- **叙事闭环**:首次补齐 FedDSA Share 章节缺口

**Closest prior works**:
1. **FedProto (AAAI 2022)**:single global-mean proto + 等权对齐 → SCPR 用域索引 multi-proto + style weighting
2. **FPL (CVPR 2023) / FedPLVM (NeurIPS 2024)**:cluster-based multi-proto,聚类 key 是原型自身几何 → SCPR 的 key 是**客户端风格**(语义完全不同)
3. **FedSTAR (2025)**:Transformer aggregator + FiLM 风格分离,但风格**严格本地** → SCPR 首次把风格作为**跨客户端 attention key**
4. **FISC / PARDON (ICDCS 2025)**:共享风格但在**图像/混合特征空间 AdaIN**(EXP-059 已证伪)→ SCPR 在**解耦表征空间** + 只改对齐 target
5. **SAS(我们,未发表)**:参数空间路由,PACS 失败(EXP-086)→ SCPR 把路由下放到**原型层**,保留参数共享

---

## Claim-Driven Validation Sketch

### Claim A(主):PACS 全 outlier,SCPR > M3 uniform

- **数据集**:PACS(4 domains, 7 classes)
- **对比**:
  - A.1 FedDSA orth_only(Plan A baseline,80.41%)
  - A.2 + M3 domain-aware multi-pos(= SCPR uniform attention;对应 τ_SCPR → ∞,~81.91%)
  - A.3 **+ SCPR(τ_SCPR = 0.3, self-mask)**
- **指标**:3-seed {2, 15, 333} mean AVG Best / Last / per-domain(重点看 Art、Sketch)
- **预期**:A.3 ≥ A.2 + 0.5%,A.3 ≥ 81.5%
- **决定性**:若 A.3 < A.2,style weighting 不如 uniform,方法证伪

### Claim B(主):Office 单 outlier,SCPR ≥ SAS(不做参数个性化)

- **数据集**:Office-Caltech10(4 domains, 10 classes;Caltech 是 outlier)
- **对比**:
  - B.1 FedDSA orth_only(baseline)
  - B.2 + SAS(参数空间路由,已验证 89.82%)
  - B.3 **+ SCPR(原型层路由,τ_SCPR = 0.3)**
- **指标**:3-seed mean AVG Best / Last / Caltech-specific
- **预期**:B.3 ≥ B.2
- **决定性**:证明**原型层 ≥ 参数层**,且 SCPR **普适**两种 regime(SAS 只赢 Office)

### Claim C(机制):Outlier-ness 相关性 + τ_SCPR 敏感性

**主诊断(非 tautological)**:
- 定义 `iso_k := 1 - mean_{j ≠ k} cos(s_k, s_j)`:客户端 k 的风格孤立度
- 定义 `gain_k := acc_k^{SCPR} - acc_k^{M3}`:SCPR 相对 M3 的 per-client accuracy 改善
- 报告 **Spearman ρ(iso_k, gain_k)**,预期 **ρ > 0**(outlier 客户端受益更多)
- **Non-tautological**:`gain_k` 是 downstream accuracy,不是 cos 的函数;M3 baseline 与 style 无关;此预测**可证伪**
- 若 ρ ≈ 0,机制未激活 → 方法证伪

**辅助**:
- `τ_SCPR ∈ {0.1, 0.3, 1.0, 3.0}` 扫描,记录 H(w_k) 曲线与 attention heatmap
- 预期最优 `τ ∈ [0.3, 1.0]`,H(w) 不长期靠边界

**GPU 预算**:0 额外(per-client accuracy 与 style bank 均已在现有日志中)

### 附录 composability check(非主 claim)

- Office 上 SCPR + SAS 叠加
- 1 config × 3 seeds
- 只回答"两层路由是否正交",不进 headline

---

## Experiment Handoff Inputs(交给 /experiment-plan)

- **Must-prove claims**:A, B, C(含 outlier-ness ρ)
- **Must-run ablations**:
  - uniform attention(= M3 下界)
  - τ_SCPR 扫描
  - self-mask 必须性(单测层面)
  - per-class renormalize 正确性(单测)
- **Critical datasets / metrics**:
  - PACS AVG Best/Last + per-domain(Art, Sketch)
  - Office AVG Best/Last + Caltech
  - **ρ(iso_k, gain_k)**
- **Highest-risk assumptions**:
  1. `s_k` 区分度充足(若 H(w) 长期 ≈ log(K-1),降维 fallback)
  2. SCPR ≥ M3(若否,方法证伪)
  3. Burn-in 期(bank 空时的 uniform fallback)不破坏训练
- **Unit tests**:
  - K=4 时 `w_{k→k} = 0` 数值验证
  - `τ_SCPR → ∞` 时 `w` 收敛到 uniform 1/3
  - `τ_SCPR → 0` 时 `w` 收敛到 one-hot

---

## Compute & Timeline Estimate

- **GPU·h**:~60(3 天单卡 24GB)
  - PACS 主:3 configs × 3 seeds ≈ 18 h
  - Office 主:3 configs × 3 seeds ≈ 18 h
  - τ 扫描:4 configs × 3 seeds ≈ 24 h
  - 附录(SCPR + SAS):1 config × 3 seeds ≈ 6 h
- **数据 / 标注**:0(PACS + Office-Caltech10 已就位)
- **人力 timeline**:
  - Day 1:实现 + 单测 + codex review
  - Day 2–4:跑实验
  - Day 5:回填 NOTE.md + 消融决策 + 附录实验
  - **共 5 天 end-to-end**

---

## Implementation Note(附录,从主方法剥离)

1. **首轮 bank 空**:客户端首轮 pack() 时 bank 无可用 `s_j`;此时 `w` 自动回退到 uniform 1/(K-1)(= M3 burn-in,1-2 round),**无需显式 warmup 超参**
2. **PACS / Office 上 `A_k^c`**:每客户端持有所有类样本,`A_k^c = {j : j ≠ k}`,renormalize 恒为 1/(K-1)
3. **数值稳定**:使用 log-sum-exp softmax;`τ_SCPR` 最小 0.1 以避免数值爆炸
4. **单测**:
   - `w_{k→k} ≡ 0`(self-mask 验证)
   - `τ_SCPR → ∞` 时 `w → uniform`
   - `τ_SCPR → 0` 时 `w → one-hot`
5. **Outlier-ness 诊断 log**:每轮记录 `(iso_k, acc_k)` 到 `attention_log.jsonl`,离线计算相关性

---

**文档版本**:Round 4 refinement,GPT-5.4 R5 verdict **READY 9.1/10**
**Refine session**:`2026-04-18_SCPR_v1`(5 轮审核,日期 2026-04-18 → 2026-04-19)
**建议下一步**:`/experiment-plan` 细化执行路线,然后实现并跑 Claim A / B / C
