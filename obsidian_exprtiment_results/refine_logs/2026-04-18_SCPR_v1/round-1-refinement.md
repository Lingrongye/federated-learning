# Round 1 Refinement — SCPR 收窄为 Self-Masked Style-Weighted M3

---

## Problem Anchor(复制自 Round 0,保持不变)

- **Bottom-line problem**:跨域联邦学习中,客户端数据来自不同视觉风格域。现有原型对齐把同类原型求全局均值后广播,使 outlier 域客户端(Caltech、Sketch)被严重稀释。
- **Must-solve bottleneck**:
  1. global-mean proto 揉糊 style → outlier 客户端丢失自身结构
  2. SAS 参数空间个性化在 PACS 全 outlier 下退化为 FedBN-like(EXP-086)
  3. FedDSA Share 章节从未落地(EXP-059/078d 均失败)
- **Non-goals**:不做风格作训练数据增强、不做分类器个性化、不加辅助损失、不改架构、不新增 trainable 组件
- **Constraints**:ResNet-18/AlexNet, FedBN, R=200 3-seed, 正交解耦保留, InfoNCE target 可改
- **Success condition**(不变):
  1. PACS AVG Best 3-seed mean ≥ 81.5%(严格优于 M3 81.91%)
  2. Office AVG Best 3-seed mean ≥ 90.5%
  3. drop ≤ 2%, <100 行代码, 0 新 trainable
  4. Share 章节首次有实证

---

## Anchor Check(评审是否造成 drift?)

**原始 bottleneck**:global mean proto 稀释 outlier;Share 章节缺口。

**评审是否让我们漂离?**
- Reviewer 建议:收窄为 **self-masked style-weighted M3** 单一机制
- 这个机制**严格命中 bottleneck**:从"global mean"换成"style-weighted 近邻原型集"作为对齐目标
- Reviewer 要求删掉 SCPR+SAS 主 claim、retrieved-mean 变体、grad 监控
- 这些都是"把 sprawl 砍掉",**不改变** anchored problem
- **结论:无 drift**,反而让方法**更纯粹地**攻击 bottleneck

**被拒绝的 drift 类建议**:无(reviewer 没有提出任何 drift 类建议)

---

## Simplicity Check

**Round 0 的"主贡献"状态**:混合两条线(retrieved mean + weighted multi-pos + SCPR+SAS composability)→ sprawl

**Round 1 的"主贡献"**(唯一):
> **Self-Masked Style-Weighted Multi-Positive InfoNCE(下称 SCPR)**:在 FedDSA 解耦架构上,把 M3 domain-aware multi-positive InfoNCE 的"等权 positives"替换为**按客户端风格相似度加权的 positives**,并**自掩码**(自己的原型不参与 attention)。

**被删除或移到附录的组件**:
1. ❌ 单原型 retrieved-mean 分支 → **删除**(和 multi-pos 数学性质不一致)
2. ❌ SCPR+SAS 叠加作主 claim → **移到附录**(只作 composability check,1 config × 3 seeds)
3. ❌ grad_CE vs grad_InfoNCE 监控 → **删除**(不是证明主 claim 所需)
4. ❌ z_sty-proto 和 (μ,σ) 并存作为 key → **冻结为 style_proto_k**(和 SAS 一致)
5. ❌ "严格泛化 FedProto 和 M3" 双重 claim → **只留一个**:uniform w → M3
6. ❌ self-mask 作为 fallback variant → **改为默认**

**剩余 mechanism 的最小性论证**:
- **单一 attention 公式**:`w_{i→j≠i} = softmax(cos(style_proto_i, style_proto_j)/τ)`
- **单一 loss 改动**:SupCon multi-pos 里每个 positive 从等权 `1/K` 变成 `w_{i→j}`
- 无新 module、无新 loss 项、无新超参(τ 继承 SAS 最优 0.3)
- 数学性质清晰:uniform w (τ→∞) ⟺ M3 equal multi-pos

---

## Changes Made(按 reviewer 优先级)

### 1. [CRITICAL] 修正 Method Specificity:收窄为单一 canonical algorithm

- **Reviewer 说**:retrieved-mean 和 multi-pos 两条主线数学性质不一致,不能共享 "worst case = M3" claim
- **Action**:砍掉 retrieved-mean,**只保留 self-masked style-weighted multi-positive**
- **Reasoning**:
  - multi-pos 分支的 uniform-w 真正退化为 M3(reviewer 承认)
  - M3 已经在 PACS 上验证 +5.09%,作为下界是 solid 的
  - 单一公式易实现、易讲、易反驳
- **Impact**:Method Specificity 从"两路并存,相互矛盾"→ "唯一公式,数学自洽"

### 2. [IMPORTANT] 修正 Contribution Quality:Share 防退化设计

- **Reviewer 说**:4-5 clients 下 w_{i→i} 会主导,Share 退化为 no-share
- **Action**:**默认 self-mask**(j ≠ i),softmax 只在 K−1 个其他 client 上计算
- **Reasoning**:
  - 自己的原型对自己没有"共享"价值(训练时本来就在拉近 z_sem 和 local proto)
  - self-mask 保证 SCPR 永远是"从别的 client 处借力"的机制
  - 若不 mask,cos(s_i, s_i)=1 是 softmax 最大值,τ 很小时 w_{i→i} 几乎=1
- **Impact**:Share 机制被"强制生效",不能退化为 FedBN-like 本地化

### 3. [IMPORTANT] 冻结接口

- **Reviewer 说**:style key、bank 更新时机、missing class 处理没冻结
- **Action**:三个决策写死在方案中:
  1. **Style key** = `style_proto_k`(z_sty 空间的客户端类均值或 batch 均值,复用 SAS/EXP-084 已实现的接口)
  2. **Bank 更新时机**:客户端每轮 `pack()` 时连同 class prototypes 一起上传(服务器在 `iterate()` 时覆盖 bank);首轮 warmup 期间 bank 为空则 fallback 到 global-mean(退化为 FedProto multi-pos ≡ M3 初始态)
  3. **Missing class**:若类 c 在客户端 j 没有样本(proto 缺失),则 w_{i→j} 对该类单独重新归一化(`w_{i→j}^c = w_{i→j} / Σ_{l ∈ have_c} w_{i→l}`),只在有 proto 的 client 上计算 softmax
- **Impact**:Method Specificity 冻结到 engineer 可以直接编码,无 ambiguity

### 4. [IMPORTANT] 砍掉 SCPR+SAS 主 claim

- **Reviewer 说**:把叠加当主 claim 会冲淡核心贡献
- **Action**:SCPR+SAS 只保留**附录 composability check**(Office 1 config × 3 seeds)
- **Reasoning**:SAS 的 Office 专属有效性仍然值得用 1 个附录表格证明 SCPR 不互斥,但不是 headline claim
- **Impact**:Venue Readiness 提升(主 claim 更 sharp)

### 5. [IMPORTANT] 删掉 grad 监控

- **Reviewer 说**:grad 监控不是证主 claim 所需
- **Action**:删掉 cos_sim(grad_CE, grad_InfoNCE) 诊断
- **保留**:仅保留 H(w_k) attention entropy(论文要写一张 attention health 图,轻量)
- **Impact**:实验 log 压缩,但 attention 可视化仍在

### 6. [MINOR] 修正"严格泛化 FedProto 和 M3"的数学 claim

- **Reviewer 说**:你只能二选一
- **Action**:主论文只声称
  > uniform w(τ→∞) 严格退化为 M3 domain-aware multi-positive InfoNCE
  
  不再声称"也泛化 FedProto"
- **Reasoning**:在 multi-pos 下,M3 和 FedProto 本来就不同(M3 有 K 个 positives,FedProto 只有 1 个 global mean positive)。砍掉误导性的"双重泛化"claim

---

## Revised Proposal(完整重写)

# 研究方案 Round 1:Self-Masked Style-Weighted Multi-Positive InfoNCE(SCPR)

> 面向跨域联邦学习的风格条件化原型对齐机制
> 基础框架:FedDSA (Decouple-Share-Align)
> 目标会议:CVPR / ICCV / NeurIPS

---

## Problem Anchor(同上)

(略,与 Round 0 一致)

---

## Technical Gap

现有方法没有在**客户端风格作为 key / 域索引原型作为 value** 的表征空间上建立跨客户端共享的检索机制。所有尝试要么擦除风格(FDSE/FedDP)、本地保留(FedSTAR/FedBN)、混合空间 AdaIN(FISC/PARDON/EXP-059)、或参数空间软聚合(SAS,PACS 退化)。

**唯一需要改的**:InfoNCE target 的构造方式。其余全冻结。

---

## Method Thesis

> **唯一 thesis**:把 M3 domain-aware multi-positive InfoNCE 的 "等权 positives" 替换为 "按客户端风格相似度加权的 positives",并自掩码(自己不参与 attention)。

- **最小充分干预**:修改 M3 的 loss 权重计算(~20 行),无新组件
- **数学自洽性**:uniform weight 严格退化为 M3
- **安全性**:
  - 最差情况 = M3(已验证 PACS +5.09%)
  - self-mask 保证 Share 机制不退化为 FedBN

---

## Contribution Focus

- **唯一主贡献**:**Self-Masked Style-Weighted Multi-Positive InfoNCE(SCPR)**
  > 在 FedDSA 解耦架构上,首次把"客户端风格"作为跨客户端 attention key,对 M3 的等权多正例进行风格加权。严格退化到已验证的 M3(+5.09%),并通过 self-mask 防止退化为本地化。

- **附录 composability check**(不是主 claim):
  > SCPR 与 SAS 可叠加,在 Office-Caltech10 上同时启用原型层和参数层路由是否进一步提升

- **非贡献**:不做风格作训练数据、不做参数路由、不做新损失项

---

## Proposed Method

### Complexity Budget

| 组件 | 状态 |
|------|------|
| 主干 ResNet-18 / AlexNet | 冻结 |
| 正交解耦(cos² + HSIC) | 复用 Plan A |
| sem_head / style_head / classifier | 复用(不个性化) |
| BN | 复用 FedBN |
| **Style bank**(z_sty 空间 per-client proto) | **复用 SAS/EXP-084 实现** |
| **Class-proto bank**(z_sem 空间 per-(class,client) proto) | **复用 M3/EXP-072 实现** |
| **SCPR attention 函数**(无参数) | **新增 ~20 行** |
| **Self-mask + renormalize 逻辑** | **新增 ~10 行** |

- 新 trainable 组件:**0**
- 新 loss 项:**0**(只改 M3 loss 内部的 positive 权重)
- 新超参:**1**(τ_SCPR,默认 0.3 继承 SAS)

### System Overview

```
┌────────────── 客户端 k ──────────────┐
│ x → backbone → h                      │
│  ├→ style_head → z_sty → style_proto  │──上传──┐
│  ├→ sem_head → z_sem                  │        │
│  │   ├→ classifier → ŷ → L_CE         │        │
│  │   └→ L_SCPR(z_sem, {p_c^j,w_{k→j}})│        │
│  └→ per-class z_sem mean → class_proto│──上传──┤
└───────────────────────────────────────┘        │
                                                  │
┌───────── Server(SCPR aggregator)─────────────┐ │
│ Style bank:  {style_proto_j}_{j=1..K}    ←───┘ │
│ Class-proto bank: {p_c^j}_{c,j=1..K}     ←─────┘
│                                                 │
│ For each target client i:                       │
│   # self-mask attention                         │
│   g_ij = cos(style_proto_i, style_proto_j)      │
│   raw = exp(g_ij / τ_SCPR) · 𝟙[j≠i]             │
│   # renormalize per class (over clients w/ p_c) │
│   w_{i→j}^c = raw_j · 𝟙[p_c^j exists]           │
│              / Σ_{l≠i, p_c^l exists} raw_l      │
│                                                 │
│ Dispatch {w_{i→j}^c, p_c^j} to client i         │
│                                                 │
│ DFE/sem_head/classifier: FedAvg                 │
│ BN: not aggregated (FedBN)                      │
└─────────────────────────────────────────────────┘
```

### Core Mechanism — Self-Masked Style-Weighted Multi-Positive InfoNCE

**Setup**(与 M3 一致,只在 weight 处改):
- 客户端 k,样本 (x_i, y_i=c),z_i = sem_head(backbone(x_i))
- 服务器提供 K 个 client 的 per-(class, client) prototypes `{p_c^j}` 和 per-client style prototypes `{style_proto_j}`

**Self-masked attention over style**:
```
g_kj = cos(style_proto_k, style_proto_j)                (j = 1..K)
raw_j = exp(g_kj / τ_SCPR) · 𝟙[j ≠ k]                   (self-mask)
```

**Per-class renormalize**(处理 missing class):
```
A_k^c = {j : j ≠ k, p_c^j exists}                       (available clients for class c)
w_{k→j}^c = raw_j / Σ_{l ∈ A_k^c} raw_l     if j ∈ A_k^c
w_{k→j}^c = 0                                otherwise
```

**SCPR loss**(对 z_i, y_i=c):
```
L_SCPR(z_i)
  = - Σ_{j ∈ A_k^c} w_{k→j}^c · log [
        exp(sim(z_i, p_c^j) / τ_nce)
        / ( exp(sim(z_i, p_c^j) / τ_nce)
            + Σ_{c'≠c, l ∈ A_k^{c'}} exp(sim(z_i, p_{c'}^l) / τ_nce) )
      ]
```

**Detach**: 所有 `p_c^j.detach()`、`style_proto_j.detach()`(原型是软锚点,不从对比损失回传梯度到 proto bank)。

### 关键数学性质(可写到论文 mechanism 小节)

- **τ_SCPR → ∞(uniform)**:`w_{k→j}^c = 1/|A_k^c|` → 严格退化为 M3 domain-aware multi-positive InfoNCE(已验证 PACS +5.09%)
- **τ_SCPR → 0**(one-hot 到最相似的非自我 client):`w_{k→j}^c = 𝟙[j = argmax_{l≠k} cos(style_k, style_l)]` → nearest-style single positive
- **self-mask 的几何含义**:Share 路径上永远不使用自己的原型,避免 w_kk 主导

### 首轮 warmup 处理

- Round < warmup(50):服务器 bank 尚未稳定,回退到 M3 等权(等价 τ→∞)
- Round ≥ warmup:SCPR 完整启用
- Bank 更新:客户端每轮 pack 时上传新 style_proto 和 class_proto,服务器直接覆盖

### Training Plan

- **骨干/数据**:PACS (AlexNet-from-scratch)、Office-Caltech10 (ResNet-18)
- **超参**(最小充分):
  - lambda_orth = 1.0, lambda_hsic = 0.1, lambda_sem = 1.0(Plan A)
  - τ_nce = 0.3(Plan A)
  - **τ_SCPR = 0.3**(默认,继承 SAS)
  - warmup = 50
  - R = 200, 3-seed {2, 15, 333}
- **无新可训练参数**,客户端训练循环无变化
- **监控**:attention entropy H(w_k) ∈ [0, log(K-1)](诊断塌缩,论文画图用)

### Failure Modes and Diagnostics

| Failure | Signal | Fallback |
|---------|--------|----------|
| Attention uniform collapse | H(w_k) > 0.95·log(K-1) 持续 >20 轮 | 把 key 从 style_proto 换成 style_proto 的 L2-normalize 后投影(仍无参数) |
| 数值稳定性(τ 过小) | loss NaN / Inf | τ_SCPR 从 0.3 上调到 1.0 |
| PACS < M3 | 3-seed AVG Best < M3(81.91%) | 检查 self-mask 是否启用、检查 attention entropy 是否塌缩 |
| 某 class 在所有 client(除 self)都缺失 | A_k^c = ∅ | 对该样本 SCPR loss 设为 0(fallback 到纯 CE) |

**内置安全网**:若所有 client 贡献被 w 稀释,最差退化到 M3 的 positive 集合(因为 uniform w → M3)。

### Novelty and Elegance Argument

- **最小机制**:20 行 attention + 10 行 self-mask/renorm,无新组件
- **数学自洽**:uniform w = M3,τ→0 = nearest-style share,有清晰退化路径
- **首次**:在 FedDSA 解耦架构上,把客户端风格作为跨客户端 attention key 嵌入 M3 multi-pos InfoNCE
- **独特 2x2 位置**:(Decouple, Style-share)格子,没有已发表工作占据
- **论文 Share 章节**:首次补齐,机制级 contribution

---

## Claim-Driven Validation Sketch(收窄为 3 主 claim + 1 附录)

### Claim A(主):PACS 全 outlier 下,SCPR 严格优于 M3 uniform sharing

- 数据集:PACS (4 domains, 7 classes)
- 对比:
  - A.1 FedDSA orth_only(Plan A baseline, 80.41%)
  - A.2 + M3 domain-aware multi-pos(对应 SCPR uniform attention;再验证 M3 81.91%)
  - A.3 **+ SCPR(τ_SCPR=0.3, self-mask)**
- 指标:3-seed {2, 15, 333} mean AVG Best / Last / per-domain(Art、Sketch)
- 预期:A.3 ≥ A.2 + 0.5%,A.3 ≥ 81.5%
- 决定性:若 A.3 < A.2,说明风格加权**不优于**等权 — 方法证伪
- 预算:3 configs × 3 seeds = 9 runs ≈ 18 GPU·h

### Claim B(主):Office 单 outlier 下,SCPR 不依赖参数个性化,性能 ≥ SAS

- 数据集:Office-Caltech10
- 对比:
  - B.1 FedDSA orth_only
  - B.2 + SAS(参数空间路由,89.82%)
  - B.3 **+ SCPR(原型层路由)**
- 指标:3-seed mean AVG Best / Last / Caltech-specific AVG
- 预期:B.3 ≥ B.2(原型层路由至少不差于参数层)
- 决定性:B.3 ≥ B.2 证明"原型层比参数层至少一样好",且 SCPR 普适于 PACS+Office(SAS 不行)
- 预算:3 configs × 3 seeds = 9 runs ≈ 18 GPU·h

### Claim C(机制):τ_SCPR 敏感性 + attention 健康度

- 数据集:PACS 3-seed
- τ ∈ {0.1, 0.3, 1.0, 3.0}
- 记录 H(w_k) 曲线、per-client avg attention 热图
- 预期:最优 τ ∈ [0.3, 1.0],H(w) 不应长期在 boundary
- 预算:4 configs × 3 seeds = 12 runs ≈ 24 GPU·h

### 附录 composability check(非主 claim)

- Office 上 SCPR + SAS 叠加
- 1 config × 3 seeds = 3 runs ≈ 6 GPU·h
- 只回答"两层路由是否正交",不进主表格

---

## Experiment Handoff Inputs

- **Must-prove claims**:A, B, C
- **Must-run ablations**:M3 baseline(=uniform SCPR 下界)、τ sweep、missing-class renormalize 正确性测试(单测)
- **Critical metrics**:PACS AVG Best/Last + per-domain, Office AVG Best/Last + Caltech
- **Highest-risk assumptions**:
  1. PACS 4-client 下 style_proto 区分度足够(若 H(w) 长期 ≈ log(K-1),需降维或换 key)
  2. SCPR ≥ M3(若否,方法证伪)
  3. 首轮 warmup 期间 bank 空时的 fallback 逻辑不破坏训练
- **Unit tests 需要**:
  - uniform τ→∞ 时 w_{i→j}^c 退化为 1/(K-1) 的数值验证
  - self-mask 下 j=k 的 weight 恒为 0
  - missing class renormalize 正确(手工 K=4 case)

---

## Compute & Timeline Estimate

- 主实验 ~54 GPU·h(PACS+Office+τ sweep)
- 附录 +6 GPU·h
- **总计 ~60 GPU·h**,单卡 3 天
- 实现 + 单测 + codex 审:1 天
- 结果回填 + 消融决策:1 天
- **总 5 天 end-to-end**

---

*Round 1 refinement 完成,主 mechanism 收窄为单一 canonical algorithm(self-masked style-weighted M3),准备发回 codex Round 2。*
