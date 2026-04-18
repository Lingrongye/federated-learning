# 研究方案 Round 0:Style-Conditioned Prototype Retrieval (SCPR)

> 面向跨域联邦学习的风格条件化原型检索机制
> 基础框架:FedDSA (Decouple-Share-Align)
> 目标会议:CVPR / ICCV / NeurIPS

---

## Problem Anchor(不可变,每轮复制)

- **Bottom-line problem**:跨域联邦学习中,客户端数据来自不同视觉风格域(照片/素描/油画/线稿等)。现有原型对齐方法(FedProto/FPL/FedPLVM)把所有客户端的同类原型求平均后广播,使得风格-outlier 客户端(Caltech、Sketch)被稀释严重,性能显著落后于 FedBN/FDSE 等域适应方法。

- **Must-solve bottleneck**:
  1. 现有 global-mean prototype 把 style 信息"揉糊",outlier 域客户端对齐时丢失自身风格结构
  2. 我们自己已验证的 SAS(Style-Aware sem_head Aggregation)在 Office-Caltech10 Caltech-outlier 场景 +1.21%,但在 PACS 全 outlier 场景 −0.65%,因参数空间个性化退化为 FedBN-like 本地化(EXP-086 诊断)
  3. 论文框架 FedDSA = Decouple-Share-Align 中,**Share 章节从未落地**:EXP-059 在 z_sem 空间做 AdaIN 注入致 PACS −2.54%,EXP-078d 在 h 空间 AdaIN→InfoNCE 致 NaN 崩溃,所有"风格共享"尝试均失败

- **Non-goals**(严格排除,已被 85 次实验证伪):
  1. 不做"风格作为训练数据增强"(不改 z_sem 特征,不改 CE 输入)
  2. 不做分类器个性化(EXP-093 sas-FH 证伪)
  3. 不加辅助损失(HSIC/PCGrad/Triplet/CKA/Uncertainty 全败)
  4. 不做架构改动(多层注入/VAE head/非对称 heads 全败)
  5. 不做训练调度复杂化(ramp-down/warmup 延长等全败)
  6. 不引入新的可训练组件(soft cap = 0 个新网络)

- **Constraints**:
  - Backbone:ResNet-18 (Office) / AlexNet-from-scratch (PACS 对齐 FDSE paper)
  - 遵守 FedBN 原则(BN 参数本地化)
  - 训练预算:R=200 rounds,3-seed {2, 15, 333},单卡 24GB
  - 保留:正交解耦(cos² + HSIC)、orth_only Plan A 基线(LR=0.05, warmup=50, λ_orth=1.0)
  - 保留:InfoNCE 对齐机制(只改 target,不改 loss 形式)
  - 不允许:修改 ResNet-18/AlexNet 骨干、增加深度、引入 VGG/CLIP 等外部网络

- **Success condition**(同时满足才算成功):
  1. **PACS 3-seed mean AVG Best ≥ 81.5%**(vs orth_only baseline 80.41%,提升 ≥ +1.1%,且严格优于 M3 孤立值 81.91%)
  2. **Office 3-seed mean AVG Best ≥ 90.5%**(vs SAS baseline 89.82%,提升 ≥ +0.7%,逼近 FDSE 90.58%)
  3. **R200 drop(AVG Best − AVG Last)≤ 2%**(训练稳定,不崩)
  4. **新增代码 < 100 行**、无新 trainable 组件
  5. **论文叙事闭环**:Share 章节首次有正向实证,Decouple-Share-Align 三章均成立

---

## Technical Gap(问题为什么现在没被解决)

### 现有方法的共性失败模式

| 流派 | 代表方法 | 对风格的态度 | 为什么不适合 |
|------|----------|--------------|--------------|
| Global-mean prototype | FedProto / FedAvg-proto | 隐式抹除 | outlier 域被稀释 |
| Cluster-based multi-proto | FPL / FedPLVM / MP-FedCL | 在**混合特征**里聚类,仍然隐式擦除 style | 聚类 key 是原型自己的几何,不是风格身份 |
| 风格擦除 | FDSE / FedSeProto / FedDP | domain feature = 噪声 | 风格信息整块丢失,下游看不见风格 |
| 风格本地保留 | FedSTAR (FiLM) / FedBN | 风格不共享 | 无跨域互助 |
| 风格共享但不解耦 | FISC / PARDON / StyleDDG / FedCCRL | 在**混合空间**做 AdaIN | 破坏语义特征,我们已经在 EXP-059 证伪 |
| 风格感知参数聚合(我们的 SAS) | FedDSA+sas | 参数空间软混合 | PACS 全 outlier 时退化为本地路径 |

**关键缺口**:还没有任何方法把"风格"作为**跨客户端共享、在解耦后的表征空间**被**客户端按需检索**的资产。这正好是 FedDSA 原本的 Share 章节口号,但我们自己的 3 次尝试都在"风格作为训练输入"这条路径上翻车,必须换**对齐目标**这条路径。

### 朴素放大的方法为什么不够

- 加客户端数/加 round/加数据:无助于"对齐目标本身是糊的"这一结构缺陷
- 扩大 sem_head/加层:EXP-042 非对称 heads 已证伪,反而 gap=15.79%
- 加预训练/VLM:和 CVPR baseline 不公平对比,且违反 constraint

### 最小充分干预

**唯一需要改的是"客户端在 InfoNCE 对齐时,目标原型怎么构造"**。其余不动:
- 特征提取、正交解耦、sem_head/classifier、FedBN、服务器 DFE 聚合、CE loss 全部保留
- 客户端上传改为"每类原型 + 本域风格向量 (μ_k, σ_k)"(风格向量本来 SAS 就在传)
- 服务器广播改为"保留所有域的 (class-proto, style-proto)",不再平均
- 客户端按本域风格 s_k 对 style-proto bank 做 softmax attention,重构出风格匹配的 class-proto 作为 InfoNCE target

---

## Method Thesis

- **一句话 thesis**:用"客户端风格 → style-prototype bank"的 attention 检索取代"全局平均",把风格信息从"被稀释的标量权重"变成"动态检索的共享资产"

- **为什么是最小充分干预**:只改对齐目标的构造方式;不增加 trainable 组件;不加损失;不碰 z_sem 特征;新增代码 <100 行

- **为什么"当下可行"**:(1) 风格向量 (μ, σ) 已经在 EXP-059/SAS 中实现并验证可用;(2) domain-indexed prototype bank 已经在 M3 中实现;(3) 所有零件都在 base 上,组合即可

---

## Contribution Focus

- **主贡献(1 个,mechanism 级)**:
  > **Style-Conditioned Prototype Retrieval (SCPR)**:在域索引的 (style_proto, class_proto) bank 上用客户端瞬时风格做 softmax attention,动态检索风格匹配的类原型作为 InfoNCE 对齐目标。这是第一个把"风格"作为跨客户端共享资产、通过查询机制接入到解耦后的语义对齐目标的方法

- **辅贡献(至多 1 个,可选)**:
  > **Style-Weighted Multi-Positive InfoNCE**:不是将 K 个原型压成一个加权均值,而是让 K 个原型都作为 positive,每个 positive 的对比权重按风格相似度加权。严格泛化 FedProto(K=1 全局均值) 和 M3 multi-positive(K 个 positive 等权)。

- **明确排除**(not contributions):
  1. 不引入"风格作为训练数据增强"(EXP-059 失败)
  2. 不引入"风格作为参数路由"(SAS 已发表过,且 PACS 失效)
  3. 不引入"新的解耦损失"(继续用 cos² + HSIC)
  4. 不引入新可训练网络

---

## Proposed Method

### Complexity Budget

| 组件 | 状态 |
|------|------|
| 主干网络 (ResNet-18 / AlexNet) | **冻结/复用**(不改结构) |
| 正交解耦 (cos² + HSIC) | **复用**(已 merge 到 Plan A) |
| sem_head (MLP) | **复用**(完全不个性化) |
| style_head | **复用** |
| classifier | **复用**(FedAvg 不个性化) |
| BN 层 | **复用**(FedBN 本地私有) |
| Style bank (μ, σ) | **复用**(SAS 已有) |
| Domain-indexed class prototype bank | **复用**(M3 已有) |
| **SCPR retrieval 模块** | **新增(仅 attention 函数,无参数)** |
| **Style-weighted multi-pos InfoNCE** | **新增(仅 loss 权重调整,无参数)** |

**Soft cap 检查**:
- 新 trainable 组件:**0** ≤ 2 ✓
- 新 loss 项:**0**(只改 target 构造) ≤ 2 ✓

**明确不做**:多层风格、VAE head、对比 head、GNN、RL 门控、外部预训练 VLM、扩散 prior。

### System Overview

```
┌────────────────────────── 客户端 k ──────────────────────────┐
│                                                               │
│  x ─► 主干(ResNet-18) ─► pool5 ─► h (1024d)                   │
│           │                      │                            │
│           │    ┌─── style_head ──┘─► z_sty ──► style_proto_k │
│           │    │                                 (本域风格 s_k)│
│           └────┴─── sem_head ────► z_sem ─┬► classifier ─► ŷ │
│                                            │                  │
│                                            │    InfoNCE target│
│                                            ▼   ┌─────────────┤
│                                          L_InfoNCE(z_sem, P*) │
│                                                               │
└───────────────▲──────────────────────────▲────────────────────┘
                │ 上传 (proto_c^k, s_k)   │ 下发 retrieved protos
                │                          │
┌───────────────┴──────────────────────────┴────────────────────┐
│                      Server (SCPR aggregator)                  │
│                                                                │
│  Domain-indexed bank:                                          │
│    Style-proto bank:   {s_k}_{k=1..K}        每域一个风格向量  │
│    Class-proto bank:   {p_c^k}_{c=1..C, k=1..K}  每(类,域)一个 │
│                                                                │
│  对每个 target client i:                                       │
│     w_{i→j} = softmax_j(cos(s_i, s_j) / τ_SCPR)                │
│     P*_c^i  = Σ_j w_{i→j} · p_c^j     (风格匹配的类原型)       │
│                                                                │
│  Dispatch {P*_c^i}_{c=1..C} 给 client i (代替 global mean)     │
│                                                                │
│  DFE/sem_head/classifier 参数:标准 FedAvg 聚合                 │
│  BN 参数:不聚合(FedBN 原则)                                    │
└────────────────────────────────────────────────────────────────┘
```

### Core Mechanism — SCPR Retrieval

**输入**:
- 客户端 k 的本域风格向量 `s_k = (μ_k, σ_k) ∈ R^{2d}`(d = feature dim,ResNet-18 是 512,拼起来 1024)
- 服务器维护 `{s_k, {p_c^k}_{c=1..C}}_{k=1..K}`,每轮在客户端 pack 时更新

**检索**:
```
对每个 target client i:
  原始 cos similarity:   g_ij = cos(s_i, s_j) ∈ [-1, 1]
  温度缩放 softmax:       w_{i→j} = exp(g_ij / τ_SCPR) / Σ_l exp(g_il / τ_SCPR)
  风格匹配类原型:          P*_c^i = Σ_j w_{i→j} · p_c^j           (∀c)
```

**关键属性**:
- 当 τ_SCPR → ∞(无风格感知):w 均匀,P*_c^i → global-mean(退化为 FedProto)
- 当 τ_SCPR → 0(完全 hard):w 是 one-hot,P*_c^i = 最相似域原型(退化为 nearest-style retrieval)
- 继承 SAS 最优 τ=0.3 作为默认,τ 扫描 {0.1, 0.3, 1.0, 3.0}

**梯度停传**:`P*_c^i.detach()` — retrieved proto 是软锚点,不向 proto bank 回传梯度(与 FedProto 里 global proto 一致)。这是关键的安全性设计,避免 EXP-078d 的 NaN(当时梯度从 InfoNCE 回流到 AdaIN-modified z_sem)。

### Supporting Component — Style-Weighted Multi-Positive InfoNCE(可选)

**动机**:硬把 K 个原型加权平均成一个单 positive,会丢失原型间的几何结构。让 K 个原型都作为 positive,权重决定 pulling 强度。

**公式**:
```
L_SCPR(z_i, y=c)
  = - Σ_j w_{i→j} · log [ exp(sim(z_i, p_c^j)/τ_nce)  /
                         (exp(sim(z_i, p_c^j)/τ_nce) + Σ_{c'≠c, l} exp(sim(z_i, p_{c'}^l)/τ_nce)) ]
```

- 当 w 均匀(高 τ_SCPR):退化为标准 SupCon multi-positive(= M3)
- 当 w one-hot(低 τ_SCPR):退化为 single positive(最相似域) + all negatives
- 中间区域:style-weighted pulling,近风格域拉力大,远风格域拉力小

**注意**:retrieved proto 变体和 weighted multi-pos 变体**都做实验**,消融选更稳的作为主方法。默认主实现是 retrieved single proto(简单,稳定)。

### Modern Primitive Usage(无)

**故意保持保守**:不引入 LLM/VLM/Diffusion/RL。原因:
- 问题本身是"联邦通信 + 原型对齐"的工程问题,不需要生成模型
- 引入 CLIP/VLM 会违反"公平基线对比"的 constraint(FedProto/FPL/FDSE 都不用 VLM)
- CVPR FL 圈重视"mechanism clean + 公平对比",额外 VLM 反而是负资产
- **Frontier leverage 体现在:将 attention-based retrieval(Transformer-era 原语)引入到 FL 原型系统**,这在 FedSTAR 之后已被承认是 FL 原型学习的正确工具

### Integration into Base

- **挂载点**:替换 `FDSE_CVPR25/algorithm/feddsa_scheduled.py` 的 `iterate()` 中的 prototype aggregation 分支
- **冻结**:backbone/sem_head/style_head/classifier/BN 全部不变,FedAvg/FedBN 规则不变
- **新增**:
  1. Server 端:`_retrieve_prototypes(target_cid)` 方法(~30 行)
  2. Server 端:去掉 `proto_global = mean(proto_c^k)`,改为 dispatch per-client retrieved protos
  3. Client 端:InfoNCE target 从 `proto_global[c]` 改为 `retrieved_protos[c]`(~5 行)
- **客户端训练流程不变**:`z_sem ← sem_head(h); L_CE = CE(classifier(z_sem), y); L_InfoNCE = infonce(z_sem, retrieved_protos)`
- **CE 路径保持原封不动**(z_sem → classifier → ŷ → CE(ŷ, y))

### Training Plan

**训练预算**:R=200 rounds,E=1(Office) / E=5(PACS),B=50,LR=0.05 (Plan A 最优)。

**超参**(最小充分):
- `lambda_orth = 1.0`(Plan A)
- `lambda_hsic = 0.1`(Plan A)
- `lambda_sem = 1.0`(InfoNCE 权重,Plan A)
- `tau_nce = 0.3`(InfoNCE 温度,Plan A 最优)
- `tau_scpr = 0.3`(SCPR retrieval 温度,和 SAS 一致作为默认)
- `warmup_rounds = 50`(Plan A)

**无新损失权重**,无新可训练参数。

**训练曲线监控**(诊断,不影响训练):
- Attention entropy `H(w_k) = - Σ_j w_{k→j} log w_{k→j}`(退化检测)
- Retrieved vs global-mean 的 L2 距离(退化检测)
- cos_sim(grad_CE, grad_InfoNCE)(梯度冲突检测,仿 EXP-077)

### Failure Modes and Diagnostics

| 失败模式 | 检测信号 | 回退 |
|---------|---------|------|
| 1. Attention 塌缩为 uniform(高维 cos 区分度不够) | H(w_k) > 0.95·log(K) 持续 > 20 轮 | 换 attention key 为 style_head 投影后的低维向量(仍然零参数,只改维度)或 `tau_scpr` 调到 0.1 |
| 2. Attention 塌缩为 one-hot(客户端完全只听自己) | H(w_k) < 0.1·log(K) 持续 > 20 轮 | `tau_scpr` 调到 1.0 软化 |
| 3. PACS 全 outlier 下性能退化 | 3-seed AVG Best < orth_only baseline | 验证是否至少达到 M3(等价 uniform attention);若否则回退到 retrieve-and-average-with-global-mean(50/50 混合) |
| 4. InfoNCE 梯度爆炸(NaN) | loss 出现 NaN/Inf | retrieved proto 更激进 detach、tau_nce 从 0.3 提到 0.5 |
| 5. Outlier 域自己的 proto 主导 attention(退化为 FedBN) | w_{k→k} > 0.9 持续多轮 | 在 attention 中 mask self(强制 j ≠ k) |

**内置安全网**:若 SCPR 失败,软回退到 M3 multi-positive(已验证 +5.09%)—— 因为 uniform attention 下 SCPR 数学上就等于 M3。这个"最差情况 = 已知好基线"的性质是 SCPR 的一个重要设计安全系数。

### Novelty and Elegance Argument

**最接近的先前工作**:
1. **FedProto (AAAI 2022)**:global mean proto,InfoNCE/MSE 对齐 — SCPR 替换其 target 构造方式,用风格 attention 检索
2. **FPL (CVPR 2023)** / **FedPLVM (NeurIPS 2024)**:FINCH 多聚类原型 — 都用原型自身几何聚类,**key 是原型**;SCPR 的 **key 是客户端风格**,语义完全不同
3. **FedSTAR (2025)**:Transformer 聚合原型 + FiLM 风格分离 — 聚合端用了 attention,但**风格严格本地**,没有跨客户端共享;SCPR 首次把风格做成跨客户端 attention key
4. **FISC/PARDON (ICDCS 2025)**:风格共享 + AdaIN — 在**图像/混合特征空间**做 AdaIN,改训练数据;SCPR 在**解耦后表征空间** + 只改对齐 target
5. **SAS (我们,未发表)**:参数空间风格软聚合 — PACS 失败;SCPR 把风格路由下放到原型层,保留"所有客户端共享同一 sem_head"的隔离性

**差异化 2×2 矩阵**:

|   | 风格不共享 | 风格共享 |
|---|-----------|---------|
| 不解耦 | FedBN, FedAvg, FedProto | FISC, PARDON, StyleDDG, FedCCRL |
| 解耦 | FedSTAR, FedSeProto, FDSE | **SCPR(首次)** |

**Elegance 论证**:
- 单一 mechanism(风格条件化检索),0 新参数,<100 行代码
- 数学上严格泛化 FedProto(τ→∞) 和 M3(uniform w,multi-pos 变体)
- 最差情况退化到 M3(+5.09%),上限通过风格 routing 继续往上拉
- 解救了 FedDSA "Share" 章节,且规避了所有已验证的失败模式(不改 z_sem、不改参数、不加 loss)

---

## Claim-Driven Validation Sketch(最小充分,≤ 3 个核心实验块)

### Claim 1(主):SCPR 在**全 outlier 分布(PACS)上**仍然有效,严格优于 M3 multi-positive 基线

- 数据集:PACS (4 域 7 类)
- 核心对比:
  - C1.1 FedDSA orth_only(Plan A baseline)
  - C1.2 FedDSA + M3 multi-positive(等价于 SCPR uniform attention,即最差退化)
  - C1.3 **FedDSA + SCPR(τ_SCPR=0.3)**
- 指标:3-seed {2, 15, 333} mean AVG Best / AVG Last / per-domain(重点看 Art、Sketch)
- 预期:C1.3 ≥ C1.2 + 0.5%,且 C1.3 ≥ 81.5%(承诺 threshold)
- 决定性:若 C1.3 < C1.2,SCPR 在风格 routing 上毫无价值,方法证伪

### Claim 2(主):SCPR 在**单 outlier 分布(Office-Caltech10)上**≥ SAS 基线

- 数据集:Office-Caltech10 (4 域 10 类,Caltech 是 outlier)
- 核心对比:
  - C2.1 FedDSA orth_only(基线)
  - C2.2 FedDSA + SAS(τ_SAS=0.3,已验证 89.82%)
  - C2.3 **FedDSA + SCPR(τ_SCPR=0.3)**
  - C2.4 FedDSA + SCPR + SAS(叠加验证正交性)
- 指标:3-seed mean AVG Best / Caltech-specific AVG
- 预期:C2.3 ≥ C2.2,C2.4 ≥ max(C2.2, C2.3)+ 0.3%
- 决定性:(a) C2.3 ≥ C2.2 证明原型层 ≥ 参数层;(b) C2.4 > C2.3 证明两层正交互补

### Claim 3(机制):τ_SCPR 敏感性与 attention 健康度

- 数据集:PACS 3-seed(小规模消融)
- 扫描 τ_SCPR ∈ {0.1, 0.3, 1.0, 3.0}
- 记录 attention entropy H(w) 曲线、retrieved vs global-mean L2 距离
- 预期:最优 τ ∈ [0.3, 1.0],H(w) ∈ [0.3·log K, 0.85·log K](既不 uniform 也不 one-hot)
- 决定性:若所有 τ 下 H(w) 都近 log K,attention 塌缩,需要降维 key(fallback 分支)

### 实验量(总预算)

- PACS 主实验:3 configs × 3 seeds = 9 runs × ~2h = **18 GPU·h**
- Office 主实验:4 configs × 3 seeds = 12 runs × ~2h = **24 GPU·h**
- τ 消融:3 configs × 3 seeds = 9 runs × ~2h = **18 GPU·h**
- **总计 ~60 GPU·h**,单卡 3 天完成

---

## Experiment Handoff Inputs(交给 /experiment-plan)

- **Must-prove claims**:Claim 1 (主), Claim 2 (主), Claim 3 (机制)
- **Must-run ablations**:
  - uniform attention (= M3) 作为下界
  - τ 扫描
  - mask self-attention 变体(防止退化到 FedBN)
  - SCPR + SAS 叠加
  - (可选)single retrieved proto vs weighted multi-pos
- **Critical datasets/metrics**:PACS AVG Best/Last + per-domain, Office AVG Best/Last + Caltech-specific
- **Highest-risk assumptions**:
  1. 高维 cos 相似度在 4-5 clients 下能产生有效的区分度 — 需要 attention entropy 诊断
  2. PACS 全 outlier 下 SCPR ≥ M3 — 若失败则方法证伪
  3. retrieved proto detach 足以避免 EXP-078d 式梯度爆炸 — 需要 NaN watch

---

## Compute & Timeline Estimate

- **GPU·h**:~60(单卡 24GB,3 天完成全部主表)
- **数据 / 标注**:0(PACS 和 Office-Caltech10 已经就位)
- **人力 timeline**:1 天实现 + codex 审 + 单测 → 3 天跑实验 → 1 天回填 NOTE.md + 消融决策,共 5 天

---

*初始方案写完,等待 Phase 2 GPT-5.4 审核。*
