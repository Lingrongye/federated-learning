# Round 1 Refinement

## Problem Anchor (verbatim)
跨域FL中，特征空间语义和风格纠缠→直接聚合产生模糊原型。现有方法要么擦除风格、要么私有保留、要么在纠缠空间共享。没有方法先解耦再共享解耦后的风格作为增强资产。

## Anchor Check
- Original bottleneck: 模糊原型 + 风格信息未被利用
- Revised method still addresses it: 是——解耦产生更纯净的语义原型（减模糊），风格共享增强泛化
- Reviewer suggestions rejected as drift: "加DINO/CLIP变体"——接受作为补充实验但不改变核心方法，毕业论文以ResNet-18为主体
- Drift risk acknowledged: 如果收益主要来自augmentation而非cleaner prototypes，需在消融中隔离

## Simplicity Check
- Dominant contribution after revision: "解耦后风格资产化共享"——一个机制，不是模块列表
- Components removed: (1) need-aware dispatch → 降级为可选消融项 (2) 精简为最核心的3个损失
- Reviewer suggestions rejected as unnecessary complexity: 无
- Why remaining mechanism is smallest adequate: 双头(解耦必需) + 风格库(共享必需) + 对比对齐(对齐必需)

## Changes Made

### 1. Method Specificity: 明确风格增强操作
- Reviewer said: z_aug = z_sem + λ*MixStyle太模糊
- Action: 改为在骨干特征层做AdaIN风格注入，而非在投影后做加法
- Reasoning: AdaIN是成熟的风格迁移算子，比向量加法更有物理意义
- Impact: 增强操作从"加噪声"变为"特征统计量替换"

### 2. Contribution Quality: 精简为一个优雅机制
- Reviewer said: 模块太多
- Action: 核心机制精简为"Decouple-Share-Align"三步，need-aware dispatch降级为可选
- Reasoning: 一篇论文一个dominant contribution
- Impact: 论文更聚焦

### 3. Frontier Leverage: 加DINO/CLIP补充实验
- Reviewer said: ResNet-18在2026过时
- Action: 主实验保持ResNet-18（公平对比），补充一组frozen DINOv2实验
- Reasoning: 证明机制在现代骨干上也有效，但不改变核心方法
- Impact: 增强paper说服力，不增加方法复杂度

### 4. 风格库定义：从"per-domain均值"到"per-domain多原型"
- Reviewer said: 单个均值太粗糙不配叫bank
- Action: 每个客户端上传K个聚类中心(K=3 via k-means on z_sty)，而非单个均值
- Reasoning: 多原型更好地表示域内风格多样性
- Impact: 风格库更丰富，"bank"名副其实

## Revised Proposal

# Decoupled Prototype Learning with Style Asset Sharing for Cross-Domain FL (v2)

## Problem Anchor
[同上，不变]

## Method Thesis
- One-sentence: 在联邦原型学习中，通过双头解耦将语义与风格分离，然后将风格原型作为可共享的跨域增强资产，在纯语义空间做原型对齐，产生更纯净的语义原型和更强的跨域泛化。
- Mechanism-first framing: 这是一个关于"FL中什么信息该以什么形式共享"的研究。参数通过FedAvg共享，语义原型通过对比对齐共享，风格原型通过资产库共享——三种信息、三种共享方式。

## Contribution Focus
- **Dominant**: "Decouple-then-Share"范式——解耦后的风格作为可共享增强资产
- **Supporting**: 正交+HSIC互补解耦正则化
- **Dropped from main**: need-aware dispatch (可选消融项)

## Revised Method

### Architecture
```
Client k:
  x → ResNet-18(BN_local) → h ∈ R^512
       ├── Hsem: Linear(512,128)+ReLU+Linear(128,128) → z_sem
       └── Hsty: Linear(512,128)+ReLU+Linear(128,128) → z_sty
```

### Step 1: Decouple (双重约束)
```
L_decouple = λ_orth * (z_sem·z_sty / (||z_sem||·||z_sty||))² + λ_hsic * HSIC(z_sem, z_sty)
```
- 正交约束: 消除线性相关
- HSIC约束: 消除非线性统计依赖 (Gaussian kernel, bandwidth = median(pairwise distances))

### Step 2: Share (风格资产共享)
**风格原型构建** (per-domain, 多原型):
```
For client k:
  Collect all {z_sty(x) | x ∈ D_k}
  Run k-means(K=3) → get 3 cluster centers as style prototypes
  Upload: {s_k^1, s_k^2, s_k^3}
```

**服务器风格库管理**:
```
B = ∪_k {s_k^1, s_k^2, s_k^3}
Dedup: remove entries with cosine_sim > 0.95
Dispatch: randomly sample M=5 style prototypes for each client (main version)
Optional: need-aware dispatch based on margin gap (ablation)
```

**风格增强** (AdaIN in feature space):
```
Given z_sem, z_sty (local), s_ext (from bank):
  # MixStyle in style space
  α ~ Beta(0.1, 0.1)
  s_mixed = α * z_sty + (1-α) * s_ext
  
  # AdaIN injection
  μ_s, σ_s = mean(s_mixed), std(s_mixed)
  z_aug = σ_s * (z_sem - mean(z_sem)) / std(z_sem) + μ_s
  
  # z_aug is a "counterfactual" semantic feature with different style
```

### Step 3: Align (语义软对齐)
```
# Semantic prototypes
P_sem_k^c = mean({z_sem(x) | y(x)=c, x ∈ D_k})  per-class
P_sem_global^c = mean_k(P_sem_k^c)  global aggregation

# InfoNCE contrastive alignment
L_sem_con = -log[ exp(sim(z_sem, P_sem_global^{y})/τ) / Σ_c exp(sim(z_sem, P_sem_global^c)/τ) ]
```

### Training
```
L_total = L_task(z_sem) + L_task(z_aug) + λ1*L_decouple + λ2*L_sem_con

# L_task applied to both original and augmented features
# This forces the classifier to be robust to style variation
```

**Aggregation**:
- Backbone conv layers + Hsem → FedAvg
- Hsty → private (not aggregated)
- BN layers → private (FedBN principle)

**Inference**: backbone + Hsem + classifier only. No style head or bank needed.

### Failure Modes
| Mode | Detection | Mitigation |
|------|-----------|------------|
| z_sem leaks domain info | Domain classifier accuracy from z_sem > chance | Increase λ_hsic |
| z_sem loses class info | Class accuracy from z_sem drops | Decrease λ_decouple |
| AdaIN creates off-manifold features | z_aug classification accuracy << z_sem | Reduce style mixing ratio |
| Style bank collapses | Pairwise similarity > 0.9 for most entries | Stronger dedup + diversity sampling |

## Validation (3 claims, leaner)

### Claim 1 (Core): Decouple+Share > alternatives
- 2×2 ablation on PACS: {decouple, no-decouple} × {share, no-share}
- Baselines: FedProto, FDSE, FISC, FedSeProto(if possible)
- Metric: avg accuracy + per-domain accuracy + domain variance

### Claim 2: Dual constraint > single
- PACS: orth-only, HSIC-only, orth+HSIC
- Diagnostic: domain/class prediction accuracy from z_sem and z_sty

### Claim 3: Style augmentation in decoupled space > entangled space
- Compare: AdaIN on z_sem with decoupled s_ext vs AdaIN on h with raw style stats
- Metric: accuracy + prototype compactness

### Optional Claim 4: Need-aware dispatch > random
- PACS/DomainNet: need-aware vs random vs no dispatch
- Report as supplementary ablation

### Supplementary: Modern backbone
- Frozen DINOv2-S + same heads → verify mechanism transfers

## Compute & Timeline
- GPU-hours: ~250h (PACS full + DomainNet subset + ablations + diagnostics, 3 seeds)
- Timeline: 实现2周 + 实验3周 + 论文2周 = 7周
