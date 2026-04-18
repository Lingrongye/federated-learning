# Round 2 Refinement

## Problem Anchor (verbatim, unchanged)
跨域FL中，特征空间语义和风格纠缠→直接聚合产生模糊原型。没有方法先解耦再共享解耦后的风格作为增强资产。

## Anchor Check
- 锚点保持：是
- 漂移风险：将"what to share in FL"收窄为"cross-domain FL prototype learning"
- 拒绝的建议：无

## Simplicity Check
- Dominant contribution unchanged: Decouple-then-Share
- Components adjusted: AdaIN位置从嵌入层移到中间特征图
- 无新增复杂度

## Changes Made

### 1. AdaIN从嵌入层移到中间特征图
- Reviewer said: 128d嵌入上做mean/std不是真正的AdaIN
- Action: 风格增强在骨干的中间特征图(conv4 output, f∈R^{512×7×7})上做AdaIN，然后再过投影头
- Reasoning: AdaIN的inductive bias是channel-wise统计量替换，需要空间维度才有意义
- Impact: 风格增强更有物理意义，与经典风格迁移一致

### 2. Framing收窄
- Reviewer said: "study of what to share"太宽泛
- Action: 标题和定位收窄为"Decoupled Prototype Learning for Cross-Domain FL"
- Impact: 声称与证据匹配

### 3. L_sem_con显式定义
- Reviewer said: InfoNCE未完全定义
- Action: 明确query=z_sem(x), positive=P_sem_global^{y(x)}, negatives={P_sem_global^c | c≠y(x)}, τ=0.1

## Revised Method (v3, key changes only)

### Architecture (updated)
```
Client k:
  x → ResNet-18 layers 1-3 → f_mid ∈ R^{256×14×14}
    → ResNet-18 layer 4 → f ∈ R^{512×7×7}
    → AdaptiveAvgPool → h ∈ R^512
       ├── Hsem(h) → z_sem ∈ R^128
       └── Hsty(h) → z_sty ∈ R^128

Style statistics are extracted from f (before pooling):
  μ_local = channel_mean(f)  ∈ R^512
  σ_local = channel_std(f)   ∈ R^512
```

### Style Prototype (revised: feature-map statistics)
```
P_sty_k = {(μ_k, σ_k)} = channel-wise mean and std of f across all samples in D_k
Optional: K=3 cluster centers of (μ, σ) pairs via k-means
Upload: (μ, σ) pairs — compact (512*2 = 1024 floats per prototype)
```

### Style Augmentation (AdaIN on feature map, proper)
```
Given f (local feature map), (μ_ext, σ_ext) from style bank:
  α ~ Beta(0.1, 0.1)
  μ_mixed = α * μ_local + (1-α) * μ_ext
  σ_mixed = α * σ_local + (1-α) * σ_ext
  
  f_aug = σ_mixed * (f - μ_local) / σ_local + μ_mixed   # Standard AdaIN
  
  h_aug = AdaptiveAvgPool(ResNet18_layer4(f_aug))  # Continue through rest of backbone
  z_sem_aug = Hsem(h_aug)
```

**注意**: 风格增强在layer3 output上做AdaIN，然后继续经过layer4→pool→Hsem。这样z_sem_aug是"相同语义但不同风格"的特征。

### Decoupling (unchanged)
```
z_sem = Hsem(h),  z_sty = Hsty(h)
L_decouple = λ_orth*cos²(z_sem, z_sty) + λ_hsic*HSIC(z_sem, z_sty)
```

### Semantic Alignment (explicit)
```
L_sem_con = -log[ exp(cos(z_sem, P_sem_global^{y}) / τ) / Σ_c exp(cos(z_sem, P_sem_global^c) / τ) ]
τ = 0.1, query = z_sem(x), positive = P_sem_global^{y(x)}, negatives = all other class prototypes
```

### Total Loss
```
L_total = L_CE(z_sem) + L_CE(z_sem_aug) + λ1*L_decouple + λ2*L_sem_con
```
4 terms, each with clear purpose:
- L_CE(z_sem): 原始分类
- L_CE(z_sem_aug): 风格增强后分类（强制风格不变性）
- L_decouple: 解耦约束
- L_sem_con: 语义原型对齐

### Communication per round
```
Upload: model params (excl BN) + P_sem_k^c (128d × C classes) + style stats (1024d × K)
Download: global params + P_sem_global + style subset (1024d × M)
Overhead vs FedProto: +style stats (~5K floats) — negligible
```

### Framing (narrowed)
- Title: "Decoupled Prototype Learning with Style Asset Sharing for Cross-Domain Federated Learning"
- One-sentence: "We decouple semantic and style in FL prototype learning, then share decoupled style statistics as cross-domain augmentation assets, producing cleaner semantic prototypes and stronger generalization."
- NOT: "a study of what to share in FL"
