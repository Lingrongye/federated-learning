# Research Proposal: Decoupled Prototype Learning with Style Asset Sharing for Cross-Domain Federated Learning

## Problem Anchor
- **Bottom-line problem**: 跨域FL中，特征空间语义和风格纠缠，直接原型聚合产生"模糊原型"，跨域泛化严重下降
- **Must-solve bottleneck**: 没有方法先解耦语义/风格，再将解耦后的风格作为可共享增强资产
- **Non-goals**: 不解决标签倾斜、模型异构、通信效率极致优化、隐私理论保证
- **Constraints**: ResNet-18, Digit-5/Office-Caltech10/PACS/DomainNet, 毕业论文+目标AAAI/CVPR
- **Success condition**: (1)PACS/DomainNet超强基线 (2)消融证明解耦+共享>单独使用 (3)解耦诊断验证有效性

## Method Thesis
在联邦原型学习中，通过双头解耦将语义与风格分离，然后将风格统计量作为可共享的跨域增强资产，在纯语义空间做原型对齐，产生更纯净的语义原型和更强的跨域泛化。

## Contribution Focus
- **Dominant**: "Decouple-then-Share"范式——首次在FL原型学习中将解耦后的风格特征视为可共享增强资源
- **Supporting**: 正交+HSIC互补解耦正则化
- **Non-contributions**: 不声称"首次内容/风格分离"，不声称"独立性保证"

## Proposed Method: Decouple → Share → Align

### Architecture
```
Client k:
  x → ResNet-18 layers1-3 → f ∈ R^{256×14×14}
    ├── [Style Stats] μ_local=channel_mean(f), σ_local=channel_std(f) ∈ R^256
    ├── [AdaIN Augmentation] f_aug = σ_mixed*(f-μ_local)/σ_local + μ_mixed
    │     where (μ_mixed, σ_mixed) = α*(μ_local,σ_local) + (1-α)*(μ_ext,σ_ext), α~Beta(0.1,0.1)
    │
    ├── f → layer4 → pool → h → Hsem(h) → z_sem ∈ R^128
    ├── f → layer4 → pool → h → Hsty(h) → z_sty ∈ R^128 (diagnostic/decoupling)
    └── f_aug → layer4 → pool → h_aug → Hsem(h_aug) → z_sem_aug ∈ R^128
```

### Step 1: Decouple
```
L_decouple = λ_orth * (z_sem·z_sty / (||z_sem||·||z_sty||))² + λ_hsic * HSIC(z_sem, z_sty)
```
- 正交约束：消除线性相关
- HSIC约束：Gaussian kernel, bandwidth=median heuristic, 消除非线性依赖

### Step 2: Share (Style Asset Bank)
```
Style prototype (per-domain): P_sty_k = {(μ_k, σ_k)} from layer3 feature stats
Optional: K=3 cluster centers of (μ,σ) pairs via k-means

Server:
  B = ∪_k {style prototypes}
  Dedup: cosine_sim > 0.95 → remove
  Dispatch: randomly sample M=5 style prototypes per client
  Warmup: 前10轮不做风格增强（积累风格库）
```

### Step 3: Align
```
Semantic prototypes: P_sem_k^c = mean({z_sem(x) | y(x)=c})
Global: P_sem_global^c = mean_k(P_sem_k^c)

L_sem_con = -log[ exp(cos(z_sem, P_sem_global^y)/τ) / Σ_c exp(cos(z_sem, P_sem_global^c)/τ) ]
τ = 0.1
```

### Total Loss
```
L_total = L_CE(z_sem) + L_CE(z_sem_aug) + λ1*L_decouple + λ2*L_sem_con
```

### Aggregation
- Backbone conv layers + Hsem → FedAvg
- Hsty → private (不聚合，仅用于解耦约束和诊断)
- BN layers → private (FedBN原则)

### Inference
仅 backbone + Hsem + classifier。风格头和风格库推理时不需要。

### Communication Overhead
Upload: params(excl BN) + semantic protos(128d×C) + style stats(256×2×K)
vs FedProto: +~5K floats per round — negligible

## Claim-Driven Validation

### Claim 1 (Core): Decouple+Share > alternatives
- 2×2 ablation: {decouple, no-decouple} × {share, no-share} on PACS
- Baselines: FedAvg, FedBN, FedProto, FPL, FDSE, FISC
- Metric: avg accuracy + per-domain accuracy + domain variance, 3 seeds

### Claim 2: Dual constraint effective
- orth-only vs HSIC-only vs orth+HSIC on PACS
- Diagnostics: domain/class prediction accuracy from z_sem and z_sty
- Prototype compactness: intra-class distance / inter-class distance

### Claim 3: Decoupled augmentation > entangled augmentation
- AdaIN on decoupled f with external style vs AdaIN on f with raw mixed stats
- Metric: accuracy + prototype compactness

### Optional: Need-aware dispatch (appendix ablation)
### Supplementary: Frozen DINOv2-S backbone

## Compute & Timeline
- GPU-hours: ~250h
- Timeline: 实现2周 + 实验3周 + 论文2周 = 7周

## Remaining Risk
- 风格统计量(μ,σ)与解耦z_sty是两个不同的风格表示——需统一叙事：Hsty用于解耦约束，(μ,σ)用于共享增强，两者互补
- 需证明收益来自"更干净的原型"而非仅仅"更多增强"
