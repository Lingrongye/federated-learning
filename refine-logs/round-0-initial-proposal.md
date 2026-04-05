# Research Proposal: Decoupled Prototype Learning with Style Asset Sharing for Cross-Domain Federated Learning

## Problem Anchor
- **Bottom-line problem**: 跨域联邦学习中，不同客户端数据来自不同域（照片/素描/油画），P(X|Y)存在显著域偏移。现有原型学习方法在混合（纠缠）特征空间中做原型聚合，产生"模糊原型"，导致跨域泛化性能严重下降。
- **Must-solve bottleneck**: 特征空间中语义信息和风格信息纠缠在一起，直接聚合导致语义污染；现有方法要么擦除风格（浪费域增强资源），要么私有保留风格（失去跨域知识共享机会），要么在纠缠空间共享风格（语义被污染）。
- **Non-goals**: 不解决标签倾斜(label skew)问题；不解决模型异构问题；不追求通信效率的极致优化；不做隐私保证的理论证明。
- **Constraints**: ResNet-18骨干，Digit-5/Office-Caltech10/PACS/DomainNet数据集，毕业论文+目标AAAI/CVPR投稿，计算资源有限。
- **Success condition**: (1)在PACS和DomainNet上超过FedProto/FPL/FDSE等强基线 (2)消融实验证明"解耦后共享"优于"不解耦共享"和"解耦后擦除" (3)解耦诊断证明语义头域无关、风格头域特定。

## Technical Gap

现有跨域FL方法对域/风格信息的处理可分为三派，均有本质缺陷：

1. **擦除派**(FDSE, FedSeProto, FedDP)：用MI/IB/DSE压缩掉域信息 → 丢弃了可用于跨域增强的风格资源
2. **私有派**(FedSTAR, FedBN)：风格仅本地保留 → 失去跨域知识共享机会
3. **不解耦共享派**(FISC, StyleDDG, FedCCRL)：在纠缠特征空间做风格迁移/共享 → 语义被风格操作污染

**Gap**：没有方法先将语义和风格显式解耦，再将解耦后的风格作为可共享资产进行跨域增强。

## Method Thesis
- **One-sentence thesis**: 在联邦原型学习中，将特征显式解耦为语义和风格两个独立空间，然后将风格原型作为可共享的数据资产进行跨域增强，使本地模型在不接触其他域原始数据的情况下"看到"多域风格变异，从而提升跨域泛化。
- **Why smallest adequate intervention**: 只需在骨干后加两个投影头+一个服务器端风格库，不改变骨干架构、不引入生成模型、不需要对抗训练。
- **Why timely**: 风格共享已被FISC/StyleDDG验证有效，但它们在纠缠空间操作；解耦技术(HSIC/正交约束)已成熟但未用于FL风格共享场景。本方法是两条成熟技术线的首次交叉。

## Contribution Focus
- **Dominant contribution**: 提出"解耦后风格资产化共享"的新范式——首次在FL原型学习中将解耦后的风格特征视为可共享增强资源，而非噪声擦除或私有保留。
- **Supporting contribution**: 正交+HSIC互补解耦约束，在几何和统计两个层面保障解耦质量。
- **Explicit non-contributions**: 不声称"首次做内容/风格分离"（FedSTAR已做）；不声称"解耦保证"（只是互补正则化）；不声称通信效率优化。

## Proposed Method

### Complexity Budget
- **Frozen/reused**: ResNet-18骨干（参数聚合），BN层本地冻结（FedBN原则）
- **New trainable components**: (1)语义投影头Hsem (2)风格投影头Hsty — 共2个，符合MAX_NEW_TRAINABLE_COMPONENTS=2
- **Tempting additions intentionally excluded**: 对抗训练（不稳定）、生成模型（太重）、Transformer聚合器（过度设计）、L_sty_con风格对比损失（审稿建议精简，消融后决定）

### System Overview

```
Client k:
  x → ResNet-18(BN_local) → h
       ├── Hsem(h) → z_sem (语义特征)
       └── Hsty(h) → z_sty (风格特征)
  
  解耦约束: L_decouple = λ_orth*cos²(z_sem,z_sty) + λ_hsic*HSIC(z_sem,z_sty)
  
  风格增强: 从服务器下发的外部风格s_ext → z_aug = z_sem + λ*MixStyle(z_sty, s_ext)
  
  分类: z_sem和z_aug → 分类器 → L_task(CE)
  
  语义对齐: L_sem_con = InfoNCE(z_sem, P_sem_global)
  
  上传: 模型参数(无BN) + P_sem_k(语义原型) + P_sty_k(风格原型,per-domain)

Server:
  聚合: 骨干+语义头 → FedAvg; 风格头 → 不聚合
  语义原型: P_sem_global = mean(P_sem_k)
  风格仓库: B ← B ∪ {P_sty_k} (余弦去重>0.95)
  按需调度: 对每个客户端k每个类c，计算margin_gap = |P_sem_k^c - P_sem_global^c|
            高gap的(k,c)对获得更多外部风格
  下发: 全局参数 + P_sem_global + 调度后的风格子集{s_ext}
```

### Core Mechanism: Style Asset Sharing in Decoupled Space

**Input**: 骨干特征h ∈ R^d  
**Output**: 语义特征z_sem ∈ R^d_sem, 风格特征z_sty ∈ R^d_sty

**Architecture**:
- Hsem: Linear(d, d_sem) + ReLU + Linear(d_sem, d_sem)
- Hsty: Linear(d, d_sty) + ReLU + Linear(d_sty, d_sty)
- d_sem = d_sty = 128（默认）

**Decoupling Loss**:
```
L_orth = (z_sem · z_sty / (||z_sem|| · ||z_sty||))²
L_HSIC = (1/n²) tr(K_sem H K_sty H)  # Gaussian kernel, bandwidth = median heuristic
L_decouple = λ_orth * L_orth + λ_hsic * L_HSIC
```

**Style Prototype (per-domain)**:
```
P_sty_k = (1/|D_k|) Σ_{x∈D_k} z_sty(x)   # 整个客户端的风格均值，非per-class
```

**Need-Aware Dispatch**:
```
For client k, class c:
  gap_k^c = ||P_sem_k^c - P_sem_global^c||_2
  dispatch_weight_k^c = softmax(gap_k^c / τ_dispatch)
  # 高gap类获得更多外部风格原型
```

**Style Augmentation via MixStyle in Decoupled Space**:
```
s_mixed = α * z_sty + (1-α) * s_ext,  α ~ Beta(0.1, 0.1)
z_aug = z_sem + λ_aug * s_mixed
# 或 z_aug = z_sem * (1 + γ(s_mixed)) + β(s_mixed)  # FiLM variant, 消融决定
```

**Why this is the main novelty**: 所有先前方法要么在纠缠空间做风格操作（FISC/FedCCRL），要么解耦后丢弃风格（FedSeProto/FedDP），要么解耦后风格私有（FedSTAR）。本方法首次在解耦后的纯风格空间做跨域风格共享增强。

### Training Plan

**Total Loss**:
```
L_total = L_task + λ1 * L_decouple + λ2 * L_sem_con
```

注意：相比开题报告去掉了L_sty_con（风格对比损失），简化为3项。消融实验中验证是否需要恢复。

**Training Recipe**:
- 本地epochs: 5 per round
- FL rounds: 100 (Digit-5), 200 (PACS/DomainNet)
- Optimizer: SGD, lr=0.01, momentum=0.9
- 风格仓库warmup: 前10轮不调度（积累足够风格原型）
- HSIC kernel bandwidth: median heuristic, 每轮重新计算

**Inference**:
- 仅使用骨干+语义头+分类器
- 不使用风格头和风格仓库（推理时不需要）
- 分类方式：最近语义原型分类 或 线性分类器（消融决定）

### Failure Modes and Diagnostics

| 失败模式 | 检测方式 | 缓解措施 |
|---------|---------|---------|
| 解耦不充分（z_sem仍含域信息） | 域预测准确率from z_sem应低 | 增大λ_hsic |
| 解耦过度（z_sem丢失类信息） | 类预测准确率from z_sem下降 | 减小λ_decouple |
| 风格注入破坏标签 | 增强样本分类准确率下降 | 减小λ_aug或限制风格混合比例 |
| 风格仓库同质化 | 仓库中风格原型余弦相似度过高 | 加强去重阈值或加入多样性采样 |
| HSIC在小batch下不稳定 | 监控L_HSIC方差 | 增大batch size或用RFF近似 |

### Novelty and Elegance Argument

**最接近工作**：
- FedSTAR：也做内容/风格分离，但风格仅本地→我们共享
- FISC/StyleDDG：也做风格共享，但不解耦→我们先解耦
- FedSeProto：也做语义/域分离，但域信息丢弃→我们保留并资产化

**本方法的独特位置**：在"解耦"和"共享"的2×2矩阵中，占据了唯一空白格：

|  | 不共享风格 | 共享风格 |
|--|-----------|---------|
| **不解耦** | FedBN, FedAvg | FISC, StyleDDG, FedCCRL |
| **解耦** | FedSTAR, FedSeProto, FDSE | **★ 本方法（首次）** |

**为什么不是模块堆叠**：每个组件都为核心claim服务——双头为解耦服务，风格仓库为共享服务，按需调度为高效共享服务。去掉任何一个都会破坏"解耦+共享"的核心范式。

## Claim-Driven Validation Sketch

### Claim 1: 解耦后共享 > 不解耦共享 > 解耦后擦除/私有

- **Minimal experiment**: PACS 4域，对比 Full method vs "仓库+不解耦"(ablation) vs "解耦+不共享"(ablation) vs FedSeProto(解耦+擦除) vs FISC(不解耦+共享)
- **Metric**: 平均准确率 + 域间方差
- **Expected evidence**: Full > 不解耦共享 > 解耦擦除 ≈ 解耦私有

### Claim 2: 双重约束(正交+HSIC)解耦质量优于单一约束

- **Minimal experiment**: PACS上，orth-only vs HSIC-only vs orth+HSIC
- **Metric**: (1)域预测from z_sem的准确率(越低越好) (2)类预测from z_sem的准确率(越高越好) (3)下游分类准确率
- **Expected evidence**: orth+HSIC在解耦质量和下游任务上均优于单一约束

### Claim 3: 按需调度优于随机调度

- **Minimal experiment**: PACS/DomainNet上，need-aware dispatch vs random dispatch vs no dispatch
- **Metric**: 平均准确率 + 困难域(如Sketch/Quickdraw)准确率
- **Expected evidence**: need-aware > random > no dispatch，且困难域提升最大

## Experiment Handoff Inputs
- **Must-prove claims**: (1)解耦+共享优于其他范式 (2)双重约束有效 (3)调度策略有效
- **Must-run ablations**: 解耦ON/OFF × 共享ON/OFF 的2×2矩阵；orth/HSIC/both；random/need-aware/none
- **Critical datasets**: PACS（必须）, DomainNet子集（必须）, Digit-5/Office-Caltech10（支撑）
- **Highest-risk assumptions**: (1)双头投影真的能分离语义和风格 (2)风格原型per-domain足够代表域风格 (3)简单MixStyle在解耦空间有效

## Compute & Timeline Estimate
- **Estimated GPU-hours**: ~200h (PACS主实验+全套消融+诊断，3 seeds)
- **Timeline**: 实现2周 + 实验3周 + 论文撰写2周 = 7周
