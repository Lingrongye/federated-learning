# Experiment Plan

**Problem**: 跨域FL中特征语义/风格纠缠导致模糊原型，跨域泛化下降  
**Method Thesis**: Decouple → Share → Align: 双头解耦 + 风格资产共享(AdaIN@layer3) + 语义对比对齐  
**Date**: 2026-04-01  
**GPU Budget**: ~250h  
**Seeds**: 3 (所有报告结果)

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|---------------|-----------------------------|---------------|
| **C1**: Decouple+Share > 单独使用任一 | 核心贡献——解耦后共享是新范式 | 2×2消融在PACS上全方法显著优于其他3个变体 | B1, B2 |
| **C2**: 正交+HSIC > 单一约束 | 支撑贡献——互补解耦有效 | orth+HSIC在解耦诊断指标和下游准确率上均优 | B3 |
| **Anti-claim**: 收益仅来自"更多增强" | 必须排除——否则解耦无意义 | "仅增强无解耦"明显弱于"解耦+增强" | B2 |

## Paper Storyline
- **Main paper必须证明**: C1(核心消融) + C2(解耦有效性) + 主表(vs强基线)
- **Appendix可支撑**: need-aware dispatch消融、DINOv2骨干、参数敏感性、通信开销
- **明确Cut**: 隐私攻击实验(讨论即可)、FedDP/FedSeProto复现(代码不公开)

---

## Experiment Blocks

### Block 1: Main Comparison (主表) — MUST-RUN

- **Claim tested**: 整体方法优于现有跨域FL方法
- **Why**: 论文Table 1，审稿人第一眼看这个
- **Datasets**:
  - PACS (4 domains: Photo/Art/Cartoon/Sketch, 7 classes, ~9991 images)
  - DomainNet subset (6 domains, 10 classes subset following FedPLVM protocol)
  - Digit-5 (5 domains: MNIST/SVHN/USPS/SynthDigits/MNIST-M, 10 classes)
  - Office-Caltech10 (4 domains: Amazon/Webcam/DSLR/Caltech, 10 classes)
- **Compared systems**:
  | 方法 | 类型 | 代码来源 |
  |------|------|---------|
  | FedAvg | 基础 | PFLlib |
  | FedProx | 基础 | PFLlib |
  | FedBN | 域适应 | PFLlib |
  | FedProto | 原型基础 | PFLlib |
  | FPL | 原型聚类 | RethinkFL |
  | FedPLVM | 原型多层聚类 | FedPLVM |
  | FDSE | 特征解耦(擦除派) | FDSE_CVPR25 |
  | FISC | 风格共享(不解耦) | 按论文复现 |
  | **Ours** | 解耦+共享 | 自实现 |
- **Metrics**: 各域Top-1准确率 + 平均准确率 + 域间标准差
- **Setup**:
  - Backbone: ResNet-18 pretrained on ImageNet (所有方法相同)
  - BN: 所有方法使用local BN (公平对比)
  - FL rounds: 100(Digit-5), 200(PACS/DomainNet/Office)
  - Local epochs: 5
  - Batch size: 32
  - LR: 0.01, SGD, momentum 0.9
  - Clients: 每个域1个客户端 (PACS: 4 clients, Digit-5: 5 clients)
  - Seeds: 3
- **Success criterion**: PACS和DomainNet上平均准确率超过所有基线，且困难域(Sketch/Quickdraw)提升明显
- **Failure interpretation**: 如果不超过FDSE/FISC，说明"解耦+共享"没有比单独"解耦"或"共享"更好→需要重新审视核心claim
- **Table/figure**: Table 1 (主表), Figure 2 (各域柱状图)
- **Priority**: MUST-RUN

---

### Block 2: Core Ablation — 2×2 Matrix — MUST-RUN

- **Claim tested**: C1 + Anti-claim
- **Why**: 论文核心消融，隔离"解耦"和"共享"各自的贡献
- **Dataset**: PACS (最重要), DomainNet subset (补充)
- **Compared variants**:
  | Variant | Decouple | Share | Description |
  |---------|----------|-------|-------------|
  | (a) Full | ✓ | ✓ | 完整方法 |
  | (b) Decouple-only | ✓ | ✗ | 双头解耦但不共享风格(风格头私有，无仓库) |
  | (c) Share-only | ✗ | ✓ | 无解耦，在原始特征上做AdaIN风格共享 |
  | (d) Neither | ✗ | ✗ | 骨干+分类器+语义原型对齐(无解耦无共享) |
- **Metrics**: 平均准确率 + 各域准确率
- **Success criterion**: Full > Decouple-only > Neither, Full > Share-only > Neither, Full显著优于(b)(c)(d)
- **Failure interpretation**:
  - 如果(c)≈Full → 解耦不重要，核心claim失败
  - 如果(b)≈Full → 共享不重要，风格仓库没意义
  - 如果(c)>(b) → 共享比解耦重要，需调整叙事
- **Table/figure**: Table 2 (消融表)
- **Priority**: MUST-RUN

---

### Block 3: Decoupling Effectiveness — Diagnostics — MUST-RUN

- **Claim tested**: C2 + 解耦质量验证
- **Why**: 审稿人最可能质疑"解耦是否真的有效"
- **Dataset**: PACS
- **Sub-experiments**:

  **(3a) 约束消融**:
  | Variant | L_orth | L_HSIC | 
  |---------|--------|--------|
  | orth-only | ✓ | ✗ |
  | HSIC-only | ✗ | ✓ |
  | orth+HSIC | ✓ | ✓ |
  | no-decouple | ✗ | ✗ |

  **(3b) 解耦诊断** (关键！比t-SNE重要):
  - 训练一个**线性域分类器**在frozen z_sem上 → 域预测准确率应接近随机(~25% for 4 domains)
  - 训练一个**线性类分类器**在frozen z_sem上 → 类预测准确率应高
  - 训练一个**线性域分类器**在frozen z_sty上 → 域预测准确率应高
  - 训练一个**线性类分类器**在frozen z_sty上 → 类预测准确率应接近随机(~14% for 7 classes)

  **(3c) 原型质量**:
  - 计算全局语义原型的类内紧凑度(intra-class avg distance)
  - 计算类间分离度(inter-class avg distance)
  - 与FedProto原型直接对比
  - 可视化: t-SNE of z_sem colored by class (应按类聚) and by domain (不应按域聚)

- **Metrics**: 域/类预测准确率交叉表、原型紧凑度比值、t-SNE
- **Success criterion**: z_sem域预测接近chance level，z_sty类预测接近chance level，orth+HSIC > 单一约束
- **Failure interpretation**: 如果z_sem仍能高准确率预测域 → 解耦不充分，需增强约束
- **Table/figure**: Table 3 (约束消融), Table 4 (诊断交叉表), Figure 3 (t-SNE)
- **Priority**: MUST-RUN

---

### Block 4: Decoupled vs Entangled Augmentation — MUST-RUN

- **Claim tested**: 在解耦空间做风格增强 > 在纠缠空间做
- **Why**: 与FISC/FedCCRL的核心差异化
- **Dataset**: PACS
- **Compared variants**:
  | Variant | Description |
  |---------|-------------|
  | Decoupled AdaIN | 本方法: 在layer3 feature上用解耦后的外部style stats做AdaIN |
  | Entangled AdaIN | 在layer3 feature上用其他客户端的原始(未解耦)feature stats做AdaIN |
  | Entangled MixStyle | 在layer3 feature上用FedCCRL式MixStyle(混合特征统计量) |
  | No augmentation | 仅解耦+对齐，无风格增强 |
- **Metrics**: 准确率 + 原型紧凑度
- **Success criterion**: Decoupled AdaIN > Entangled AdaIN ≈ Entangled MixStyle > No augmentation
- **Failure interpretation**: 如果Entangled AdaIN ≈ Decoupled AdaIN → 解耦对增强无额外价值
- **Table/figure**: Table 5 (增强方式对比)
- **Priority**: MUST-RUN

---

### Block 5: Overhead & Scalability — NICE-TO-HAVE (main paper一段)

- **Claim tested**: 方法开销合理
- **Dataset**: PACS
- **Metrics**:
  - 每轮通信量 (bytes): 与FedAvg/FedProto对比
  - 服务器风格仓库内存 (MB)
  - 单轮训练时间 (seconds): 与FedProto对比
  - 随域数量增加的扩展性
- **Table/figure**: Table 6 (开销表)
- **Priority**: NICE-TO-HAVE (but reviewers will ask)

---

### Block 6: Optional Ablations — APPENDIX

**(6a) Need-aware dispatch vs Random dispatch vs No dispatch**
- Dataset: PACS + DomainNet
- 如果need-aware显著优于random → 升级为main paper

**(6b) Style bank size: K=1 vs K=3 vs K=5 cluster centers per client**
- Dataset: PACS

**(6c) 超参敏感性: λ_orth, λ_hsic, λ2(L_sem_con), τ(温度)**
- Dataset: PACS, 固定其他参扫描每个

**(6d) Frozen DINOv2-S backbone**
- 同样的双头+风格仓库，验证机制迁移到现代骨干

**(6e) Warmup轮数: 0 vs 5 vs 10 vs 20**

**Priority**: NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0: Sanity** | 数据加载正确，单client过拟合，损失下降 | 1 client PACS-Photo, 10 epochs | 损失<0.5, acc>80% | 2h | 低 |
| **M1: Baseline** | 复现FedProto/FedBN/FDSE在PACS上 | 3 baselines × 3 seeds | 数字与论文报告一致(±2%) | 30h | 中——FDSE可能复现困难 |
| **M2: Full Method** | 我们的方法在PACS上跑通 | 1 config × 3 seeds | 准确率>FedProto | 15h | 中——HSIC实现可能有bug |
| **M3: Core Ablation** | 2×2消融 + 约束消融 | 4+3 variants × 3 seeds = 21 runs | Full > 所有variants | 50h | 关键——如果(c)≈Full则需重新评估 |
| **M4: Diagnostics** | 解耦诊断 + 原型质量 | 4 linear probes + compactness | z_sem域预测<30% | 5h | 中——可能解耦不充分 |
| **M5: Main Table** | 全基线在PACS+DomainNet | 8 baselines × 2 datasets × 3 seeds | PACS/DomainNet均超SOTA | 80h | 高——DomainNet耗时 |
| **M6: Polish** | 增强对比 + 开销 + 可选消融 | Block 4 + 5 + 6(选做) | 结果一致支撑story | 60h | 低 |

**总计**: ~242h，在250h预算内

### 关键Stop/Go Gates

```
M0 通过 → M1
M1 基线复现成功 → M2
M2 方法跑通且>FedProto → M3 (如果不超过FedProto，先debug)
M3 消融:
  - Full > Decouple-only AND Full > Share-only → GO (核心claim成立)
  - Share-only ≈ Full → STOP (解耦对共享无额外价值，需重新评估)
  - Decouple-only ≈ Full → STOP (共享无额外价值，需重新评估)
M4 诊断:
  - z_sem域预测<35% → GO
  - z_sem域预测>50% → STOP (解耦不充分，需调整约束)
M5 主表通过 → M6 → 论文撰写
```

---

## Compute and Data Budget

| Item | GPU-hours | Notes |
|------|-----------|-------|
| M0 Sanity | 2h | 1× quick |
| M1 Baselines | 30h | 3 methods × 2 datasets × 3 seeds × ~1.5h |
| M2 Full method | 15h | 1 config × 2 datasets × 3 seeds × ~2.5h |
| M3 Ablations | 50h | 7 variants × 2 datasets × 3 seeds × ~1.2h |
| M4 Diagnostics | 5h | Linear probes are fast |
| M5 Main table | 80h | 8 baselines × 2 datasets × 3 seeds |
| M6 Polish | 60h | Block 4 + 5 + selected 6 |
| **Total** | **~242h** | Within 250h budget |

**Data**: PACS/DomainNet/Digit-5/Office-Caltech10均为公开数据集，无需标注  
**最大瓶颈**: M5 (80h) — DomainNet训练慢。可先只跑PACS，DomainNet用subset加速

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| HSIC实现不稳定(小batch) | 中 | 高 | 用RFF近似或增大batch size到64 |
| 解耦不充分(z_sem仍含域信息) | 中 | 高 | 先跑M4诊断，如有问题增大λ_hsic或加GRL辅助 |
| FDSE基线复现困难 | 中 | 中 | 使用官方代码FDSE_CVPR25/，严格按论文设置 |
| FISC无官方代码 | 高 | 中 | 按论文描述复现核心机制，或退而求其次用FedCCRL |
| DomainNet训练时间超预算 | 中 | 中 | 使用10-class subset，减少FL rounds到100 |
| 增强破坏标签(off-manifold) | 低 | 中 | 监控z_sem_aug分类准确率，必要时限制α范围 |

---

## Hyperparameter Defaults

| Param | Default | Search Range (if tuning) |
|-------|---------|-------------------------|
| λ_orth | 1.0 | {0.1, 0.5, 1.0, 2.0} |
| λ_hsic | 0.1 | {0.01, 0.05, 0.1, 0.5} |
| λ2 (L_sem_con) | 1.0 | {0.1, 0.5, 1.0, 2.0} |
| τ (temperature) | 0.1 | {0.05, 0.1, 0.2} |
| α distribution | Beta(0.1, 0.1) | Beta(0.1,0.1) / Beta(0.3,0.3) / Beta(1,1) |
| λ_aug (augmentation weight) | 1.0 | implicit via L_CE(z_sem_aug) |
| d_sem, d_sty | 128 | {64, 128, 256} |
| Style bank K (clusters) | 3 | {1, 3, 5} |
| Style dispatch M (per client) | 5 | {3, 5, 10} |
| Warmup rounds | 10 | {0, 5, 10, 20} |

---

## Final Checklist
- [x] Main paper tables covered (Table 1-5)
- [x] Novelty isolated (Block 2: 2×2 ablation)
- [x] Simplicity defended (去掉了need-aware dispatch，消融决定是否恢复)
- [x] Anti-claim addressed (Block 2中"Share-only"排除"仅增强"解释)
- [x] Diagnostics prove decoupling (Block 3)
- [x] Nice-to-have separated from must-run
- [x] Budget within 250h
