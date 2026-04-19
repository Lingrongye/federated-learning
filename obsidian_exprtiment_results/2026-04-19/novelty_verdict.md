# Novelty Check 综合 Verdict — 2026-04-19

## Top-2 候选 novelty 对比

| 维度 | Top-1: FedDSA-ETF + TTA | Top-2: FedDSA-TIES |
|------|------------------------|-------------------|
| Dim 1 最严重重叠 | FedDEAP (55%, 不同赛道) | MPSF-FL (80%, 正面硬撞) |
| Dim 2 | FL+TTA+disentangle PARTIAL 35% | FedSECA sign-election OCCUPIED 65% |
| Dim 3 | Orth+Fixed classifier CLEAR <25% ✅ | Gradient sign FL OCCUPIED |
| 综合 verdict | **GO WITH MODIFICATIONS** | REJECT / MAJOR REDESIGN |
| 风险等级 | 中 (避 FedDEAP 赛道即可) | 高 (需重定义核心贡献) |

## 决定:采纳 Top-1 "FedDSA-ETA" (ETF + TTA)

### 6 条差异化 (来自 novelty-check agent 建议)

1. **Kill 旧 headline**:不主打"FL+ETF+decouple 第一个"(被 FedDEAP 抢了)
2. **强化 moat**:ResNet-18 from-scratch,不碰 CLIP/ViT (FedDEAP 的地盘)
3. **TTA 为主贡献**:SATA-gate + T3A-proto 的完整组合无人占坑
4. **单侧 ETF**:只在 z_sem 用 ETF,z_sty 保持 Gaussian μ,σ bank (避免双 ETF 与 FedDEAP 撞)
5. **主表必备对比**:FedDEAP (复现到 ResNet) / FedDG-MoE / FedETF / FedSTAR / FediOS / SATA (centralized)
6. **Related Work 必引**:StyleDDG, FedCA, FISC/PARDON (share-but-not-decouple 基线)

## 新 Framing

**标题候选**:
- "Orthogonal-Decoupled Federated Learning with Neural-Collapse Inspired ETF Head and Style-Aware Test-Time Adaptation"
- 简称 **FedDSA-ETA** (ETA = ETF + TTA)

**一句话定位**:
> 首个在 feature-skew 跨域联邦学习中,组合 (a) 正交双头解耦 + (b) 固定单纯形 ETF 分类头 (仅 z_sem) + (c) 基于风格仓库的 SATA 可靠性 gate + (d) T3A 在线原型分类的**完全 backprop-free 测试时推理框架**。

**三个独立贡献 (可分别消融)**:
1. **训练端**: 把 sem_classifier 改为 Fixed ETF (与 L_orth 几何互补)
2. **推理端 - 可靠性**: SATA style-exchange gate (复用现有 style_bank + z_sty)
3. **推理端 - 分类**: T3A on-the-fly prototype (复用 z_sem 与 ETF 顶点)

## 下一步

进入 Task 56,用 `/research-refine` skill 做 5 轮 GPT-5.4 xhigh 打磨,目标 READY ≥9.0/10。

Refine 重点审查:
- ETF 几何证明 (与 L_orth 的互补性是否形式化)
- z_sty Gaussian bank vs FedDEAP 双 ETF 的差异足不足
- SATA gate + T3A proto 融合的数学一致性
- ResNet-18 from-scratch 不引入 CLIP 的可行性
- PACS 4-outlier 下 Fixed ETF 是否可能 hurt (风险评估)
