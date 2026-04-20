# Landscape Survey — Federated Domain Generalization × 风格/语义解耦

**作者**: FedDSA-SGPA 调研备忘
**时间**: 2026-04-19
**目的**: 为 "FedDSA-SGPA 在 PACS 失败 / Linear+whitening 磨掉风格 / 统计解耦不够" 的三连痛点寻找相关工作与差异化空间
**输入背景**:
- 我们方案: 双头解耦 (z_sem + z_sty) + 服务器 pooled whitening 广播 + Fixed ETF classifier
- 诊断: Office Linear+whitening +6.20pp, PACS -1.49pp; PACS z_sty_norm R10=3.12 → R200=0.15 (磨掉 95%)
- 核心怀疑: 统计约束 (L_orth cos²=0 + HSIC) 没告诉模型"什么是风格", PACS 的风格本身是类别判别信号

---

## Section 1 — 联邦域泛化 / 跨域联邦学习 (已拥挤区)

### 1.1 FedSDAF — 源域感知框架 (arXiv 2025)
- **URL**: arxiv.org/abs/2505.02515, github.com/pizzareapers/FedSDAF
- **方法**: 双适配器 (Domain-Aware Adapter 本地保留 + Domain-Invariant Adapter 共享) + 双向知识蒸馏。核心发现: 完整源域学到的特征泛化优于直接学目标域
- **vs FedDSA-SGPA**: 都做 "保留域特定 + 共享域无关", 但 FedSDAF 用**参数级适配器**切分 (parameter-efficient), 我们用**特征级双头**. 它没做 whitening, 没处理 PACS/Office 风格语义性差异

### 1.2 Fed-DIP — 隐式解耦 + 上下文 Prompting (OpenReview 2025)
- **URL**: openreview.net/forum?id=VbmV3rs284
- **方法**: Implicit Decoupling Distillation (逐 patch logit 蒸馏做细粒度分离) + CAPE 动态生成 text prompt
- **vs FedDSA-SGPA**: 它靠 logit 空间隐式解耦, 没有几何约束; 依赖预训练 VLM; SOTA on PACS/VLCS/OfficeHome/DomainNet. **和我们互补**: 它可能解决了我们担心的"风格即信号"问题, 因为不强制擦除风格

### 1.3 FedOMG — 服务器端梯度匹配 (arXiv 2501.14653, 2025)
- **方法**: 服务器通过本地梯度内积最大化寻找不变梯度方向, data-free
- **vs 我们**: 它做梯度空间对齐, 我们做特征空间解耦. 可作为正交加法

### 1.4 FedAlign — MixStyle + 多级对齐 (arXiv 2501.15486, 2025)
- **方法**: MixStyle 在 channel-wise μ/σ 间插值做跨客户端增强 + 多级 alignment 目标
- **vs 我们**: 它走"共享但不解耦"路线 (FISC/PARDON 同派), 风格是增强资产但不做显式 z_sty 分离

### 1.5 FedDG-MoE — 测试时 MoE 融合 (CVPRW 2025)
- **URL**: openaccess.thecvf.com/content/CVPR2025W/FedVision/papers/Radwan_FedDG-MoE_...
- **方法**: 冻结 ViT 骨干 + Kronecker MoE adapter, 测试时融合
- **vs 我们**: 依赖大预训练 ViT, 我们是 ResNet-18 从头训, 不直接竞争

---

## Section 2 — 显式监督风格解耦 / Style-Content Disentanglement

### 2.1 SCFlow — 流模型隐式解耦 (ICCV 2025) ★★★
- **URL**: arxiv.org/abs/2508.03402, github.com/CompVis/SCFlow
- **方法**: Flow matching 学可逆的 style-content 合并映射, **只训练合并任务**即自动获得分离能力. 关键洞见: 合并是 well-defined 任务, 反向就能分离
- **vs 我们**: 我们的 L_orth+HSIC 是**对称约束**, 不知道方向; SCFlow 用**任务驱动**定义解耦, 这正是我们担心的"统计约束没有语义指导"的替代思路. 计算太重 (Flow matching) 不能直接移植到 FL, 但思想启发: **用一个"可逆重构"loss 代替纯统计约束**

### 2.2 Invariant Representation via Decoupling Style and Spurious Features (arXiv 2312.06226)
- **方法**: 对抗网络 + 多环境优化, 从图像中分离 style 和 spurious features
- **vs 我们**: 用对抗方向, 我们用几何方向. 它强调"风格也是 spurious 的一种"但要分开处理

### 2.3 Style Mixup Enhanced Disentanglement (Medical Image Analysis 2024)
- **URL**: sciencedirect.com/science/article/abs/pii/S1361841524003657
- **方法**: 特征空间中对 style factor 做凸组合生成混合域 + pixel-wise consistency
- **vs 我们**: 医学图像, 有两个不同 modality 做监督信号 (CT vs MRI label 明确), 我们 PACS 只有 domain label 可用

### 2.4 Deep Feature Disentanglement for Supervised Contrastive (Cognitive Computation 2025)
- **方法**: 用对比学习把 deep feature 分成 common (class-defining) + style (within-class variation)
- **vs 我们**: 思路类似但**基于 class label 监督**, 这是我们可以直接借鉴的 — PACS 里同一类别跨域的 z_sem 应该一致, z_sty 应该不同, 这比 cos² 约束信息丰富得多

### 2.5 Content-Style Disentangled Representation (OpenReview 2025)
- **方法**: 构建 image-text paired dataset, 用文本作风格监督
- **vs 我们**: 需要多模态对齐; 但思想可借鉴: **用 domain label 作 z_sty 的分类目标** (类似 DANN 但反向用)

---

## Section 3 — 风格即信号 / "Domain-Invariant" 何时失败

### 3.1 Style Blind Domain Generalized Semantic Segmentation (arXiv 2403.06122)
- **方法**: Covariance Alignment + 语义一致性对比, 明确指出**特征归一化会同时移除语义内容**因为 content 与 style 纠缠
- **核心观点**: "feature normalization tends to confuse semantic features in the process of constraining feature space distribution" → **这直接印证我们 PACS 失败的原因**
- **vs 我们**: 他们用 covariance alignment 替代直接 whitening; 可考虑引入到我们的服务器聚合

### 3.2 Correlated Style Uncertainty for DG (PMC 2024)
- **URL**: pmc.ncbi.nlm.nih.gov/articles/PMC11230655
- **方法**: style 不是独立同分布噪声, 而是有相关结构的 uncertainty, 用相关协方差建模
- **vs 我们**: pooled whitening 假设风格独立, 他们说风格有 correlation. 对 PACS (art/sketch/cartoon 风格差异大但类别一致) 这个假设更合理

### 3.3 Exploring Semantic Consistency and Style Diversity for DG Segmentation (arXiv 2412.12050, 2024)
- **方法**: 语义一致性 + 风格多样化**并行**, 不做解耦但保留风格多样性
- **关键 insight**: 不是所有场景都要去除风格 — 当 style shift 与 label 无关时才擦除, 与 label 相关时要保留

### 3.4 Suppressing Style-Sensitive Features via Randomly Erasing (Springer 2021)
- **URL**: springer/chapter/10.1007/978-3-030-88013-2_25
- **方法**: 不是"擦除所有风格", 而是通过随机擦除**style-sensitive channels**, 对 content-sensitive channels 保留

### 3.5 Frequency Decomposition to Tap Single Domain Potential (arXiv 2304.07261)
- **观点**: 低频=内容, 高频=风格; 但 sketch/cartoon 的"风格"其实大部分在**低频结构**里 (线条稀疏, 色彩单薄), 传统 freq-based DG 方法会误伤

---

## Section 4 — 互信息最小化 / 信息瓶颈解耦

### 4.1 FedIB — 重加权信息瓶颈 FL (Inf. Sciences 2024)
- **URL**: sciencedirect.com/science/article/abs/pii/S0020025524007394
- **方法**: 跨客户端用 IB 提取域不变表示, 消除伪不变特征 (pseudo-invariant)
- **vs 我们**: 纯粹压缩域信息 (擦除派, 和 FedSeProto/FedDP 同派). 我们要保留 z_sty

### 4.2 FedSeProto / FedDP (ECAI 2024 / TMC 2025)
- **方法**: MI 分离后**丢弃域特征** → 全局域无关原型
- **vs 我们**: 态度完全相反 — 他们丢弃域信息, 我们资产化

### 4.3 A Generalized Information Bottleneck Theory (arXiv 2509.26327, 2025)
- **方法**: 把 IB 原理通过 "synergy" 重新表述 — 只有联合处理多特征才能得到的信息
- **vs 我们**: 理论工具. 可以用来证明 "z_sem 和 z_sty 联合能用于分类, 但单独 z_sem 不行" 是 PACS 场景的特征

### 4.4 Scale-Invariant Information Bottleneck for DG (ESwA 2025)
- **方法**: 自适应捕捉全局不变信息, 同时保留 finer-grained details
- **vs 我们**: 类似 "selective whitening" 思想, 不是全局擦除而是自适应

### 4.5 IB-D2GAT — 动态图 IB (TPAMI 2025)
- **方法**: 时空分布偏移下的 variant/invariant 模式
- **vs 我们**: 领域不同 (时空图) 但思路可借鉴 — 动态判断哪些维度是 invariant, 哪些是 variant

---

## Section 5 — 对抗解耦 in FL / DANN 联邦版本

### 5.1 FedPall — 原型对抗 + 协作 (ICCV 2025) ★★★
- **URL**: arxiv.org/abs/2507.04781, openaccess.thecvf.com/content/ICCV2025/...
- **方法**: 服务器 Amplifier (MLP) 放大异构信息, 客户端用 KL 减少. 原型对比 + 原型混合特征 + Bernoulli mask 隐私保护
- **vs 我们**: **最接近的对抗派 FL 工作**. 区别: 他们在混合特征空间做对抗 (不解耦), 我们先解耦后共享. 能否借鉴: **服务器 Amplifier 思想可以用来监督 z_sty** — 让 Amplifier 只能从 z_sty 识别 domain, 从 z_sem 无法识别 → 这是 DANN 的联邦版, 且是**带约束的 DANN** (只反向 z_sem 不反向 z_sty)

### 5.2 ADCOL — Adversarial Collaborative Learning (ICML 2023)
- **URL**: proceedings.mlr.press/v202/li23j/
- **方法**: 客户端学共同表示分布 vs 服务器学区分 party ID. 最早的 FL + DANN 原型
- **vs 我们**: 它擦除所有客户端差异, 我们要保留 z_sty 的差异

### 5.3 Federated Adversarial Domain Adaptation (arXiv 1911.02054)
- **方法**: 最早把 DANN 搬到 FL; representation disentanglement + dynamic attention mask on gradients
- **vs 我们**: 老 baseline, 2019 年的; 但"基于梯度 attention 的 disentanglement" 思路对我们有用 — **可以学 gradient-level 的 z_sem vs z_sty 分离**, 而不是 feature-level

### 5.4 FedDAG — 联邦对抗生成医学影像 DG (arXiv 2501.13967, 2025)
- **方法**: 对抗生成 + 联邦, 专门医学影像
- **vs 我们**: 生成式路线, 依赖 GAN 稳定性

### 5.5 Federated Unsupervised DG via Global-Local Alignment (arXiv 2405.16304, 2024)
- **方法**: 梯度 global-local 对齐, 不用 label
- **vs 我们**: 标签模式不同 (unsupervised)

---

## Section 6 — DANN + FL 组合 / 梯度反转联邦实现

### 6.1 FedDSPG — 域特定软 Prompt 生成 (arXiv 2509.20807, 2025)
- **方法**: 对每个域生成 soft prompt, 通过 generator-discriminator adversarial game 学到 diverse DSP
- **vs 我们**: 用 adversarial 训练**生成风格**, 这和我们"风格仓库"概念相似但他们是可学 prompt, 我们是采样 μ/σ

### 6.2 MCGDM — Multi-source Collaborative Gradient Discrepancy (AAAI 2024)
- **URL**: github.com/weiyikang/FedGM
- **方法**: 域内 + 域间梯度匹配, 最小化梯度差异
- **vs 我们**: 梯度级对齐, 我们是特征级. 可作为辅助损失

### 6.3 FedCCRL — Cross-Client Representation Learning (arXiv 2410.11267, 2024)
- **方法**: MixStyle 跨客户端 + AugMix + supervised contrastive + JS divergence
- **vs 我们**: 混合空间 MixStyle (不解耦), 和 FISC 同派

### 6.4 StyleDDG — 去中心化风格共享 (arXiv 2504.06235, 2025)
- **URL**: arxiv.org/abs/2504.06235
- **方法**: P2P 网络风格共享 + 收敛性证明 O(1/√K). 第一个有收敛保证的 style-based decentralized DG
- **vs 我们**: 去中心化架构; 但**没解耦**, 直接共享 batch style 统计量

### 6.5 PARDON / FISC — 插值风格迁移 (ICDCS 2025)
- **URL**: github.com/judydnguyen/pardon-feddg
- **方法**: 客户端 FINCH 聚类 μ/σ → 服务器 FINCH 聚类取中位数 → AdaIN 迁移到客户端
- **vs 我们**: 已在 CLAUDE.md 详细分析. 不解耦, 图像空间 AdaIN

### 6.6 RobustNet — Instance Selective Whitening (CVPR 2021)
- **方法**: 选择性白化只擦除 style-encoded covariance, 保留 content-encoded. 显式监督: 用 photometric transform 对比区分哪些 covariance 维度是 style
- **vs 我们**: **直接解决我们 PACS 的痛点** — 不是全擦风格, 是**选择性擦除**. 我们可以直接借鉴: 服务器不是广播 Σ_inv_sqrt, 而是 per-channel 加权广播, style-sensitive 通道强 whitening, content-sensitive 通道弱/不 whitening

---

## Gap Analysis — 我们能打哪

### 已拥挤 (别做同质)
| 方向 | 代表 | 状态 |
|------|------|------|
| 共享但不解耦的风格迁移 | FISC/PARDON, StyleDDG, FedCCRL, FedAlign, CCST | **7+ 篇同类**, 做到饱和 |
| 擦除派 (丢弃域信息) | FDSE, FedSeProto, FedDP, FedIB, FediOS | **6+ 篇**, 趋势饱和 |
| Prompt-based FedDG (依赖 VLM) | FedDSPG, PLAN, Fed-DIP, FedAPT | **5+ 篇新 (2024-2025)**, 预训练大模型流 |
| 纯几何正交解耦 (cos²) | FediOS, FedSeProto, **我们原方案** | 少有 3 篇, 但面临"统计约束不够"的共同质疑 |

### 有空白 (可切入)
1. **Selective/Conditional Whitening in FL**: RobustNet 2021 证明选择性白化 > 全局白化, 但**FL 场景完全没人做**. 我们可以做 "风格敏感度自适应 whitening"
2. **DANN 监督 + 几何解耦**: 现有对抗 FL (FedPall/ADCOL) 全擦除域, 没人做"带约束的 DANN" (让 z_sem 不识别 domain, 但 z_sty 必须识别 domain). 这正是 Deep Feature Disentanglement for SCL (Cog. Comp. 2025) 的思路但没搬到 FL
3. **数据集自适应白化策略**: 没人正面讨论 "Office 风格弱 → whitening 有效, PACS 风格强 → whitening 磨掉语义" 的边界. 我们的诊断实验是原生素材, 可以做 "Style Strength Detector + 自适应 whitening 策略"
4. **Flow-based / 可逆解耦 in FL**: SCFlow 是 ICCV 2025 但没 FL 版本. 但太重, 暂不推荐
5. **统计约束 + 语义监督混合**: 现有解耦 90% 是单一路线 (纯统计 / 纯对抗 / 纯重构). 我们可以做 cos²+HSIC+DANN-guided + class-invariant consistency 的**混合 loss**

### 推荐打法 (按可行性)
**Option A (主推, 低风险): "Style-Aware Selective Whitening"**
- 把我们的 pooled whitening 换成 RobustNet 风格的 selective whitening
- 用 class-level covariance vs 全局 covariance 识别哪些通道是 style-sensitive
- PACS/Office 同时 work, 用一个机制解决两个数据集

**Option B (激进, 高 novelty): "Constrained DANN + 双头"**
- z_sem 接 gradient reversal 到 domain classifier (DANN 本职)
- z_sty 接 **正向** domain classifier (监督 z_sty 必须能识别 domain)
- 这是对 Deep Feature Disentanglement (Cog. Comp. 2025) 的 FL 扩展, 也是对 FedPall Amplifier 思想的反转应用
- 直接回答审稿人"你的解耦监督信号在哪"

**Option C (理论补丁): 引入"风格强度自适应"**
- 对每个数据集训练一个 Style Strength Detector (z_sty 的 signal-to-noise ratio)
- Office (低强度) → 强 whitening, PACS (高强度) → 弱 whitening
- 解决 "一套方法应两个数据集" 问题

### 我们的"首次"claim 仍成立
在 "解耦 + 风格资产化 + FL" 这个 2×2 交叉点 (CLAUDE.md §12 矩阵), **FedDSA-SGPA 仍是唯一代表**. FedSDAF/Fed-DIP 是 adapter/prompt 路线, Fed-Pall 是对抗不解耦, StyleDDG/FISC 是共享不解耦. 2024-2026 两年内**没有新工作抢占这个象限**, 这是好消息.

但**需要解耦的语义监督升级** — 建议 Option A (selective whitening) + 部分 Option B (z_sty 的正向 domain 监督), 至少**回应"统计约束无法区分风格/语义" 的根本质疑**.

---

## 源码汇总 (可复现 baselines)

| 论文 | 代码 | 用途 |
|------|------|------|
| FedSDAF | github.com/pizzareapers/FedSDAF | dual-adapter baseline |
| FedPall | arxiv.org/abs/2507.04781 | adversarial FL baseline |
| FediOS | arxiv.org/abs/2311.18559 | orthogonal subspace 对比 |
| PARDON/FISC | github.com/judydnguyen/pardon-feddg | 共享不解耦对比 |
| FedCCRL | arxiv.org/html/2410.11267 | MixStyle FL |
| CCST | github.com/JeremyCJM/CCST | AdaIN FL 老 baseline (WACV 2023) |
| RobustNet (ISW) | 提供选择性白化思路 | 非 FL, 但我们可直接借鉴 |
| SCFlow | github.com/CompVis/SCFlow | 流模型解耦思想 |
| FedGM (MCGDM) | github.com/weiyikang/FedGM | 梯度对齐 baseline |
| MCGDM | AAAI 2024 | 梯度对齐 |

---

## Sources (参考链接)

- [FedSDAF arXiv 2505.02515](https://arxiv.org/abs/2505.02515)
- [FedPall arXiv 2507.04781](https://arxiv.org/abs/2507.04781)
- [Fed-DIP OpenReview](https://openreview.net/forum?id=VbmV3rs284)
- [FedOMG arXiv 2501.14653](https://arxiv.org/abs/2501.14653)
- [FedAlign arXiv 2501.15486](https://arxiv.org/html/2501.15486)
- [FedDG-MoE CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/FedVision/papers/Radwan_FedDG-MoE_Test-Time_Mixture-of-Experts_Fusion_for_Federated_Domain_Generalization_CVPRW_2025_paper.pdf)
- [SCFlow ICCV 2025](https://arxiv.org/abs/2508.03402)
- [Invariant Decoupling Style & Spurious](https://arxiv.org/html/2312.06226)
- [Style Mixup Disentanglement MIA 2024](https://www.sciencedirect.com/science/article/abs/pii/S1361841524003657)
- [Style Blind DG Segmentation arXiv 2403.06122](https://arxiv.org/html/2403.06122v1)
- [Correlated Style Uncertainty PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230655/)
- [Semantic Consistency & Style Diversity arXiv 2412.12050](https://arxiv.org/html/2412.12050v1)
- [FedIB Inf. Sciences](https://www.sciencedirect.com/science/article/abs/pii/S0020025524007394)
- [Generalized IB Theory arXiv 2509.26327](https://arxiv.org/abs/2509.26327)
- [FedPall ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_FedPall_Prototype-based_Adversarial_and_Collaborative_Learning_for_Federated_Learning_with_ICCV_2025_paper.pdf)
- [ADCOL ICML 2023](https://proceedings.mlr.press/v202/li23j/li23j.pdf)
- [Federated Adversarial Domain Adaptation](https://arxiv.org/html/1911.02054)
- [FedDAG arXiv 2501.13967](https://arxiv.org/html/2501.13967v1)
- [FedDSPG arXiv 2509.20807](https://arxiv.org/abs/2509.20807)
- [MCGDM AAAI 2024](https://arxiv.org/html/2401.10272v1)
- [FedCCRL arXiv 2410.11267](https://arxiv.org/html/2410.11267v1)
- [StyleDDG arXiv 2504.06235](https://arxiv.org/abs/2504.06235)
- [RobustNet ISW CVPR 2021](https://ar5iv.labs.arxiv.org/html/2103.15597)
- [PARDON-FedDG github](https://github.com/judydnguyen/pardon-feddg)
- [FediOS arXiv 2311.18559](https://arxiv.org/abs/2311.18559)
- [Switchable Whitening ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Pan_Switchable_Whitening_for_Deep_Representation_Learning_ICCV_2019_paper.pdf)
- [Federated DG Survey TUWien 2025](https://dsg.tuwien.ac.at/team/sd/papers/Journal_paper_2025_S_Dustdar_Federated.pdf)
- [CCST WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Chen_Federated_Domain_Generalization_for_Image_Recognition_via_Cross-Client_Style_Transfer_WACV_2023_paper.pdf)
- [Deep Feature Disentanglement SCL Cog. Comp. 2025](https://link.springer.com/article/10.1007/s12559-025-10430-4)
