# 文献综述: Purify-then-Share 3-Stage Pipeline 调研

> 调研日期: 2026-04-20
> 目的: 为 FedDSA-SGPA 的 Purify (MLP adversary) + Prototype Alignment (drift-penalized InfoNCE) + pure z_sty AdaIN 3-stage 方案查找 prior, 评估 novelty
> 方法: 限定 2023-2026 为主, 保留必要的经典奠基工作 (Moyer 2018, HEX 2019)

---

## 主题 1: Purify → Share 完整 pipeline 的 FL DG 方法

我们要找的是 "先严格解耦,再跨客户端共享风格" 的完整 stage-by-stage 方法。结论: **没找到直接对应的 prior**,现有工作都是单阶段或另一范式。

| # | 论文 | 年份/venue | 核心一句话 | 和我们差异 | 代码 |
|---|------|-----------|-----------|-----------|------|
| 1 | **FISC/PARDON**: Federated DG via Interpolative Style Transfer and Contrastive Learning | ICDCS 2025 | FINCH 提取客户端风格(μ,σ)→服务器再聚类→中位数插值风格→AdaIN 迁移+三元组对比 | **不解耦**,在混合特征/图像空间直接操作,风格向量带 class 信息 | https://github.com/judydnguyen/pardon-feddg |
| 2 | **Fed-DIP**: Federated DG by Synergizing Implicit Disentanglement and Context-Aware Prompting | OpenReview 2025 | MIDD (Multi-scale Implicit Decoupling Distillation) + CAPE prompt encoder,通过 logit 比较实现隐式解耦 | 隐式解耦 (无显式解耦损失),基于 prompt 不是特征空间风格共享 | 未开源 |
| 3 | **DiPrompT**: Disentangled Prompt Tuning for Multiple Latent DG in FL | CVPR 2024 | G-Prompt (global 共享) + D-Prompt (domain specific) 解耦, 动态 query 自动匹配 domain label | **不共享** domain prompt,用 prompt 不是风格统计量 | https://arxiv.org/abs/2403.08506 |
| 4 | **FedSDAF**: Source Domain-Aware Federated DG | arXiv 2025 | 双 adapter: Domain-Aware (本地) + Domain-Invariant (共享) + 双向知识蒸馏 BKD | **不共享** Domain-Aware adapter,用 adapter 不是风格特征 | https://github.com/pizzareapers/FedSDAF |
| 5 | **Multi-Source Collaborative Style Augmentation (MCSAD)** for Federated DG | arXiv 2025 | 多源风格增强扩展风格空间 + 跨域特征对齐 + 类别关系集成蒸馏 | **不解耦**, 风格增强在原始特征空间直接操作 | 未开源 |
| 6 | **StableFDG**: Style and Attention Based Learning for Federated DG | NeurIPS 2023 | Style sharing + shifting + exploration; attention-based feature highlighter | **不解耦**,风格共享在原始 CNN 特征层次 | 未见 code |
| 7 | **FediOS**: Decoupling Orthogonal Subspaces for Personalization | arXiv 2023 (ML 2025 接收) | 双 extractor (generic + personalized) + fixed orthogonal projection + shared head | 解耦 generic/personalized 不是 semantic/style;**不共享** personalized,没有 style swap | 见 Springer ML 2025 |

**小结**: FISC/PARDON 最接近 Share 阶段,但完全跳过 Purify;Fed-DIP/DiPrompT 做了解耦但不共享 domain prompt;FedSDAF 做了决解耦但域感知 adapter 完全本地保留。**没有"先 MLP adversarial 真解耦 → 再跨客户端 feature-space 风格共享"的 3-stage 组合**。

---

## 主题 2: 对抗纯化风格特征 (class-exclusion from style)

验证: adversarial 到底能否真解耦 (我们 probe 96% 显示双头假解耦)。

| # | 论文 | 年份/venue | 核心一句话 | 和我们差异 | 代码 |
|---|------|-----------|-----------|-----------|------|
| 1 | **Invariant Representations without Adversarial Training** (Moyer) | NeurIPS 2018 | **证明对抗训练有时 counter-productive**,用 MI upper-bound 单目标代替 iterative minimax | 我们正好要加 adversarial,本文反其道;可借鉴 MI upper bound 作辅助 | https://github.com/dcmoyer/inv-rep |
| 2 | **HEX**: Learning Robust Representations by Projecting Superficial Statistics Out | ICLR 2019 (oral) | NGLCM 提取纹理 → 投影到与 NGLCM representation 正交的子空间 | 不用对抗,用 orthogonal projection; 我们可补充这种几何约束 | https://github.com/HaohanWang/HEX |
| 3 | **Learning Not to Learn (LNTL)**: Training DNN with Biased Data | CVPR 2019 | min-max adversarial + MI minimization 两阶段移除 bias 信息 | 我们的 class-adv-on-sty 就是这个结构;LNTL 是经典参考 | https://github.com/feidfoe/learning-not-to-learn |
| 4 | **CLUB**: Contrastive Log-ratio Upper Bound of Mutual Information | ICML 2020 | 提供 MI **upper bound** (可最小化),比 MINE 更适合解耦 | 我们现在用 HSIC,可替换为 CLUB 做更强非线性独立性约束 | https://github.com/Linear95/CLUB |
| 5 | **FedSaaS**: Class-Consistency Federated Semantic Segmentation via Global Prototype Supervision and Local Adversarial Harmonization | IJCAI 2025 | client 端对抗协调 global/local branch + 多层对比 | 对抗在 segmentation 场景的 class-consistency;启发: 对抗可以用在 global/local **分支协调** 而非 purify | https://www.ijcai.org/proceedings/2025/770 |
| 6 | **Invariant Representations through Adversarial Forgetting** | AAAI 2020 | 经典 Moyer 后续,引入 "forgetting mechanism" 改进 adversarial | 给出 adversarial 的另一种稳定化方式;我们可借鉴 forgetting gate | https://arxiv.org/abs/1911.04060 |
| 7 | **FedSeProto**: Learning Semantic Prototype in Federated Learning | ECAI 2024 | MI 最小化分离 semantic/domain 特征 + KD 蒸馏 + 语义原型对齐 | 用 MI (非 adversarial) 做 purify,**丢弃 domain 特征** (我们要保留) | 未见 code |

**关键发现**:
- Moyer 2018 明确警告 adversarial 有时 counter-productive,建议 MI 单目标
- 但 LNTL/LearnAdvForgetting 证明有 stabilization 机制的对抗是可行的
- FedSeProto 和我们方向相反 (丢 vs 保留 domain),但验证 "MI 最小化可以真解耦" 的可行性
- 我们现在只有 cos² + HSIC;可以补 **CLUB MI upper bound** 或 **HEX orthogonal projection** 作为 adversarial 的辅助 / 替代

---

## 主题 3: FL 场景下的 Prototype Drift Constraint

InfoNCE 崩盘本质是 client-specific 原型漂移过度。文献已有哪些 drift 控制?

| # | 论文 | 年份/venue | 核心一句话 | 和我们差异 | 代码 |
|---|------|-----------|-----------|-----------|------|
| 1 | **FedProto**: Federated Prototype Learning across Heterogeneous Clients | AAAI 2022 | 原型代替参数通信,本地 CE + λ·MSE(local_feat, global_proto) | **MSE 锚点**就是 drift penalty 的最早形式; FPL/FedDSA 都继承 | https://github.com/yuetan031/FedProto |
| 2 | **FPL (RethinkFL)**: Rethinking Federated Learning with Domain Shift: A Prototype View | CVPR 2023 | 构建 cluster prototypes + unbiased proto,**consistency reg 对齐 unbiased proto** | FPL 就用了 MSE 锚点防漂移 (我们 mode4 借鉴),tau=0.02 | https://github.com/WenkeHuang/RethinkFL |
| 3 | **FedPLVM**: Taming Cross-Domain Representation Variance in FPL | NeurIPS 2024 | dual-level FINCH + α-sparsity (cos^0.25) + correction MSE | α-sparsity 就是弱化正例梯度防崩;tau=0.07, α=0.25 | https://github.com/jcui12345/FedPLVM |
| 4 | **DCFL**: Decoupled Contrastive Learning for FL | arXiv 2025 | **理论证明**标准对比损失 O(M^{-1/2}) 误差在 FL 小 batch 下不可忽略,分解为 alignment + uniformity 独立调控 | 给出理论依据我们为什么 InfoNCE 崩;建议 λ_a=0.9, λ_u=0.1 | https://arxiv.org/abs/2508.04005 |
| 5 | **FedDPA**: Dynamic Prototypical Alignment for FL with Non-IID | Electronics (MDPI) 2025 | **adaptive regularization**,根据 client-specific 数据异质程度调惩罚力度 | drift penalty 动态化;我们现在是固定 λ_sem | 未开源 |
| 6 | **FedAli**: Personalized FL with Aligned Prototypes through Optimal Transport | arXiv 2024 | ALP layer + OT assignment; 可无标签预训练 prototype | 用 OT 做 soft matching, 比 MSE 硬锚点更灵活 | https://github.com/getalp/FedAli |
| 7 | **Federated Optimization with Doubly Regularized Drift Correction** | arXiv 2024 | 理论性 drift correction,双重正则提供收敛保证 | 理论参考;没有 prototype 层面的 drift constraint | https://arxiv.org/abs/2404.08447 |

**关键发现**:
- FedProto 的 MSE 锚点是最简单的 drift penalty (tuned λ)
- FPL + FedPLVM 通过 **unbiased proto / α-sparsity** 做更精细的 drift 控制
- DCFL 从理论角度证明我们 InfoNCE 失败不是偶然,而是标准对比损失在 FL 小样本下的根本缺陷;建议 alignment λ=0.9, uniformity λ=0.1
- FedAli 的 **OT assignment** 是比 MSE 硬锚点更好的选择 (soft matching)

---

## 主题 4: Feature-space AdaIN / Style Swap in FL

我们 Stage 3 要在 pure z_sty 上做 feature-level AdaIN。文献里谁做过?

| # | 论文 | 年份/venue | 核心一句话 | 和我们差异 | 代码 |
|---|------|-----------|-----------|-----------|------|
| 1 | **CCST**: Federated DG via Cross-Client Style Transfer | WACV 2023 | VGG 风格提取 + **图像空间 AdaIN**,传输 style moments (μ,σ) | **图像空间**非 feature 空间,需要 VGG decoder;**不解耦** | https://github.com/JeremyCJM/CCST |
| 2 | **FedFA**: Federated Feature Augmentation | ICLR 2023 | 每 client 用 **(μ,σ) 统计量做 feature-level 高斯增强**,方差用全域统计量校准 | **feature-space 增强**但是 Gaussian sampling 不是 swap; **不解耦** | https://arxiv.org/abs/2301.12995 |
| 3 | **FedCCRL**: Federated DG with Cross-Client Representation Learning | NeurIPS 2024 | **MixStyle** 跨客户端风格迁移 + AugMix 扰动 + 对比对齐 | 传 channel-wise (μ,σ);**MixStyle 在混合特征空间**,不解耦 | https://github.com/SanphouWang/FedCCRL |
| 4 | **StyleDDG**: Decentralized DG with Style Sharing | ECML PKDD 2025 | P2P 去中心化 style sharing + 首个风格共享 DG 的收敛保证 O(1/√K) | 去中心化 + 理论分析;**不解耦**,风格在混合特征空间 | https://arxiv.org/abs/2504.06235 |
| 5 | **FedSTAR** (previously FedSAR): Federated Style-Aware Transformer Aggregation | arXiv 2025 | StyleFiLM 风格调制 + Transformer class-wise attention 聚合 content prototype | **显式** content/style 解耦但 **风格严格本地保留**;Transformer 聚合 content | 未见 code |
| 6 | **FedStyle**: Style-Based FL Crowdsourcing Framework for Art Commissions | arXiv 2024 | 学风格表示 + 对比学习构建风格表示空间 | 极端风格异质 (每 client 一种艺术风格),不是通用 FedDG;**不解耦** | https://arxiv.org/abs/2404.16336 |
| 7 | **FedCA**: Federated DG for Medical Segmentation via Cross-Client Style Transfer | Expert Systems 2026 (early) | 跨客户端 feature style transfer + adaptive style alignment | medical segmentation 场景;**不解耦** | ScienceDirect (closed) |

**关键发现**:
- **CCST 在图像空间 AdaIN,FedCCRL 在浅层特征空间做 MixStyle**,都不解耦,FedFA 做的是高斯增强不是 swap
- **没有直接找到 "先严格 purify 解耦 → 再在 pure z_sty 上做 feature-level AdaIN" 的工作**
- FedSTAR 最接近 (显式解耦 content/style) 但**不共享风格**
- 我们方案 = 填补 "解耦 + 共享 + feature-level" 三者交集空白

---

## 主题 5: Purify 质量的 nonlinear probe 评估

我们的 linear probe 显示 96% (意味着假解耦)。文献里 MLP / kernel / MI estimator 哪个可靠?

| # | 论文 | 年份/venue | 核心一句话 | 和我们差异 | 代码 |
|---|------|-----------|-----------|-----------|------|
| 1 | **MINE**: Mutual Information Neural Estimation | ICML 2018 | 基于 Donsker-Varadhan 的 MI 下界,神经网络可训练 | MI **下界**,适合最大化 MI (不适合 minimize) | https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation- |
| 2 | **CLUB**: Contrastive Log-ratio Upper Bound of MI | ICML 2020 | MI **upper bound**,适合 MI minimization;O(N²) 速度快 | **直接可用于我们 purify**;比 HSIC 更强 (非线性) | https://github.com/Linear95/CLUB |
| 3 | **Mutual Information Estimation via Normalizing Flows** (Butakov) | NeurIPS 2024 | normalizing flow 估计 MI,高维更稳定 | 最新 MI estimator; overkill for 128d | https://proceedings.neurips.cc/paper_files/paper/2024/file/05a2d9ef0ae6f249737c1e4cce724a0c-Paper-Conference.pdf |
| 4 | **MIG**: Mutual Information Gap (disentanglement metric) | 多次被引 ICLR/NeurIPS | 每个 latent 对每个 generating factor 的 MI 差值作为解耦度量 | 标准 disentanglement 度量;可直接套用评估我们双头 | TF Disentanglement Lib |
| 5 | **sisPCA**: Supervised Independent Subspace PCA (HSIC-based disentanglement) | NeurIPS 2024 | HSIC 分解子空间,显式 supervised 独立性约束 | 我们已用 HSIC;可借鉴 subspace structure | https://proceedings.neurips.cc/paper_files/paper/2024/file/41ca8a0eb2bc4927a499b910934b9b81-Paper-Conference.pdf |
| 6 | **Identifiability Guarantees for Causal Disentanglement from Purely Observational Data** | NeurIPS 2024 | 给出 nonlinear causal disentanglement 的 identifiability 充分条件 | 理论证据: **什么条件下解耦可识别** (additive Gaussian noise + linear mixing) | https://neurips.cc/virtual/2024/poster/95550 |
| 7 | **Nonlinear ICA with Auxiliary Variables** (Halva 2024 & 相关 ICLR 2024 unified framework) | AISTATS 2024 / ICLR 2024 | 提供 multi-view nonlinear ICA / weakly-supervised disentanglement 的统一框架 | 理论基础: 在何种监督下可严格识别 | https://proceedings.mlr.press/v238/halva24a/halva24a.pdf |

**关键发现**:
- **linear probe 不够**, 必须用 **MLP probe + MINE/CLUB MI estimator** 评估
- CLUB 比 MINE 更适合 purify (MI upper bound → 可优化最小化)
- MIG 是标准解耦度量,可直接作为评估指标写进论文
- **Identifiability 理论**告诉我们: 没有 auxiliary signal (domain label / class label 监督),nonlinear ICA 本质不可识别 → 这给了我们 adversarial supervision 的理论依据

---

## Gap Analysis: 我们 3-stage 组合的 Novelty

### 直接 prior 检索结果

**没有找到直接 prior**。检索覆盖: CVPR/NeurIPS/ICLR/ICML/AAAI/IJCAI/CVPR-W 2023-2026,以及 arXiv 最新 preprint。最接近的对手方案是:

| 对手方案 | 相似度 | 关键缺失 |
|---------|-------|---------|
| FedSTAR | 高 (做 content/style 解耦) | 风格严格本地保留,不共享 |
| FISC/PARDON | 高 (跨客户端风格共享 + 对比) | 不解耦,直接在混合空间操作 |
| Fed-DIP | 中 (隐式解耦) | 用 prompt 不是特征空间 AdaIN |
| FedSeProto | 中 (MI purify) | 丢弃 domain 特征 (我们要保留做 augmentation) |
| FediOS | 中 (正交子空间) | 分 generic/personalized 不是 sem/sty;不共享 personalized |
| FedCCRL | 中 (cross-client style + contrastive) | 不解耦, MixStyle 直接在混合特征 |

### 我们的 3-stage 组合的 Novelty 定位

**Novel 三要素并置**:

1. **Purify (MLP adversary on top of cos²+HSIC)**: LNTL/HEX 在集中式有做,但 **FL 场景 + 双向 adversarial (class-adv on sty + dom-adv on sem)** 尚未见。FedSeProto 用 MI 单向 purify (丢 domain), 我们双向且保留。

2. **Prototype alignment with drift penalty on pure z_sem**: DCFL 理论支撑 + FedProto/FPL MSE 锚点 + FedPLVM α-sparsity + FedAli OT matching → 组合起来没有 prior。尤其 "**先确保 z_sem 是真纯 class 信号, 再上 InfoNCE**" 这个前置条件没有工作强调。

3. **Feature-space AdaIN on pure z_sty**: CCST 在图像空间; FedCCRL/MCSAD 在浅层特征但不解耦; FedFA 在特征但是 Gaussian sampling。"**先严格解耦得到 pure z_sty → 在 128d 特征空间做 cross-client AdaIN swap**" 没有直接对手。

### 3-stage 组合的 novelty 质量评估

- **高**: 3-stage 串联方案本身 (purify → align → style share) 在 FedDG 里首次明确提出
- **中**: 每一 stage 单独看都有 prior, 但组合系统性 (每阶段互相约束) novel
- **主要风险**: 
  - Moyer 2018 警告 adversarial counter-productive → 必须加 stabilization (CLUB MI / orthogonal projection 辅助)
  - DCFL 证明 InfoNCE 在 FL 有限样本下不稳 → 必须加 drift penalty + λ_u 远小于 λ_a
  - Identifiability 理论要求 auxiliary signal 才能真解耦 → 我们的 (y, d) 双监督符合条件,但 MLP probe 评估必要

### 下一步建议

1. **优先补 CLUB MI minimization**替代/辅助 HSIC (主题 2/5)
2. **MLP probe + MIG 指标**评估 Stage 1 purify 质量 (主题 5),不能只看 linear probe
3. **drift penalty 参考 FedAli OT** 而不只是 FedProto MSE (主题 3)
4. **Stage 3 AdaIN 参考 CCST feature recipe**(主题 4),但做在 128d 特征空间
5. **消融对比必跑**: FedSTAR (解耦不共享) + FISC (共享不解耦) + 我们 (解耦共享) → 证明两者组合 > 单一

---

## 参考文献 (URL 列表)

**主题 1 (Purify→Share pipeline)**
- [FISC/PARDON](https://arxiv.org/abs/2410.22622)
- [Fed-DIP](https://openreview.net/forum?id=VbmV3rs284)
- [DiPrompT](https://arxiv.org/abs/2403.08506)
- [FedSDAF](https://arxiv.org/abs/2505.02515)
- [MCSAD](https://arxiv.org/abs/2505.10152)
- [StableFDG](https://arxiv.org/abs/2311.00227)
- [FediOS](https://arxiv.org/abs/2311.18559)

**主题 2 (Adversarial Purify)**
- [Moyer 2018](https://arxiv.org/abs/1805.09458)
- [HEX](https://github.com/HaohanWang/HEX)
- [LNTL CVPR 2019](https://arxiv.org/abs/1812.10352)
- [CLUB](https://arxiv.org/abs/2006.12013)
- [FedSaaS](https://arxiv.org/abs/2505.09385)
- [Invariant Reps via Adversarial Forgetting](https://arxiv.org/abs/1911.04060)
- [FedSeProto ECAI 2024](https://ebooks.iospress.nl/doi/10.3233/FAIA240731)

**主题 3 (Prototype Drift)**
- [FedProto](https://arxiv.org/abs/2105.00243)
- [FPL CVPR 2023](https://github.com/WenkeHuang/RethinkFL)
- [FedPLVM NeurIPS 2024](https://arxiv.org/abs/2403.09048)
- [DCFL](https://arxiv.org/abs/2508.04005)
- [FedDPA](https://www.mdpi.com/2079-9292/14/16/3286)
- [FedAli](https://arxiv.org/abs/2411.10595)
- [Doubly Regularized Drift](https://arxiv.org/abs/2404.08447)

**主题 4 (Feature-space Style Swap)**
- [CCST WACV 2023](https://arxiv.org/abs/2210.00912)
- [FedFA ICLR 2023](https://arxiv.org/abs/2301.12995)
- [FedCCRL NeurIPS 2024](https://arxiv.org/abs/2410.11267)
- [StyleDDG](https://arxiv.org/abs/2504.06235)
- [FedSTAR](https://arxiv.org/abs/2511.18841)
- [FedStyle](https://arxiv.org/abs/2404.16336)

**主题 5 (Disentanglement Evaluation)**
- [MINE](https://arxiv.org/abs/1801.04062)
- [CLUB](https://arxiv.org/abs/2006.12013)
- [MI via Normalizing Flows NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/05a2d9ef0ae6f249737c1e4cce724a0c-Paper-Conference.pdf)
- [sisPCA NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/41ca8a0eb2bc4927a499b910934b9b81-Paper-Conference.pdf)
- [Causal Disentanglement Identifiability NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95550)
- [Unified Multi-view Nonlinear ICA ICLR 2024](https://research-explorer.ista.ac.at/download/14946/18995/2024_ICLR_Yao.pdf)

---

**调研结论**: 3-stage 组合 novelty **成立**,文献里无直接 prior。风险点在于每 stage 单独都有警告 (Moyer 对抗警告 / DCFL InfoNCE 警告 / identifiability 理论要求),因此 **稳定化辅助机制必不可少** (CLUB / OT / drift penalty)。MLP probe 评估解耦质量是关键差异化实验。
