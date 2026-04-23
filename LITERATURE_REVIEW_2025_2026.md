# 跨域联邦学习最新文献调研报告（2025-2026）

**调研时间**：2026-04-01  
**调研方向**：跨域联邦学习（Cross-Domain Federated Learning）最新方法与突破方向  
**覆盖范围**：2024下半年 ~ 2026年初，主要关注顶会（ICLR/CVPR/ICML/ICCV/IJCAI/AAAI/NeurIPS）及高质量arXiv预印本

---

## 一、调研摘要

本次调研在已分析方法（FedAvg, FedProx, SCAFFOLD, MOON, FedBN, FedProto, FPL, FedPLVM, MP-FedCL, I2PFL, FDSE, FedPall, FedFSL-CFRD, FedSTAR, DCFL, FedALA, FedAPA）基础上，新发现了以下几个重要方向和方法：

### 核心发现

| 类别 | 新发现方法 | 会议/期刊 | 核心亮点 |
|------|-----------|----------|---------|
| **预训练模型+原型** | MPFT | ICLR 2025 | 预训练模型+多域原型微调，单轮通信收敛 |
| **双适配器解耦** | FedSDAF | arXiv 2025 | 域感知适配器+域不变适配器+双向知识蒸馏 |
| **原型+记忆蒸馏** | FedCPD | IJCAI 2025 | 原型增强+特征蒸馏防遗忘，泛化提升10.4% |
| **跨客户端表示学习** | FedCCRL | arXiv 2024 | MixStyle跨域迁移+AugMix扰动，增加域多样性 |
| **跨客户端特征对齐** | FedAlign | CVPR Workshop 2025 | 双阶段对齐（嵌入+预测），轻量隐私保护 |
| **热插拔模块化FL** | HPFL | ICLR 2025 | 骨干+插件架构，插件市场按需检索 |
| **风格特征调度+KD** | FedDG-SFD | ICONIP 2024/2026 | 风格特征调度器增强知识蒸馏 |
| **演化分布偏移** | FedEvolve/FedEvp | 2026期刊 | 捕获时间演化模式，原型持续更新 |
| **VLM+Prompt学习** | FedTPG / pFedMoAP | ICLR 2025 / IJCAI 2025 | CLIP+联邦Prompt，文本驱动跨域泛化 |
| **扩散模型增强** | Gen-FedSD | 2025 | Stable Diffusion生成合成数据填补分布差异 |
| **自适应聚类聚合** | FedACA | 2026 | 自适应K-means原型优化+客户端聚类+分类器融合 |

---

## 二、重要新方法详细分析

### 2.1 MPFT — 多域原型联邦微调 (ICLR 2025) ★★★

- **来源**：[Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning](https://openreview.net/forum?id=3wEGdrV5Cb)
- **作者**：Jingyuan Zhang, Yiyang Duan, Shuaicheng Niu, Yang Cao, Wei Yang Bryan Lim
- **核心思路**：在预训练模型（如CLIP）基础上，每个客户端生成domain-specific原型（而非模型参数），上传到服务器构成原型训练数据集，模拟集中式学习
- **关键创新**：
  1. **预训练模型+原型微调**：利用预训练表示的强泛化力，原型注入域特定信息
  2. **单轮通信收敛**：原型一次上传即可完成微调，通信效率极高
  3. **差分隐私保护**：对原型施加DP噪声保护隐私
- **实验**：in-domain和out-of-domain准确率均显著优于传统方法
- **与你的关联**：★★★ — 预训练模型+原型是一个全新范式，可以考虑将你的解耦原型方案建立在预训练模型（CLIP/ViT）之上

---

### 2.2 FedSDAF — 源域感知联邦域泛化 (arXiv 2025) ★★★

- **来源**：[FedSDAF: Leveraging Source Domain Awareness for Enhanced Federated Domain Generalization](https://arxiv.org/abs/2505.02515)
- **代码**：https://github.com/pizzareapers/FedSDAF
- **核心思路**：双适配器架构解耦"本地专长"与"全局泛化共识"
- **关键创新**：
  1. **Domain-Aware Adapter（本地保留）**：提取并保护每个源域的独特判别性知识
  2. **Domain-Invariant Adapter（全局共享）**：跨客户端构建鲁棒的全局共识
  3. **双向知识蒸馏（Bidirectional KD）**：两个适配器之间高效知识交换
  4. **核心发现**：完整源域学习的特征比直接从目标域学的特征泛化更好
- **实验**：OfficeHome, PACS, VLCS, DomainNet 上显著超越现有FedDG方法
- **与你的关联**：★★★ — 双适配器思路与你的双头解耦异曲同工，但它用的是适配器而非原型头。双向KD机制值得借鉴

---

### 2.3 FedCPD — 原型增强+记忆蒸馏 (IJCAI 2025) ★★☆

- **来源**：[FedCPD: Personalized Federated Learning with Prototype-Enhanced Representation and Memory Distillation](https://www.ijcai.org/proceedings/2025/0612.pdf)
- **核心思路**：用特征蒸馏保留历史信息防止遗忘 + 原型学习增强类区分
- **关键创新**：
  1. **Memory Distillation**：防止联邦训练中的"历史信息遗忘"
  2. **Prototype-Enhanced Representation**：识别多样特征，增强类间区分
  3. 泛化提升10.40%，个性化提升4.90%
- **与你的关联**：★★☆ — 蒸馏+原型的结合思路可借鉴，特别是防遗忘机制在跨域场景中可能有用

---

### 2.4 FedCCRL — 跨客户端表示学习 (arXiv 2024) ★★☆

- **来源**：[FedCCRL: Federated Domain Generalization with Cross-Client Representation Learning](https://arxiv.org/abs/2410.11267)
- **代码**：https://github.com/SanphouWang/FedCCRL
- **核心思路**：通过跨客户端域迁移和特征扰动增加本地域多样性
- **关键创新**：
  1. **MixStyle跨域迁移**：将MixStyle适配到联邦场景，传递域特定特征
  2. **AugMix域不变扰动**：扰动域不变特征增加多样性
  3. 两个模块协同增加本地训练的域多样性
- **实验**：PACS, OfficeHome, miniDomainNet 达到SOTA
- **与你的关联**：★★☆ — MixStyle跨域迁移思路与你的"全局风格仓库+风格交换"有相似之处，但它不做显式解耦

---

### 2.5 FedAlign — 跨客户端特征对齐 (CVPR Workshop 2025) ★★☆

- **来源**：[FedAlign: Federated Domain Generalization with Cross-Client Feature Alignment](https://arxiv.org/abs/2501.15486)
- **核心思路**：双阶段对齐（特征嵌入对齐 + 预测对齐）促进域不变特征学习
- **关键创新**：
  1. **Cross-Client Feature Extension**：域不变特征扰动 + 选择性跨客户端特征传递
  2. **Dual-Stage Alignment**：同时对齐嵌入和预测，蒸馏域不变特征
  3. 轻量级、隐私保护、低通信开销
- **与你的关联**：★★ — 双阶段对齐思路可参考，但该方法不做特征解耦

---

### 2.6 HPFL — 热插拔联邦学习 (ICLR 2025) ★★☆

- **来源**：[Hot-Pluggable Federated Learning: Bridging General and Personalized FL via Dynamic Selection](https://openreview.net/forum?id=B8akWa62Da)
- **核心思路**：模型分为骨干+插件模块，客户端训练个性化插件，服务器端维护插件市场
- **关键创新**：
  1. **模块化架构**：共享骨干 + 个性化插件
  2. **Plug-in Market**：服务器维护模块库，客户端按需检索合适插件
  3. **动态选择**：推理时选择最佳插件组合增强泛化
  4. 差分隐私保护 + 显著优于GFL和PFL方法
- **与你的关联**：★★☆ — "插件市场"思路与你的"全局风格仓库"有概念上的呼应，可以考虑将风格仓库做成可检索的插件式设计

---

### 2.7 FedDG-SFD — 风格特征调度器 (ICONIP 2024) ★★☆

- **来源**：[A Federated Domain Generalization Method by Enhancing Knowledge Distillation with Stylistic Feature Dispatcher](https://link.springer.com/chapter/10.1007/978-981-96-7036-9_16)
- **核心思路**：用风格特征调度器增强跨域知识蒸馏
- **关键创新**：风格特征不再被简单擦除或私有化，而是通过"调度器"有目的地分配和利用
- **与你的关联**：★★★ — 直接相关！这个方法也在做"风格资产化"，需要仔细对比与你方案的差异

---

### 2.8 FedEvolve / FedEvp — 演化分布偏移 (2026期刊) ★☆☆

- **来源**：[Federated Learning Under Evolving Distribution Shifts](https://pmc.ncbi.nlm.nih.gov/articles/PMC12839774/)
- **核心思路**：处理随时间演化的域偏移（非静态域偏移）
- **关键创新**：
  1. **FedEvolve**：学习两个不同的表示映射，捕获连续数据域之间的转换
  2. **FedEvp**：学习单一的域不变表示，通过持续更新的原型对齐当前数据与历史域
- **与你的关联**：★☆☆ — 关注点不同（时序演化 vs 静态跨域），但原型持续更新的思路可借鉴

---

### 2.9 联邦Prompt学习（VLM方向）★★☆

#### FedTPG (ICLR 2024, 2025验证)
- **来源**：[Federated Text-Driven Prompt Generation for Vision-Language Models](https://openreview.net/forum?id=NW31gAylIm)
- **方法**：文本驱动的prompt生成网络，根据类名动态创建prompt

#### pFedMoAP (ICLR 2025)
- **来源**：[Mixture of Experts Made Personalized: Federated Prompt Learning for Vision-Language Models](https://openreview.net/forum?id=xiDJaTim3P)
- **方法**：MoE框架个性化prompt学习过程

#### PLAN — Prompt学习+联邦域泛化
- **来源**：[Federated Domain Generalization via Prompt Learning and Aggregation](https://ieeexplore.ieee.org/document/11358406/)
- **方法**：Prompt作为知识传递桥梁，替代传统参数聚合

**与你的关联**：★★☆ — VLM+Prompt是一个全新范式，如果骨干网络换成CLIP/ViT，可以用prompt替代原型做跨域对齐。这是一个潜在的更前沿方向。

---

### 2.10 扩散模型增强方向 ★☆☆

#### Gen-FedSD — Stable Diffusion增强FL
- **方法**：每个客户端构造文本prompt → 预训练Stable Diffusion生成合成样本 → 填补分布差异
- **与你的关联**：★☆☆ — 计算开销大，但"生成式域增强"的思路比简单特征加法更强

#### 反馈驱动联邦域适应
- **方法**：服务器端域分析检测分布偏移 → 生成反馈信号(Grad-CAM/SHAP) → 客户端用生成模型(VAE/Diffusion/LM)增强
- **与你的关联**：★☆☆ — 服务器反馈引导客户端增强的思路新颖

---

### 2.11 其他值得关注的方法

#### FedACA — 自适应分类器聚合 (2026)
- **方法**：自适应K-means改进原型质量 + 客户端聚类匹配 + 个性化分类器融合
- **效果**：异构场景下准确率提升9.66%

#### FedPKD — 联邦原型知识蒸馏 (2025)
- **方法**：将本地模型抽象为原型表示和软标签进行服务器端聚合，大幅减少通信开销

#### FedCA — 跨客户端风格迁移+自适应风格对齐 (2026)
- **来源**：[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426003076)
- **方法**：用于医学图像分割的跨客户端特征风格迁移+自适应风格对齐
- **与你的关联**：★★ — 也在做跨客户端风格迁移！虽然是医学领域

#### FedDistr — 数据分布解纠缠 (2025)
- **来源**：[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0004370225001742)
- **方法**：用扩散模型解耦和恢复数据分布，实现单轮通信联邦学习

---

## 三、技术趋势总结

### 3.1 六大新趋势

```
2025-2026 跨域联邦学习技术趋势
═══════════════════════════════════

趋势1：预训练模型(VLM/CLIP)进入FL
  ├── MPFT: 预训练+多域原型微调
  ├── PLAN/FedTPG: Prompt替代参数聚合
  └── pFedMoAP: MoE个性化Prompt

趋势2：适配器(Adapter)替代全模型聚合
  ├── FedSDAF: 双适配器(域感知+域不变)
  ├── HPFL: 骨干+可插拔模块市场
  └── FedRLoRA: 残差低秩适配

趋势3：跨客户端特征交换/增强
  ├── FedCCRL: MixStyle域迁移+AugMix扰动
  ├── FedAlign: 双阶段嵌入+预测对齐
  └── FedCA: 跨客户端风格迁移

趋势4：风格从"噪声"到"资产"
  ├── FedDG-SFD: 风格特征调度器
  ├── FedCA: 自适应风格对齐
  └── 你的方案: 全局风格仓库(仍有差异化空间)

趋势5：蒸馏+原型结合
  ├── FedCPD: 记忆蒸馏防遗忘+原型增强
  ├── FedPKD: 原型知识蒸馏
  └── FedSDAF: 双向知识蒸馏

趋势6：生成式域增强
  ├── Gen-FedSD: Stable Diffusion合成
  ├── FedDistr: 扩散模型解耦分布
  └── 反馈驱动框架: 服务器引导客户端生成
```

### 3.2 对你原有方案的影响

| 你的模块 | 新发现的竞争/互补方法 | 启示 |
|---------|---------------------|------|
| **正交双头解耦** | FedSDAF双适配器、HPFL骨干+插件 | 可考虑用Adapter替代投影头，更灵活 |
| **全局风格仓库** | FedDG-SFD风格调度器、FedCA风格迁移、HPFL插件市场 | 需要与FedDG-SFD仔细区分创新点 |
| **语义软对齐** | FedAlign双阶段对齐、FedCPD蒸馏+原型 | 可加入蒸馏防遗忘机制 |
| **整体框架** | MPFT预训练+原型 | 考虑在CLIP/ViT基础上做解耦原型 |

---

## 四、文献详细表

| 方法 | 会议/期刊 | 年份 | 核心方法 | 关键结果 | 代码 | 与你的关联度 |
|------|----------|------|---------|---------|------|------------|
| MPFT | ICLR 2025 | 2025 | 预训练模型+多域原型微调 | 单轮通信收敛，in/out-domain均提升 | - | ★★★ |
| FedSDAF | arXiv | 2025 | 双适配器+双向KD | OfficeHome/PACS/VLCS/DomainNet SOTA | [GitHub](https://github.com/pizzareapers/FedSDAF) | ★★★ |
| FedDG-SFD | ICONIP 2024 | 2024/2026 | 风格特征调度器+KD | 域泛化增强 | - | ★★★ |
| FedCPD | IJCAI 2025 | 2025 | 原型增强+记忆蒸馏 | 泛化+10.4%, 个性化+4.9% | - | ★★☆ |
| FedCCRL | arXiv | 2024 | MixStyle+AugMix跨域增强 | PACS/OfficeHome/miniDN SOTA | [GitHub](https://github.com/SanphouWang/FedCCRL) | ★★☆ |
| FedAlign | CVPR-W 2025 | 2025 | 双阶段特征+预测对齐 | 强域对齐效果 | - | ★★☆ |
| HPFL | ICLR 2025 | 2025 | 骨干+插件市场+动态选择 | 显著优于GFL/PFL | - | ★★☆ |
| FedTPG | ICLR 2024 | 2024 | 文本驱动Prompt生成 | 跨类泛化强 | - | ★★ |
| pFedMoAP | ICLR 2025 | 2025 | MoE个性化Prompt | VLM联邦个性化 | - | ★★ |
| FedCA | Expert Systems | 2026 | 跨客户端风格迁移+风格对齐 | 医学图像域泛化 | - | ★★ |
| FedACA | Neurocomputing | 2026 | 自适应聚类+分类器融合 | 异构+9.66% | - | ★☆☆ |
| FedEvolve | 期刊 | 2026 | 演化域偏移建模 | 时序泛化 | - | ★☆☆ |
| Gen-FedSD | - | 2025 | SD生成合成数据 | 填补分布差异 | - | ★☆☆ |
| FedDistr | AI Journal | 2025 | 扩散模型解耦分布 | 单轮通信 | - | ★☆☆ |
| FedPKD | Computing | 2025 | 原型知识蒸馏 | 减少通信 | - | ★☆☆ |

---

## 五、对你方案改进的建议方向

基于本次调研，以下是几个值得考虑的改进方向：

### 方向A：预训练模型+解耦原型（融合MPFT思路）
- 在CLIP/ViT预训练骨干上做解耦原型学习
- 原型注入域特定信息，Prompt做跨域对齐
- **优势**：预训练表示泛化力强，原型通信效率高
- **风险**：可能偏离原有"从头训练"的实验设置

### 方向B：适配器式解耦（融合FedSDAF思路）
- 用Adapter替代投影头做域感知/域不变分离
- 双向知识蒸馏替代简单正交约束
- **优势**：更灵活，参数量更小
- **风险**：与FedSDAF差异化不够

### 方向C：风格调度器+原型解耦（融合FedDG-SFD思路）
- 在你的双头解耦基础上，加入风格调度器机制
- 风格不是简单存储和交换，而是有目的地调度给需要的客户端
- **优势**：与你原有方案兼容，增量式改进
- **风险**：需要确认与FedDG-SFD的差异

### 方向D：跨客户端域增强+解耦（融合FedCCRL思路）
- 在解耦空间中做MixStyle跨域迁移
- 在语义空间做对齐，在风格空间做MixStyle增强
- **优势**：解耦后的MixStyle比混合特征空间更可控
- **风险**：实现复杂度增加

### 方向E：模块化插件式风格仓库（融合HPFL思路）
- 风格仓库做成可检索的"插件市场"
- 客户端按需检索合适的风格插件增强泛化
- **优势**：概念新颖，实用性强
- **风险**：检索机制设计复杂

---

## 六、推荐下一步行动

1. **精读 MPFT 和 FedSDAF 论文** — 这两篇与你的方案最相关且来自顶会
2. **精读 FedDG-SFD** — 直接竞争者，需要仔细区分
3. **下载 FedSDAF 和 FedCCRL 代码** — 有开源实现可参考
4. **重新评估方案** — 考虑是否将骨干换成预训练模型，或将解耦方式从投影头改为适配器
5. **确定差异化创新点** — 在"风格资产化"这个核心创新上，与FedDG-SFD做出明确区分

---

*本报告基于2026年4月1日的网络搜索结果生成，可能遗漏部分最新工作*

Sources:
- [MPFT - ICLR 2025](https://openreview.net/forum?id=3wEGdrV5Cb)
- [FedSDAF - arXiv](https://arxiv.org/abs/2505.02515)
- [FedSDAF GitHub](https://github.com/pizzareapers/FedSDAF)
- [FedCPD - IJCAI 2025](https://www.ijcai.org/proceedings/2025/0612.pdf)
- [FedCCRL - arXiv](https://arxiv.org/abs/2410.11267)
- [FedCCRL GitHub](https://github.com/SanphouWang/FedCCRL)
- [FedAlign - CVPR Workshop 2025](https://arxiv.org/abs/2501.15486)
- [HPFL - ICLR 2025](https://openreview.net/forum?id=B8akWa62Da)
- [FedDG-SFD - Springer](https://link.springer.com/chapter/10.1007/978-981-96-7036-9_16)
- [FedEvolve](https://pmc.ncbi.nlm.nih.gov/articles/PMC12839774/)
- [FedTPG - ICLR](https://openreview.net/forum?id=NW31gAylIm)
- [pFedMoAP - ICLR 2025](https://openreview.net/forum?id=xiDJaTim3P)
- [PLAN - IEEE](https://ieeexplore.ieee.org/document/11358406/)
- [FedCA - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426003076)
- [FedACA - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231226003693)
- [Federated Domain Generalization Survey - IEEE](https://ieeexplore.ieee.org/document/11130884/)
- [FedDistr - AI Journal](https://www.sciencedirect.com/science/article/abs/pii/S0004370225001742)
