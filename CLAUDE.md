# 面向跨域联邦学习的解耦原型学习 — 研究综述与方向指南

## 一、研究背景与问题定义

**核心问题**：联邦学习（FL）中，不同客户端的数据来自不同域（如照片/素描/油画），导致条件特征分布 P(X|Y) 存在显著差异（域偏移/Feature Skew）。传统参数聚合方法（FedAvg）在此场景下性能严重下降。

**关键区分**：
- **标签倾斜（Label Skew）**：不同客户端拥有不同类别分布
- **特征倾斜（Feature Skew / Domain Shift）**：同一类别在不同客户端呈现不同特征分布 — **本课题聚焦此问题**

---

## 二、论文深度分析

### 2.1 基础联邦学习方法

#### FedAvg (McMahan et al., 2017) — 联邦学习奠基
- **方法**：各客户端本地SGD训练，服务器按数据量加权平均聚合模型参数
- **局限**：简单欧氏平均在域偏移场景下导致全局模型收敛到差解（负迁移）
- **源码位置**：`PFLlib/system/flcore/clients/clientavg.py`

#### FedProx (Li et al., 2020) — 近端正则化
- **方法**：在本地目标函数中添加近端项 `μ/2 ||w - w_global||²`，限制本地更新偏离全局模型
- **局限**：全局正则化对域偏移场景过于刚性，无法区分语义信息和域特定信息

#### SCAFFOLD (Karimireddy et al., 2020) — 方差缩减
- **方法**：引入控制变量（control variates）纠正本地更新偏差，减少客户端漂移
- **局限**：需额外通信控制变量，且对特征分布差异处理不足

#### MOON (Li et al., 2021) — 模型级对比学习
- **方法**：最大化本地模型与全局模型嵌入的相似度，同时远离上一轮本地模型
- **创新**：首次将对比学习引入FL纠正客户端漂移
- **局限**：模型级对比粒度太粗，无法处理细粒度的域偏移

#### FedBN (Li et al., 2021) — 局部BN
- **方法**：保持BN层本地私有，不参与聚合，保留各客户端的本地统计特征
- **创新**：简单有效地保留域特定统计信息
- **局限**：仅在BN层做域适应，深层特征仍然混合了语义和风格

---

### 2.2 联邦原型学习方法

#### FedProto (Tan et al., AAAI 2022) — 原型学习先驱
- **核心idea**：用类原型（类特征均值）代替模型参数进行通信
- **方法**：
  - 本地：提取特征 → 计算类原型（均值）→ 上传原型
  - 全局：聚合各客户端同类原型 → 下发全局原型
  - 训练损失：CE + λ·MSE(local_feature, global_proto)
- **优势**：通信高效、支持异构模型
- **局限**：
  1. 简单欧氏均值在跨域场景导致"模糊原型"
  2. MSE硬对齐破坏本地流形
  3. 单一原型无法捕获多模态分布
- **源码分析**：`PFLlib/system/flcore/clients/clientproto.py` — 清晰展示了MSE损失逼近全局原型的机制，测试时用最近原型分类

#### FPL (Huang et al., CVPR 2023) — 重新思考域偏移下的原型学习
- **核心idea**：构建聚类原型 + 无偏原型，提供丰富域知识
- **方法**：
  - 用FINCH聚类算法对各客户端同类原型进行自动聚类，获得多个聚类原型
  - 分层InfoNCE损失：拉近样本与同类聚类原型，推开异类原型
  - MSE正则化：对齐本地特征与无偏均值原型
- **创新**：首次在FL中引入多原型机制处理域偏移
- **局限**：
  1. 聚类基于混合特征，未分离语义与风格
  2. FINCH聚类对异常值敏感
  3. 欧氏均值作为"无偏原型"在强域偏移下仍有偏
- **源码分析**：`RethinkFL/models/fpl.py` — `hierarchical_info_loss`实现分层对比损失，`proto_aggregation`用FINCH聚类

#### FedPLVM (Wang et al., NeurIPS 2024) — 驯服跨域表征方差
- **核心idea**：双层原型聚类 + α-稀疏原型损失，解决"困难域"学习不公平问题
- **方法**：
  - **本地聚类**：基于FINCH生成本地多聚类原型
  - **全局聚类**：服务器对收集的原型再次FINCH聚类，减少通信开销
  - **α-稀疏损失**：`cosine_similarity^α` (0<α<1)，增强类间稀疏性
  - 类内修正项：减小同类样本特征距离
- **创新**：关注域间学习难度不平等，α-稀疏机制增强困难域
- **局限**：
  1. 仍基于混合特征聚类，未解耦语义与风格
  2. α参数敏感性
- **源码分析**：`FedPLVM/utils/update.py` — `criterion_InfoNCE_alpha`实现α-稀疏损失

#### MP-FedCL (Qiao et al., IoT Journal 2024) — 多原型对比学习
- **核心idea**：用k-means生成每类多个原型，构建全局原型池
- **方法**：
  - 本地k-means聚类 → 多原型
  - 全局聚合为原型池
  - 有监督对比学习引导本地训练
- **创新**：首次同时处理标签和特征非IID
- **局限**：k值需要预设，缺乏自适应性

#### I2PFL (Le et al., 2025) — 域内域间原型
- **核心idea**：同时考虑域内（intra-domain）和域间（inter-domain）原型
- **方法**：
  - **域内原型**：MixUp增强 + 对齐，增强本地特征多样性
  - **域间原型**：重加权方案——距均值远的原型获得更高权重，缓解主导域偏置
  - GPCL（广义原型对比学习）+ APA（增强原型对齐）
- **创新**：首次关注域内变异对原型学习的影响
- **局限**：重加权方案的鲁棒性有待验证

---

### 2.3 特征解耦与域适应方法

#### FDSE (Wang et al., CVPR 2025) — 域偏移擦除器 ★重要参考
- **核心idea**：将模型每层分解为DFE（域无关特征提取器）+ DSE（域特定偏移擦除器），迭代去偏
- **方法**：
  - **层分解**：每个卷积层 → DFE提取核心特征 + DSE擦除域偏移
  - **一致性正则化**：拉近DSE输出的本地统计量与全局一致统计量
  - **差异化聚合**：DFE公平聚合（类似FedAvg），DSE基于相似度个性化聚合
  - DSE使用分组深度卷积（cheap linear operations），仅为DFE的1/94大小
- **创新**：
  1. 混合视角——将个性化用于增强共识（而非对抗）
  2. 迭代逐层去偏移（vs 仅在某一层做域适应）
  3. 二次规划（QP）优化DFE聚合权重
- **实验**：Office-Caltech10、PACS、DomainNet，显著优于FedBN、FedProto等
- **局限**：
  1. 将域信息视为"噪声"擦除，未利用风格作为增强资源
  2. 层分解架构固定比例G，灵活性有限
- **源码分析**：`FDSE_CVPR25/algorithm/fdse.py`
  - `DSEConv/DSELinear`类实现层分解
  - Server端 `iterate()` 实现差异化聚合 + QP优化
  - Client端 `train()` 实现一致性正则化（KL散度对齐BN统计量）

#### FedPall (Zhang et al., ICCV 2025) — 原型对抗协作学习 ★重要参考
- **核心idea**：对抗学习统一特征空间 + 协作学习增强类信息
- **方法**：
  - **Amplifier**：服务器训练的MLP，放大客户端间异构信息
  - **对抗学习**：客户端用KL散度减少Amplifier识别的异构信息
  - **原型对比**：InfoNCE拉近本地特征与全局原型，推开异类
  - **原型混合特征**：αz + (1-α)G^k → 上传服务器训练全局分类器
  - **全局分类器下放**：替换本地分类器，再做本地微调
- **创新**：
  1. 对抗+协作双重机制
  2. 原型混合特征 + Bernoulli mask保护隐私
  3. 全局分类器提供跨域视角
- **实验**：Digits、Office-10、PACS，整体SOTA
- **局限**：
  1. 对抗训练在FL中不稳定
  2. 隐私风险（上传混合特征）
  3. 未显式解耦语义与风格

#### FedFSL-CFRD (Wang et al., AAAI 2025) — 联邦少样本学习的协同特征解耦
- **核心idea**：在客户端层面和类别层面协同进行特征表示解耦
- **方法**：
  - **第一级解耦（client-wise）**：原始特征 → 全局共性特征 + 本地个性特征 + 个性化偏置
  - **第二级解耦（class-wise）**：度量空间中，通过类特定重构和类无关重构分离判别性特征
- **创新**：双层次协同解耦（全局/本地 + 类相关/类无关）
- **局限**：
  1. 针对少样本场景设计，普通FL适用性有限
  2. 隐式重构机制，解耦程度难以量化

#### FedSAR/FedSTAR (Jeon et al., 2025) — 风格感知Transformer聚合 ★重要参考
- **核心idea**：显式内容-风格分离 + Transformer注意力聚合原型
- **方法**：
  - **StyleFiLM模块**：用风格向量(γ, β)调制内容特征 `y = γ·x + β`
  - 只上传内容原型，风格向量保留本地
  - **Transformer聚合器**：类嵌入 + 客户端身份嵌入 + 内容原型 → 多头注意力加权聚合
- **创新**：
  1. 首个在原型FL中做显式内容-风格分离的框架
  2. 注意力替代简单平均，学习客户端贡献权重
  3. 通信高效（仅传输紧凑内容原型）
- **局限**：
  1. **风格严格本地私有，未跨域共享** — 这是你的课题突破口！
  2. 风格分离基于FiLM（仿射变换），表达能力有限
  3. 未利用风格做数据增强

---

### 2.4 对比学习与聚合优化方法

#### FedALA (Zhang et al., AAAI 2023) — 自适应局部聚合
- **方法**：客户端学习逐元素聚合权重，自适应融合本地和全局模型
- **关联**：与你的课题互补——可用于优化双头模型的参数聚合

#### FedAPA (Sun et al., IJCAI 2025) — 服务器端梯度自适应聚合
- **方法**：服务器端基于客户端参数变化的梯度更新聚合权重
- **创新**：将 `||Δθ||²/2` 作为本地损失代理，集中式效率高

#### DCFL (Kim et al., 2025) — 解耦对比学习
- **核心idea**：标准对比损失在FL有限样本下不适用，将其解耦为对齐+均匀性两个独立目标
- **方法**：`L = λ_a·L_alignment + λ_u·L_uniformity`
- **创新**：理论证明标准对比损失的渐近假设与FL有限样本冲突
- **关联**：你的语义对比损失设计可借鉴此思路

#### Cross-Domain Federated Semantic Communication
- **方向**：将联邦学习与语义通信结合，关注跨域语义传输
- **关联度较低**：偏通信方向，但域适应思想可借鉴

---

## 三、开题报告方案评估

### 3.1 方案概述
提出"双原型解耦 + 全局风格仓库 + 语义软对齐"三大模块：
1. **正交约束双头解耦**：语义头(Hsem) + 风格头(Hsty)，余弦正交损失
2. **全局风格仓库**：收集各域风格原型，跨域风格交换增强 `z_aug = z_con + λ·s_ext`
3. **语义软对齐**：对比损失替代MSE硬对齐

### 3.2 创新性评估 ✅

| 创新点 | 对比现有工作 | 评估 |
|--------|-------------|------|
| 显式正交双原型 | FedSTAR用FiLM隐式分离，FedFSL-CFRD用重构隐式分离 | **强创新** — 首次在FL原型学习中引入几何正交硬约束 |
| 全局风格仓库 | FedSTAR风格严格本地私有，FDSE擦除风格 | **核心创新** — 将风格从"噪声"变为"资产"，跨域共享 |
| 风格交换增强 | I2PFL用MixUp增强，FedPall混合特征 | **有创新** — 特征层面隐式数据增强，保护隐私 |
| 语义软对齐 | FedProto用MSE，FPL用InfoNCE | **增量创新** — 借鉴已有，但在解耦语义空间做更纯净 |

### 3.3 可行性评估

**优势：**
- 技术路径清晰，每个模块有明确的数学定义
- 有现成源码框架（PFLlib、FDSE、RethinkFL）可复用
- 数据集标准（Digit-5、Office-Caltech10、PACS、DomainNet）
- 基线方法源码齐全

**风险点：**
1. **正交约束的实际效果**：余弦正交 ≠ 信息正交，可能存在信息泄漏
   - 建议：加入互信息最小化约束或梯度反转层作为辅助
2. **风格仓库的隐私保护**：风格原型虽不是原始数据，但可能携带域标识信息
   - 建议：加入差分隐私噪声或风格聚合匿名化
3. **风格交换的效果**：简单加法 `z_aug = z_con + λ·s_ext` 可能不够
   - 建议：考虑AdaIN风格注入或FiLM调制
4. **超参数过多**：λ1, λ2, λ3 + 正交损失 + 对比温度τ
   - 建议：消融实验逐步验证

### 3.4 总体结论
> **方向可行，创新点成立，核心贡献明确。** 全局风格仓库是最大亮点，填补了FedSTAR等方法"风格仅本地"的空白。建议在风格注入方式（简单加法 → AdaIN/FiLM）和正交约束的信息完备性上做进一步探索。

---

## 四、推荐研究路线

### Phase 1：基线复现与环境搭建（1-2周）
1. 基于 PFLlib 框架搭建实验环境
2. 在 Digit-5 数据集上复现 FedProto、FPL、FedBN 基线
3. 验证跨域场景下各方法的性能下降

### Phase 2：双头解耦模块开发（2-3周）
1. 实现双头架构（在ResNet-18骨干后分叉）
2. 实现正交约束损失 `L_orth = (cos_sim(z_sem, z_sty))²`
3. 验证t-SNE可视化解耦效果
4. **关键实验**：对比有/无正交约束的特征分离效果

### Phase 3：全局风格仓库与增强（2周）
1. 实现服务器端风格仓库收集、去重、采样
2. 实现风格交换增强机制
3. **建议改进**：从简单加法开始，逐步尝试：
   - `z_aug = z_con + λ·s_ext` （基础版）
   - AdaIN: `z_aug = s_std * (z_con - z_mean) / z_std + s_mean`
   - FiLM: `z_aug = γ_sty * z_con + β_sty`

### Phase 4：语义软对齐与聚合策略（1-2周）
1. 实现对比损失的语义对齐
2. 实现差异化参数聚合（语义头+骨干聚合，风格头私有）
3. **建议借鉴 DCFL** 的解耦对比设计

### Phase 5：完整实验与论文撰写（3-4周）
1. 在 Digit-5、Office-Caltech10、PACS 上跑完整实验
2. 如时间允许，加入 DomainNet（更大规模验证）
3. 消融实验：逐模块去除验证贡献
4. 参数敏感性分析
5. 可视化分析（t-SNE、风格仓库内容、原型质量）

---

## 五、进一步研究方向建议

### 5.1 更深层的创新点拓展
1. **动态风格仓库管理**：引入多样性度量，自动控制仓库容量和采样策略
2. **条件风格生成**：用VAE/CVAE根据语义条件生成风格向量，而非仅采样已有风格
3. **多粒度解耦**：不仅分离语义/风格，还可分出"结构/纹理/颜色"等多层次
4. **自适应正交松弛**：早期训练允许一定耦合，后期逐步加强正交约束

### 5.2 值得关注的技术趋势
1. **DCFL的解耦对比学习**：为FL量身定制的对比目标，可直接集成
2. **Transformer聚合**（来自FedSTAR）：注意力机制替代简单平均
3. **FDSE的迭代去偏移思想**：逐层处理域偏移可能比最后一层解耦更有效

---

## 六、关键源码参考

| 代码仓库 | 对应论文 | 核心文件 | 用途 |
|----------|---------|---------|------|
| `PFLlib/` | 联邦学习通用框架 | `system/flcore/clients/clientproto.py` | 实验框架基础 |
| `FDSE_CVPR25/` | FDSE (CVPR 2025) | `algorithm/fdse.py` | 层分解、差异化聚合参考 |
| `RethinkFL/` | FPL (CVPR 2023) | `models/fpl.py`, `models/fedproto.py` | 原型聚类、对比损失参考 |
| `FedPLVM/` | FedPLVM (NeurIPS 2024) | `utils/update.py` | α-稀疏损失、InfoNCE实现 |

---

## 七、实验标准配置

**数据集**：
- Digit-5: MNIST, SVHN, USPS, SynthDigits, MNIST-M（5个域，10类）
- Office-Caltech10: Amazon, Webcam, DSLR, Caltech（4个域，10类）
- PACS: Photo, Art, Cartoon, Sketch（4个域，7类）
- DomainNet（可选）: 6个域，345类子集

**基线方法**：FedAvg, FedProx, FedBN, FedProto, FPL, FedPLVM, FDSE, FedPall, MOON, FedALA

**评估指标**：各域Top-1准确率、平均准确率、域间方差

**骨干网络**：ResNet-18 (with FedBN原则保留本地BN)

---

---

## 八、Agent深度分析补充（关键实验数据与细节）

### 8.1 FDSE 实验关键数据
| 方法 | DomainNet | Office-Caltech10 | PACS |
|------|-----------|-------------------|------|
| FedAvg | 69.17 | 82.60 | 74.30 |
| FedBN | 74.75 | 83.08 | 81.58 |
| Ditto | 75.18 | 84.12 | 82.02 |
| **FDSE** | **76.77** | **87.15** | **83.81** |

- FDSE参数量仅0.65×10^7（基线1.30×10^7），通信量减半，FLOPs减半
- 层分解中DSE模块参数量仅为DFE的1/94
- 服务器端QP求解有计算开销（使用cvxopt，需重定向stdout抑制输出）
- 未见域泛化：Office-Caltech10上75.52%（FedBN仅66.60%）

### 8.2 FedPall 实验关键数据
| 数据集 | FedPall | 第二名 |
|--------|---------|--------|
| Office-10 | **67.5%** | ADCOL 64.5% |
| Digits | **88.7%** | FedBN 87.6% |
| PACS | **60.6%** | FedBN 59.5% |

- 对抗+协作双机制：Amplifier(MLP)放大异构信息，客户端用KL散度减小异构信息
- 原型混合+Bernoulli mask隐私保护：互信息2.10（优于高斯噪声2.96）
- 通信开销仅FedAvg的10.6%
- **局限**：假设所有客户端相同类别，仅4-5个客户端

### 8.3 FedProto 关键细节
- 通信参数量对比：MNIST上 4K vs FedAvg 430K（降低100倍+）
- 推理方式：最近原型分类（非神经网络分类器）
- 提供了非凸条件下的收敛性证明

### 8.4 FedPLVM 关键细节
- Digit-5困难域SVHN：42.08%（FPL仅36.78%，+5.3%）
- 双层聚类vs全局聚类vs均值：69.48 > 66.40 > 63.54
- α-sparsity最优α=0.25，最优τ=0.07
- 全局聚类将原型数量压缩至1/5

### 8.5 FedSTAR 关键细节
- 内容-风格分解方式：投影到全局原型方向=内容，正交残差=风格
- StyleFiLM: `h' = h * (1 + γ(s)) + β(s)`
- 骨干网络MobileNetV3，100客户端
- **关键限制**：DomainNet仅16.03%、CIFAR-100仅29.17%（绝对精度低）

### 8.6 DCFL 理论贡献
- 证明标准对比损失有限样本误差O(M^{-1/2})在FL小数据集下不可忽略
- 最优超参：λ_a=0.9, λ_u=0.1（对齐力远大于排斥力）
- λ_u过大导致训练崩溃（α=0.3时DCFL-SW降至20.18%）
- 与服务器端方法正交，可灵活组合

### 8.7 FedFSL-CFRD 关键细节
- 双层解耦消融：Basic 51.32% → +Client-Wise 51.96% → +Class-Wise 53.29% → CFRD 56.66%
- 全局偏差特征：初始化阶段聚合后冻结，后续不更新
- 基于ProtoNet + ResNet-12

### 8.8 I2PFL 关键细节
- 原型重加权：距均值越远权重越大 → 缓解主导域偏置
- EMA平滑更新广义原型（β=0.99）
- Office-10上提升最大：72.84%（FedPLVM 67.85%，+4.99%）

---

## 九、技术脉络总结图

```
联邦学习跨域问题技术演进路线
═══════════════════════════════

[基础方法层]
FedAvg → FedProx → SCAFFOLD → MOON
  │         │         │         │
  └─────────┴─────────┴─────────┘
              问题：无法处理域偏移
                    ↓
[原型学习层] ←── FedBN(保留本地BN统计量)
FedProto(单原型+MSE) → FPL(FINCH聚类+InfoNCE)
   │                        │
   ├→ MP-FedCL(k-means多原型)  ├→ FedPLVM(双层聚类+α-sparsity)
   │                        │
   └→ I2PFL(域内MixUp+域间重加权)
              问题：特征纠缠（语义与风格混合）
                    ↓
[特征解耦层]
FDSE(层分解DFE+DSE)   FedPall(对抗+协作)
FedFSL-CFRD(双层重构)  FedSTAR(FiLM风格分离)
              问题：风格仅被擦除或保留本地
                    ↓
[你的方案 ★]
正交双头解耦 + 全局风格仓库(风格资产化) + 语义软对齐
```

---

## 十、与开题方案最相关的对比方法优先级

| 优先级 | 方法 | 原因 |
|--------|------|------|
| ★★★ | FedSTAR | 最接近竞争者：也做内容-风格分离，但风格仅本地 |
| ★★★ | FDSE | 层分解思路可借鉴，差异化聚合可参考 |
| ★★★ | FedPLVM | 同赛道：多原型+对比损失，实验设置可复用 |
| ★★☆ | FedPall | 对抗+协作机制参考，原型混合增强参考 |
| ★★☆ | FPL | 基础对比方法，必须作为基线 |
| ★★☆ | I2PFL | 域内+域间双视角可借鉴 |
| ★☆☆ | FedProto | 基础基线 |
| ★☆☆ | DCFL | 对比损失设计参考 |
| ★☆☆ | FedAPA | 聚合策略参考 |

*文档创建时间：2026-03-31*
*最后更新：2026-04-01（含新文献调研+方案精炼+代码实现）*

---

## 十一、2025-2026新发现论文补充分析

### 11.1 MPFT — 多域原型联邦微调 (ICLR 2025) ★★★
- **方法**：预训练模型(CLIP)+多域原型微调，每客户端生成domain-specific原型上传构成原型训练数据集
- **创新**：单轮通信收敛，差分隐私保护原型
- **与我们的关联**：预训练+原型是全新范式，但我们走ResNet-18从头训练路线
- **来源**：https://openreview.net/forum?id=3wEGdrV5Cb

### 11.2 FedSDAF — 源域感知联邦域泛化 (arXiv 2025) ★★★
- **方法**：双适配器(域感知+域不变)+双向知识蒸馏
- **创新**：发现完整源域学习的特征比直接从目标域学的泛化更好
- **与我们的差异**：它用适配器而非原型头，不做风格共享
- **代码**：https://github.com/pizzareapers/FedSDAF

### 11.3 FedCPD — 原型增强+记忆蒸馏 (IJCAI 2025) ★★☆
- **方法**：特征蒸馏防遗忘+原型学习增强类区分
- **效果**：泛化提升10.4%，个性化提升4.9%

### 11.4 FedCCRL — 跨客户端表示学习 (arXiv 2024) ★★☆
- **方法**：MixStyle跨域迁移+AugMix扰动增加域多样性
- **与我们的差异**：在纠缠(混合)特征空间做MixStyle，不解耦
- **代码**：https://github.com/SanphouWang/FedCCRL

### 11.5 FedAlign — 跨客户端特征对齐 (CVPR Workshop 2025) ★★☆
- **方法**：双阶段嵌入+预测对齐，轻量隐私保护

### 11.6 HPFL — 热插拔联邦学习 (ICLR 2025) ★★☆
- **方法**：骨干+插件市场+动态选择，差分隐私保护
- **概念启发**：风格仓库可类比为"插件市场"

### 11.7 FedDG-SFD — 风格特征调度器 (ICONIP 2024) ★★★
- **方法**：风格特征调度器增强跨域知识蒸馏
- **与我们的差异**：它有调度但不做显式内容/风格解耦

### 11.8 FISC/PARDON — 插值风格迁移+对比学习 (ICDCS 2025) ★★★
- **方法**：FINCH聚类提取各客户端风格统计量(μ,σ)→服务器FINCH再聚类→取中位数为插值风格→客户端用AdaIN做风格迁移+三元组对比学习
- **架构**：ResNet50骨干+VGG19编码器(relu4-1提取风格)+解码器(AdaIN风格迁移)
- **损失**：L_CE + 0.75*L_triplet + 0.2*L_reg
- **数据集**：PACS, OfficeHome, FEMNIST, IWildCam
- **与我们的核心差异**：FISC共享风格但**不做显式内容/风格解耦**——在混合特征空间做风格迁移。我们先解耦再共享，风格操作在纯净空间进行
- **代码**：https://github.com/judydnguyen/pardon-feddg
- **源码位置**：`PARDON-FedDG/`

### 11.9 StyleDDG — 去中心化风格共享 (arXiv 2025) ★★☆
- **方法**：P2P网络风格共享+风格探索策略+共识聚合
- **理论**：首个风格共享DG的收敛保证O(1/√K)
- **与我们的差异**：去中心化且不解耦

### 11.10 FedSeProto — 联邦语义原型学习 (ECAI 2024) ★★★
- **方法**：互信息最小化分离语义/域特征+知识蒸馏+语义原型对齐
- **作者**：Yanyi Lai, Lele Fu, Chuan Chen (中山大学)
- **与我们的核心差异**：MI分离后**丢弃域特征**，我们保留并共享。对域信息的态度完全相反
- **来源**：https://ebooks.iospress.nl/doi/10.3233/FAIA240731

### 11.11 FedDP — 联邦域无关原型学习 (IEEE TMC 2025) ★★★
- **方法**：信息瓶颈消除域特定信息+全局域无关原型+表示/参数空间双对齐
- **作者**：同FedSeProto课题组(Lele Fu等)
- **与我们的核心差异**：信息瓶颈**压缩掉**域信息，我们用双头**保留**并资产化

### 11.12 FedSA/FedLSA — 可学习语义锚点 (AAAI 2025) ★★☆
- **方法**：预定义类锚点通过可训练嵌入层投射到语义空间+margin增强对比+分类器校准
- **作者**：同FedSeProto课题组
- **效果**：CIFAR-10 90.88%, CIFAR-100 54.39%

### 11.13 FediOS — 正交子空间解耦 (ML 2025) ★★☆
- **方法**：固定正交投影将特征分离到通用和个性化子空间
- **与我们的差异**：分的是generic/personalized而非semantic/style，且不做风格共享

### 11.14 PDKD — 原型分解知识蒸馏 (IEEE TMM 2024) ★☆☆
- **方法**：SVD分解聚合后的原型→判别性原型+泛化性原型
- **与我们的差异**：后端SVD分解 vs 我们的前端双头分解

### 11.15 SCFlow — 流模型风格内容解耦 (ICCV 2025) ★☆☆
- **方法**：Flow Matching做可逆的风格-内容合并/分离，只训练合并任务即可获得分离能力
- **启发**：最前沿的解耦方法，但计算开销太大不适合FL

---

## 十二、竞争者四派分类（核心定位依据）

```
对域/风格特征的态度分类：

"擦除派"（域特征是噪声，要消除）
  ├── FDSE (CVPR 2025): 层分解+DSE擦除域偏移
  ├── FedSeProto (ECAI 2024): MI分离后丢弃域特征
  ├── FedDP (TMC 2025): 信息瓶颈压缩掉域信息
  └── FediOS (ML 2025): 正交投影到通用子空间

"私有派"（域特征有用但不能共享）
  ├── FedSTAR: FiLM分离后风格仅本地
  ├── FedBN: BN统计量仅本地
  └── FedSDAF: 域感知适配器仅本地

"共享但不解耦派"（风格可共享但在混合空间操作）
  ├── FISC/PARDON (ICDCS 2025): 插值风格迁移（不解耦）
  ├── StyleDDG: 去中心化风格共享（不解耦）
  ├── FedCCRL: MixStyle（不解耦）
  └── CCST (WACV 2023): AdaIN图像空间迁移

★ 我们的方案："解耦+资产化派"（首次！）
  └── 显式双头解耦 + 风格保留 + 全局风格仓库共享 + 语义纯净对齐
```

### 2×2差异化矩阵

|  | 不共享风格 | 共享风格 |
|--|-----------|---------|
| **不解耦** | FedBN, FedAvg, FedProto | FISC, StyleDDG, FedCCRL |
| **解耦** | FedSTAR, FedSeProto, FDSE | **★ FedDSA（我们，首次）** |

---

## 十三、当前方案：FedDSA (Decouple-Share-Align)

### 13.1 核心贡献
首次在联邦原型学习中，将解耦后的风格特征视为可共享的数据资产进行跨域增强，而非作为噪声擦除或私有保留。

### 13.2 三步机制

| 步骤 | 模块 | 作用 |
|------|------|------|
| **Decouple** | 正交约束cos²+HSIC核独立性 | 线性+非线性分离语义与风格 |
| **Share** | 全局风格仓库(μ,σ)+AdaIN增强 | 风格作为跨域增强资产共享 |
| **Align** | InfoNCE对比损失 | 语义特征与全局原型软对齐 |

### 13.3 损失函数
```
L_total = L_CE(z_sem) + L_CE(z_sem_aug) + λ_orth*L_orth + λ_hsic*L_HSIC + λ_sem*L_InfoNCE
```

### 13.4 差异化聚合
- 骨干conv层 + 语义头 + 语义分类器 → FedAvg聚合
- 风格头 → 不聚合（私有）
- BN层 → 不聚合（FedBN原则）

### 13.5 GPT-5.4评审评分
- 方案精炼：6.3/10 → 7.4/10 → **7.8/10 (READY)**
- 严苛审稿：4→5/10（需实验证据提升）

---

## 十四、代码实现状态

### 14.1 已实现
| 文件 | 内容 | 状态 |
|------|------|------|
| `PFLlib/system/flcore/clients/clientdsa.py` | 客户端：双头解耦+增强+对齐 | ✅ 已实现+review修复 |
| `PFLlib/system/flcore/servers/serverdsa.py` | 服务器：差异化聚合+风格仓库 | ✅ 已实现+review修复 |
| `PFLlib/system/main.py` | 注册FedDSA+超参数 | ✅ 已注册 |
| `PFLlib/dataset/generate_PACS.py` | PACS数据集生成 | ✅ 已实现 |

### 14.2 Code Review修复（GPT-5.4审查）
- [CRITICAL] 任务损失改为通过semantic_head→sem_classifier路径
- [CRITICAL] BN参数本地化（FedBN式set_parameters）
- [MAJOR] LR scheduler重绑定、原型加权聚合、HSIC数值稳定性、在线特征累加、feat_dim自动推断、active_clients过滤
- [MINOR] 解耦权重独立化、InfoNCE向量化

---

## 十五、关键源码参考（更新版）

| 代码仓库 | 对应论文 | 核心文件 | 用途 |
|----------|---------|---------|------|
| `PFLlib/` | 联邦学习通用框架 | `system/flcore/clients/clientproto.py` | 实验框架基础，FedDSA基于此开发 |
| `FDSE_CVPR25/` | FDSE (CVPR 2025) | `algorithm/fdse.py` | 层分解、差异化聚合参考 |
| `RethinkFL/` | FPL (CVPR 2023) | `models/fpl.py` | 原型聚类(FINCH)、InfoNCE实现参考 |
| `FedPLVM/` | FedPLVM (NeurIPS 2024) | `utils/update.py` | α-稀疏损失、InfoNCE实现参考 |
| `PARDON-FedDG/` | FISC/PARDON (ICDCS 2025) | `src/client.py`, `src/server.py` | 风格共享基线，FINCH聚类+AdaIN参考 |

### PARDON-FedDG 核心代码结构
```
PARDON-FedDG/
├── main.py                 # 入口：加载配置+数据+初始化客户端/服务器
├── src/
│   ├── server.py          # DGServer: FINCH聚类风格→中位数插值→FedAvg聚合
│   ├── client.py          # DGClient: VGG提取风格→AdaIN迁移→三元组对比训练
│   ├── models.py          # ResNet50骨干+线性分类器
│   ├── functions.py       # adaIN_StyleStat_ContentFeat() 核心AdaIN算子
│   ├── adain_net.py       # 预训练VGG编码器+解码器
│   ├── finch.py           # FINCH层次聚类
│   └── datasets.py        # PACS/OfficeHome/FEMNIST数据集
├── config/FISC/PACS/      # PACS实验配置
└── style_stats/           # 缓存的风格统计量
```

### PARDON vs FedDSA 关键对比
| 维度 | PARDON (FISC) | FedDSA (我们) |
|------|--------------|--------------|
| 解耦 | 不解耦，混合空间操作 | 正交+HSIC双头解耦 |
| 风格提取 | VGG relu4-1特征+FINCH聚类 | 骨干pooled特征(μ,σ) |
| 风格共享 | FINCH聚类→中位数→单一插值风格 | 风格仓库→多风格随机dispatch |
| 增强方式 | VGG编码→AdaIN→解码回图像空间 | 特征空间直接AdaIN |
| 对比损失 | 三元组损失(margin=0.3) | InfoNCE(温度τ=0.1) |
| 骨干 | ResNet50 (2048d) | ResNet-18 (512d) |
| 额外网络 | 需要预训练VGG+解码器 | 不需要额外网络 |

---

## 十六、实验计划基线

### 必比基线（主表）
| 方法 | 类型 | 代码来源 |
|------|------|---------|
| FedAvg | 基础 | PFLlib |
| FedProx | 基础正则化 | PFLlib |
| FedBN | 域适应(BN本地) | PFLlib |
| FedProto | 原型基础 | PFLlib |
| FPL | 原型聚类+InfoNCE | RethinkFL |
| FedPLVM | 多层聚类+α-sparsity | FedPLVM |
| FDSE | 擦除派代表 | FDSE_CVPR25 |

### 差异化关键基线
| 方法 | 2×2位置 | 代码 |
|------|---------|------|
| FISC/PARDON | 共享但不解耦 | PARDON-FedDG |
| FDSE | 解耦但不共享(擦除) | FDSE_CVPR25 |

### 可选加分基线
FedPall, MOON, FedSTAR(需复现), FedSeProto(代码不公开)

---

## 十七、开发与同步工作流

### 17.1 环境架构

```
Windows 本地 (D:\桌面文件\联邦学习\)
  ├── Claude Code / Cursor 编辑代码
  ├── Git 仓库 → https://github.com/Lingrongye/federated-learning
  └── WSL (Ubuntu 20.04)
       └── rr (Road Runner) → rsync + SSH 同步执行
                ↓
服务器 (lab-lry: 222.201.145.9:22, user lry)
  ├── GPU: 双卡 RTX 3090 (各24GB)
  ├── 项目目录: /home/lry/code/federated-learning
  └── 共享服务器（其他用户: wjc, syy, tfs）
```

### 17.2 rr 配置

- **全局配置**: WSL `~/.rr/config.yaml`
- **项目配置**: `.rr.yaml`（定义 host、remote_dir、exclude 规则）
- **SSH 配置**: WSL `~/.ssh/config` → Host lab-lry, 密钥 `~/.ssh/id_ed25519`

### 17.3 日常工作流命令（WSL 终端执行）

```bash
cd "/mnt/d/桌面文件/联邦学习"

# 推送代码到服务器并执行命令
rr run "cd PFLlib/system && CUDA_VISIBLE_DEVICES=1 python main.py -algo FedDSA -data PACS ..."

# 只推送文件不执行
rr sync

# 只执行命令不推送
rr exec "nvidia-smi"

# 执行并拉回结果文件
rr run "cd PFLlib/system && python main.py ..." --pull experiments/results/metrics.json

# 从服务器拉文件到本地（排除数据集等大文件）
bash sync_pull.sh

# 拉之前预览
bash sync_pull.sh --dry-run
```

### 17.4 同步排除规则

以下目录/文件在 `.rr.yaml`、`sync_pull.sh`、`.gitignore` 三处保持一致排除：

| 排除项 | 原因 |
|--------|------|
| `.git` | 避免 git 仓库冲突 |
| `无关文件/` | 周报、文档等无关内容 |
| `Qwen3-VL-4B-mlc/`, `mlc-qwen3-vl/` | 模型权重（2.4GB） |
| `papers/` | PDF论文（193MB） |
| `PFLlib/dataset/MNIST/`, `PFLlib/dataset/utils/LEAF/` | 数据集原始数据 |
| `PFLlib/dataset/*/rawdata,train,test` | 各数据集生成的数据文件 |
| `RethinkFL/data/` | RethinkFL数据（244MB） |
| `PARDON-FedDG/style_stats/` | 风格统计缓存 |
| `__pycache__/`, `*.pyc` | Python 缓存 |
| `wandb/` | 实验追踪日志 |

**注意：以下目录需要同步，不排除：**
- `*.log` — 训练日志
- `checkpoints/` — 模型检查点
- `runs/` — TensorBoard 运行记录
- `experiments/` — 实验记录

### 17.5 换行符规则

`.gitattributes` 统一所有文本文件为 LF（`* text=auto eol=lf`），避免 Windows CRLF 与 Linux LF 差异导致 rsync 同步后产生假修改。

### 17.6 GPU 使用注意

- 共享服务器，先用 `rr exec "nvidia-smi"` 检查 GPU 占用
- 指定空闲卡：`CUDA_VISIBLE_DEVICES=0` 或 `CUDA_VISIBLE_DEVICES=1`
- 小实验 batch_size 调小防止 OOM
