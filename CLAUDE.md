# 面向跨域联邦学习的解耦原型学习 — 研究综述与方向指南

## 零零、给数据的硬性规则 (强制,任何报告/对话都适用)

**任何时候给用户数据 / 表格 / 数字之前,必须先解释 3 件事:**

1. **这个数字是怎么算的** (用什么 input、用什么公式、用了多少 round/seed)
2. **这个数字代表什么含义** (大值 = 好还是坏?边界值是什么?)
3. **这个数字的边界 / 阈值** (range 5-10 = 健康,>10 = 异常,< 1 = 没意义,等等)

**禁止做法:**
- ❌ 直接甩一张表格不解释列名 / 单位
- ❌ 用专业术语 (CKA / LOO cos sim / dispatch ratio / silhouette / drift) 不先翻译
- ❌ 给"range = 0.02 ❌"但不告诉用户"这个 range 是什么变量算出来的"

**正确做法:**
- ✅ "X 是 daa_freqs_i / sample_shares_i 算出来的(每个 client 一个数),代表 DaA 给这个 client 升降权多少"
- ✅ "range = 10 个 client 数字的 max − min,大 = 区分对待,小 = 几乎 uniform"
- ✅ "Office range 0.2 = 健康差异化,PACS range 0.02 = 退化成 FedAvg"
- ✅ 表格只放最重要的 3-5 个对比维度,其他细节省略或放折叠区

**指标速查表 (常用诊断指标的解释,任何报告中第一次出现都要点出来):**

| 指标 | 怎么算 | 大值 | 小值 | 健康范围 |
|------|--------|:--:|:--:|:--:|
| **AVG Best** | 100 轮中 4 域简单平均最大的那一轮的值 | 高=好 | — | 跟 FDSE 阈值比 |
| **AVG Last** | R100 的 4 域简单平均 | 高=好 | — | 跟 Best gap < 3 = 稳定 |
| **best→last drift** | last - best (per domain) | 接近 0 = 稳 | 大负 = 后期飘 | gap > 5 = 不健康 |
| **dispatch ratio** | daa_freqs_i / sample_shares_i (每 client 一个) | >1 升权 | <1 降权 | 看 range |
| **dispatch range** | 10 client ratio 的 max−min | 大 = 差异化 | 小 = uniform | 0.1+ 健康, < 0.05 失效 |
| **effective contribution** | daa_freqs × grad_l2 (per client) | 高 = 这 client 真发声 | 低 = 被 wash | 看 4 域是不是 uniform |
| **LOO cos sim** | client_proto vs leave-one-out 9 client 平均的 cos | >0.85 同化 | <0.5 outlier | **0.5-0.85 健康, >0.95 创新失效** |
| **CKA cross-method** | 两个 method 的 feature matrix 的 linear CKA | >0.85 殊途同归 | <0.5 学得不一样 | **<0.7 才能拉开差距** |
| **per-domain trajectory std** | 后 30 round per-domain acc 的 std | 大 = 训练晃动 | 小 = 稳 | < 3 稳定, > 5 不稳 |

**适用范围**: 任何 cold path 诊断、obsidian 笔记、对话式实验汇报。

---

## 零、实验胜负判定硬性要求 (强制,最高优先级)

**实验成功的唯一标准**: 3-seed {2,15,333} mean AVG Best accuracy **必须超过 FDSE 本地复现 baseline** (不是论文数字,而是我们 env 下同 seed 同 config 的真实复现):

| Dataset | FDSE paper | **FDSE 本地复现** | **必须达到** |
|---------|:---------:|:----------------:|:-----------:|
| **PACS** AVG Best | 82.17 | **79.91** | 3-seed mean > 79.91 |
| **Office-Caltech10** AVG Best | 91.58 | **90.58** | 3-seed mean > 90.58 |

**当前进展** (截至 2026-04-21, 3-seed {2,15,333} R200):

| Metric | orth_only | FDSE | Δ |
|--------|:--------:|:----:|:--:|
| PACS AVG Best | **80.64** | 79.91 | ✅ **+0.73** |
| PACS AVG Last | **79.98** | 77.55 | ✅ **+2.43** |
| Office AVG Best | 89.09 | **90.58** | ❌ **-1.49** |
| Office AVG Last | 87.32 | **89.22** | ❌ **-1.90** |

**战场判决**:
- **PACS 全面领先**: Best/Last 都赢,VIB 只需保持不退即可
- **Office 双指标落后**: Best -1.49, Last -1.90, **必须涨至少 1.5-2pp**
- **真正的攻坚是 Office** (Best + Last 都要攻)

### 硬性禁止 (不得违反,违反即失败)

1. ❌ **禁止换数据集**: 必须 PACS + Office-Caltech10,不得因为打不过 FDSE 就 pivot 到 FEMNIST / Rotated MNIST / Camelyon17 等"容易打"的场景
2. ❌ **禁止改 paper 叙事为"诊断论文"**: 必须是 accuracy 直接胜 FDSE 的 method paper,不接受 "FDSE 没做严格评估所以我们赢"
3. ❌ **禁止用 non-inferior / 持平叙事**: 不接受 "accuracy 不差 FDSE 太多 + probe 好" 的妥协定位
4. ❌ **禁止"有创新就算赢"**: Novelty 是前提,**accuracy 数字超 FDSE 才是胜利**

### 方案迭代判决

- **3-seed mean accuracy 没达到 FDSE 阈值** → **立即换方法**,不浪费 GPU 在小幅 incremental 优化
- **3-seed mean accuracy ≥ FDSE 阈值** → 进入 paper 阶段,把 probe / 诊断 / novelty 作为 **bonus contribution**
- **PACS 过但 Office 没过** (或反之) → 视情况,但**两个都要过才算完全胜利**

### Probe 和诊断的定位

- ✅ **Paper bonus 贡献**: 作为 "method 胜 FDSE 同时更严格评估"
- ❌ **绝不作主卖点**: 如果 accuracy 没胜,probe/诊断救不回来

### 当前 baseline 差距 (截至 2026-04-21)

| 当前状态 | PACS | Office |
|---------|:----:|:------:|
| orth_only 3-seed mean | 80.64 | 89.09 |
| FDSE 阈值 | 82.17 | 91.58 |
| **必须涨** | **+1.53** | **+2.49** |

任何新方案 proposal 必须预期 / 实测达到上表涨幅,否则立刻 kill。

---

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

> **⚠️ 论文同时报 ALL(加权)和 AVG(简单平均)两个指标。下表为 ALL 指标。**

| 方法 | DomainNet ALL | Office-Caltech10 ALL | PACS ALL |
|------|-----------|-------------------|------|
| FedAvg | 69.17 | 82.60 | 74.30 |
| FedBN | 74.75 | 83.08 | 81.58 |
| Ditto | 75.18 | 84.12 | 82.02 |
| **FDSE** | **76.77** | **87.15** | **83.81** |

| 方法 | DomainNet AVG | Office-Caltech10 AVG | PACS AVG |
|------|-----------|-------------------|------|
| FedAvg | 67.53 | 86.26 | 72.10 |
| FedBN | 72.25 | 87.01 | 79.47 |
| Ditto | 72.82 | 88.72 | 80.03 |
| **FDSE** | **74.50** | **91.58** | **82.17** |

> 注意:Office-Caltech10 上 AVG >> ALL,因为 DSLR 域样本极少(157)但准确率高,在 AVG(等权)中贡献大,在 ALL(按量加权)中被稀释。

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

### 17.0 本地 Python 环境

**本地 Python（Windows）**: `D:/anaconda/python.exe`（conda base，已装 pdfplumber/pypdf/pandas/numpy 等常用包）
- 读 PDF/处理数据/本地脚本统一用此路径
- 其他 conda env: `D:\anaconda\envs\{pfllib, pytorch, py38, py311, ...}`

### 17.0.1 Git commit message 规则（强制）

**所有 `git commit -m` 信息必须用中文书写**。

- ✅ 正确: `git commit -m "结果: Office 补 s=15，orth_only 3-seed AVG 88.64 vs FDSE 90.58"`
- ❌ 错误: `git commit -m "results: Office add s=15, orth_only AVG 88.64"`

例外：前缀可用英文约定词（`config:` / `results:` / `docs:` / `fix:` / `feat:`），但主体描述必须中文。例如：
- `git commit -m "config: EXP-080 正交超参扫（L_orth + HSIC + LR 变体）"`
- `git commit -m "fix: 修正 orth_only R200 的 s=333 崩盘误判"`

**HEREDOC 多行 commit 也必须中文**：
```bash
git commit -m "$(cat <<'EOF'
修正: R200 full 数据推翻 R181 快照结论

- orth_only 3-seed mean last 73.87（非之前以为的 80.7）
- bell_60_30 才是 PACS 最稳方案 (last 79.29)
- s=333 在 R200 时崩到 65.34%
EOF
)"
```

### 17.1 环境架构

```
Windows 本地 (D:\桌面文件\联邦学习\)
  ├── Claude Code / Cursor 编辑代码
  ├── Git 仓库 → https://github.com/Lingrongye/federated-learning
  └── WSL (Ubuntu 20.04)
       └── SSH 直连各服务器
                ↓
服务器集群:
  ├── seetacloud (主力): AutoDL 按量 GPU
  │   ├── SSH: Host seetacloud (WSL ~/.ssh/config)
  │   ├── 项目: /root/autodl-tmp/federated-learning
  │   ├── Python: /root/miniconda3/bin/python
  │   └── GitHub 代理: source /root/clashctl/scripts/cmd/clashctl.sh && clashctl on
  │
  ├── seetacloud2 (辅助): AutoDL 4090 24GB
  │   ├── SSH: Host seetacloud2 (port 19385, connect.westb.seetacloud.com)
  │   ├── 项目: /root/autodl-tmp/federated-learning (克隆实例)
  │   ├── Python: /root/miniconda3/bin/python
  │   └── GitHub 代理: 同上
  │
  ├── lab-lry (备用): 实验室共享服务器
  │   ├── SSH: Host lab-lry (222.201.145.9:22, user lry)
  │   ├── GPU: 双卡 RTX 3090 (各24GB, 常被 wjc 占用)
  │   └── 项目: /home/lry/code/federated-learning
  │
  └── scut-hpc (校级集群, 详见 17.9): SCUT HPC hpckapok1
      ├── SSH: 202230040034@202.38.252.202 (login01-04 任选)
      ├── GPU: gpuA800 队列 30 节点 × 4× A800-SXM4-80GB
      ├── 调度器: Slurm 22.05.10 (必须 srun/sbatch, 不能直跑)
      └── 项目: ~/projects/federated-learning (/share/home/202230040034/)
```

### 17.2 代码同步方式：Git 双向同步

**唯一同步方式是 Git commit + push/pull。**

```
[本地编辑代码] → git commit → git push → GitHub
                                            ↓
[服务器] ← git pull ← ← ← ← ← ← ← ← ← ←┘
    ↓
[服务器跑实验，产生结果]
    ↓
[服务器] → git add → git commit → git push → GitHub
                                                ↓
[本地] ← git pull ← ← ← ← ← ← ← ← ← ← ← ←┘
```

### 17.3 在服务器上执行命令

**Claude Code 通过 WSL SSH 直连服务器执行命令（不使用 rr）。**

```bash
# 执行方式
wsl bash -lc "ssh seetacloud '<命令>'"
wsl bash -lc "ssh seetacloud2 '<命令>'"

# 执行脚本（推荐，避免引号转义问题）
wsl bash -lc 'ssh seetacloud bash < "/mnt/d/桌面文件/联邦学习/rr/脚本.sh"'

# 常用命令
wsl bash -lc "ssh seetacloud 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'"
wsl bash -lc "ssh seetacloud 'ps -eo pid,etime,cmd | grep run_single | grep -v grep'"
wsl bash -lc "ssh seetacloud 'cd /root/autodl-tmp/federated-learning && git log --oneline -3'"
```

### 17.4 跑实验完整流程

**强制规则：本地 commit 后必须在服务器 git pull，确认同步后才能启动实验。**
**强制规则：新增/修改算法代码后，必须先经过以下验证才能启动实验。**

```
0. 本地: 写代码 → python -c "import ast; ast.parse(...)" 语法检查
1. 本地: 写单元测试 (test_*.py) → 验证梯度流/数据流/边界情况 → ALL PASS
2. 本地: codex exec 代码审查 → 确认无 bug/设计缺陷 → 修复所有 Important+ issues
3. 本地: 写 config + NOTE.md → git commit → git push
4. 服务器: 开代理 → git pull （必须确认 Fast-forward 成功）
5. 服务器: 检查 GPU 空闲 (nvidia-smi)
6. 服务器: nohup 后台启动实验
7. 定期: 通过 log 文件检查进度
8. 实验完成: 收集结果到 EXP 目录 (collect_results.py)
9. 服务器: git add → git commit → git push
10. 本地: git pull → 更新 NOTE.md 回填结果
```

**实验前验证清单 (steps 0-2)**:
- [ ] `ast.parse()` 语法检查通过
- [ ] 单元测试覆盖: 模型前向/损失计算/梯度流向/边界情况/数值稳定性
- [ ] Codex 代码审查: 无 Critical/Important 未修复的 issue
- [ ] Config 参数数量和顺序与 algo_para 解析一致

**seetacloud 服务器 GitHub 代理**：
```bash
source /root/clashctl/scripts/cmd/clashctl.sh && clashctl on > /dev/null 2>&1
cd /root/autodl-tmp/federated-learning && git pull --no-rebase origin main
```

### 17.4.1 实验输出结构

**flgo 框架默认输出**（不可改）：
- JSON: `FDSE_CVPR25/task/{TASK}/record/{algo}_{params}_{seed}.json`
- LOG: `FDSE_CVPR25/task/{TASK}/log/{timestamp}{algo}_{params}_{seed}.log`

**实验目录结构**（每个 EXP 自包含）：
```
experiments/ablation/EXP-052_lr_grid_search/
├── NOTE.md              ← 实验说明+配置+结论
├── results/             ← 收集的 json (用 collect_results.py)
└── logs/                ← 收集的 log
```

**收集结果命令**（实验完成后执行）：
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
python collect_results.py --exp EXP-052 --task PACS_c4 --algorithm feddsa --seed 2
```

### 17.4.2 启动实验的标准模板

```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
EXP_DIR=../../experiments/ablation/EXP-052_lr_grid_search

# 确保输出目录存在
mkdir -p $EXP_DIR/results $EXP_DIR/logs

# 启动实验，stdout 输出到实验目录
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 \
  --config ./config/pacs/feddsa_lr005.yml --logger PerRunLogger --seed 2 \
  > $EXP_DIR/terminal_s2.log 2>&1 &
```

### 17.5 .gitignore 与大文件排除

**原则：只排除真正的大文件，日志和实验产物都入Git以便双向同步。**

排除（不入Git）：

| 排除项 | 原因 |
|--------|------|
| `无关文件/` | 周报、文档等无关内容 |
| `Qwen3-VL-4B-mlc/`, `mlc-qwen3-vl/` | 模型权重（2.4GB） |
| `papers/` | PDF论文（193MB） |
| `PFLlib/dataset/*/rawdata,train,test` | 各数据集生成的数据文件 |
| `RethinkFL/data/` | RethinkFL数据（244MB） |
| `PARDON-FedDG/style_stats/` | 风格统计缓存 |
| `__pycache__/`, `*.pyc` | Python 缓存 |
| `wandb/` | 实验追踪日志（文件量大） |
| `*.pth`, `*.pt`, `*.ckpt` | 模型检查点文件（单个数百MB） |

需要入Git（可同步）：

| 入Git项 | 原因 |
|---------|------|
| `experiments/` | 实验配置、笔记、metrics.json、terminal.log |
| `*.log` | 训练日志，需要读取分析 |
| `runs/` | TensorBoard记录 |
| `checkpoints/`目录结构 | 保留目录，模型文件被*.pth排除 |

### 17.6 换行符规则

`.gitattributes` 统一所有文本文件为 LF（`* text=auto eol=lf`），避免 Windows CRLF 与 Linux LF 差异导致 Git 产生假修改。

### 17.7 GPU 使用注意

- 共享服务器，先用 `rr exec "nvidia-smi"` 检查 GPU 占用
- 指定空闲卡：`CUDA_VISIBLE_DEVICES=0` 或 `CUDA_VISIBLE_DEVICES=1`
- 小实验 batch_size 调小防止 OOM
- 长时间实验用 `nohup ... &` 后台运行，避免SSH断连中断

### 17.8 GPU 并行原则（强制，最高优先级）

**核心原则**：**只要 GPU 有空闲显存就并行 launch，不要做人为的 wave 串行等待。**

#### 禁止

- ❌ 写 "Wave 1 (6 runs) → wait → Wave 2 (6 runs) → wait → Wave 3 (4 runs) × 3 批" 这种 **chained wait dispatcher**。即使每批的 runs 完成时间接近，后一批 launch 前也会浪费 GPU idle 时间（task 长度不均匀时浪费更严重）。
- ❌ 固定 `MAX_PARALLEL=6` 这种静态常量 — 显存占用取决于 config（E=1 vs E=5、batch size、proto dim）
- ❌ 在 Wave 内用 `wait` 等**所有** runs 完成才进下一批

#### 应该

- ✅ **Greedy scheduler**：按 `nvidia-smi --query-gpu=memory.free` 动态 launch。每完成一个 slot 立即补下一个 task。
- ✅ **显存阈值 `MIN_FREE_MB`** 按 config 估算：Office E=1 ~2.5GB，PACS E=5 ~4.5GB，DomainNet E=5 ~4GB。留 500MB 安全余量。
- ✅ 同一 GPU 可混跑不同数据集（Office + PACS），只要总显存不超 24GB
- ✅ Launch 后 `sleep 15-20s` 让进程 ramp up 完整显存再做下一次 check（避免 nvidia-smi 读到还没分配满的 free memory）

#### 标准 Greedy Launcher 模板

```bash
#!/bin/bash
TASKS=(
    "label1|task|algo|config|seed"
    "label2|task|algo|config|seed"
    ...
)
MIN_FREE_MB=4500  # 按单 run 显存估, 留余量
for task in "${TASKS[@]}"; do
    IFS="|" read -r label t algo config seed <<< "$task"
    while true; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (free=${free_mb}MB)"
            CUDA_VISIBLE_DEVICES=0 $PY run_single.py \
                --task $t --algorithm $algo --gpu 0 \
                --config ./$config --logger PerRunLogger --seed $seed \
                > $LOG/${label}.log 2>&1 &
            sleep 20  # ramp up 再 check 下一个
            break
        fi
        sleep 45
    done
done
wait
```

#### 为什么重要

- 24 runs 串行 4 批 × 7h = **21h wall**
- 24 runs greedy 按显存动态 = **~14-16h wall**（平均 6-8 并行不阻塞）
- **节省 5-7h wall 时间 = 每次实验提前半天出结果**

#### 历史错误

- EXP-119 Wave 3 原本写成 3 批 × 4 runs × 7h = 21h PACS wall。实际 Wave 2 Office 只用 13.5GB/24GB，Wave 3 的 PACS (~4GB each) 完全可以**立刻并行** 2-3 个到 Wave 2 里，不用等 Wave 2 完成。这个错误让实验 wall 多 5h。已改为 greedy launcher。

### 17.9 SCUT HPC 集群 (hpckapok1)

> **校级共享 HPC，跟 AutoDL/seetacloud 不同：必须经 Slurm 调度器申请节点才能跑代码。** 登录节点没 GPU、计算节点没外网，工作流要绕这两个限制。

#### 17.9.1 基本信息

| 项 | 值 |
|---|---|
| 门户 (web) | https://hpckapok1.scut.edu.cn |
| 登录节点 | 202.38.252.202-205 (login01-04, 任选一台) |
| **账号 / 密码** | `202230040034` / 学校密码 |
| **课题组 (Slurm account)** | **`a_csxlzhang`** (扣费从这个组余额扣, 不是个人) |
| 家目录 | `/share/home/202230040034/` (NFS, 9.1PB 共享盘, 暂无单用户配额) |
| OS / 调度器 | Rocky Linux 8.6 / Slurm 22.05.10 |
| 登录认证方式 | **账号密码** (集群 1) |

#### 17.9.2 队列 + 计费 (单价 / 节点配置)

| 队列 | 卡 / 资源 | 节点数 | 单价 | 备注 |
|---|---|:--:|:--:|---|
| **`gpuA800`** | **4× A800-SXM4-80GB** + 36C + 1TB RAM | 30 (10 idle) | **3.80 元/卡·h** + CPU 0.04/核·h | **首选, CUDA 原生** |
| `cpuXeon6458` (默认) | Xeon 6458 64C + 500GB | 316 (194 idle) | 0.04 元/核·h | 准备数据/调试用, 极便宜 |
| `cpuFatSR950` | 192C + 5.9TB RAM | 3 (2 idle) | 0.10 元/核·h | 大内存才用 |
| `cpuHygon7380` | 海光 64C + 257GB | 16 | 0.01 元/核·h | 国产 CPU, **少数库可能编译失败, 慎用** |
| `gpuMi210` | 8× **AMD MI210 64GB ROCm** | 3 | 2.00 元/卡·h | **避开** — ROCm 生态, torch+cu* 不通 |
| `gpuHygonZ100` | 8× 国产 GPU | 5 | 1.00 元/卡·h | **避开** — 生态比 MI210 还差 |

**关键计费规则**:
- 登录节点**完全免费** (写代码 / git / pip install / rsync 全免费, 随便用)
- **`salloc` / `sbatch` 一旦分到节点就开始扣**, 直到作业结束 / `scancel` / `--time` 到点
- 扣费公式: `(GPU 卡数 × GPU 单价 + CPU 核数 × 0.04) × 实际秒数`, **从课题组 `a_csxlzhang` 余额扣**
- 节省: `--cpus-per-task=4` 而不是默认 8, dataloader 4 worker 够用, CPU 部分省一半
- 余额查询: 门户 https://hpckapok1.scut.edu.cn (没有命令行 `mybalance` 之类的工具)

**实测**: g01n03 节点的 A800 driver 版本 = **550.78 (CUDA 12.4)**, **torch 2.1.2+cu121 完全兼容**, capability 8.0 (Ampere)。比 sc3 4090 显存大 3.3 倍 (80GB vs 24GB)。

#### 17.9.3 网络拓扑 (重要 — 决定工作流)

| 方向 | 状态 | 说明 |
|---|---|---|
| HPC 登录节点 → PyPI | ✅ 直通 | `pip install` 不需要代理 |
| HPC 登录节点 → GitHub | ❌ 超时 | **必须开 clash 代理** |
| HPC 登录节点 → AutoDL sc3 (out) | ✅ 通 | HPC 主动 `rsync sc3:...` 拉文件 OK |
| AutoDL → HPC SSH:22 (in) | ❌ 封 | **sc3 不能反向推到 HPC**, 只能 HPC 主动拉 |
| **计算节点** → 任何外网 | ❌ **完全无外网** | 装环境/git/pip **必须在登录节点完成**, 计算节点只能跑代码 |

**铁律**: 所有需要外网的操作 (git pull / pip install / rsync 拉数据) **必须在登录节点做**, 然后再 srun/sbatch 到计算节点跑训练。

#### 17.9.4 已部署环境 (`/share/home/202230040034/`)

```
~/miniconda3/                              # conda 24.1.2 (从清华源装)
~/miniconda3/envs/f2dc/                    # Python 3.10
   ├── torch 2.1.2+cu121 (注意: 不是 sc3 的 cu118, A800 driver 550 兼容)
   ├── numpy 1.26.4 (强制 < 2, 否则 torch ABI 冲突)
   ├── flgo 0.4.3 (sc3 是 0.4.4, 这是 PyPI 版本, 兼容)
   ├── cvxopt 1.3.3, scipy 1.15.3, tqdm 4.64.1
   └── pandas / matplotlib / seaborn / sklearn / pyyaml / networkx
~/clashctl/                                # mihomo 后台运行, port 7890
~/projects/federated-learning/             # shallow clone @ commit c31bfe3
~/projects/federated-learning/FDSE_CVPR25/task/
   ├── PACS_c4/                            # 404M, flgo 格式
   ├── office_caltech10_c4/                # 117M
   └── digit5_c5/                          # 7.4M
~/.proxyrc                                 # source 后启用代理
~/.bashrc                                  # alias proxyon / proxyoff
~/.ssh/config                              # alias `sc3` (HPC → sc3 免密已配)
```

**别名**: `proxyon` 开代理 (export http_proxy + https_proxy), `proxyoff` 关。

#### 17.9.5 操作模板

**登录 (从 Mac/WSL)**:
```bash
ssh 202230040034@202.38.252.202   # 输密码进 login01
# 或者本地 ~/.ssh/config 加 alias:
# Host scut-login01
#     HostName 202.38.252.202
#     User 202230040034
```

**登录后激活环境** (每次新 shell):
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f2dc
proxyon                            # 需要访问 GitHub 时才开
cd ~/projects/federated-learning
```

**git pull (走代理)**:
```bash
proxyon && cd ~/projects/federated-learning && git pull
# shallow clone 后第一次 push 前必须先解 shallow:
# git fetch --unshallow
```

**从 sc3 拉新代码/数据 (不走代理, 已配免密)**:
```bash
rsync -azP sc3:/root/autodl-tmp/federated-learning/F2DC/ ~/projects/federated-learning/F2DC/
```

**交互式拿 GPU 调试 (短时长, 用 srun)**:
```bash
srun -p gpuA800 --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash
# 进去后直接 nvidia-smi, python 等; --time 到点自动释放
```

**批量提交正式实验 (用 sbatch)**:
```bash
# 模板: ~/projects/federated-learning/scripts/scut/train.slurm
#!/bin/bash
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4              # 不要默认 8, 省一半 CPU 费
#SBATCH --mem=32G
#SBATCH --time=08:00:00                 # 必填上限, 到点自动结束防漏跑
#SBATCH --job-name=feddsa_pacs
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate f2dc
cd ~/projects/federated-learning/FDSE_CVPR25
python run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_xxx.yml --logger PerRunLogger --seed 2

# 提交:
sbatch train.slurm                  # 返回 job_id
squeue -u $USER                     # 看自己作业 (PD=pending, R=running)
scancel <job_id>                    # 取消
tail -f logs/<job_id>.out           # 实时看输出
```

**多 config 扫参**:
```bash
for cfg in config/pacs/feddsa_*.yml; do
    for seed in 2 15 333; do
        sbatch --export=ALL,CFG=$cfg,SEED=$seed train.slurm
    done
done
# 数十个 sbatch 一次投, Slurm 自动按 idle 节点排队
```

#### 17.9.6 Slurm 常用命令

```bash
sinfo -p gpuA800 -o "%P %a %t %D %N"    # 队列状态 (idle/alloc/drain)
squeue -u $USER                         # 自己的作业列表
scancel <job_id>                        # 取消单个作业
scancel -u $USER                        # 取消自己所有作业 (慎用)
sshare -U                               # 自己账户的 fair-share 使用
sacct -u $USER --starttime=2026-04-01   # 历史作业 + 实际扣费
```

#### 17.9.7 跟 sc3 比的取舍

| 维度 | sc3 (AutoDL 4090 24GB) | scut-hpc (A800 80GB) |
|---|---|---|
| 单卡价格 | ~1.6-2 元/h | 3.80 元/h + CPU 费 |
| 显存 | 24GB | **80GB** (3.3 倍) |
| 速度 (FP32 ResNet-18) | 略快 (Ada vs Ampere) | 略慢 (~70% 速度) |
| 启动延迟 | 立即 | 排队 0-30 分钟 |
| 并行控制 | greedy launcher 自由 | Slurm 调度, 每作业独占 |
| 出网 | 完全自由 | 计算节点零外网 |
| **适合场景** | 主力 / debug / greedy 扫参 | **大批量 ablation** (一次投十几个 sbatch 让 Slurm 排) / **大模型** (>24GB 显存时) |
| **不适合** | 80GB 模型 | 实时 debug / 需要持续监控 |

**当前策略**: 主力 sc3, **scut-hpc 当 ablation 扫参 + 大显存 fallback** (一次性投多个 sbatch)。**禁止把 scut-hpc 当主力**, 因为排队 + 计算节点无外网拖慢迭代节奏。

#### 17.9.8 经验/坑

- **shallow clone 不能直接 push**: 我们当前 `~/projects/federated-learning/` 是 `git clone --depth 1`, push 前必须先 `git fetch --unshallow` (走代理拉完整 git history)。如果只在 HPC 跑实验、不在 HPC commit, 可以无视。
- **大仓库 git clone 走代理易失败**: 代理切节点会 RPC 中断, 用 `--depth 1` + `git config --global http.postBuffer 1048576000` 才稳。
- **NumPy 必须 < 2.0**: torch 2.1.2 用 NumPy 1.x ABI 编译, 装 NumPy 2.x 会 crash。
- **计算节点 `pip install` 不会通**: 网络全封, 必须登录节点装好。
- **CUDA 版本**: A800 driver 550.78 = CUDA 12.4, **cu121/cu118 都兼容**, 没必要重装 cu118。

---

## 十八、Obsidian 知识库同步工作流

### 18.1 Obsidian 目录结构

```
D:\桌面文件\联邦学习\obsidian_exprtiment_results\
├── 知识笔记/                     ← 概念解释、算法说明、学习笔记
│   ├── 解释InfoNCE对比损失.md
│   ├── Alpha-Sparsity数学推导.md
│   └── ...
├── 2026-04-16/                   ← 按日期的实验记录
│   ├── EXP-076_orth_only.md      ← 从 experiments/ 同步的 NOTE.md
│   ├── EXP-077_mse_anchor.md
│   └── daily_summary.md          ← 每日实验总结
└── 2026-04-17/
    └── ...
```

### 18.2 四个 Skill

| Skill | 触发 | 用途 |
|-------|------|------|
| `/experiment-sync` | 自动 hook + 手动 | NOTE.md → Obsidian 对应日期目录 |
| `/daily-experiment-summary` | "总结实验" | 全服务器实验报告 → daily_summary.md |
| `/save-explanation` | "记笔记" / "保存到笔记" | 解释性内容 → 知识笔记/ |

### 18.2.1 ⚠️ 关键实验发现备忘 — 最重要的规则

**每天的 Obsidian 日期目录下必须维护一个 `关键实验发现备忘.md` 文件。**

规则（极其重要，必须严格遵守）：

1. **追加写入，绝不覆盖** — 每次发现新结论，在文件末尾追加一个新的 `## 发现 N:` 章节
2. **触发条件** — 对话中出现以下任何情况时必须写入：
   - 发现实验结果中的关键趋势（如"cos_sim 穿零"、"某方法崩溃"）
   - 得出新的因果结论（如"InfoNCE 是崩溃元凶"）
   - 文献调研发现关键对比（如"FPL 用了 MSE 锚点我们没有"）
   - 发现之前结论有误（如"EXP-070 协同效应是 peak 幻觉"）
   - 任何可能影响后续方向决策的发现
3. **格式** — 每条发现包含：标题、时间、具体数据、因果分析
4. **位置** — `obsidian_exprtiment_results/{YYYY-MM-DD}/关键实验发现备忘.md`
5. **如果文件不存在则创建，如果存在则在末尾追加**

### 18.2.2 NOTE.md 同步的日期规则

**NOTE.md 必须按内容中的日期（而非今天的日期）分到对应的文件夹。**

- 读取 NOTE.md 内容，找到第一个 `YYYY-MM-DD` 格式的日期
- 以该日期作为 Obsidian 中的文件夹名
- 如果找不到日期，使用文件修改时间
- **不要把所有文件都堆到今天的文件夹下**

### 18.2.3 已做实验总览.md 维护规则（强制）

**文件位置**：`obsidian_exprtiment_results/已做实验总览.md`

**两条硬性规则：**

1. **每次新实验完成后必须追加一条**：在文件末尾（`缺失编号说明` 之前）按 `## EXP-XXX — 标题` 格式追加，格式与文件内现有条目一致。

2. **描述必须自说明，禁止使用内部版本号**：
   - ❌ 禁止：「V4 with seed=15」「FedDSA+ 延后阶段」「基于 V3 配置」
   - ✅ 要求：把实际配置写清楚，任何时候拿起来看都能明白做的是什么，不需要翻别的文档
   - 示例 ✅：「warmup=50 + 全量权重（λ_orth=1.0/λ_hsic=0.1/λ_sem=1.0）配置下换 seed=15 方差检验」
   - 如果实验是在某个基线配置上做变动，直接写出那个基线的关键参数，不要用版本号代替

---

### 18.3 NOTE.md 写作规范

每个实验的 NOTE.md **必须包含以下内容**：

1. **变体通俗解释**：用 2-3 句话解释这个变体做了什么改动、为什么要这么改。
   - 例：M3 = "不把所有域的原型平均成一个，而是保留每个域自己的原型，让 InfoNCE 同时拉近所有同类域的原型"
   - 例：mode 6 = "给 InfoNCE 加两道安全阀：MSE 锚点防止特征飘太远，alpha-sparsity 弱化正例梯度避免跟 CE 打架"
2. **技术细节**：损失公式、关键超参数、与基线的区别
3. **实验配置**：config 文件名、seeds、服务器、GPU
4. **结果**：per-seed 的 max/last/drop + cos_sim（如有）
5. **结论**：一句话判断（有效/无效/部分有效）+ 原因分析
6. 最重要的比如要有`YYYY-MM-DD`的日期说明你是什么时候创建的！！！！非常重要

#### 18.3.1 NOTE.md 语言风格（强制）

**写给自己看的实验笔记，不是给审稿人看的学术文档。**

- ✅ 用大白话描述做了什么：「把分类器也按风格加权」「只个性化特征映射层，分类器还是全局平均」
- ❌ 不要堆学术黑话：~~counterfactual baseline~~ → 「对照实验」；~~thesis falsified~~ → 「假设不成立」；~~routing signal~~ → 「按什么标准分配」
- ✅ 配置表用「xx 怎么聚合」这种白话列名，不要 "style-conditioned aggregation scope"
- ✅ 结论直接说「分类器不该个性化」「τ=0.3 还是最好」，不要包装成 "negative result validates the mechanism hypothesis"
- ✅ 失败实验直接写「❌ 失败」+ 用一两句话说为什么失败，不要写一大段 "this constitutes an important contribution as..."
- ❌ 不要在 NOTE.md 里放论文级别的 claim-driven validation sketch、decision rule 公式、reviewer 评审记录等（这些放 refine-logs）
- ✅ 表格标题用中文（「各域准确率」而非 "per-domain accuracy"），技术术语（sas、FedAvg、sem_head、L_orth 等）保留英文
- ✅ **每个结果表必须带对照行 + Δ 行**：第一行放对照基线（如方案A），第二行放本实验结果，第三行放 Δ（差值）。一眼看出比基线好还是差。不要让读者自己翻到别的表格找对照数字
- ✅ **汇总表必须有全指标**：ALL B/L + AVG B/L + Caltech(或对应 outlier 域) B/L，不能只放一个 AVG Best。3-seed mean 的汇总表每列都要有值
- ✅ **对照行不允许写"—"**：如果之前的实验记录没有某个指标的数据，必须 SSH 到服务器从 JSON record 里提取真实值再填。宁可多花 1 分钟查数据，也不要留空让读者猜

### 18.3.2 知识笔记双版本规范 (强制)

**写到 `obsidian_exprtiment_results/知识笔记/` 下的概念笔记必须有两个版本**:

1. **大白话版** (放在文档前半 or 独立 `大白话_xxx.md` 文件):
   - **禁止**堆学术术语 (如 "Neural Collapse NC2 性质")
   - 用**比喻**: "Fixed ETF 就像老师一开始就定好 10 个标准姿势,学生照着画就行"
   - 用**对比表**: 普通做法 vs 我们做法,一眼看出改了啥
   - 用**一句话记住**: 每个方案尾部加"一句话总结"
   - 假设读者 **6 个月后回来看**,不记得任何上下文
   - 术语出现时**立即括号解释**或拆到下一句讲

2. **学术版** (放在文档后半 or 独立 `xxx方案_xxx.md` 文件):
   - 完整数学公式 (L_orth = cos²(z_sem, z_sty) 这种)
   - 引用 top-venue 论文 (SATA IVC'25 / T3A NeurIPS'21 Spot)
   - Claim/novelty/差异化表述, 准备 reviewer
   - 与 FINAL_PROPOSAL / refine logs 对齐

**触发条件**: 用户说 "解释一下"、"记笔记"、"保存到笔记"、"不要学术"、"大白话" 时,**必须**同时产出两个版本 (或至少标注这个版本的风格)。

**示例**:
- ✅ `obsidian_exprtiment_results/知识笔记/FedDSA-SGPA方案_风格门控原型校正.md` (学术版, 含完整公式)
- ✅ `obsidian_exprtiment_results/知识笔记/大白话_我们所有方案.md` (大白话版, 比喻 + 对比表)
- ❌ 只写学术版让用户自己翻译 (违反规则)
- ❌ 只写大白话让 reviewer 看不懂 (信息不完整)

**技术术语处理规范** (两个版本都适用):
- **第一次出现**的非 common 术语必须括号解释: "pooled whitening (白化: 数学操作, 把数据从'鸡蛋形' 拉成'球形')"
- 缩写必须首次展开: "ETF (Equiangular Tight Frame, 等角紧密框架)"
- 代码名保留英文: `L_orth`, `z_sem`, `nn.Linear`, `FedAvg` 这些不翻译
- 数字一致: 两个版本报的 accuracy / loss 等数字完全一致, 只在**描述语气**上不同

### 18.4 查看实验进度时的同步要求

**当用户让我查看实验进度时，必须同时**：
1. SSH 到服务器提取最新 accuracy 数据
2. 更新对应的 NOTE.md（回填结果）
3. 自动触发 experiment-sync 同步到 Obsidian
4. 如果实验已完成（log 中有 "End"），在 NOTE.md 中填写最终结论

### 18.5 实验变体速查表

> 方便快速回忆每个变体是什么

| 变体 | 全称 | 一句话解释 |
|------|------|-----------|
| M0 | Fixed Alpha | 固定增强强度，消融基线 |
| M1 | Adaptive Aug | 根据域差异自动调节增强强度 |
| M3 | Domain-Aware Proto | 保留每域原型，不做跨域平均 |
| M4 | Dual Alignment | 域内 cosine + 跨域 InfoNCE 双对齐 |
| M5 | Style Contrastive | M4 + z_sty 做域对比学习 |
| M6 | Delta-FiLM | 跨域风格差异做 FiLM 调制 |
| mode 0 | Orth Only | 纯正交解耦，无增强无对齐 |
| mode 1 | Bell-curve | InfoNCE 先升后降（钟形） |
| mode 2 | Cutoff | InfoNCE R80 后硬关闭 |
| mode 3 | Always-on | L_orth 全开 + InfoNCE 始终不关 |
| mode 4 | MSE Anchor | 标准 InfoNCE + MSE 锚点（FPL 式） |
| mode 5 | Alpha-Sparsity | cos^0.25 弱化正例梯度（FedPLVM 式） |
| mode 6 | MSE + Alpha | 双安全阀组合（**R50 最佳 82.2%**） |
| mode 7 | Detach Aug | 增强走对比不走 CE（PARDON 式） |

---

## 十九、关键实验发现备忘（2026-04-16 更新）

### 19.1 梯度冲突是核心问题

- cos_sim(grad_CE, grad_InfoNCE) 在 R50 穿零 → CE 和 InfoNCE 打架
- 所有没有"安全阀"的 InfoNCE 变体都会在后期崩溃
- 三道安全阀：MSE 锚点 / alpha-sparsity / 梯度截断（triplet margin）

### 19.2 文献审查关键结论

- **FPL**：InfoNCE + MSE 锚点，tau=0.02，FINCH 多原型
- **FedPLVM**：alpha-sparsity (α=0.25)，correction MSE，tau=0.07
- **PARDON**：Triplet loss (margin=0.3)，图像空间 AdaIN (no_grad)，CE 不用增强特征
- **MixStyle**：浅层增强有效，深层（res4）暴跌 -7%
- **共识**：我们在最深层做 AdaIN + CE 回传 = 违反所有安全原则

### 19.3 EXP-070 消融的真相

- "Decouple only" 用了 warmup=9999 → L_orth 权重仅 0.02 → 等于纯 CE
- "+Share" 2/3 seed 的 peak 在 warmup 激活前就出现了 → Share 没贡献
- "协同效应"是基于 peak 值的幻觉，按 final 排序结论完全反转
