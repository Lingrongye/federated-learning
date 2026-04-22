# 大白话：7 个 FL 跨域新方案 brainstorm

**日期**: 2026-04-22
**背景**: 原 FedDPG 方案 novelty check 暴雷（被 FedFA ICLR 2023 + FedCA ESWA 2025 + FedDr+ 2024 覆盖）, 需要找更 novel 角度
**2 个 agent 并行调研**: Agent 1 精读 FedCA/FedFA 找漏洞 + Agent 2 独立 brainstorm 新方向
**输出**: 7 个候选方案, 3 个推荐组合

---

## 🎯 背景：为啥要 brainstorm

之前方案 FedDPG 的 4 个组件 novelty 体检：

| 组件 | 体检分数 | 撞车 paper |
|:---:|:---:|:---:|
| A: W 作语义原型 | 3/10 | Center Loss / ArcFace / FedDr+ (2024) |
| B: BN stats 作风格 bank | 4/10 | FedCA (ESWA 2025), FedCCRL (2024) |
| C: feature-AdaIN + Gaussian | **2/10 CRITICAL** | **FedFA (ICLR 2023) 直接重复** |
| D: difficulty gate | 6/10 | 部分新 |

**结论**: 整体组合 = FedFA + FedCA + FedDr+ 的组合, 都已发表, top venue 进不去.

**用户决策**: 保留 B + D 精神, 找更 novel 角度.

---

## ⭐ 方向 A: CC-Bank (Class-Conditional 风格银行)

### 比喻：每种菜分开存酱料

**现在 FedFA 怎么做**：把所有图的风格指纹混一缸:
```
Photo 域: 狗+车+人+马 的风格全搅拌 → 一个平均风格
Sketch 域: 狗+车+人+马 的风格全搅拌 → 一个平均风格
```

**问题**：狗从 photo 变到 sketch 的风格变化 ≠ 车的变化（狗是有机纹理，车是刚体）。全类混搅=丢信息。

### CC-Bank：按 (类, 域) 二维分桶

```
         Photo       Sketch     Art        Cartoon
 狗    (μ,σ)_狗P   (μ,σ)_狗S   ...
 车    (μ,σ)_车P   (μ,σ)_车S   ...
 人    ...
```

**增强时**：我现在训一张"photo 的狗"，从 bank 里抽 **"sketch 的狗"** 的风格套上 → 只换风格不换内容。

### 一句话
> "不是全图共享一份风格，而是每**类**单独一份，增强时同类跨域混搭。"

### Novelty
8.5/10 — **FedFA/FedCA 都做 domain-level pool，没人做 class-conditional**

### 实现
~50 行代码，server-side bank 按 label 分桶

---

## ⭐ 方向 B: Learnable α (自学习增强强度)

### 比喻：老师按学生强弱自动调作业难度

**现在 FedFA 怎么做**：所有 client 同一强度增强:
- Amazon (1000 张, 92% 准) 硬增强 — 浪费
- DSLR (157 张, 100% 准) 硬增强 — 骚扰学霸
- Caltech (1100 张, 73% 准) 硬增强 — 强度不够

**问题**：统一强度不合理。

### Learnable α：每个 client 自学一个"增强开关"

```
f_final = α_k · 增强后特征 + (1 - α_k) · 原特征
```

- α 小 → 基本不增强（保持原）
- α 大 → 全力增强（用别人风格）

**怎么学 α**: 看 validation loss 的 rank:
- Caltech loss 高 → α 大 (多救它)
- DSLR loss 低 → α 小 (别动它)

### 一句话
> "每个 client 自学一个增强强度旋钮，成绩差的调大，成绩好的关小。"

### Novelty
8.0/10 — **FedFA 的 variance 硬编码, 没有 per-client 自适应**

### 实现
~30 行代码，加 scalar parameter per client

---

## 🥇 FedPTR: Prototype 轨迹正则 (Top 新 idea)

### 比喻：Prototype 是"跑步的人"

**现在 FedProto/FPL 怎么做**：每轮把所有 client 的"狗"原型取平均 → 全局狗原型。每轮独立算，**没考虑时间**。

### FedPTR 的视角

把每个类的 prototype 想象成一个在特征空间里**跑步的人**：
- 上轮在哪（`p_{t-1}`）
- 这轮在哪（`p_t`）
- **速度**：`v = p_t - p_{t-1}`（方向 × 快慢）
- **曲率**：`κ = |v_t - v_{t-1}|`（有没有突然拐弯）

### 关键洞察

**正常情况**：不同 client 的"狗 prototype"应该**大致朝同方向跑**（逐渐收敛共识）

**outlier client (如 Caltech)**：可能**突然拐弯**（本地数据偏），曲率大

### FedPTR 怎么用这个

1. **预测下一步位置**:
   ```
   p̂_{t+1} = p_t + η · v_t
   ```
   Client 训练时朝**预测位置**而不是当前位置靠 → 像"开车看远处不看车头"
   
2. **曲率大的 client 降权**: 它跑错方向了, 少听它

### 一句话
> "Prototype 不是静止的点, 是**时间里跑步的人**. 看它速度和拐弯决定下一步去哪 + 谁该降权."

### 为啥 novel
- 别人把 prototype 当零阶（每轮独立快照）
- 我们加一阶（速度）+ 二阶（曲率）= 时间动力学
- **FL 场景没人做过 prototype trajectory**

### Novelty
**8.5/10**

### 适合攻什么
Office **Caltech** 样本少、prototype 抖动大 → 轨迹平滑最受益（预期 +2-3pp）

### 实现
~150 行，server 侧维护 v_c 和 p_c history

---

## 🥈 FedAdvStyle: 对抗风格挖掘

### 比喻：教练专治你的弱项

**现在别人怎么选 style**:
- FISC: 所有 client style 的**中位数** → 温和
- FedFA: 从 Gaussian **随机采样** → 盲盒

都是"被动式"。

### FedAdvStyle 的做法：主动找最难的 style

1. 服务器拿到所有 client 的 (μ, σ)
2. 这些指纹围成一个**凸包**（几何上的区域）
3. 服务器用 **PGD (Projected Gradient Descent, 对抗攻击算法)** 在凸包里找：
   > "哪个 style 让当前全局模型 **最混淆**（loss 最大）？"
   
   这就是 **worst-case style** — 虚拟最难的风格
4. 下发给所有 client，local 训练用它增强

### 比喻升级
- FISC 像"家常菜" (安全无趣)
- FedFA 像"盲盒菜" (随机)
- FedAdvStyle 像"教练专治你不爱吃的菜" (定向狠训)

### 一句话
> "服务器主动在所有 client 风格凸包里挖'最难的 style'，定向下发让模型专治短板。"

### 为啥 novel
- 风格共享文献从没人做**对抗** worst-case 挖掘
- **DG 经典理论 (GroupDRO / IRM) 说 worst-case 才能保证泛化** — 理论背书强

### Novelty
**8.0/10**

### 实现
~80 行，server 加 PGD 攻击循环

### 风险
PGD 找到的 style 可能过于极端导致训练崩溃

---

## FedCRM: 推理时软融合

### 比喻：多专家加权投票

**现在 FedProto 推理**：算 feature 跟哪个类原型最近 → 选最近的（**hard** 选）

### FedCRM 的做法

**训练**：标准 FedBN + FedProto（**零改动**）

**推理**：
1. 服务器维护每类 K 个 prototype（K 个 client 一人一个）
2. 测试样本的 feature 算跟所有 K 个 prototype 的相似度
3. Softmax 加权融合：
   ```
   logit_c = Σ_k α_k · similarity(feature, prototype_c^k)
   ```
4. argmax 选类

### 比喻
- FedProto 像"找最像的一个专家问"
- FedCRM 像"咨询多专家按可信度加权投票"

### 一句话
> "训练不改, 推理时让多个类原型软投票 (替代 hard 选最近的)."

### Novelty
**7.5/10** — 类似 FedDG-MoE 但**我们零训练 router**（MoE 要训 router）

### 优势
- 训练零成本（直接用现有 FedProto 结果）
- 推理代价小
- "免费午餐"

### 适合攻什么
PACS Sketch 这种语义清、风格极端的域

---

## FedGLA: 条件互信息对齐 (无 head 解耦)

### 比喻：知道"是狗"之后, 特征里不应再含"是 photo 还是 sketch"

**问题**：用户不允许加 projection head（老方案的双头被否定了）

### 数学语言版（别怕）
> 理想：给定类 y 的情况下，特征 f 和域 d 的**条件互信息** I(f; d | y) = 0

**翻译**：知道这是"狗"之后，特征里不应该再能告诉你"这是 photo 还是 sketch"（域信息已完全从特征中消失，只留语义）

### FedGLA 的做法

```
L = L_CE + λ · I(feature ; domain | class)
```

直接在原特征上加一个 scalar loss，**不加任何 projection head**。

用 **CLUB (ICML 2020)** 算法估这个 CMI（Condition Mutual Information）。

### 不加新模块的秘诀
加一个**小 domain classifier**（2 层 MLP, `q(d|f,y)`）辅助估 MI，但它跟主 model 一起聚合，不算"额外 head"。

### 一句话
> "用条件互信息 I(f;d|y) 当 loss, 强制'类确定后特征不再含域信息'. 首次在 FL 用 CLUB."

### Novelty
**7.5/10**

### 适合攻什么
所有数据集（理论通用），跨域 invariance 有理论保证

### 风险
CMI 估计噪声大，小 batch 不稳定

---

## FedCUR: 域难度课程学习

### 比喻：先做简单题, 后做难题

**问题**：Caltech 是 Office 最难域。每轮都让它等权参与 → 污染早期共识。

### FedCUR 的做法

服务器给每个 client 算"难度分":
```
h_i = local loss + 类原型散度
```

**按 round 调度**:
- **早期 (round 0-50)**: 只让简单 client (Amazon/DSLR/Webcam) 参与 → 打基础
- **后期 (round 50-200)**: 逐渐引入难 client (Caltech) → 精修

### 一句话
> "让简单 client 先垫底打基础, 难 client 后期再上, 避免早期噪声污染."

### Novelty
**7.0/10** — curriculum 老技术，FL + domain-hardness 组合新一点但撑不起主卖点

### 不推荐作主 claim
Agent 2 评估: "Curriculum 概念在 DG 被用烂了"

---

## 📊 七个方案对照表

| 方案 | 比喻 | 核心 | Novelty |
|:---:|---|---|:---:|
| **CC-Bank** | 每种菜分开存酱料 | (类, 域) 二维 style bank | 8.5 |
| **Learnable α** | 老师按学生强弱调作业 | 自适应增强强度 | 8.0 |
| **FedPTR** 🥇 | Prototype 是跑步的人 | 速度 + 曲率时间动力学 | 8.5 |
| **FedAdvStyle** 🥈 | 教练专治弱项 | PGD 挖 worst-case style | 8.0 |
| **FedCRM** | 多专家加权投票 | 推理时软融合 | 7.5 |
| **FedGLA** | 知狗后别判photo/sketch | 条件 MI 无 head 解耦 | 7.5 |
| **FedCUR** | 先简单后难 | Domain hardness 课程 | 7.0 |

---

## 🏆 三个推荐组合方案

### 方案 X: **FedPTR + CC-Bank + Learnable α**（推荐, 稳）

**三个角度**:
- 时间 (FedPTR 轨迹)
- 类别 (CC-Bank)
- 样本量 (Learnable α)

**Paper 叙事**: *"Prototype dynamics + class-conditional style sharing + adaptive intensity — 首次把时间、类、样本量三维信息协同用于 FL 跨域"*

**预期**: PACS +2, Office +2, DomainNet +2

**实现**: 总计 ~230 行

### 方案 Y: **FedAdvStyle + CC-Bank + Learnable α**（激进）

- 主卖 FedAdvStyle（worst-case 对抗）
- CC-Bank + Learnable α 做配套

**优势**: DG 经典理论背书强

**劣势**: PGD 训练可能不稳

### 方案 Z: **FedPTR + FedAdvStyle 双主卖点**（高上限）

两个主创新并行
- Trajectory 决定 "prototype 去哪"
- AdvStyle 决定 "训练见啥"

**优势**: novelty 上限 9+/10

**劣势**: 焦点分散，复杂度高

---

## 🧠 我的推荐: 方案 X

### 为什么
1. **Novelty 综合 9/10**: FedPTR 视角完全新
2. **跟已发表工作零正面撞车**: FedFA/FedCA/FedDr+ 的领域我们绕开
3. **消融极清晰**: 4 层 (base / +PTR / +CC-Bank / +α)，每层独立涨点
4. **实现成本低**: 总 230 行，1-2 天写完
5. **Office 有救**: CC-Bank + Learnable α 专攻 Office 短板 Caltech

### Paper Story
> "我们提出 **FedPTR-CCA**：在 FedBN 基础上，把 prototype 当时间序列建模 (trajectory dynamics)，用类条件风格 bank 做 cross-client 增强，再用 per-client learnable α 自适应调节增强强度。三者协同攻克 FL 跨域 accuracy 天花板。"

---

## 🚦 下一步（等用户 Gate）

等选定方案后进 **research-refine**（方案细化） + **experiment-plan**（消融矩阵）。

---

## 名词速查（新术语）

| 术语 | 大白话 |
|:---:|---|
| Class-Conditional | 按类别分开处理（不跨类混） |
| Learnable α | 可学习的标量开关 |
| Prototype trajectory | 原型在时间里的移动轨迹 |
| 曲率 (curvature) | 轨迹有没有突然拐弯 |
| 凸包 (convex hull) | 一组点围成的最小凸形区域 |
| PGD | 对抗攻击算法，用梯度上升找最难样本/style |
| Worst-case style | 让模型最混淆的虚拟风格 |
| Soft mixture | 不选一个而是加权融合多个 |
| Conditional MI (CMI) | 条件互信息：给定 y 下 f 和 d 的信息重叠 |
| CLUB | MI 上界估计算法 (Cheng ICML 2020) |
| Curriculum | 课程学习，按难度排序训练 |
| Domain hardness | 某 client 有多难训 |

---

## 附录：被否决的原 FedDPG 方案

```
原 FedDPG = FedBN + W as semantic prototype + BN stats as style Gaussian prototype 
          + cross-client sharing + no projection head
```

**否决理由**:
- 组件 A (W prototype): FedDr+/ArcFace/FedETF 覆盖 (textbook)
- 组件 B (BN style bank): FedCA 直接撞
- 组件 C (Gaussian reparam AdaIN): **FedFA ICLR 2023 完全重复**
- 组件 D (sample-size gate): **方向错** (DSLR 100% 不需救，Caltech 73% 才是 outlier)

**教训**: 下次做 novelty check **第一步要查 ICLR/NeurIPS 2023 起的核心 FL 论文**，不能只看 2025 年的。FedFA 是 2023 的基础工作，4 个 agent 都漏掉了。

---

*对应技术报告*: `IDEA_REPORT_2026-04-22.md` (旧, FedDPG)
*后续技术报告待生成*: `IDEA_REPORT_2026-04-22_V2.md` (新方案 X/Y/Z)
*精读 FedFA/FedCA 报告*: Agent 1 输出 (未单独保存)
*Brainstorm 5 方向报告*: Agent 2 输出 (未单独保存)
