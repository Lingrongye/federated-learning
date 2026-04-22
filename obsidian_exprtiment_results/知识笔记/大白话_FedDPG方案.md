# 大白话版：FedDPG 新方案（破釜沉舟 2026-04-22）

## 一句话总结

> **"不加任何新模块。直接把分类器的权重当语义原型，把 BN 层的 (μ, σ) 当风格原型。语义拉近，风格跨域混搭。"**

---

## 为什么要大改？

我们之前的 FedDSA 系列（双头架构 + 正交损失 + whitening 等）：
- ✅ PACS 比 FDSE 高 +0.73
- ✅ DomainNet 比 FDSE 高 +0.28
- ❌ **Office 差 FDSE -1.41**（攻不下）

更气人的是，实验证据显示：
- 正交损失 lo=1 vs lo=0 差 ≤0.3pp（**不显著**）
- 我们所谓"解耦创新"几乎不贡献 accuracy
- 真正 +1pp over FedBN 的来源是**降维瓶颈** (1024→128→128)，不是正交头

**结论**：老方案的"创新"被数据打脸。要彻底换思路。

---

## 新方案核心思想（比喻版）

### 老方案像什么
```
FedBN（简单 BN 本地）
  + 双头投影（多加了 2 个头，1 个做语义 1 个做风格）
  + 正交损失（逼两个头的输出垂直）
  + whitening（给特征白化）
  + ETF fixed classifier（分类器固定成正三角形）
  + ...（一堆料堆上去）
```

像**在蛋糕上不停加装饰**，但蛋糕本身没做好。

### 新方案像什么
```
FedBN（简单 BN 本地）
  + 分类器的权重 W 自然就是"语义原型"（每行一个类的代表）
  + BN 每层的 (μ, σ) 自然就是"风格原型"（每 client 一个指纹）
  + 跨 client 把风格指纹交换着用
  + 小样本的 client 少用别人的风格（防过扰动）
```

**像把本来就有的东西合理利用**，不加新料。

---

## 四个组件大白话

### 🅰 组件 A：分类器权重 = 语义原型

**啥意思**：
- 每个分类器 `Linear(1024 → 7)` 的 weight shape 是 `[7, 1024]`
- **每一行**就是一个类的 "典型表示"（语义 prototype）
- 比如第 0 行就是"狗"这个类的 1024 维向量表示

**要做啥**：训练时强制让每张图的 feature 朝自己类的那一行靠拢。
```
图片是狗 → 特征 h 要跟 W[0]（狗那一行）相似
图片是长颈鹿 → 特征 h 要跟 W[2]（长颈鹿那一行）相似
```

**为什么算 novelty**：
- FedProto 用"特征均值"当原型（要多传一套东西）
- 我们用 **W 本身**（分类器本来就有，零成本）
- FL 跨域场景下没人这么做过

**数学**：`L_A = 1 - cos(h, stop_grad(W[y]))` 

**stop_grad 关键**：W 只通过 CE 学，L_A 只驱动 encoder（不让 encoder 和 classifier 互相"作弊"凑近）

### 🅱 组件 B：BN (μ, σ) = 风格原型（跨 client 共享）

**啥意思**：
- FedBN 里每层 BN 都有 `running_mean` 和 `running_var`（记录这个 client 的统计量）
- 这俩就是 **该 client 所在域的风格指纹**（μ=颜色偏移，σ=对比度）
- 每个 client 有 6 层 BN，每层一对 (μ, σ)

**要做啥**：
- 每轮 server 收集所有 client 的 BN stats → 建 "风格银行" Ψ
- Ψ 不广播回去（保持 FedBN 本地性），只作为**增强资源**

**为什么算 novelty**：
- FedBN 把 BN stats 本地**私有**（不共享）→ 我们**反向**：共享作资产
- FISC/CCST 共享 VGG 特征的 style，**不叫 prototype** 也不是 BN
- **首次把 BN stats 概念化为 style distribution prototype**

**数学**：`Ψ = {(μ_c^l, σ_c^l)} for l in 6 BN layers, c in received clients`

### 🅲 组件 C：跨 client 风格混搭（feature-level AdaIN + 高斯采样）

**啥意思**：
- 训练时 50% 概率从别的 client 的风格里采一个，"借来用"
- 假设我是 Webcam client，这轮借 DSLR 的风格：
  ```
  h_aug = σ_DSLR · (h - μ_webcam) / σ_webcam + μ_DSLR
  ```
  （把我的 feature 去掉自己的风格，套上 DSLR 的风格）
- 然后让模型对"被换了风格的 feature"也要分类正确

**为什么跟以前失败不一样**：
- **以前 EXP-040/059/061 做过 AdaIN 都失败** ⚠️
- 失败原因：直接从 style bank 里抽一个样本，方差大不稳定
- **新做法**：假设每 client 风格是 Gaussian N(μ, σ²)，从里面 **reparameterize 采样**
  ```
  μ̂ ~ N(μ, 0.01·σ)   # 小扰动不崩
  σ̂ ~ N(σ, 0.01·σ)
  ```
- 稳定很多（Gaussian 保证平滑可微）

**数学**：`L_C = CE(classifier(h_aug), y)`

### 🅳 组件 D：小样本域自适应门控（救 DSLR）

**啥意思**：
- Office 的 DSLR 只有 157 张图，比 Amazon 958 张少 6 倍
- 如果对 DSLR 做同样强度的风格混搭 → 本来就少的数据被扰动成噪声
- **解决**：按 client 数据量动态调 gate
  ```
  gate(|D_k|) = 1 - exp(-|D_k| / 300)
  DSLR (157):  gate ≈ 0.41  → 几乎不混搭
  Amazon (958): gate ≈ 0.96 → 正常混搭
  ```

**为什么 novelty**：
- 老方案 EXP-067/68 做过基于 **style distance** 的 gate（失败）
- 我们做基于 **sample size** 的 gate（没人做过）
- 直接对准 Office DSLR 这种 outlier 小域

---

## 预期效果（每个组件独立涨点）

| 配置 | PACS | Office | DomainNet |
|:---:|:---:|:---:|:---:|
| FedBN baseline | 79.01 | 88.68 | 72.08 |
| +A (W 锚) | 79.8 | 89.3 | 72.7 |
| +B (BN 风格原型) | 80.5 | 89.9 | 73.1 |
| +C (风格混搭) | 81.5 | 90.8 | 73.8 |
| +D (小域 gate) | **82.0** | **92.6** | **74.3** |
| **vs FDSE baseline** | **+0.1** | **+2.0** | **+2.1** |

**目标**：所有 3 数据集都 **>FDSE +2pp**（尤其 Office 终于能攻下来）

---

## ⚠ 历史警示：组件 C 的魔咒

**警告**：feature-level AdaIN 在我们项目里**失败过 4 次**：

| 实验 | 结果 |
|:---:|:---|
| EXP-040/044 多层 AdaIN (mid+final) | 多 seed 80.74 < 原版 81.29 |
| EXP-059 stylehead bank AdaIN | PACS **-2.54%** |
| EXP-061 style sharing 诊断套件 | Office 全部失败 |
| EXP-047A/D style aug 调权 | gap 恶化 |

**新方案为什么希望打破魔咒**：
1. **Gaussian reparameterize** 替代直接 bank 采样 → 扰动平滑
2. **纯 FedBN 架构**（无双头干扰）→ 少一层复杂度
3. 源头是 **BN running stats**（多轮稳定） → 比 feature bank（单次样本）稳
4. **组件 D gate** 专治小域崩溃（EXP-061 Office 崩的那种）

**Stage 1 smoke test 关键**：Office seed=2 R200 必须 > FedBN 88.68。如果不行，回滚到 A+B only。

---

## 一句话记住

> **"把分类器权重 W 叫语义原型，把 BN 层 (μ,σ) 叫风格原型；语义拉近，风格混搭。不加任何新模块，全是 FedBN 原生产物。"**

---

## 名词速查（给未来的自己）

| 术语 | 大白话 |
|:---:|---|
| classifier.weight W | 分类器最后那层的权重矩阵，每一行就是一个类 |
| semantic prototype | "这个类长啥样"的代表向量 |
| BN running mean/var | Batch Norm 层记的平均值和方差，跟 client 的域风格有关 |
| style prototype | "这个 client 的风格长啥样"的代表向量对 (μ, σ) |
| reparameterization | 从分布里采样但让梯度能回传的技术（VAE 里的经典技巧）|
| AdaIN | 把一组特征的风格 (mean/std) 换成另一组的操作 |
| gate | 门控，一个 0-1 之间的系数，控制某机制开多少 |
| stop_grad | 这个方向梯度不传，防止两端互相"作弊" |

---

## 跟 FDSE 的根本区别（一张图）

```
FDSE: 域信息是噪声, 层层擦掉
       ┌─ conv1 ─ DSE1 擦 ─┐
       │                   │
       │─ conv2 ─ DSE2 擦 ─│
       │                   │  复杂 QP 聚合
       │─ conv3 ─ DSE3 擦 ─│
       └─ ...              ┘

FedDPG: 域信息是资产, 跨 client 共享着用
       ┌─ conv1 ─ BN (μ₁,σ₁)──┐
       │                       │ 汇聚成风格银行 Ψ
       │─ conv2 ─ BN (μ₂,σ₂)──│
       │                       │ 训练时从 Ψ 采别 client 的 (μ,σ)
       │─ conv3 ─ BN (μ₃,σ₃)──│ 做 feature-level 混搭
       └─ ...                  ┘
```

FDSE 想让模型"看不见域"；我们想让模型"看过所有域"。

---

## 下一步（Stage 1）

1. 写代码 `feddpg.py` 基于 `fedbn.py`，加 ~150 行（主要是 style bank 管理 + 4 个 loss）
2. 写 config `feddpg_office_r200.yml`
3. Office seed=2 R200 smoke test
4. Go/No-Go 节点：
   - Best > 89.0 → 继续全量 3-seed 消融
   - Best < 88.5 → 回滚到 A+B only（砍 AdaIN）
   - Best < 88.0 → 彻底放弃，回 brainstorm

预计 Stage 1 在 1-2h 内得到答案。

---

*本笔记生成日期: 2026-04-22*
*对应详细技术报告: `IDEA_REPORT_2026-04-22.md`*
*5 个 agent 并行调研产出的最终方案*
