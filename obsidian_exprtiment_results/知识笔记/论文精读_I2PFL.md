# 论文精读 — I2PFL (CVPR 2025)

> **阅读日期**: 2026-04-23
> **状态**: 精读完成,13 页全 (含 supplementary)

---

## 1. Paper 基本信息

| 字段 | 内容 |
|------|------|
| **标题** | Mitigating Domain Shift in Federated Learning via Intra- and Inter-Domain Prototypes |
| **作者** | Huy Q. Le¹, Ye Lin Tun¹, Yu Qiao¹, Minh N. H. Nguyen², Keon Oh Kim¹, Choong Seon Hong¹ |
| **机构** | ¹Kyung Hee University (韩国庆熙大学) / ²Vietnam-Korea Univ. of IT (越南) |
| **venue** | CVPR 2025 |
| **arXiv ID** | 2501.08021 (v2, 2025-03-10) |
| **URL** | https://arxiv.org/abs/2501.08021 |

**TL;DR**: 在 FL 跨域场景下,用"局部 feature-level MixUp 产生 **intra-domain 增强原型**(APA 对齐) + 服务器端按'距均值越远权重越大'的 **inter-domain 原型重加权**(GPCL 对比)"两套原型机制,打败 FPL / FedPLVM,Office-10 上涨 4.99%,PACS 上 82.25% AVG。

---

## 2. 核心问题 + 动机 (大白话)

### 他们观察到什么问题?

FL 跨域两个 level 的不平衡同时存在:
- **inter-domain (域间)**: Photo 大象 vs Cartoon 大象,**同类异域**特征分布完全不同
- **intra-domain (域内)**: 即使在同一个 Photo 域,不同 image 的背景 / lighting / pose 也差异巨大

### FedProto / FPL / FedPLVM 哪里不够?

| 方法 | intra-domain | inter-domain | 问题 |
|------|--------------|--------------|------|
| FedProto (AAAI'22) | ❌ 没做 | 简单均值 | 有 dominant domain bias |
| FPL (CVPR'23) | ❌ 没做 | FINCH 聚类多原型 | 聚类对 outlier 敏感 + 只顾服务器侧 |
| FedPLVM (NeurIPS'24) | ❌ 没做 | 双层聚类 + α-sparsity | 同上 + 仍然"服务器一点论" |

💡 **他们的切入点**: 之前方法**全都只在服务器做 inter-domain 原型聚合**,漏了**客户端本地**就可以做 intra-domain 的事 → 在 client 加 MixUp augmented prototype + server 加距离加权,**两个 level 同时打**。

---

## 3. 方法精读 — 5 个组件独立讲

### 组件 1: Intra-domain Prototype MixUp (feature-level)

#### 直觉 — 为什么 feature 级不是 image 级?

Image-level MixUp (原版 ZhangICLR18) 在 raw pixel 上混两张图 → 产生一张"两只猫头叠起来"的伪图,语义乱、domain-specific 信息也混乱。
Feature-level MixUp 在 encoder 提取后的 pooled feature 上做线性插值 → **语义更稳定、domain-specific 的细节被 encoder pooling 抽象过**,产生的增强 prototype 更 robust。

#### 精确公式

对同一客户端 $m$ 内两个样本 $h_i, h_j$:

$$\tilde h_i = \gamma h_i + (1-\gamma) h_j$$

- $\gamma \sim \text{Beta}(\alpha, \alpha)$, $\alpha \in (0, \infty)$
- 🔑 **最优 $\alpha$**: Digits/Office-10 为 $\alpha=0.4$, PACS 未明说但从 Fig.6 看也约 0.4

#### $h_j$ 怎么选?

论文原文 (§3.4): "$h_j$ is the feature of **random data sample $x_j$ from different semantic class** on $D_m$"

⚠️ 注意: **异类** mixup (different class),不是同类。这个选择是反直觉的 — 他们要的是"跨类的模糊",增加 decision boundary 的平滑度 (类似 input MixUp 原版思路)。

#### 在哪一层做?

在 `feature_extractor` 的**输出端**(pooled feature $h = f(x) \in \mathbb{R}^d$)。Digits/Office-10 用 ResNet-10 → $d$ 未明说,PACS 用 ResNet-18 → $d = 512$。**不在中间层做** (类似 MixStyle 那种浅层 BN 统计量换的做法不一样)。

#### 伪代码 (5 行)

```python
# inside local training, per batch
h = feature_extractor(x)                         # [B, d]
idx = torch.randperm(B)
j = idx[(y[idx] != y)].nonzero()[:B]             # 选异类
gamma = Beta(alpha, alpha).sample([B, 1])
h_tilde = gamma * h + (1 - gamma) * h[j]         # feature MixUp
```

#### augmented prototype 聚合

$$\tilde p_m^k = \frac{1}{|S_m^k|} \sum_{i \in S_m^k} \tilde h_i$$

其中 $S_m^k$ 是客户端 $m$ 上 class $k$ 的样本索引集。**与原 prototype $p_m^k$ 并列计算,不替代**。

---

### 组件 2: Inter-domain Prototype Reweighting

#### 直觉 — 为什么离均值远的给高权重?

- 如果 PACS 里 Photo 占 3 个 client、Sketch 占 4、Cartoon 只占 1,简单均值会被 Photo+Sketch 拉走,Cartoon 的 prototype 信号被淹没
- **"离均值越远 = 说明这个域有独特贡献 = 必须重视"**
- 这和 "variance reduction" 的直觉一致 (Fig.3 的圆圈示意: 远距的小圆圈 $d_m^k$ 大,权重高)

#### 精确公式 (Eq. 2-4)

**Step 1 — 本地原型** (每个客户端 $m$ 对 class $k$):
$$p_m^k = \frac{1}{|S_m^k|} \sum_{i \in S_m^k} h_i$$

**Step 2 — 初始均值** (服务器):
$$\mu^k = \frac{1}{M} \sum_{m=1}^M p_m^k$$

**Step 3 — 距离**:
$$d_m^k = \|p_m^k - \mu^k\|_2^2, \quad d^k = \sum_m d_m^k$$

**Step 4 — 距离加权聚合**:
$$g^k = \sum_{m=1}^M \frac{d_m^k}{d^k} p_m^k \in \mathbb{R}^d$$

#### 🔑 和 FPL / FedProto 对比

| 方法 | 聚合方式 | 公式 |
|------|---------|------|
| FedProto | simple averaging | $g^k = \frac{1}{M}\sum_m p_m^k$ |
| FPL | FINCH clustering → 多原型 | 聚类得到 $\{g^{k,c}\}_c$ |
| **I2PFL** | distance-based reweighting | $g^k = \sum_m \frac{d_m^k}{d^k} p_m^k$ |

⚠️ 注意: reweighting **仍然是单原型**,不是多原型。比 FPL 的 FINCH 聚类简单得多,但表现更好 (Table 5: PACS 82.25 vs FPL 80.16)。

---

### 组件 3: EMA (Exponential Moving Average)

#### 位置 — server

每轮服务器更新 generalized prototype 后,用上一轮的做 EMA smoothing:

$$G^{t+1} = \beta G^{t+1} + (1 - \beta) G^t$$

(⚠️ 公式里的 $G^{t+1}$ 在等号两边出现看似有 typo — 实际应理解为: 用本轮**新算出**的 $G_{\text{new}}^{t+1}$ 和**上一轮**的 $G^t$ 做 EMA, 即 $G^{t+1} \leftarrow \beta \cdot G_{\text{new}}^{t+1} + (1-\beta) G^t$)

#### 更新什么?

只更新 **generalized prototype $G$** (server 存), 不更新模型参数。

#### 为什么 0.99?

- Fig. 8 sensitivity: β ∈ {0.95, 0.99, 0.999}
- **最优 β = 0.99** (3 个数据集一致)
- β = 0.95 → 太快,prototype 震荡
- β = 0.999 → 太慢,跟不上训练

### 组件 4: GPCL (Generalized Prototype Contrastive Learning)

#### 直觉

把 generalized prototype $G = \{g^1, ..., g^K\}$ 当作 InfoNCE 的 memory bank: 本地 feature $h$ 向**同类 $g^+$** 靠拢,远离所有**异类** $g^k$。

#### 精确公式 (Eq. 6)

$$\mathcal{L}_{\text{GPCL}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(s(h_i, g^+)/\tau)}{\sum_{g^k \in G} \exp(s(h_i, g^k)/\tau)}$$

- $s(u, v) = u^\top v / \|u\| \|v\|$ = cosine similarity
- $g^+$ = 和 $h_i$ 同 class 的 generalized prototype
- 分母 sum over **所有 K 个 class 的 $g^k$**
- 🔑 **最优 $\tau$**: Digits=0.07, Office-10=0.02, PACS=0.04 (Fig. 6)

#### Positive/Negative 定义

- **Positive**: 1 个 (本样本所属 class 的 $g^+$)
- **Negative**: K-1 个 (其他所有 class 的 $g^k$)
- ⚠️ **没用多原型**: 和 FPL (一个 class 有多个聚类原型) 不同, I2PFL 一个 class 就一个 $g^k$

#### 和普通 InfoNCE 区别

|  | 标准 InfoNCE (MoCo) | GPCL (I2PFL) |
|--|---------------------|--------------|
| Memory bank | 样本 feature queue | **K 个 class 级 prototype** |
| Positive | 同 image 另一 view | **同 class 的 $g^+$** |
| Negative | queue 中所有其他样本 | K-1 个异类 $g^k$ |
| 规模 | 4096~65536 | **K (很小, PACS=7)** |

→ 本质上是 **Supervised Contrastive + Prototype Memory** 的结合,粒度极粗但稳定。

---

### 组件 5: APA (Augmented Prototype Alignment)

#### 精确公式 (Eq. 9)

$$\mathcal{L}_{\text{APA}} = \sum_k \|h_m^k - \tilde p_m^k\|_2^2$$

- $h_m^k$ = 客户端 $m$ 上 class $k$ 样本的 feature (本步 batch 的均值,非全局)
- $\tilde p_m^k$ = 组件 1 产生的 augmented prototype
- **MSE loss**,不是 InfoNCE

#### 为什么不做 "augmented CE"?

augmented feature 上直接跑 CE 会让 classifier 拟合这些**伪样本的 label** (而 label 是异类混合后不清晰的),反而破坏。用 MSE 把**原 feature** 拉向 **augmented prototype** 更安全 — 只在 representation 层面做 regularization。

> ⚠️ 这点和我们的 FedDSA 很像: 增强要做,但不要用增强 feature 过 CE (PARDON 也是这思路)。

---

### 总损失 (Eq. 10)

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}}_{\text{标准}} + \underbrace{\lambda_{\text{intra}} \mathcal{L}_{\text{APA}}}_{\text{Intra}} + \underbrace{\lambda_{\text{inter}} \mathcal{L}_{\text{GPCL}}}_{\text{Inter}}$$

🔑 **最优超参** (Fig. 7):

| Dataset | $\lambda_{\text{intra}}$ | $\lambda_{\text{inter}}$ |
|---------|:------------------------:|:------------------------:|
| Digits | 10 | 1 |
| Office-10 | 10 | 1 |
| PACS | **2** | 1 |

→ **Intra 权重 >> Inter 权重** (除 PACS) 说明 augmented prototype 的 regularization 比 contrastive 更起作用。PACS 数据量小且域差异大,Intra 不能压太重。

---

## 4. 算法流程 (Alg.1 中文化)

```
输入: 通信轮数 T, 本地 epoch R, 客户端数 M, 本地数据 D_m,
      特征提取器 f, 分类器 g
输出: 全局模型 θ_T

服务器执行:
for t = 0 .. T-1:
    for m = 0 .. M-1:
        θ_t^m, p_m = LocalUpdate(θ_t, G^t)
    # 初始均值
    μ^k = (1/M) Σ p_m^k
    # 距离重加权
    d_m^k = ‖p_m^k - μ^k‖², d^k = Σ d_m^k
    g^k = Σ (d_m^k / d^k) · p_m^k
    G = [g^1, ..., g^K]
    # EMA
    G^{t+1} = β · G + (1-β) · G^t
    # 模型聚合 (FedAvg-like)
    θ_{t+1} = Σ (|D_m|/|D|) · θ_t^m

客户端执行 LocalUpdate(θ_t, G^t):
for r = 0 .. R-1:
    for each batch {x_i, y_i} in D_m:
        h_i = f(x_i); z_cls = g(h_i)
        # MixUp 增强 (异类对)
        h̃_i = γ·h_i + (1-γ)·h_j
        # augmented prototype
        p̃_m^k = (1/|S_m^k|) Σ h̃_i
        # 三项 loss
        L_GPCL ← (h_i, G^t)          # Eq. 6
        L_APA  ← (h_m, p̃_m)          # Eq. 9
        L_CE   ← (z_cls, y)
        L = L_CE + λ_intra·L_APA + λ_inter·L_GPCL
        θ_t^m ← θ_t^m - η ∇L

# epoch 结束后计算原始原型上传
p_m^k = (1/|S_m^k|) Σ_{i ∈ S_m^k} h_i
return θ_t^m, p_m
```

---

## 5. 实验 Setup

| 项 | 配置 |
|----|------|
| Backbone | ResNet-10 (Digits/Office-10), **ResNet-18 (PACS)** |
| Rounds T | **100** |
| Local epoch E | **10** |
| Batch size | 32 (Digits/Office-10), **16 (PACS)** |
| Optimizer | SGD, lr=0.01, weight_decay=1e-5 |
| EMA β | 0.99 |
| Seeds | 固定 seed, **重复 3 次**,报 last 5 rounds 平均 |
| 评估 | Top-1 acc |

**PACS 客户端分布 (非均匀, 10 client)**:
| 域 | P | A | C | S |
|----|:-:|:-:|:-:|:-:|
| client 数 | 3 | 2 | **1** | 4 |

⚠️ **Cartoon 只有 1 个 client** → 这是"少数派域"考验,reweighting 的重点就是救它。

**采样率**: Digits 1%, Office-10 20%, **PACS 30%**

---

## 6. PACS 完整数字

### 6.1 Main Table 2 — 所有 baseline (PACS, domain-shift)

| 方法 | P | A | C | S | **Avg** |
|------|:-:|:-:|:-:|:-:|:-:|
| FedAvg | 81.65 | 68.07 | 72.84 | 87.14 | 77.43 |
| FedProx | 80.67 | 67.59 | 75.41 | 88.92 | 78.15 |
| FedDyn | 83.27 | 67.85 | 74.44 | 88.36 | 78.48 |
| MOON | 84.64 | 73.21 | 74.70 | 91.85 | 81.10 |
| FedProc | 83.18 | 70.27 | 75.23 | 94.29 | 80.71 |
| FedProto | 89.29 | 71.08 | 73.59 | 87.83 | 80.45 |
| FPL | 85.27 | 71.40 | 74.96 | 90.83 | 80.62 |
| FedPLVM | 86.70 | 73.00 | 76.86 | 90.64 | 81.80 |
| **I2PFL** | **87.85** | **73.29** | 75.66 | **92.20** | **82.25** |

🔑 **I2PFL 相对 MOON 第二名 Avg 涨 +1.15%** (但落后于 MOON 在 Photo?不,MOON 84.64 < I2PFL 87.85 是赢的)。
🔑 **Cartoon (1 client, 最弱域)**: FedPLVM 76.86 是最高,I2PFL 75.66 排第二 (**-1.20**) → reweighting 对"唯一 client 的孤立域"反而不是最优,FedPLVM 的双层聚类更稳。

### 6.2 Table 4 — 消融 (PACS)

| Variant | P | A | C | S | Avg |
|---------|:-:|:-:|:-:|:-:|:-:|
| w/o (L_GPCL, L_APA) = FedAvg | 81.65 | 68.07 | 72.84 | 87.14 | 77.43 |
| w/o L_APA (只留 GPCL) | 85.43 | 68.49 | 75.25 | 88.31 | 79.37 |
| w/o L_GPCL (只留 APA) | 85.33 | **71.64** | 75.89 | 89.87 | 80.68 |
| w/o EMA | 87.00 | 69.64 | 73.79 | 87.53 | 79.49 |
| **Full I2PFL** | **87.85** | **73.29** | **75.66** | **92.20** | **82.25** |

🔑 **每组件对 Art (A) 贡献 (从 FedAvg 基线的 68.07 起)**:
- 加 GPCL only: 68.07 → 68.49 (**+0.42**) — 单独用对比几乎无效
- 加 APA only: 68.07 → 71.64 (**+3.57**) — **APA 是主力**
- 加 EMA (其他全留): 73.29 → 69.64 (**-3.65**) ← 去掉 EMA 反而降,说明 EMA 对 A 帮助 3.65
- Full = 73.29 (**+5.22**)

**Art 最弱 (68.07) 被 APA 拉起 3.57, EMA 再托 3.65 → 最终 +5.22.**

### 6.3 Table 5 — Reweighting vs Clustering vs Avg (PACS)

| Generalized Prototype | P | A | C | S | Avg |
|-----------------------|:-:|:-:|:-:|:-:|:-:|
| Averaging (FedProto 式) | 85.69 | 71.12 | 74.77 | 88.02 | 79.90 |
| Clustering (FPL 式 FINCH) | 86.19 | 71.84 | 73.61 | 89.00 | 80.16 |
| **Reweighting (ours)** | **87.85** | **73.29** | **75.66** | **92.20** | **82.25** |

🔑 **Reweighting 比 Clustering 高 +2.09, 比 Averaging 高 +2.35** — 简单的距离权重 > 复杂的 FINCH 聚类,这是非常干净的对比结果。

### 6.4 Table 6 — Input MixUp vs Feature MixUp (PACS)

| Intra-prototype | P | A | C | S | Avg |
|-----------------|:-:|:-:|:-:|:-:|:-:|
| w/o MixUp | 85.41 | 70.53 | 71.85 | 91.57 | 79.84 |
| MixUp (Input) | 87.13 | 71.67 | 73.22 | 91.82 | 80.96 |
| **MixUp (Feature, ours)** | **87.85** | **73.29** | **75.66** | **92.20** | **82.25** |

🔑 feature-level 比 input-level MixUp 高 **+1.29** 平均。feature 空间的 mixup 更 robust。

### 6.5 Hyper-parameter sensitivity (Fig.6-8)

| 超参 | PACS 最优 | 范围 |
|------|:--------:|:----:|
| τ (GPCL 温度) | **0.04** | {0.02, 0.04, 0.07, 0.1} |
| α (Beta 分布) | **~0.2-0.4** | {0.2, 0.4, 0.6, 0.8, 1.0} |
| β (EMA) | **0.99** | {0.95, 0.99, 0.999} |
| λ_intra | **2** | {1, 2, 5, 10} |
| λ_inter | **1** | {1, 2, 5, 10} |

⚠️ PACS 的 λ_intra=2 远小于 Digits/Office 的 10 — **数据越小越难,不能压太重 APA loss**。

---

## 7. 借鉴给 FedDSA-SGPA 的判断

### 7.0 我们现有架构回顾

```
x → AlexNet encoder (1024d)
    ├── semantic_head   (1024 → 128) → z_sem (128d)
    │                                    └─→ sem_classifier (128 → 7)
    └── style_head      (1024 → 128) → z_sty (128d)
         └─→ style_bank (per-client μ, σ) → pooled whitening
```

### 7.1 MixUp 套在哪一层?

**答: 套在 `z_sem` (128d 语义空间) 上, 不套在 encoder 1024d.**

**理由**:
1. I2PFL 是把 MixUp 套在**最终 pooled feature** (encoder 的 pooled 输出)
2. 我们的"最终 task-relevant 空间"是 `z_sem` (语义空间,专门为分类优化)
3. 在 1024d 混会把 style 信息也混进去,破坏我们的解耦前提
4. 在 `z_sem` 混等于**在纯语义空间做增强** — 这正是我们想要的 "pure semantic augmentation"

```python
# 建议的伪代码 (嵌入 clientdsa.py local_train 里)
h = encoder(x)                            # [B, 1024]
z_sem = semantic_head(h)                  # [B, 128]
# I2PFL-style intra MixUp on z_sem
idx_neg = _sample_different_class(y)
gamma = Beta(0.4, 0.4).sample([B, 1]).to(device)
z_sem_tilde = gamma * z_sem + (1-gamma) * z_sem[idx_neg]  # 异类混
```

### 7.2 I2PFL 的 prototype 对应我们什么?

| I2PFL 概念 | 我们的对应 |
|-----------|-----------|
| local prototype $p_m^k$ | **client $m$ 的 z_sem class-mean**: $\bar z_{sem}^{m,k}$ |
| initial mean $\mu^k$ | 所有 client 的 $\bar z_{sem}^{m,k}$ 的均值 |
| generalized prototype $g^k$ | distance-reweighted 后的 class-$k$ 语义原型 |
| augmented prototype $\tilde p_m^k$ | MixUp 后 $\tilde z_{sem}$ 的 class-mean |

**存储位置**: 加一个 `proto_bank` dict,key=class_idx, value=128d 向量。**和 style_bank 独立**,不要复用。

### 7.3 复用 style_bank 还是单独加 proto_bank?

**答: 单独加 `proto_bank` (class-indexed), 不复用 style_bank (client-indexed).**

**理由**:
- style_bank 是 **per-client** 的 $(\mu, \sigma)$ → 服务 pooled whitening
- proto_bank 是 **per-class** 的 $g^k$ → 服务 GPCL/APA
- 两者语义不同,不能混。这也符合 I2PFL 的设计: `P = {p_m}` 和 `G = {g^k}` 完全分开

### 7.4 实施优先级排序

按"边际收益 / 实施复杂度"打分:

| 优先级 | 组件 | PACS 收益 | 实施复杂度 | 总评分 |
|:------:|------|:--------:|:---------:|:------:|
| **P1** | **L_APA (feature MixUp + MSE 对齐)** | +3.57 (A) | 低 (纯 client 侧) | ⭐⭐⭐⭐⭐ |
| **P2** | **Reweighting (inter-domain)** | +2.35 vs avg | 低 (server 侧加个 for 循环) | ⭐⭐⭐⭐⭐ |
| **P3** | **EMA β=0.99** | +3.65 (A) | 极低 (server 侧一行) | ⭐⭐⭐⭐ |
| **P4** | L_GPCL (InfoNCE) | **只有 +0.42 (A)** | 中 (需要调 τ) | ⭐⭐ |

🔑 **决策**: **先做 P1+P2+P3 三件套**,**不做 GPCL**。原因:
1. GPCL 在 PACS 上单独加收益极小 (+0.42 Art)
2. 我们已经有 orthogonal + HSIC,再加 InfoNCE 容易梯度冲突 (参考 §19.1 的历史教训)
3. APA (MSE 锚点) 比 GPCL 更安全 — 和 FPL 的 MSE + InfoNCE 组合思路一致

### 7.5 完整嵌入 pseudo code (50 行内)

```python
# clientdsa.py — local_train, 核心新增
class ClientDSA(Client):
    def __init__(self, ...):
        self.lambda_apa = 2.0           # PACS 最优
        self.alpha_mixup = 0.4          # Beta 分布参数
        self.ema_beta = 0.99            # EMA (server 侧用)

    def _mixup_different_class(self, z_sem, y):
        """feature-level MixUp, 异类对"""
        B = z_sem.shape[0]
        # 对每个 i, 在 batch 内找 y != y[i] 的样本
        idx_neg = torch.zeros(B, dtype=torch.long, device=z_sem.device)
        for i in range(B):
            mask = (y != y[i]).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                idx_neg[i] = i  # fallback 自身
            else:
                idx_neg[i] = mask[torch.randint(len(mask), (1,))]
        gamma = torch.distributions.Beta(self.alpha_mixup, self.alpha_mixup)\
                    .sample([B, 1]).to(z_sem.device)
        return gamma * z_sem + (1 - gamma) * z_sem[idx_neg]

    def _compute_class_mean(self, z, y):
        """按 class 算 mean, 返回 [num_classes, d] 的 tensor (NaN 表示无样本)"""
        out = z.new_full((self.num_classes, z.shape[1]), float('nan'))
        for c in y.unique():
            out[c] = z[y == c].mean(dim=0)
        return out

    def local_train_step(self, x, y):
        h = self.encoder(x)                              # [B, 1024]
        z_sem = self.semantic_head(h)                    # [B, 128]
        z_sty = self.style_head(h)                       # [B, 128]
        logits = self.sem_classifier(z_sem)              # [B, 7]

        # 已有 losses
        L_CE   = F.cross_entropy(logits, y)
        L_orth = cos_sq(z_sem, z_sty)
        L_HSIC = hsic(z_sem, z_sty)

        # === I2PFL P1+P3 新增: APA ===
        z_sem_tilde = self._mixup_different_class(z_sem.detach(), y)  # detach 防 MixUp 反传
        proto_tilde_local = self._compute_class_mean(z_sem_tilde, y)  # [K, 128]
        proto_local       = self._compute_class_mean(z_sem, y)
        valid_mask = ~torch.isnan(proto_local).any(dim=1)
        L_APA = F.mse_loss(proto_local[valid_mask], proto_tilde_local[valid_mask])

        L = L_CE + self.lambda_orth * L_orth + self.lambda_hsic * L_HSIC \
              + self.lambda_apa * L_APA
        return L


# serverdsa.py — P2 新增 reweighting + EMA
class ServerDSA(Server):
    def aggregate_prototypes(self, client_protos):
        """client_protos: list of dict {class_k: [d]}"""
        # Step 1: initial mean
        mu = {k: torch.stack([c[k] for c in client_protos if k in c]).mean(0)
              for k in self.all_classes}
        # Step 2: distance
        d_m_k = {(m, k): (p - mu[k]).pow(2).sum()
                 for m, c in enumerate(client_protos) for k, p in c.items()}
        # Step 3: reweighted aggregation
        g_new = {}
        for k in self.all_classes:
            d_sum = sum(d_m_k[(m, k)] for m in range(len(client_protos)) if k in client_protos[m])
            g_new[k] = sum(d_m_k[(m, k)] / d_sum * client_protos[m][k]
                           for m in range(len(client_protos)) if k in client_protos[m])
        # Step 4: EMA
        if not hasattr(self, 'g_prev'):
            self.g_prev = g_new
        else:
            for k in self.all_classes:
                g_new[k] = self.ema_beta * g_new[k] + (1 - self.ema_beta) * self.g_prev[k]
            self.g_prev = g_new
        return g_new
```

### 7.6 ⚠️ 需要注意的陷阱

1. **异类采样 fallback**: 如果 batch 内 class 全一样 (极端情况),`idx_neg = i` 导致 $\tilde z_i = z_i$ → L_APA=0, 安全无副作用
2. **`z_sem.detach()` in MixUp**: MixUp 源 feature 要 detach, 防止增强梯度污染原 feature 的梯度方向 (这是 PARDON 的 style_stat 也用的 trick)
3. **EMA 冷启动**: 第一轮还没 `g_prev`, 直接用 new 值
4. **nan 过滤**: 小 batch 可能某些 class 没样本,class_mean 会 NaN,需 mask 掉
5. **通信开销**: 每 client 每轮多传 `[K, d] = [7, 128] = 896 floats ≈ 3.5KB` — **几乎可忽略**

### 7.7 推荐 PACS 起跑超参 (直接用 I2PFL 最优)

```yaml
lambda_orth: 1.0        # 保留我们现有
lambda_hsic: 0.1        # 保留
lambda_apa: 2.0         # 新 (I2PFL PACS 最优 λ_intra)
alpha_mixup: 0.4        # 新 (Beta 分布, I2PFL 默认)
ema_beta: 0.99          # 新 (I2PFL 最优)
# 不加 GPCL (跳过 L_inter)
```

---

## 8. 一句话总结

> **I2PFL = 客户端做 feature-level 异类 MixUp 产生 augmented prototype 并 MSE 对齐 (APA, λ=2) + 服务器按"离均值越远越重要"加权聚合 class 原型 (reweighting) + EMA 平滑 (β=0.99);简单三板斧打败了 FPL 的 FINCH 聚类和 FedPLVM 的双层聚类,PACS 82.25% AVG,最大贡献者是 APA 不是 GPCL.**

---

## 📌 3 行核心发现

1. **APA (feature MixUp + MSE) 远比 GPCL (InfoNCE) 有用** — 单独 APA 在 PACS Art 涨 +3.57,单独 GPCL 只涨 +0.42; 我们抄 APA 不抄 GPCL 最划算。
2. **Distance-based reweighting 比 FINCH 聚类更强也更简单** — 一行 $g^k = \sum_m \frac{d_m^k}{d^k} p_m^k$ 就比 FPL/FedPLVM 的多原型聚类高 2.35%; 在服务器侧替换我们现在的简单均值可以直接涨点。
3. **PACS 的 λ_intra=2 远小于 Office/Digits 的 10** — 数据难 → APA 要小权重; 我们的 FedDSA 已经在 PACS 领先但 Office 落后 1.49,Office 场景上 I2PFL 把 λ_intra 拉到 10 这个信号非常值得参考。
