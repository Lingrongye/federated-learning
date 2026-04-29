# 论文精读 · F2DC (CVPR 2026)

> **精读日期**: 2026-04-23
> **精读目的**: 评估 F2DC 的 "Calibrate 而非 Eliminate" 思想能否借鉴到我们 FedDSA 方案 (目标打过 FDSE baseline PACS 79.91 / Office 90.58)
> **PDF 位置**: `D:\桌面文件\联邦学习\papers\F2DC_2603.14238.pdf`
> **代码**: https://github.com/mala-lab/F2DC

---

## 1. 基本信息

| 字段 | 内容 |
|------|------|
| **标题** | Domain-Skewed Federated Learning with Feature Decoupling and Calibration |
| **作者** | Huan Wang, Jun Shen (Wollongong), Jun Yan, **Guansong Pang** (SMU) |
| **会议** | CVPR 2026 (arXiv 2603.14238v1, 2026-03-15) |
| **代码** | https://github.com/mala-lab/F2DC |
| **数据集** | Digits-4, Office-Caltech, PACS |
| **Backbone** | ResNet-10, d=512 |

**TL;DR**: 在 domain-skewed FL 场景下,**不要擦除 domain bias**(FDSE 的做法),因为 domain-related feature 里面其实**藏着宝贵的 class-relevant clue**。F2DC 用 DFD (Decoupler) 把 feature 分为 robust 和 related 两路,再用 DFC (Corrector) 把 related 部分**校准**(不是丢弃)成 `f*`,最后 `f+ + f*` 合并喂给分类器。同时服务器端用 Domain-aware Aggregation 按域分布差异赋权。PACS AVG 达 **76.47** vs FDSE 73.13 (两者同 setup)。

🔑 **核心数字**: PACS AVG 76.47 (vs FDSE 73.13, FedHEAL 73.34, FedAvg 66.39)

---

## 2. 核心问题 + 动机 (大白话)

### 2.1 F2DC 对 FDSE 的批评

FDSE 是典型的 **"擦除派"**(elimination-based):把 domain-specific bias 当噪声**删掉/去激活**,保留 domain-agnostic 部分。

F2DC 作者说 FDSE 有个根本问题:

> *"these 'elimination-based' methods risk discarding additional class-relevant information that is entangled with these domain-specific features, e.g., FDSE misses the antlers and head of the giraffe in the cartoon and sketch domains in Fig.2-Middle."*

用 Grad-CAM 可视化:FDSE 在 Cartoon/Sketch 域认长颈鹿时**看不到鹿角和头** — 因为这部分特征在 cartoon/sketch 里是用笔触(stroke style, 典型 domain feature)勾勒的,FDSE 把笔触擦了就连带把头也擦了。

### 2.2 "Calibrate 而非 Eliminate" 精确含义

**Elimination 哲学**(FDSE / FedSeProto / FedDP / FediOS):
- Domain-related feature = 噪声
- 操作 = 擦除 / 压缩 / 投影到正交子空间
- 风险 = **误删**纠缠在 domain context 里的 class-relevant 信号

**Calibration 哲学**(F2DC):
- Domain-related feature = **脏但有用** — 里面同时混了 domain bias(笔触风格)+ class 线索(物体轮廓)
- 操作 = **不丢弃**,而是送进一个 Corrector MLP "洗"一次,把 class 信号显式激活出来
- 结果 = `f* = f⁻ + correction`,再合并 `f⁺` 喂分类器

💡 **关键差异**: F2DC 不是说 domain feature 要全部保留,而是说**擦除会连带把 class 信号一起 kill**,所以要先用 Corrector 把 class 信号从 domain context 中"抢救"出来。

### 2.3 为什么传统去偏不够

从 **dimensional collapse** 角度解释 (论文 Fig.1 的 motivation):
- Vanilla FL: 每个 client 只见一个域,local feature 协方差矩阵的奇异值**大量趋近 0**(表征塌缩到低维子空间)
- 聚合后 global model 依然被单一域偏置锚死,**"一致表征"只在少数维度上达成共识**
- Elimination 方法只是让那个低维子空间更"干净",但**没有拉回失去的维度**
- F2DC 通过 calibration 恢复了 domain-related 维度中的 class 信息 → 奇异值分布更均匀(Fig.1-Bottom)

---

## 3. 方法精读 (按组件拆分)

### 3.1 符号约定

- Client $k$ 的样本 $(x_i, y_i) \in D_k$,$y_i \in \{1, ..., C\}$
- Feature extractor 拆成两段:$r_k^B$ (backbone, 4 个 conv 层)+ $r_k^F$ (pooling + flatten)
- $f_i = r_k^B(x_i) \in \mathbb{R}^{C \times H \times W}$ (ResNet-10 on PACS: $512 \times 16 \times 16$, 这里 C 是 channel 数,不是类别数 — 论文符号冲突 🫠)
- Classifier $h(\theta_k; \cdot): \mathbb{R}^d \to \mathbb{R}^{C_{cls}}$

### 3.2 组件 A: DFD (Domain Feature Decoupler) — Sec 4.1

**核心任务**: 把 feature map $f_i$ 的每个 cell 标注为 "domain-robust" 或 "domain-related"。

**实现**:
1. **Attribution network** $\mathcal{A}_D$: 一个 **两层 CNN**(BN + ReLU),输入 $f_i$,输出 attribution map $S_i = \mathcal{A}_D(f_i) \in \mathbb{R}^{C \times H \times W}$
2. **Gumbel-Concrete mask** (可微离散化,Eq.3):
$$
M_i = \frac{e^{\sigma^{-1}(\log(\epsilon(S_i)) + g_a)}}{e^{\sigma^{-1}(\log(\epsilon(S_i)) + g_a)} + e^{\sigma^{-1}(\log(1 - \epsilon(S_i)) + g_b)}}
$$
其中 $g_a, g_b$ 是 Gumbel noise,$\sigma \to 0$ 趋近硬离散化,$\epsilon$ 是 Sigmoid
3. **Feature 切分** (Eq.4):
$$
f_i^+ = M_i \odot f_i, \quad f_i^- = (1 - M_i) \odot f_i
$$
$f_i^+$ = domain-robust feature, $f_i^-$ = domain-related feature

**训练目标** $\mathcal{L}_{DFD}$ (Eq.5, 双目标):
$$
\mathcal{L}_{DFD} = \underbrace{\log(\exp(s(l_i^+, l_i^-)/\tau))}_{\text{Separability}} - \underbrace{(y_i \log \delta(m(l_i^+)) + \hat y_i \log \delta(m(l_i^-)))}_{\text{Discriminability}}
$$

- $l_i^\pm = r_k^F(f_i^\pm) \in \mathbb{R}^d$ (pooled+flatten 后)
- $s(\cdot)$ = cosine similarity
- $m$ = 一个**单层 MLP** $\mathbb{R}^d \to \mathbb{R}^{C_{cls}}$ (auxiliary classifier)
- $\hat y_i$ = **除 GT 外置信度最高的错标签** (要求 $m(l_i^-)$ 预测到错标签)
- $\tau = 0.06$ (最优,Fig.6)

🔑 **精髓**: Separability 推开 $f_i^+$ 和 $f_i^-$(cosine 越大 loss 越大 → 鼓励正交),Discriminability 要求 $f_i^+$ 能分对(显式让 DFD 把 class 信号推到 $f_i^+$),$f_i^-$ **故意**预测到错标签(迫使 DFD 把 domain 偏置留给 $f_i^-$)。

💡 **难点**: `y_i log δ(m(l_i^+))` 这是标准 CE — 把 class 信号聚到 `f+`。`ŷ_i log δ(m(l_i^-))` 这是**反向 CE** — 让 `f-` 故意倾向错类。这是一种**对抗式教学**:DFD 必须学会把 class 丢进 `f+` 才能让两个 aux classifier 都满意。

**和 FDSE 差异**:

| 维度 | FDSE (CVPR 2025) | F2DC (CVPR 2026) |
|------|------------------|------------------|
| 分解粒度 | **层内** — 每个 conv 层拆成 DFE + DSE 两个子模块 | **特征级** — 整个 feature map 被 mask 分成两路 |
| 分解方式 | 架构级 (两个不同的小 conv) | 可微 mask (Gumbel) |
| 处理 domain 部分 | **擦除** DSE (基于 KL 对齐全局统计量) | **校准** 送进 DFC |
| 参数量 | DSE = DFE 的 1/94 | DFD ≈ 两层小 CNN |

**和我们 FedDSA 差异**:

| 维度 | FedDSA (我们) | F2DC |
|------|--------------|------|
| 解耦位置 | pooled 后 1024d → 两个 128d Linear head | pooled 前,feature map 层 |
| 解耦约束 | `L_orth = cos²(z_sem, z_sty)` + HSIC | Gumbel mask + separability + discriminability |
| Style 载体 | pooled 后的 128d 向量 z_sty,(μ,σ) 作为风格指纹 | feature map $f^-$,3D tensor |
| 全局对齐 | Style bank 跨 client 广播 | DFC 私有,不共享 |

### 3.3 组件 B: DFC (Domain Feature Corrector) — Sec 4.2 **核心贡献**

**任务**: 把 $f_i^-$ 校准成 $f_i^*$,让它从"domain 偏置为主"变成"class 线索补充"。

**架构** (同 DFD,两层 CNN + BN + ReLU):

**残差式修正**(Eq.6):
$$
f_i^* = f_i^- + (1 - M_i) \odot \mathcal{A}_C(f_i^-)
$$

- 注意 `(1 - M_i)` mask 再乘一次 — 确保 corrector **只在 domain-related 位置做修正**,不动 robust 位置
- **残差连接**:$f^*$ 从 $f^-$ 出发,只加增量 — 比直接输出更稳定

**训练目标** $\mathcal{L}_{DFC}$ (Eq.7):
$$
\mathcal{L}_{DFC} = -y_i \log \delta(m(l_i^*)), \quad l_i^* = r_k^F(f_i^*)
$$

**用同一个 aux classifier $m$**,要求 $f_i^*$ 能预测到**正确**类。对比:
- $m(l^+)$: 应预测 $y_i$ (DFD 正向目标)
- $m(l^-)$: 应预测 $\hat y_i$ (DFD 反向目标)
- $m(l^*)$: 应预测 $y_i$ (DFC 目标 — 把 class 信号"洗"出来)

**最终 feature** (Eq.8 上方):
$$
\tilde f_i = f_i^+ + f_i^* \in \mathbb{R}^{C \times H \times W}
$$

然后喂 pooling + classifier:
$$
\ell_i = h(\theta_k; r_k^F(\tilde f_i))
$$

### 3.4 **为什么是 "Calibrate 而不 Eliminate"?**

**原文引用** (Sec 4.1 末尾 / Introduction):

> *"this domain context can contain some inherently entangled class-relevant information and noisy domain biases, on which $\mathcal{A}_D$ cannot perfectly isolate all discriminative signals into $f^+$, i.e., $\mathcal{A}_D$ inevitably relegates those 'mixed' features into $f^-$. Therefore, the resulting $f_i^-$ is often a complex mixture of domain artifacts and valuable class-relevant clues."*

> *"rather than eliminating domain-specific biases as recent domain-skewed FL methods do, we argue that these biases are inherently entangled with valuable class-relevant information, once properly calibrated, can be leveraged to produce more consistent decisions across domains."*

**大白话**: DFD 再强也不可能 100% 分干净 — class 信号和 domain 偏置天生纠缠。所以 $f^-$ 里**总有一些漏网的 class 信号**。丢了它就亏了(FDSE 亏在这),所以用一个专门的 Corrector 把它**洗**一遍再加回来。

**数学上 calibrate 等价于**: 一个**信号恢复 (signal recovery)** 操作 — 给定一个 corrupted signal $f^-$ (= domain bias + 残余 class signal),用一个监督网络 $\mathcal{A}_C$ 学习从中提取 class-relevant 残差:
$$
\mathcal{A}_C(f^-) \approx \text{class-relevant-residual}(f^-)
$$

验证证据 (Table 7): $f^*$ 比 $f^-$ 在 PACS AVG 从 57.87 涨到 **73.49**(+15.62pp!),证明 $f^-$ 里确实藏着巨大的 class 信号,单用 $f^+$ (75.13) 涨到 $\tilde f = f^+ + f^*$ 后 76.47 — 虽然只涨 1.34 但说明两者**互补**。

### 3.5 组件 C: DaA (Domain-aware Aggregation) — Sec 4.3

**目标**: 服务器端**不要**简单按数据量加权,而是**按域代表性**加权。

**Step 1 — 定义 ideal 参考分布** (均匀域分布):
$$
G = [1/Q, 1/Q, ..., 1/Q] \in \mathbb{R}^{C_{cls}}
$$
$Q$ 是域数 (PACS Q=4)。

**Step 2 — 定义 client k 的 domain distribution**:
$$
B_k = [n_k^1/N, ..., n_k^{C_{cls}}/N] \in \mathbb{R}^{C_{cls}}
$$
注意:在 domain skew (label consistent) 假设下 $n_k^c \approx n_k / C_{cls}$,所以 $B_k$ 其实每个分量都是 $n_k / N$ 左右 — 这个设计只对**数据量差异大**的 client 有区分力。

**Step 3 — 计算 domain discrepancy** (Eq.10, 欧氏距离):
$$
d_k = \sqrt{\frac{1}{2} \sum_{c=1}^{C_{cls}} (B_k^c - G^c)^2}
$$

**Step 4 — 加权** (Eq.11):
$$
p_k = \frac{\epsilon(\alpha \cdot n_k/N - \beta \cdot d_k)}{\sum_{j=1}^K \epsilon(\alpha \cdot n_j/N - \beta \cdot d_j)}
$$
$\epsilon$ = Sigmoid,$\alpha = 1.0$,$\beta = 0.4$ (最优)。

💡 **解读**: $\alpha$ 偏向大 client(保留 FedAvg 的 data-size weighting),$\beta$ **惩罚**偏离均匀分布的 client(domain 偏的 client 权重低)。这个 weighting 不像 FedDisco 用全局 label distribution,而是**一个 scalar 距离**(所以不如 FedDisco 细腻)。

**聚合**:
$$
w^* = \sum_{k=1}^K p_k \cdot w_k
$$

**重要**: DFD $\mathcal{A}_D$ / DFC $\mathcal{A}_C$ / aux classifier $m$ **全部保留本地**,不参与聚合。服务器只聚合主干 + 主分类器。

### 3.6 组件 D: Loss 总式

**每个 client 的训练目标** (Eq.9):
$$
\mathcal{L} = \mathcal{L}_{CE} + \frac{1}{|L|} \sum_{j=1}^{|L|} (\lambda_1 \cdot \mathcal{L}_{DFD}^{L_j} + \lambda_2 \cdot \mathcal{L}_{DFC}^{L_j})
$$

- $\mathcal{L}_{CE}$: 主分类器的 CE
- $|L| = 1$ (default, 只在最后一个 backbone 层后插 DFD/DFC)
- $\lambda_1 = 0.8$, $\lambda_2 = 1.0$

**所有 loss 共用同一个 aux MLP $m$**(注意:这个 $m$ 可能是 F2DC 训练稳定的关键 — 不是每个 feature 独立分类器)。

---

## 4. 算法流程 (Alg.1 中文)

```
输入: 通信轮数 R=100, 本地 epoch E=10, 客户端数 K, 客户端 k 的数据 D_k 和模型 w_k
输出: 最终全局模型 w*_R

for r = 1, 2, ..., R:
    K_r = 随机选择的客户端子集
    for each client k ∈ K_r 并行:
        w_k ← LocalUpdating(w*_r)
    end for

    # 域感知聚合 (Sec 4.3)
    d_k ← 用 Eq.10 从 (B_k, G) 算 domain discrepancy
    p_k ← 用 Eq.11 从 (d_k, n_k) 算权重
    w*_{r+1} ← Σ p_k · w_k

return w*_R

---

LocalUpdating(w*_r):
    w_k ← w*_r + {A_D, A_C, m}  # 本地组件插回来
    for epoch = 1, ..., E:
        for batch b ⊂ D_k:
            # DFD (Sec 4.1)
            M_i ← A_D(f_i) 经过 Gumbel (Eq.3)
            f+_i, f-_i ← 切分 (Eq.4)
            L_DFD(f+_i, f-_i, y_i, ŷ_i, m) per Eq.5

            # DFC (Sec 4.2)
            f*_i ← A_C(f-_i, M_i) 残差修正 (Eq.6)
            L_DFC(f*_i, y_i, m) per Eq.7

            # CE (主任务)
            L_CE(ℓ_i, y_i) per Eq.8  # ℓ_i = h(θ_k; r^F_k(f+_i + f*_i))

            L = L_CE + (λ_1 L_DFD + λ_2 L_DFC)
            w_k ← w_k - η ∇L
        end for
    end for

    w_k ← w_k \ {A_D, A_C, m}  # 本地组件剥离,不回传
    return w_k 到服务器
```

---

## 5. 实验 Setup (与我们不一样!) 🚨

| 维度 | F2DC | FedDSA (我们) / FDSE 本地复现 |
|------|------|------------------------------|
| Backbone | **ResNet-10** (d=512) | ResNet-18 / AlexNet |
| PACS Client 数 | **10** | 4 (每域 1 client) |
| PACS 域→client 映射 | P:2, AP:3, Ct:2, Sk:3 | 1:1 (4 client) |
| Data 使用量 | **30% of PACS** | 100% |
| Office-10 Client 数 | 10 (C:3, A:2, W:2, D:3) | 4 |
| Office-10 Data 使用量 | **20%** | 100% |
| Digits Client 数 | 20 (M:3, U:6, SV:6, SY:5) | — |
| Digits Data 使用量 | **1%** (极少!) | — |
| Rounds R | 100 | 200 |
| Local epoch E | 10 | 5 |
| Batch size | 64 | 24~64 |
| Optimizer | SGD, lr=0.01, momentum=0.9, wd=1e-5 | SGD (类似) |

🚨 **关键发现**: F2DC 的 PACS setup (**10 client, 30% data**) 比我们的 setup (**4 client, 100% data**) **困难得多** — 这就解释了为什么 F2DC 报的 FDSE 数字是 **73.13**,而我们复现 FDSE 是 **79.91**。

**不能直接比较** F2DC 的 76.47 和我们的 80.64 — 两者在完全不同的数据规模和异构度上。

---

## 6. PACS 完整主表 (F2DC 的 setup, 不等于我们的)

### 6.1 PACS (10 client, 30% data, R=100, ResNet-10)

| 方法 | Photo | Art | Cartoon | Sketch | **AVG↑** | STD↓ |
|------|:-----:|:---:|:-------:|:------:|:--------:|:----:|
| FedAvg | 60.98 | 52.49 | 74.63 | 77.46 | 66.39 | 11.74 |
| FedProx | 61.88 | 57.40 | 76.34 | 80.14 | 68.94 | 11.00 |
| MOON | 59.72 | 50.29 | 70.35 | 70.22 | 62.64 | 9.63 |
| FPL [CVPR'23] | 66.67 | 58.38 | 79.75 | 77.59 | 70.59 | 9.95 |
| FedTGP [AAAI'24] | 60.68 | 56.06 | 75.27 | 78.03 | 67.51 | 10.78 |
| FedRCL [CVPR'24] | 61.37 | 55.17 | 70.78 | 76.57 | 65.97 | 9.54 |
| FedHEAL [CVPR'24] | 70.39 | 65.82 | 80.57 | 76.61 | 73.34 | 6.54 |
| FedSA [AAAI'25] | 68.55 | 60.42 | 78.32 | 75.50 | 70.69 | 7.98 |
| FDSE [CVPR'25] | 69.27 | 65.46 | 78.65 | 79.12 | **73.13** | 6.83 |
| **F2DC (Ours)** | **75.15** | **68.71** | **79.91** | **82.11** | **76.47** | **5.83** |

### 6.2 Office-Caltech (10 client, 20% data, R=100)

| 方法 | C | A | W | D | **AVG↑** | STD↓ |
|------|:---:|:---:|:---:|:---:|:--------:|:----:|
| FedAvg | 59.82 | 65.26 | 51.72 | 46.67 | 55.86 | 8.27 |
| FDSE | 60.39 | 66.80 | 58.31 | 67.20 | 63.18 | 4.50 |
| **F2DC** | **62.95** | **68.42** | **64.79** | **71.12** | **66.82** | **3.65** |

### 6.3 Digits (20 client, 1% data)

| 方法 | M | U | SV | SY | **AVG↑** | STD↓ |
|------|:---:|:---:|:---:|:---:|:--------:|:----:|
| FedAvg | 96.04 | 89.84 | 88.04 | 51.05 | 81.24 | 20.42 |
| FDSE | 95.17 | 90.34 | 90.98 | 60.13 | 84.15 | 16.19 |
| **F2DC** | **97.75** | **92.94** | 90.53 | **67.69** | **87.23** | **13.36** |

### 6.4 FDSE 数字差异的解释

🔑 **F2DC 报的 FDSE PACS 73.13 vs 我们本地复现 79.91 (3-seed R200)**:
- F2DC 只用 **30% PACS 数据** → FDSE 欠拟合
- F2DC client 数 10 而非 4 → 每个 client 数据更少
- F2DC 只跑 R=100 → FDSE 未充分收敛 (我们跑 R=200)
- 这三个因素叠加让 FDSE 掉 **~7pp**,所以 **F2DC 报的 76.47 ≈ 我们 setup 下 ~80+**

**启示**: F2DC 的 "+3.34pp over FDSE" 在我们的 setup 下**未必能 reproduce**(因为我们的 FDSE 已经接近 paper 上限了)。

---

## 7. Ablation (关键!)

### 7.1 Table 6 — 组件消融 (PACS)

| $L_{DFD}$ | $L_{DFC}$ | DaA |     P     |    AP     |    Ct     |    Sk     |      **AVG↑**      |   STD↓   |
| :-------: | :-------: | :-: | :-------: | :-------: | :-------: | :-------: | :----------------: | :------: |
|     -     |     -     |  -  |   60.98   |   52.49   |   74.63   |   77.46   |       66.39        |  11.74   |
|     ✓     |     -     |  -  |   61.28   |   57.64   |   75.71   |   79.09   |   68.43 (+2.04)    |  10.15   |
|     ✓     |     ✓     |  -  |   72.95   |   61.52   |   79.06   |   80.13   |   73.41 (+7.02)    |   8.35   |
|     ✓     |     -     |  ✓  |   72.55   |   65.44   |   77.61   |   78.95   |   73.64 (+7.25)    |   6.12   |
|     -     |     -     |  ✓  |   74.86   |   65.90   |   79.70   |   80.86   |   75.33 (+8.94)    |   6.80   |
|     ✓     |     ✓     |  ✓  | **75.15** | **68.71** | **79.91** | **82.11** | **76.47 (+10.08)** | **5.83** |

🔑 **关键发现**:
1. **DaA 单独就涨 +8.94pp** (!) — 在 F2DC 的 setup (10 client + 30% data) 下,**简单的域加权聚合**就是最大贡献者。这是因为 10 个 client 数据量不均衡放大了 DaA 的作用。
2. DFD 单独只涨 +2.04,加 DFC 才涨到 +7.02 → **DFC 才是 decoupling 发光的关键**
3. **对 Art 的贡献特别大**: Art +16.22 (最难的域)

### 7.2 Table 7 — $f$ 各成分消融 (PACS)

| 用哪个 feature 喂分类器 | P | AP | Ct | Sk | **AVG↑** | STD↓ |
|-------------------------|:-:|:--:|:--:|:--:|:--------:|:----:|
| $f^+$ 单用 | 74.21 | 67.52 | 78.35 | 80.47 | 75.13 | 5.90 |
| $f^-$ 单用 | 59.45 | 50.09 | 52.56 | 69.37 | 57.87 | 8.63 |
| $f^*$ 单用 (校准后) | 74.83 | 64.05 | 75.80 | 79.29 | 73.49 | 6.71 |
| $\tilde f = f^+ + f^*$ | **75.15** | **68.71** | **79.91** | **82.11** | **76.47** | 5.83 |

🔑 **核心 takeaway**:
- $f^-$ 单用只 57.87 (很差) → 证明 DFD 确实把 domain 偏置隔离到 $f^-$
- $f^*$ (校准后的 $f^-$) 涨到 73.49 (**+15.62**) → **这 15.62pp 就是 Calibration 思想的净收益**
- $\tilde f = f^+ + f^*$ 比 $f^+$ 单独涨 **+1.34** → DFC 提供的是 **互补**信息,不是替代

### 7.3 Table 3 — Modularity (关键!)

F2DC 的 DFD + DFC 可以**插到其他 FL 方法**作为 plugin:

| 基线方法 | P | AP | Ct | Sk | **AVG↑** |
|---------|:-:|:--:|:--:|:--:|:--------:|
| FedAvg + DFD + DFC | 74.86 (+13.88) | 65.90 (+13.41) | 79.70 (+5.07) | 80.86 (+3.40) | 75.33 (**+8.94**) |
| FPL + DFD + DFC | 76.85 (+10.18) | 64.46 (+6.08) | 80.02 (+0.27) | 80.75 (+3.16) | 75.52 (+4.93) |
| FedHEAL + DFD + DFC | 75.56 (+5.17) | 65.95 (+0.13) | 81.27 (+0.70) | 77.48 (+0.87) | 75.06 (+1.72) |
| FDSE + DFD + DFC | 72.25 (+2.98) | 65.99 (+0.53) | 80.56 (+1.91) | 80.36 (+1.24) | 74.79 (+1.66) |

💡 **启示**: DFD + DFC **plug-and-play**,对 FedAvg 涨最多 (+8.94), 对 FDSE 只涨 +1.66 (因为 FDSE 本身已经做了解耦 — 冲突)。

### 7.4 超参敏感性

- **$\tau$ (Eq.5 Gumbel 温度)**: 最优 0.06, 区间 [0.05, 0.07] 稳定
- **$\sigma$ (Eq.3 mask smoothness)**: 最优 0.1
- **$\alpha, \beta$ (Eq.11 聚合权重)**: $\alpha \in [0.8, 1.2]$, $\beta \in [0.3, 0.5]$
- **$\lambda_1, \lambda_2$ (Eq.9 loss 权重)**: $\lambda_1 = 0.8, \lambda_2 = 1.0$,对极端值敏感

---

## 8. 能借鉴给我们 (重点,至少 800 字)

### 8.1 定位对照

**我们 FedDSA 现有架构**:
```
AlexNet encoder
  → pooled feature (1024d)
  ├─→ semantic_head (Linear 128d)  ["domain-robust" 对应物]
  └─→ style_head (Linear 128d)     ["domain-related" 对应物]
  + sem_classifier(128d → num_classes)

+ style_bank: 跨 client 广播 (μ, σ) 作为 style asset
+ AdaIN augmentation: z_aug = γ·z_sem + β  (γ,β 来自 bank 采样)
+ L_orth = cos²(z_sem, z_sty), L_hsic, L_infonce
```

**F2DC 架构**:
```
ResNet-10 backbone
  → feature map f (512 × H × W)
  ├─→ A_D (两层 CNN) → mask M
  ├─→ f^+ = M ⊙ f    [domain-robust]
  └─→ f^- = (1-M) ⊙ f [domain-related]
  → A_C (两层 CNN) → f* = f^- + (1-M) ⊙ A_C(f^-)  [calibrated]
  → classifier(f^+ + f*)

+ aux MLP m (shared across DFD/DFC)
+ DaA: 服务器按 domain discrepancy 加权聚合
```

### 8.2 **F2DC 与 FedDSA 的根本哲学差异**

| 维度 | F2DC (私有 calibrate) | FedDSA (跨共享 augment) |
|------|----------------------|-------------------------|
| **Style 去向** | 本地 DFC 网络消化 → 变成 $f^*$ 喂回分类器 | 上传 (μ,σ) 到 style_bank → 跨 client 广播 |
| **Style 用途** | 校正 (recover class signal) | 增强 (cross-domain augment) |
| **隐私模型** | DFC/DFD 严格本地,只传主干 | style 统计量跨 client 共享 |
| **核心假设** | domain-related feature **里藏着 class 线索**, 要洗出来 | domain-related feature **本身有用**, 给别人增强 |
| **Novelty 卖点** | Calibrate vs Eliminate | Share vs Private |

**这两条路径可以组合!** 下面给 3 个优先级方案。

### 8.3 能借鉴的 3 个方案(优先级从高到低)

---

#### **方案 1 (⭐⭐⭐ HIGH PRIORITY): 给 z_sty 加一个 Corrector head (F2DC 思想本地化)**

**动机**: 我们目前 style_head 输出的 z_sty 只用于 (a) 正交损失约束 (b) 产出 (μ,σ) 进 style_bank。但 z_sty 里**可能也藏着 class 信号** — 我们从来没利用过!F2DC 证明这部分可挖出 **+15.62pp** 的价值。

**Office baseline 差距 -1.49 ~ -1.90**,这可能是我们缺的那一块。

**实现**:
```python
# PFLlib/system/flcore/clients/clientdsa.py 新增
self.corrector_head = nn.Sequential(
    nn.Linear(128, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 128)  # 输出校正量
)
self.aux_classifier = nn.Linear(128, num_classes)

# forward
z_sem = semantic_head(pooled)   # [B, 128]
z_sty = style_head(pooled)      # [B, 128]

# F2DC 式 calibration: 残差修正 style feature
z_sty_corrected = z_sty + self.corrector_head(z_sty)

# Calibrate 监督信号: 让校正后的 style 能预测正确 class
logits_corrected = self.aux_classifier(z_sty_corrected)
L_calibrate = F.cross_entropy(logits_corrected, labels)

# 最终分类器输入: z_sem + z_sty_corrected (类似 F2DC 的 tilde f)
z_final = z_sem + z_sty_corrected
logits = sem_classifier(z_final)
L_CE = F.cross_entropy(logits, labels)

# Total
L_total = L_CE + λ_cal * L_calibrate + λ_orth * L_orth + ...
```

**预期收益**:
- Office: +1.5~2.5pp (填上 -1.49 差距)
- PACS: 稳中有升,不退 (+0.3~0.8)
- 额外收益: 有一个可 probe 的 "style → class" 信号路径

**成本**: 两个小 Linear + 一个 BN, << 1% 参数量。aux_classifier 每 epoch 多一次 CE forward,< 5% 训练时间。

**风险**:
- L_CE 和 L_calibrate 可能梯度冲突 (类似我们之前遇到的 InfoNCE 问题) → 需要 warmup, λ_cal 从小开始
- Corrector 可能在 z_sty 已经被 L_orth 约束得很"瘦"时学不到东西 → 可能要放松 L_orth
- style_bank 还要不要用?建议 **保留**,corrector 作用于 **本地** z_sty,style_bank 作用于 **z_sem 增强**,两者在不同维度

---

#### **方案 2 (⭐⭐ MEDIUM): 用 F2DC 的 DaA 替换我们的 FedAvg 聚合**

**动机**: DaA 在 F2DC 独立贡献 +8.94pp (Table 6),而我们现在用的是 FedAvg (按 n_k 加权)。虽然我们 setup 只有 4 个 client 和均匀数据量,DaA 收益会小很多,但值得一试。

**实现** (修改 `PFLlib/system/flcore/servers/serverdsa.py`):

```python
def aggregate(self):
    # 原 FedAvg: weights = [n_k / N for n_k in client_data_sizes]
    # 改 DaA
    n_sizes = [c.train_samples for c in self.selected_clients]
    N = sum(n_sizes)

    # Step 1: B_k 用 label distribution 近似 (PFLlib 有 per-client label counts)
    # Step 2: G = uniform 1/num_classes
    C = self.num_classes
    G = np.ones(C) / C
    d_list = []
    for c in self.selected_clients:
        B_k = c.label_distribution  # shape [C], sum = 1
        d_k = np.sqrt(0.5 * np.sum((B_k - G) ** 2))
        d_list.append(d_k)

    # Step 3: p_k per Eq.11
    alpha, beta = 1.0, 0.4
    raw = [sigmoid(alpha * (n/N) - beta * d) for n, d in zip(n_sizes, d_list)]
    p_list = [r / sum(raw) for r in raw]

    # Step 4: 加权聚合 (用 p_list 代替原 n_k/N)
    for p_k, client in zip(p_list, self.selected_clients):
        self.add_parameters(p_k, client)
```

**预期收益**:
- PACS / Office: +0.2~0.5pp (我们 client 数少+数据均匀,DaA 收益有限)
- 代价: 几乎零,几十行代码

**风险**:
- 我们的 4-client setup 下 B_k 几乎不变,d_k 信号弱 → DaA 退化成 FedAvg
- 可以考虑 **改 B_k 为 domain distribution** (每个 client 的 "我是哪个域" 标识),这样 d_k 信号才强
- 在 **domain-aware** 设定下,d_k 可改为 "client k 的 domain 与其他 clients 的 domain 差异"

---

#### **方案 3 (⭐ LOW): DFD 的 Gumbel mask + adversarial aux classifier 替换我们的 L_orth + L_hsic**

**动机**: 我们的 `L_orth = cos²(z_sem, z_sty) + HSIC` 是**软正交**,F2DC 的 DFD 用 Gumbel mask **硬分 feature map** + **对抗式 aux classifier** (让 z_sty 故意预测错类) 是更强的分离信号。

**但这个方案 invasiveness 最大** — 要改回到 feature map 层做 mask,而不是 pooled 后做双 head。等于重写 encoder。

**实现要点**:
1. 把 AlexNet pooled 前的 feature map 取出来 (B, C, H, W)
2. 加一个小 CNN `A_D` 输出 attribution map → Gumbel mask
3. 分 $f^+, f^-$,然后各自 pool+Linear
4. aux MLP $m$: $z^+$ 预测 $y_i$,$z^-$ 预测 $\hat y_i$ (错标签)
5. 损失加 Separability + Discriminability (Eq.5)

**预期收益**: 不清楚。F2DC 在 Table 6 里 **L_DFD 单独只涨 +2.04**, 说明 mask 本身没那么强。**DFC 才是价值所在** (方案 1 是对的方向)。

**成本**: 改 encoder forward,改损失计算,调参又是一整个空间。

**建议**: **暂缓**,等方案 1 验证完 Calibration 思想有没有用再考虑。

---

### 8.4 Novelty 差异化论述 (paper writing 用)

**在 paper 里如何描述我们 vs F2DC**:

> "F2DC (CVPR 2026) argues that domain-related features contain entangled class-relevant information, which should be **calibrated locally** via a Domain Feature Corrector rather than eliminated. However, F2DC treats style as a **private local asset** — corrected features are only used by the emitting client. We build on this insight but push further: style statistics carry **cross-client** value. Our FedDSA introduces a **global style bank** where (μ, σ) of decoupled style features are broadcast and consumed by all clients as an augmentation asset (AdaIN-based style swap). This addresses the complementary question F2DC does not: *what if the class-relevant signal entangled in your style can help my training?*"

**2×2 矩阵更新**:

|  | 不共享风格 | 共享风格 |
|--|-----------|---------|
| **不解耦** | FedBN, FedAvg | FISC, StyleDDG |
| **解耦 + 擦除** | FDSE, FediOS | — |
| **解耦 + 校准** | **F2DC (new!)** | **★ FedDSA + Calibrator (我们扩展版)** |
| **解耦 + 保留** | FedSTAR, FedSeProto | **FedDSA (现版)** |

★ 如果实现方案 1,FedDSA 成为**唯一**同时做 "cross-client share" + "local calibrate" 的方法,novelty 最强。

### 8.5 优先级和行动建议

| 方案 | 优先级 | 实现时间 | 预期收益 | Office 是否能过线? |
|------|:------:|:--------:|:--------:|:------------------:|
| **方案 1** (Corrector head) | ⭐⭐⭐ | 2-3h + 1 天调参 | +1.5~2.5pp | **有希望** |
| 方案 2 (DaA 聚合) | ⭐⭐ | 1-2h | +0.2~0.5pp | 不够 |
| 方案 3 (重写 DFD) | ⭐ | 2-3 天 | 不清楚 | 不推荐 |

**推荐行动**:
1. **先做方案 1**,5 个 seed {2, 15, 333, 42, 777} 在 Office 上跑,如果 3-seed mean > 90.58 就 ship,作为 paper 关键 contribution
2. 如果方案 1 work,**再叠加**方案 2 看能不能把 PACS 从 80.64 推到 82+
3. 方案 3 只在方案 1 失败时考虑 (作为"真正的 F2DC 移植")

---

## 9. 一句话总结

F2DC 最大的贡献是**"Calibrate 而非 Eliminate"**哲学 — 证明了即使是强的 domain-skewed 方法 (FDSE) 也在**误删**藏在 domain-related feature 中的 class 线索,而用一个小的残差 Corrector 就能**抢救 +15.62pp**。对我们最大的启示是:**我们现在的 z_sty 只做约束和增强素材,完全没做"信号挽回"** — 给 z_sty 加一个 Corrector head + aux classifier 监督,极有可能填上 Office baseline 的 -1.49 差距,且和我们 style_bank 的跨共享是**正交**的两个贡献,可以同时成立。

---

**精读人**: Claude (基于 pdfplumber 提取的完整 PDF 文本)
**总字数**: ~5800 字

---

## 10. 2026-04-29 重大纠正 (基于 PDF 实际图表 + 代码 verify)

### 10.1 纠正: setup 跟我们 F2DC 项目 100% 一致

> ⚠️ **本笔记上面 §5 写的 "我们 4 client, 100% data" 是早期 FedDSA 项目的对比, 不适用于现在 F2DC 项目**.

**直接对照 PDF page 6 + 我们 dataset 代码** (`F2DC/datasets/officecaltech.py:65 percent_dict={'caltech':0.2, 'amazon':0.2, 'webcam':0.2, 'dslr':0.2}`, `F2DC/datasets/pacs.py:66 percent_dict={'photo':0.3, 'art':0.3, 'cartoon':0.3, 'sketch':0.3}`):

| 维度 | 论文 | 我们 F2DC 项目 | 一致 |
|---|:--:|:--:|:--:|
| Backbone | ResNet-10 d=512 | ResNet-10 d=512 | ✅ |
| **Office Client** | **C:3, A:2, W:2, D:3 (10 client)** | **C:3, A:2, W:2, D:3 (10 client)** | ✅ |
| **Office data** | **20% per client** | **20% per client** | ✅ |
| **PACS Client** | **P:2, AP:3, Ct:2, Sk:3 (10 client)** | **P:2, AP:3, Ct:2, Sk:3 (10 client)** | ✅ |
| **PACS data** | **30% per client** | **30% per client** | ✅ |
| Sequential 不重叠 | 是 | 是 (`not_used_index_dict`) | ✅ |
| R=100, E=10 | ✅ | ✅ | ✅ |

→ **完全一致, 数字可以直接比较**.

### 10.2 纠正: DaA 单独贡献不是 +8.94pp, 是 +1.14 ~ +2.04pp

> ⚠️ **本笔记上面 §7.1 写的 "DaA 单独 +8.94pp" 是误读 Table 6**.

**仔细看 PDF Image Table 6** (PACS bottom half):

| L_DFD | L_DFC | DaA | AVG | 增量解读 |
|:--:|:--:|:--:|:--:|---|
| ❌ | ❌ | ❌ | 66.39 | FedAvg baseline |
| ✅ | ❌ | ❌ | 68.43 | +DFD 单独 (+2.04) |
| ✅ | ✅ | ❌ | 73.41 | +DFD+DFC (+7.02 over baseline, DFC 单独 +4.98) |
| ✅ | ❌ | ✅ | 73.64 | +DFD+DaA |
| ❌ | ✅ | ✅ | 75.33 | +DFC+DaA (**+8.94 over baseline, 是 DFC+DaA 加起来**, 不是 DaA 单独!) |
| ✅ | ✅ | ✅ | 76.47 | full F2DC |

**DaA 单独贡献正确算法**:
- vanilla → +DaA only (无 DFD/DFC): 66.39 → ? (Table 6 没列这一行) — 可推算
- (+DFD+DFC) → full: 73.41 → 76.47 = **+3.06pp** (DaA 在 DFD+DFC 上的边际增量)
- 或 (+DFD) → (+DFD+DaA): 68.43 → 73.64 = **+5.21pp**

→ **DaA 真实单独贡献约 +1.14 ~ +5.21pp**, 不是 +8.94pp.

### 10.3 纠正: Office full F2DC = 66.82, 不是其他数字

PDF Image Table 6 Office (top half) 实际值:

| 配置 | C | A | W | D | **AVG** | STD |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg (vanilla) | 59.82 | 65.26 | 51.72 | 46.67 | **55.86** | 8.27 |
| +DaA | 60.25 | 65.79 | 54.39 | 50.33 | **57.69** | 6.76 |
| +DFC+DaA | 66.68 | 67.53 | 56.55 | 60.02 | **62.70** | 5.30 |
| +DFD+DaA | 63.71 | 68.21 | 59.07 | 67.62 | **64.65** | 4.22 |
| +DFD+DFC | 60.84 | 67.79 | 62.10 | 70.45 | **65.29** | 4.57 |
| **full F2DC** | 62.95 | 68.42 | 64.79 | 71.12 | **66.82** | 3.65 |

**DaA 贡献 (Office, vs 论文)**:
- vanilla → +DaA: 55.86 → 57.69 = **+1.83pp**
- (+DFD+DFC) → full: 65.29 → 66.82 = **+1.53pp**

### 10.4 我们 F2DC 项目实测 vs 论文真实数据对比

#### Office (我们 sc3_v2 R100 主表)

| 项 | 论文 | 我们 | Δ (我们 - 论文) |
|---|:--:|:--:|:--:|
| FedAvg | 55.86 | 57.90 | +2.04 (我们略高) |
| F2DC w/o DaA (= +DFD+DFC) | 65.29 | 60.56 | **-4.73** ⚠️ |
| F2DC full | 66.82 | 63.55 | **-3.27** ⚠️ |
| **DaA 增量 (full vs w/o DaA)** | **+1.53** | **+2.99** | **+1.46** (我们 DaA 涨更多) |

#### PACS (我们 sc3_v2 R100 主表)

| 项 | 论文 | 我们 | Δ |
|---|:--:|:--:|:--:|
| FedAvg | 66.39 | 69.22 | +2.83 |
| F2DC w/o DaA | 73.41 | 71.02 | **-2.39** ⚠️ |
| F2DC full | 76.47 | 72.68 (s=15) | **-3.79** ⚠️ |
| **DaA 增量** | **+3.06** | **+1.66** | -1.40 (我们略弱) |

### 10.5 重大新发现 (paper-grade 重要)

1. **我们 F2DC 复现度只到 ~95% 准** (Office full 63.55 vs 论文 66.82, **-3.27pp**) — 复现质量值得改进
2. **DaA 实现方向正确**, 增量在 office 跟 PACS 都是正向 (+1.66 ~ +2.99pp), 跟论文 +1.53 ~ +3.06pp 同量级
3. **真问题不是 DaA, 是 vanilla F2DC backbone** — F2DC w/o DaA 比论文低 2.4-4.7pp
4. **可能原因**:
   - τ (Eq.5 温度) 论文最优 0.06, 我们用默认 0.1 (论文 Fig 6 显示 τ=0.1 时 PACS 75.83 vs 76.47 最优)
   - λ1/λ2 论文最优 0.8/1.0, 我们用 1.0/1.0
   - F2DC v2 复现 vs 原作者 release 实现差异
   - Random partition seed 不同 (3-2-2-3 fixed allocation 但 sequential partition 时随机种子不同, 每 client 分到的 20%/30% 不同)

### 10.6 下一步行动建议

1. **超参 sweep**: 我们 F2DC 跑 τ ∈ {0.05, 0.06, 0.07}, λ1=0.8, λ2=1.0 的 ablation, 看能否把 vanilla F2DC 提到论文水平 65.29 / 73.41
2. **如果能复现到 95%+**: PG-DFC 的 +0.55-0.75 边际收益就是真信号 (不是 baseline 偏低带来的虚假领先)
3. **paper writing**: 主表数据 vanilla F2DC 用我们自己复现的, 不直接 cite 论文 66.82 (因为 setup 一致下复现度差异是真问题, reviewer 会问)

---

**纠正人**: Claude (基于 PDF Image #13 直接读图 + dataset 代码 verify)
**纠正日期**: 2026-04-29
