# 论文精读：FedCCRL — Federated Domain Generalization with Cross-Client Representation Learning

> 精读时间：2026-04-23
> 读者角度：我们在做 **FedDA（personalized + 本地域测试）**，想借鉴 FedCCRL 的 **CCDT（MixStyle 跨客户端变体）** 移植到 FedDSA-SGPA 的 AlexNet pipeline
> 关键判断：FedCCRL 是 **FedDG LOO（leave-one-domain-out）** 场景，数字**不可直比**我们的 FedDA 设定，但 **CCDT 跨客户端风格混合的机械思想完全可以移植**

---

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | FedCCRL: Federated Domain Generalization with Cross-Client Representation Learning |
| 作者 | Xinpeng Wang, Xiaoying Tang (CUHK-Shenzhen) |
| arXiv | 2410.11267 (v3, 2024-11-04) |
| Venue | 预印本（未明确 venue，看格式像投 CVPR workshop 或 arXiv-only） |
| 代码 | https://github.com/SanphouWang/FedCCRL |
| TL;DR | 轻量 FDG 方法：客户端上传少量样本统计量 (μ,σ) → 服务器构建 style pool → 客户端用 **CCDT（跨客户端 MixStyle）+ DIFP（AugMix）** 做双重增强 + **SupCon + JS divergence** 双阶段对齐，不加额外网络不做对抗，MobileNetV3 backbone，PACS LOO avg 82.46% SOTA |

---

## 2. 核心问题 + 动机

**问题**：FL 场景下做 DG 有两大痛点：
1. **隐私约束**：传统 DG 依赖跨域数据共享（adversarial learning / representation alignment）→ FL 禁止
2. **每个 client 数据量小 + 域单一**：即便能做 representation alignment 也因为 domain diversity 不够而效果差

**现有方法的问题**：
- **Federated adversarial（FedADG）**：discriminator 带来大量计算开销 + model collapse 风险
- **Federated representation alignment（FedSR）**：轻量但单 client 域多样性不足 → 效果有限
- **Federated style transfer（CCST）**：基于 AdaIN + 预训练 VGG + 传输高维特征 embedding → 通信/计算爆炸 + **传 embedding 仍可被解码器反推原图** → 隐私泄漏

**FedCCRL 的动机**：
- 想要 **轻量（无额外网络、无对抗）+ 隐私安全（不传特征 embedding、不传图片）+ 提升跨域多样性** 三合一
- 核心观察：**per-channel (μ, σ) 的统计量**是 **最小充分 style 描述**，传输代价几乎 0，隐私风险低（无法反推原图）

---

## 3. 方法精读

### 3.1 整体架构

```
Client i:
  x → [CCDT] → x^(1) → [DIFP(AugMix)] → x^(1)_final
             ↘ [CCDT] → x^(2) → [DIFP(AugMix)] → x^(2)_final
  同时保留原图 x

  三份输入 {x, x^(1), x^(2)} 都过 encoder h → 得到 {Z, Z^(1), Z^(2)}
  再过 classifier g → 得到 {Ŷ, Ŷ^(1), Ŷ^(2)}

  Loss:
    L_CLS = (1/3) * Σ CE(Ŷ', Y)   对三份输出都算 CE
    L_RA  = 0.5 * (L_SupCon(Z^(1), Z) + L_SupCon(Z^(2), Z))   原图 vs 增强 的 SupCon
    L_JS  = (1/3) * Σ KL(Ŷ', Ȳ)   三份输出的 softmax 对齐到其均值

  L = L_CLS + λ_1 * L_RA + λ_2 * L_JS
```

**通信协议**（关键！）：
- 每轮 client i 上传 **两份东西**：(1) 模型参数 θ_i^t （FedAvg 聚合）；(2) **本地样本统计量的一个子集 F_i**（服务器拼成 global style pool）
- 服务器把 F_pool = ∪_i F_i 整体广播，再让 client i 取 F_pool^i = F_pool \ F_i（**只用别的 client 的 style**，不用自己的）

---

### 3.2 组件 A：CCDT（Cross-Client Domain Transfer）★★★ 核心借鉴点

这是**我们最想移植到 FedDSA 的东西**，精读其每一步。

#### 3.2.1 Statistics 定义（Eq. 6, 7）

对一个样本 `x ∈ R^{C×H×W}`（**输入图像，3 通道，224×224**），逐通道计算：

```
μ(x)_c = (1/(HW)) * Σ_{h,w} x_{c,h,w}               # (C,) vector
σ(x)_c = sqrt( (1/(HW)) * Σ_{h,w} (x_{c,h,w} - μ(x)_c)^2 )   # (C,) vector
```

**关键**：统计量是在**图像像素空间**（3 通道）算的，不是在 CNN 某个中间层的 feature map 上。这是为了配合 MixStyle 的 input-level 应用方式。

⚠️ 原 MixStyle (Zhou ICLR'21) 是在 **CNN 中间 feature map** 上算（比如 ResNet 的 layer1/layer2 后），FedCCRL 把它**挪到了输入层**。这样做有利有弊：
- ✅ 好处：不需要中间层 hook，代码极简；传输的 stats 只有 3 通道 × 2 = 6 个 float
- ❌ 坏处：输入层做 affine 后的图像视觉扭曲大（见论文 Fig 3），风格迁移比 feature 层"粗暴"

#### 3.2.2 Upload Ratio r（Eq. 8）

Client i 每轮上传的统计量集合：
```
F_i = { (μ(x_j), σ(x_j)) | x_j ∈ D_i, j = 1, 2, ..., ⌈r * n_i⌉ }
```

- `r ∈ (0, 1)` 是 hyperparameter
- **论文默认 r = 0.1**，即每轮每个 client 只上传 10% 样本的 stats
- 选哪些样本？论文没明说，看 github 代码应该是**随机抽样或取前 ⌈r·n_i⌉ 个**（非 learned selection）

**Ablation（Figure 6）**：即便 r 非常小（论文显示 r 小到极小值仍然效果稳定），性能几乎不掉 → 证明 **style pool 不需要覆盖所有样本**，几个代表性 style 就够

**通信代价**：PACS 每 client ~600 图 × 10% × 6 floats × 4 bytes ≈ **1.4 KB/round** — 完全可忽略

#### 3.2.3 Server 构建 Style Pool

```python
# Server side
F_pool = union of all F_i from all K*M clients

# 关键：给 client i 下发 F_pool^i = F_pool \ F_i （排除自己的 stats）
for i in clients:
    F_pool_for_i = F_pool - F_i
    send(F_pool_for_i to client i)
```

为什么排除自己的？→ 避免自我风格迁移（空操作），强制 style 来自**其他域**。

#### 3.2.4 CCDT 算法（Algorithm 2，逐行精读）

```python
# Input: batch X = (B, C, H, W), style pool F_pool^i, α = 0.1 (Beta parameter)
# Output: style-transferred batch X_CCDT

X_hat = []
for x in X:                                   # x: (C, H, W)
    # Step 1: compute per-sample stats
    mu_x = x.mean(dim=(1,2))                  # (C,)
    sigma_x = x.std(dim=(1,2))                # (C,)

    # Step 2: instance normalize (去掉自己的 style)
    x_tilde = (x - mu_x[:, None, None]) / sigma_x[:, None, None]   # 白化后的"纯内容"

    # Step 3: UNIFORMLY sample ONE peer style from pool
    (mu_prime, sigma_prime) = uniform_sample(F_pool^i)
    # 即只用**一个**其他 client 的一个样本的 stats,不是取 batch 内其他样本

    # Step 4: sample interpolation weight
    lam ~ Beta(α, α) with α=0.1
    # ⚠️ α=0.1 让 Beta 分布极度 U 型,采到的 λ 大概率接近 0 或 1
    # 这意味着每次要么几乎完全用 peer style,要么几乎不变,稀有时才是真正的插值

    # Step 5: mix stats (注意论文公式变量名有点混乱)
    gamma_mix = lam * mu_prime + (1 - lam) * mu_x          # 新 mean
    beta_mix  = lam * sigma_prime + (1 - lam) * sigma_x    # 新 std
    # ⚠️ 论文把 σ 叫 β_mix,把 μ 叫 γ_mix,和 BN 的 γ/β 是反的,注意区分

    # Step 6: re-style
    x_hat = gamma_mix * x_tilde + beta_mix[:, None, None]
    X_hat.append(x_hat)

X_CCDT = stack(X_hat)
```

**关键实现细节**：
- **α = 0.1**：Beta(0.1, 0.1) 是 U 型分布，λ 大概率接近 0 或 1。**有 ~80% 概率做"强风格替换"或"几乎不变"**
- **Uniform sample 一个 peer**：不是加权平均所有 peer，不是取 batch 内样本，而是**每次从全局 pool 随机抽一个**
- **Trigger probability = 1.0**：每个样本每次 forward **都做** CCDT（原 MixStyle 有 p=0.5 的触发概率，FedCCRL 没有）

#### 3.2.5 💡 和原 MixStyle (Zhou ICLR'21) 的具体差异

| 维度 | 原 MixStyle (ICLR'21) | FedCCRL CCDT |
|------|----------------------|--------------|
| **操作层** | CNN **中间 feature** (e.g., ResNet layer1/2 后) | **输入图像**像素空间 (3 通道) |
| **Mix 来源** | 同 batch 内**随机 permute**另一个样本 | **跨 client 的 style pool** uniform sample |
| **Pool 大小** | 就是当前 batch (B 个样本) | 全局累积的 ⌈r·N⌉ 个样本 stats |
| **触发概率** | p=0.5 每次随机是否做 | p=1.0 总是做 |
| **Beta α** | 0.1（同） | 0.1（同） |
| **Normalize** | 用 per-instance feature stats 白化 | 用 per-image pixel stats 白化 |
| **通信代价** | N/A（单机方法） | ⌈r·n_i⌉ × 6 floats，~1.4KB/round |

**核心精神不变**：用 Beta 分布插值两组 (μ, σ) 来构造新风格。
**FL 适配改动**：把"同 batch 内 permute"改成"从全局 style pool 抽"，这是 FL 场景下的自然延伸。

---

### 3.3 组件 B：DIFP（Domain-Invariant Feature Perturbation via AugMix）

**目的**：CCDT 扰动 domain-specific（颜色、风格），DIFP 扰动 domain-invariant（形状、纹理）—— 两者互补。

**做法**：直接用 AugMix（Hendrycks ICLR'20）这个现成工具（Algorithm 3 在附录）。

```python
# AugMix on single sample x
# 参数 β = 1.0 (Beta/Dirichlet parameter)

x_aug = zeros_like(x)
k ~ Uniform{1, 2, 3}                             # 1-3 条增强链
w = Dirichlet(β, ..., β)                         # k 个链的权重

for i in range(k):
    op1, op2, op3 ~ uniform sample from O        # 从操作集合 O 抽 3 个
    op12 = op2 ∘ op1
    op123 = op3 ∘ op2 ∘ op1
    chain ~ Uniform{op1, op12, op123}            # 随机选链长度
    x_aug += w_i * chain(x)                      # element-wise 加权累加

m ~ Beta(β, β)                                   # 原图 vs 增强图 的权重
x_hat = m * x + (1-m) * x_aug                    # 最终混合
```

操作集合 O：与原 AugMix 相同（rotate, shear, translate, posterize, solarize, autocontrast, equalize, ...）

**CCDT ∘ DIFP 的顺序**（Eq. 3, `M = CCDT ∘ DIFP`）：
- 先 CCDT 再 DIFP（从下往上读操作符：`M(X) = CCDT(DIFP(X))`？或 `M(X) = DIFP(CCDT(X))`？）
- 实际从 Figure 3 的可视化看：**先 CCDT（换颜色），再 DIFP（加几何扰动）**
- 所以 `M(X) = DIFP(CCDT(X))` 更符合图示

---

### 3.4 组件 C：L_RA（Representation Alignment，SupCon 式）

**目的**：在 representation space 让 {原图, 增强图} 的同类特征靠近，异类推开。

#### 公式（Eq. 11, 12）

对两个 batch 的表征 `Z', Z'' ∈ R^{B×V}`（V 是 representation 维度）和标签 `Y', Y''`：

```
Z = cat(Z', Z'')  ∈ R^{2B × V}
Y = cat(Y', Y'')  ∈ R^{2B}

I = {1, ..., 2B}, A(i) = I \ {i}, P(i) = {p ∈ A(i) | y_p = y_i}

s(Z_i, Z_p) = exp( cos_sim(Z_i, Z_p) / τ )

L_SC(Z', Z'') = Σ_{i ∈ I} (-1/|P(i)|) Σ_{p ∈ P(i)} log[ s(Z_i, Z_p) / Σ_{a ∈ A(i)} s(Z_i, Z_a) ]
```

**对齐哪两对**：
```
L_RA = 0.5 * ( L_SC(Z^(1), Z) + L_SC(Z^(2), Z) )
```

即 **两个增强 batch 分别对齐原始 batch**，不是两个增强 batch 之间对齐。

**超参**：
- `τ = 0.1`（温度）
- `λ_1` 默认值论文里做 λ_2 ablation 时固定在 0.1（评估 λ_2 时），评估 λ_1 时 λ_2 固定为 1.0

---

### 3.5 组件 D：L_JS（Prediction Alignment via Jensen-Shannon Divergence）

**目的**：在 **prediction (softmax) 空间**保证三份输入（原图 + 两份增强）的输出一致。

#### 公式（Eq. 13, 14）

```
Ȳ = (1/3) * (Ŷ + Ŷ^(1) + Ŷ^(2))          # 三个预测分布的均值

L_JS = (1/3) * ( KL(Ŷ, Ȳ) + KL(Ŷ^(1), Ȳ) + KL(Ŷ^(2), Ȳ) )
```

其中 `Ŷ = softmax(g(Z))` — 即 **softmax 概率分布**（不是 logits）。

**注意**：JS divergence 天然对称 + bounded，比单纯 KL 稳定。论文没明说温度参数，应该是 softmax 默认温度 = 1。

---

### 3.6 Total Loss（Eq. 15, 16）

```
L_CLS = (1/3) * Σ_{Y' ∈ {Ŷ, Ŷ^(1), Ŷ^(2)}} CE(Y', Y)   # 三份输出都做 CE
L_RA = 0.5 * (L_SC(Z^(1), Z) + L_SC(Z^(2), Z))
L_JS = (1/3) * (KL(Ŷ,Ȳ) + KL(Ŷ^(1),Ȳ) + KL(Ŷ^(2),Ȳ))

L = L_CLS + λ_1 * L_RA + λ_2 * L_JS
```

**超参默认**：λ_1 = 0.1, λ_2 = 1.0（从 Figure 6 推测）。ablation 显示对超参**不敏感**，在合理范围（0.01 ~ 10）都差不多。

---

## 4. 算法流程（Algorithm 1）

```
Input: r ∈ (0,1), λ_1, λ_2, T rounds, E epochs
Output: θ

1. Server initializes f, broadcasts to clients
2. for round t = 1..T:
3.    for each client i:
4.        compute F_i (⌈r·n_i⌉ style stats)
5.        upload F_i to server
6.    Server constructs F_pool, distributes F_pool^i = F_pool \ F_i to client i
7.    for each client i:
8.        for each epoch e = 1..E:
9.            for each batch X:
10.                X^(1), X^(2) = M(X), M(X)  # CCDT + DIFP
11.                Z, Z^(1), Z^(2) = h(X), h(X^(1)), h(X^(2))
12.                Ŷ, Ŷ^(1), Ŷ^(2) = g(Z), g(Z^(1)), g(Z^(2))
13.                compute L_RA, L_JS, L_CLS
14.                L = L_CLS + λ_1 * L_RA + λ_2 * L_JS
15.                update θ_i via Adam
16.        upload θ_i^t to server
17.    θ^{t+1} = FedAvg({θ_i^t})
```

---

## 5. 实验 Setup

| 参数 | 值 |
|------|-----|
| Backbone | **MobileNetV3-Large**（最后 FC 当 classifier，前面当 encoder） |
| Communication rounds T | **10**（很少！这是 FedDG LOO 特征，样本训练轮次够用） |
| Local epochs E | **3** |
| Optimizer | Adam, lr=0.001, cosine scheduler |
| α (Beta for CCDT) | **0.1** |
| β (Beta/Dirichlet for DIFP/AugMix) | **1.0** |
| τ (SupCon temperature) | **0.1** |
| Upload ratio r | **0.1** |
| λ_1, λ_2 | λ_1=0.1, λ_2=1.0 |
| 图像大小 | PACS/OfficeHome 224×224, miniDomainNet 128×128 |
| 种子重复 | 3 次独立实验取平均 |
| 数据集 | PACS, OfficeHome, miniDomainNet |
| Client partition | 每个 source domain 分到 K 个 client（K 可变），LOO 留一个域测试 |
| 对比 baseline | FedAvg, FedProx, FedADG, GA, FedSR, FedIIR, CCST |

**关键**：只训 **10 rounds × 3 epochs = 30 epoch equivalent**，比我们 R200 短得多 → 也印证 MobileNetV3 + pretrained weights 是基础（论文没明说，但 10 rounds 这么快收敛只可能是预训练起步）。

---

## 6. PACS LOO 结果表（Table 1，6 clients 设置）

**重要语义澄清**：PACS LOO = **留一个域当测试**，剩下 3 个域的 data 分到 6 个 client 训练。比如"P 列" = 留 Photo 不训，用 A+C+S 训练后 Photo zero-shot 测。

| Method | P | A | C | S | PACS Avg |
|--------|---|---|---|---|----------|
| FedAvg | 90.48 | 69.19 | 76.15 | 71.62 | 76.86 |
| FedProx | 90.60 | 69.09 | 74.32 | 70.15 | 76.04 |
| FedADG | 91.02 | 65.04 | 72.95 | 65.82 | 73.71 |
| GA | 92.81 | 66.50 | 76.19 | 69.56 | 76.26 |
| FedSR | 91.74 | 72.36 | 75.55 | 70.48 | 77.53 |
| FedIIR | 91.26 | 71.73 | 77.94 | 71.32 | 78.06 |
| CCST | 90.16 | 76.37 | 76.11 | 78.58 | 80.30 |
| **FedCCRL** | **93.83** | **79.15** | **77.9** | **78.95** | **82.46** |

**OfficeHome LOO Avg（4 域）**：FedCCRL 68.31 vs FedAvg 66.37 （Δ +1.94）
**miniDomainNet LOO Avg（4 域）**：FedCCRL 62.14 vs FedAvg 59.77 （Δ +2.37）

**⚠️ 我们的数字不可直比**：
- FedCCRL 场景：**训练时看不到 target domain**，zero-shot
- 我们 FedDSA 场景：**target domain 也有参训 client**，本地 test
- FedCCRL 的 78.95 on Sketch 是 "zero-shot sketch"，我们的 80.64 PACS AVG 是 "train-and-test on same domain partition"
- **两套分数分布在两条完全不同的评估曲线上**

---

## 7. Ablation（Table 2，6 clients, r=0.1）

| CCDT | DIFP | L_RA | L_JS | PACS | OfficeHome | miniDomainNet |
|:----:|:----:|:----:|:----:|:----:|:----------:|:-------------:|
|      |      |      |      | 79.92 | 66.58 | 59.57 |
| ✓    |      |      |      | 79.32 | 66.84 | 59.64 |
|      | ✓    |      |      | 80.52 | 67.34 | 60.82 |
| ✓    | ✓    |      |      | 80.48 | 67.29 | 60.40 |
| ✓    |      | ✓    |      | 80.95 | 66.93 | 61.43 |
|      | ✓    | ✓    |      | 80.62 | 67.29 | 60.94 |
| ✓    | ✓    | ✓    |      | 81.18 | 67.81 | 61.47 |
| ✓    |      |      | ✓    | 81.11 | 67.58 | 61.83 |
|      | ✓    |      | ✓    | 80.38 | 67.68 | 61.53 |
| ✓    | ✓    |      | ✓    | 80.90 | 68.08 | 61.49 |
| ✓    |      | ✓    | ✓    | 80.97 | 67.92 | 61.94 |
|      | ✓    | ✓    | ✓    | 81.11 | 67.58 | 61.83 |
| ✓    | ✓    | ✓    | ✓    | **82.46** | **68.31** | **62.14** |

**关键发现**：
1. **单独加 CCDT 在 PACS 上轻微负面**（79.92 → 79.32, -0.6）— 说明单纯风格混合**不够**
2. **单独加 DIFP 正面**（79.92 → 80.52, +0.6）— AugMix 的图像扰动有普适效果
3. **CCDT + DIFP 组合**（79.92 → 80.48, +0.56）— 协同效果微弱
4. **必须加 L_RA 或 L_JS** 才能把 CCDT 的贡献释放出来（80.48 → 81.18 with L_RA, 或 → 80.90 with L_JS）
5. **四项全加**最佳 82.46

**启示**：**CCDT 不能单独用！** 必须配合 representation-level 或 prediction-level 的 alignment loss 来**消化**增强带来的 feature 漂移。这对我们非常重要。

---

## 8. 能借鉴给我们 FedDSA-SGPA（FedDA 场景）

### 8.1 场景差异再强调

| 维度 | FedCCRL（FedDG） | 我们 FedDSA（FedDA） |
|------|------------------|---------------------|
| 场景 | Leave-one-domain-out，test domain 不参训 | 所有 4 域都参训，本地 test |
| Client 数 | 6 (PACS)、10、14 等可变 | **4**（每域 1 client） |
| Backbone | MobileNetV3-Large（预训练） | **AlexNet**（从头训） |
| 训练长度 | 10 rounds × 3 epochs | R200 rounds × E5 |
| 数据量 | 每 client ~500 图（3 域分到 6 client） | 每 client 一个完整域（~1000-3000 图） |

### 8.2 CCDT 能不能套到我们？**能，而且 natural fit**

**关键观察**：我们已经有 `style_bank`（per-client pooled feature μ, σ，128-d 或 512-d），它本身就是 CCDT 所需的 peer style pool！

**两条可行路径**：

#### Path A：在输入图像空间做（照搬 FedCCRL）
- 每 client 上传样本图像的 per-channel (μ, σ)（3 维 × 2 = 6 floats）
- Server 广播 F_pool
- Client forward 时先做 CCDT 再过 AlexNet
- **问题**：我们已经有复杂的 style_bank 机制，再加一个 input-level stats 通信**冗余**

#### Path B：在 AlexNet 内部某层 feature map 上做 ★ 推荐
- 利用**已有的** style_bank（pooled 1024d feature 的 μ/σ）或在 AlexNet 的 **conv1 / conv2 / conv3 的 feature map** 上算 per-channel stats 并共享
- **这更接近原始 MixStyle 的精神**（feature-level），且能复用我们现有的跨 client 通信机制
- **问题**：在哪一层？原 MixStyle 研究表明：**浅层（conv1/conv2）有效，深层（conv4/conv5）有害**

### 8.3 在 AlexNet 哪层做 MixStyle？

**AlexNet-BN 结构**（我们 pipeline）：
```
conv1 (11x11, 64)  → bn1 → relu → maxpool
conv2 (5x5, 192)   → bn2 → relu → maxpool
conv3 (3x3, 384)   → bn3 → relu
conv4 (3x3, 256)   → bn4 → relu
conv5 (3x3, 256)   → bn5 → relu → maxpool
flatten → fc6 (4096) → bn6 → relu → dropout
         → fc7 (4096) → bn7 → relu → dropout
         → pooled (1024)    ★ 这是 semantic_head 输入点
         → semantic_head (128d)
```

**MixStyle 放在哪**（结合 Zhou ICLR'21 的消融）：
- **bn1 后**：最浅层，风格扰动最自然，**最有可能正面**
- **bn2 后**：中等，文献中也表现不错
- **bn3 后**：开始接近语义，临界点
- **bn5 后**：已经是语义特征，**会崩**
- **pooled 1024d 后**（我们 style_bank 当前位置）：**不适合**做 MixStyle，因为这里已经高度语义化，扰动会伤 semantic_head

**结论**：**推荐在 bn1 或 bn2 后**做跨 client MixStyle，**不要在 pooled 或 semantic_head 层**做。

### 8.4 DIFP (AugMix) 能加吗？

**成本**：CPU-based 图像增强，开销中等。每样本每次 forward 生成两份增强图 → **batch 实际变成 3× size**，GPU 显存翻 3 倍。

**判断**：
- ✅ 思想有价值：扰动 domain-invariant feature（形状、纹理），和 CCDT 互补
- ❌ 我们 AlexNet 输入已经有常规 augmentation（RandomCrop/HFlip），再叠 AugMix 边际收益不大
- ❌ batch 变 3 倍，E5×R200 长训练会显存吃紧
- **判决**：**先不加 DIFP**。如果 CCDT 单独加了发现效果提升但还不够，再考虑。

### 8.5 L_RA (SupCon) / L_JS 能加吗？会不会和 orth+HSIC 冲突？

**分析**：
- 我们当前 loss：`L_CE + λ_orth * L_orth + λ_hsic * L_HSIC + (可选) L_InfoNCE`
- FedCCRL 的 L_RA 本质上是 **SupCon（监督对比）**，对齐原图 vs 增强图的**表征**
- 我们已有的 L_InfoNCE（semantic 特征 vs 全局原型）和 L_RA 角色**不同**：
  - L_InfoNCE：semantic 特征 vs **类原型**
  - L_RA（SupCon）：原图特征 vs **增强图特征**（同类拉近）

**兼容性判断**：
- ✅ **可以共存**：目标不冲突，L_RA 强制 invariance on style perturbation，L_InfoNCE 强制 consistency with global proto
- ⚠️ 但**梯度竞争风险**：SupCon 和 InfoNCE 都用 softmax+NCE 形式，可能梯度互相干扰
- **Ablation 建议**：先测 CCDT + L_CE（无 L_RA/L_JS）的简单版本，再决定是否加 alignment loss

**L_JS 判断**：
- 目标：让 {原图, 增强1, 增强2} 的 softmax 输出一致
- ✅ 和我们目标不冲突
- ❌ 但我们没有"三份输入"的设定，要改成 "{原图, CCDT增强图}" 两份即可
- **判决**：**可作为 bonus 添加**，逻辑上温和有效

### 8.6 给 20 行 Pseudo Code：AlexNet 浅层 MixStyle + style_bank 采样

```python
# file: PFLlib/system/flcore/trainmodel/alexnet_mixstyle.py
# 插入到 AlexNetEncoder.forward，在 bn1 之后 / bn2 之前做 MixStyle

class AlexNetEncoderWithCCDT(nn.Module):
    def __init__(self, mixstyle_alpha=0.1, mixstyle_prob=0.5, mixstyle_layer='bn1'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, 4, 2)
        self.bn1   = nn.BatchNorm2d(64)
        # ... conv2-5, bn2-5 ...
        self.alpha = mixstyle_alpha
        self.prob  = mixstyle_prob  # 触发概率
        self.mix_layer = mixstyle_layer
        # style_bank_peer: List[(mu, sigma)], shape each (64,) for bn1 layer
        # 由 Server 在每轮开始时通过 set_style_bank() 注入
        self.style_bank_peer = None

    def set_style_bank(self, peer_stats):
        """peer_stats: list of (mu, sigma) tuples from OTHER clients, each (C,)"""
        self.style_bank_peer = peer_stats

    def ccdt_layer(self, feat):
        """feat: (B, C, H, W) after bn1. 应用跨 client MixStyle."""
        if not self.training or self.style_bank_peer is None:
            return feat
        if torch.rand(1).item() > self.prob:   # p=0.5 触发
            return feat
        B, C, H, W = feat.shape
        # 计算当前 batch per-sample stats
        mu    = feat.mean(dim=[2,3], keepdim=True)       # (B, C, 1, 1)
        sigma = feat.std(dim=[2,3], keepdim=True) + 1e-6  # (B, C, 1, 1)
        feat_norm = (feat - mu) / sigma                   # 白化
        # 从 peer bank 采样 B 个 peer stats (每样本独立采一个 peer)
        idx = torch.randint(0, len(self.style_bank_peer), (B,))
        peer_mu    = torch.stack([self.style_bank_peer[i][0] for i in idx]).to(feat.device)  # (B, C)
        peer_sigma = torch.stack([self.style_bank_peer[i][1] for i in idx]).to(feat.device)  # (B, C)
        peer_mu    = peer_mu[:, :, None, None]
        peer_sigma = peer_sigma[:, :, None, None]
        # Beta 插值系数
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(feat.device)
        mu_mix    = lam * peer_mu    + (1 - lam) * mu
        sigma_mix = lam * peer_sigma + (1 - lam) * sigma
        return sigma_mix * feat_norm + mu_mix

    def forward(self, x):
        h = self.bn1(self.conv1(x))
        if self.mix_layer == 'bn1':
            h = self.ccdt_layer(h)              # ★ 在这里注入
        h = F.relu(h); h = F.max_pool2d(h, 3, 2)
        h = self.bn2(self.conv2(h))
        if self.mix_layer == 'bn2':
            h = self.ccdt_layer(h)              # 或者在 bn2 后
        h = F.relu(h); h = F.max_pool2d(h, 3, 2)
        # ... rest conv3-5, fc6, fc7, pooled ...
        return pooled_1024d

# Server 端：
# 每轮结束后,每个 client 上传当前 bn1 层的 per-channel mean/std (在完整本地数据上聚合)
# Server 拼成 F_pool,广播给每个 client 其"非自身"的部分
# client_i.encoder.set_style_bank(F_pool \ F_i)
```

**实现要点**：
1. **插入点**：bn1 之后（推荐首选）或 bn2 之后
2. **Peer stats 生成**：每 client 每轮在**全部本地数据**上跑一遍 forward 到 bn1，聚合整个域的 per-channel μ/σ（1 份 stats per client per round），或按 FedCCRL 思路按 r=0.1 采样上传多份 stats
3. **Server 通信**：`(C,)` 向量 × 2（μ/σ）= 64 × 2 = **128 floats per client per round** — 几乎 0 开销
4. **触发概率 p=0.5**：比 FedCCRL 的 p=1.0 更保守，降低梯度噪声
5. **和 style_bank 的区别**：我们现在的 style_bank 是 **pooled 1024d 层的 stats**（用于 SGPA 的 gated proto），要**新增一个 bn1/bn2 层的 stats**（用于 CCDT），两套 bank 并行

**关键实验设计**：
- `mix_at=none`（baseline，我们当前 orth_only）
- `mix_at=bn1, α=0.1, p=0.5`
- `mix_at=bn2, α=0.1, p=0.5`
- `mix_at=bn1, α=0.1, p=1.0` （激进版）
- `mix_at=bn3, α=0.1, p=0.5` （对照，验证深层崩）

---

## 9. 一句话总结

**FedCCRL = 跨 client MixStyle（在输入图像像素空间做 Beta 插值风格混合）+ AugMix + SupCon + JSD，用极低通信代价（每 client 每轮 < 2KB 的 stats 上传）在 MobileNetV3 上把 FedDG LOO PACS avg 打到 82.46，核心机械可以以 feature-level 形式（在 AlexNet bn1/bn2 之后）移植到我们 FedDSA，风险在于需要配 alignment loss（L_RA 或 L_JS）才能把 CCDT 的收益真正释放出来，单加 CCDT 在 PACS 上反而 -0.6。**
