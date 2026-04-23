# 论文精读 FedAlign — Federated Domain Generalization with Cross-Client Feature Alignment

> **精读时间**: 2026-04-23
> **PDF**: `D:\桌面文件\联邦学习\papers\FedAlign_2501.15486.pdf`
> **作者**: Sunny Gupta, Vinay Sutar, Varunav Singh, Amit Sethi (IIT Bombay)
> **arXiv**: 2501.15486 (2025-01-26)
> **Venue**: arxiv preprint (未见标明 venue, 投稿 target 应为 IJCAI / AAAI-Workshop 级别)

---

## 1. 基本信息 + TL;DR

**标题**: FedAlign: Federated Domain Generalization with Cross-Client Feature Alignment

**TL;DR** (一句话):
在 **只跑 10 轮通信、3 epoch/client** 的极端 communication-constrained FedDG setting 下, 用 **MixStyle 跨 client 特征增强 + SupCon + MSE + JSD** 三重对齐,在 PACS LOO 达到 Avg 82.96 (FedAvg 76.10, CCST 79.85), 主打"轻量 + 不泄隐私 + 不用 VGG 预训练"。

**设定关键点**:
- MobileNetV3-Large backbone (边缘设备友好)
- **R=10 通信轮** (!), E=3 local epochs, Adam LR=0.001 cosine decay
- 4 client FedDG LOO (leave-one-domain-out), 每 client 一个 domain
- Upload ratio r=0.1 (每轮只传 10% 参数)

---

## 2. 核心问题 + 动机

### 2.1 要解决什么

**FedDG (Federated Domain Generalization)**: K 个 client 每个持有一个不同 domain, 目标模型泛化到 **训练时完全看不到** 的 target domain。

**本文核心痛点**: 现有 FDG 方法 (FedSR / FedADG / CCST) 要么:
- **对抗训练不稳** (FedADG 的 domain classifier 容易 collapse)
- **通信开销大** (CCST 用 VGG19 提风格 + AdaIN 图像生成, 要传 high-dim embedding)
- **依赖预训练模型** (CCST 的 VGG 已经在 ImageNet 上见过 target domain, 违反 DG 原则)
- **限制 domain 多样性** (每 client 只有一个 domain 的数据, 本地学不出泛化特征)

### 2.2 为什么 R=10 这么极端

**Answer**: Edge / cross-device FL 场景 (移动端 / IoT)。这种场景下:
- 每个 client 是手机 / IoT device, 电量 + 带宽都受限
- 不可能跑 FedAvg 那种 200-500 轮
- 也就不可能做 heavy adversarial training
- 所以作者选 MobileNetV3-Large + Adam (非 SGD) + cosine decay, 目标是 "少数几轮就收敛"

**对我们 FedDA 的启示**: 我们跑 200 轮 SGD on ResNet-18, 设定完全不同, **baseline 数字不可直比**。FedAlign 在 R=10 的 MobileNetV3 上跑 PACS LOO 82.96, 不代表在 R=200 ResNet-18 FedDA setting 下也能到 82。这是两个 universe。

---

## 3. 方法精读

### 3.1 组件 A — Style Mixing (MixStyle augmentation)

**借鉴自**: MixStyle (Zhou et al. 2021)

**公式** (每 channel 独立):

对 batch 内样本 x_i, 算 channel-wise mean/std:

```
μ(x)_c = (1/HW) Σ_h Σ_w x_{c,h,w}
σ(x)_c = sqrt((1/HW) Σ_h Σ_w (x_{c,h,w} - μ(x)_c)^2)
```

对任意两个 sample x_i, x_j, 用 Beta 分布采样 λ ~ Beta(α, α) 做插值:

```
γ_mix = λ · μ(x_i) + (1-λ) · μ(x_j)
β_mix = λ · σ(x_i) + (1-λ) · σ(x_j)
x_aug = γ_mix · (x_i - μ(x_i)) / σ(x_i) + β_mix
```

**触发概率**: 每个 batch 生成 **两个** augmented batch `X^(1) = M(X), X^(2) = M(X)` (独立采样两次), 原 batch X 保留。所以前向 3 次 (X, X^(1), X^(2))。

**层位置**: 论文没明确说, 但 MixStyle 原论文在 res1/res2/res3 (浅-中层), **不在最深 res4**。作者说扩展了 MixStyle 两点:
1. **Clustering**: 按 style statistics 聚类特征, 帮助找 domain-invariant 代表
2. **Probabilistic sampling weights**: 按 feature variance 加权采样, 让 "难样本" 和 "欠表达 domain" 被采中概率更高

### 3.2 跨 client 特征增强 ("cross-client" 到底在哪里)

**这是本文 novelty 点但写得极其含糊的地方**。

论文原文 (Figure 2 caption 和 Sec 3.3 标题):
- "Clients share local model parameters **and sample statistics** with the server, which aggregates and redistributes them"
- "cross-client feature extension module broadens local domain representations through domain-invariant feature perturbation and selective cross-client feature transfer"

**我的解读**: MixStyle 本来是 batch **内** 插值 (只用本地数据), FedAlign 把它扩展成 **每 client 上传 (μ, σ) style statistics 到 server, server 聚合后再下发, client 在 MixStyle 时可以和其他 client 的 style 插值**。这就是所谓的 "cross-client"。

但论文 **没给明确的算法步骤**说"接收 remote style 后怎么用", 也没说上传什么粒度的 statistics (per-layer? per-sample? per-class?)。Algorithm 1 里 line 15 就是 `X^(1) = M(X), X^(2) = M(X)`, 没有 cross-client 的任何标识。

**结论**: 这是本文叙事问题, novelty claim "cross-client feature extension" 和 method 具体实现 **存在 gap**。可能实际实现就是 per-client 的 MixStyle, 然后 style statistics 被上传做某种聚合但对 MixStyle 本身没影响。

### 3.3 组件 B — Representation Alignment (L_RA)

**L_RA = L_SC + L_RC**

#### L_SC — Supervised Contrastive Loss (SupCon)

对齐 **同 label** 的样本的 representation (Z, Z^(1), Z^(2)):

```
L_SC = Σ_{i∈I} [ -1/|P(i)| · Σ_{p∈P(i)} log ( exp(sim(z_i, z_p)/τ) / Σ_{a∈A(i)} exp(sim(z_i, z_a)/τ) ) ]
```

- P(i): 同 label 的样本 index set
- sim: cosine similarity
- τ = 0.1 (temperature)

**和我们的 L_InfoNCE 的关系**: 基本上一样, SupCon 就是 supervised 版 InfoNCE, 把同 label 的所有样本都作 positive。

#### L_RC — Representation Consistency Loss

**MSE 锚点**:

```
L_RC = 1/|mix_feat| · Σ || h(X) - h(X_aug) ||^2
```

强制 original feature 和 augmented feature 在欧氏空间 "不要差太远"。等价于 FPL 论文里的 MSE anchor。

### 3.4 组件 C — JSD Consistency (Prediction Alignment)

对 3 个 prediction Y, Y^(1), Y^(2), 算平均 Y̅ = (Y + Y^(1) + Y^(2)) / 3, 然后每个 prediction 和 Y̅ 的 KL:

```
L_JS = 1/3 · [ KL(Y || Y̅) + KL(Y^(1) || Y̅) + KL(Y^(2) || Y̅) ]
```

**作用**: 强制 original 和两个 augmented 的分类概率分布相互一致。这其实就是 JSD 的定义 (JSD 就是 mixture distribution 和每个 component 的平均 KL)。

**和 L_SC 的区别**: L_SC 在 feature 层面对齐, L_JS 在 prediction (softmax 后) 层面对齐。两者互补。

### 3.5 组件 D — 整体训练 (通信轮少 R=10 下怎么收敛)

**Total loss**:
```
L = L_CLS + λ_1 · L_RA + λ_2 · L_JS
   = L_CLS + λ_1 · (L_SC + L_RC) + λ_2 · L_JS
```

**关键**: 为什么 10 轮能收敛?
1. **MobileNetV3-Large 预训练** (ImageNet), backbone 已经有强 feature
2. **Adam + cosine decay** 自适应学习率, 比 SGD 快
3. **每 batch 3 次 forward + 3 loss 联合**, 等效 multi-task learning, 单 epoch 信号密度高
4. **E=3 local epochs**, 每轮 client 都更新 3 epoch, 10 轮总计 30 local epoch 的等效
5. 其实作者没强调 "收敛", 82.96 AVG 也不算特别惊艳 (我们 orth_only 在 FedDA 4-client 1-domain setting 下 80.64, 差距没那么大)

**Aggregation**: 标准 FedAvg 按数据量加权 (Algorithm 1 line 9): `θ_{t+1} = (1/N) Σ n_k · θ_{k,t+1}`

**Upload ratio r=0.1**: 每轮只上传 10% 参数 (可能是 top-k gradient sparsification), 降低通信成本。但论文 **没说是 magnitude top-k 还是 random mask**。

---

## 4. 算法流程 (Algorithm 1 原文简化)

```
Server:
  for t = 1..T=10:
    select subset C_t of clients
    broadcast θ_t
    collect θ_{k,t+1} from each k in C_t
    aggregate: θ_{t+1} = Σ (n_k/N) θ_{k,t+1}

Client k (on receiving θ_t):
  for e = 1..E=3:
    for batch X:
      X^(1) = MixStyle(X), X^(2) = MixStyle(X)
      Z, Z^(1), Z^(2) = h(X), h(X^(1)), h(X^(2))
      Ŷ, Ŷ^(1), Ŷ^(2) = g(Z), g(Z^(1)), g(Z^(2))
      L_CLS = CE(Ŷ, y)  # 注: 原文只用 Ŷ 算 CE, 不用 Ŷ^(1)/Ŷ^(2)
      L_SC = 0.5 · (L_SC(Z^(1), Z) + L_SC(Z^(2), Z))
      L_RC = MSE(h(X), h(X_aug))
      L_RA = L_SC + L_RC
      L_JS = (1/3) · [KL(Ŷ||Ȳ) + KL(Ŷ^(1)||Ȳ) + KL(Ŷ^(2)||Ȳ)]
      L = L_CLS + λ_1 · L_RA + λ_2 · L_JS
      SGD step with Adam(lr=0.001)
  return θ_{k,t+1}
```

---

## 5. 实验 setup

- **Backbone**: MobileNetV3-Large (为什么? edge-device 场景, 参数少 ~5M, 比 ResNet-50 轻 10x)
  - 最后 fc 层 = classifier g, 前面的都是 encoder h
- **Protocol**: Leave-One-Domain-Out (LOO), 4 domain 中 3 个做 source (分给 3 client), 1 个做 target 测试
- **Clients**: 每 domain 一个 client, 4 client (和我们 FedDA setting 同)
- **R=10** communication rounds
- **E=3** local epochs/round
- **Optimizer**: Adam, lr=0.001, cosine decay
- **Batch**: PACS/OfficeHome 224x224, miniDomainNet 128x128 (batch size 未明确, 应是默认 32 或 64)
- **Upload ratio r=0.1** (每轮传 10% 参数)
- **τ = 0.1** for SupCon
- **每算法跑 3 seeds, 平均**

---

## 6. PACS LOO 结果表

Table 1 — PACS test acc 每列是 LOO 的 target domain:

| Algorithm | P (target) | A (target) | C (target) | S (target) | Avg |
|-----------|:----------:|:----------:|:----------:|:----------:|:----:|
| FedAvg | 90.30 | 69.95 | 75.70 | 71.80 | 76.10 |
| FedProx | 91.21 | 69.84 | 73.15 | 70.87 | 75.92 |
| FedADG | 89.52 | 63.54 | 71.45 | 64.32 | 72.20 |
| GA | 92.98 | 66.01 | 76.43 | 69.23 | 75.81 |
| FedSR | 91.42 | 72.68 | 76.01 | 70.73 | 76.84 |
| FedIIR | 91.53 | 71.25 | 78.61 | 71.78 | 77.89 |
| CCST | 89.93 | 76.18 | 75.97 | 78.34 | 79.85 |
| **FedAlign** | **93.11** | **80.57** | **77.94** | **80.20** | **82.96** |

**关键 semantics 确认** (原文 Table 1 列标 "P A C S"):

**⚠️ 重要修正**: 原文列标是 **P A C S** 顺序, 列头 "P" 对应的数字 93.11 **是当 P=Photo 作为 unseen target** 时的 zero-shot test acc, 用 A+C+S 作 source。**不是** 我之前以为的"用 P 训练后测 A"。

所以 PACS LOO 的完整解读:
- **Photo unseen**: 用 Art+Cartoon+Sketch 源训练 → Photo test 93.11 (task 最容易, Photo 是 natural)
- **Art unseen**: 用 Cartoon+Photo+Sketch 源训练 → Art test 80.57
- **Cartoon unseen**: 用 Art+Photo+Sketch 源训练 → Cartoon test 77.94
- **Sketch unseen**: 用 Art+Cartoon+Photo 源训练 → Sketch test 80.20
- **Avg = 82.96**

**但原题目说 "Art unseen target 93.11 (用 C/P/S 源)"** — 我要纠正用户这个理解: **93.11 是 Photo unseen, 不是 Art**。Art unseen 是 80.57。作者其他方法列顺序也是 P A C S, 都验证 Photo 是最容易的 target domain (FedAvg 也 90.30)。

---

## 7. Ablation

**论文没有专门的 ablation table**, 但在 Sec 5 提了:

- **Figure 4**: 不同 client 数量下 (10 / 20 / 50), FedAlign 依然保持领先, 其他方法随 client 数增加性能下降
- **Figure 5** (原文 caption 提到但我没见到具体数字): FedAlign 在 upload ratio r=0.1 下仍维持高 acc, 证明通信效率

**这是本文 **实验部分的重大缺陷** **:
- 没有 L_SC / L_RC / L_JS 三者的逐项消融
- 没有 "cross-client feature sharing" vs "本地 MixStyle" 的对比 (这个本应是 novelty 点!)
- 没有对 λ_1, λ_2, τ 的 sensitivity analysis
- 没有对 MixStyle 的 "Clustering" 和 "Probabilistic sampling weights" 两个 claim 改进的消融

**推断**: 主要效果来自 **MixStyle + SupCon**, "cross-client" 可能贡献不大所以作者没做消融。

---

## 8. 能借鉴给我们 FedDA (AlexNet / ResNet-18, R=200) 的地方

### 8.1 Style Mixing 能不能套 AlexNet?

**能, 但有注意事项**。

- MixStyle 原论文只在 res1/res2/res3 插入有效 (浅-中层), 深层 (res4) **反而 -7%** 暴跌。AlexNet 的对应位置应该是 conv1-conv3 之间。
- **不能在最后 pooled feature 做 MixStyle** (和我们目前 AdaIN 在 pool 后做的 EXP 逻辑差不多, 都有风险)
- **和 SGPA / FedDA 已有的 AdaIN 增强重复** — 要么换掉 AdaIN 用 MixStyle, 要么只加 SupCon + JSD 这两个 loss
- 我们的 backbone 需要改:
  ```python
  # AlexNet feature 后手动插 MixStyle hook, 但注意要在 early layer
  # 否则破坏 semantic
  ```

### 8.2 L_RA (SupCon + MSE) 能加吗? 和 L_orth / L_HSIC 冲突吗?

**SupCon 部分: 能加, 但要小心梯度冲突**。

- SupCon = L_InfoNCE 的 supervised 版。我们 EXP-119/120 反复验证 InfoNCE 和 CE **梯度冲突 cos_sim 会穿零**, 加 SupCon 大概率遇到同样问题。
- 如果加, 必须配合 **MSE anchor (L_RC)**, 这是 FPL / FedPLVM / FedAlign 三家都用的 "安全阀"
- 和 L_orth / L_HSIC **理论上不冲突**: L_orth/L_HSIC 约束 z_sem 和 z_sty 正交, SupCon 约束同 label 的 z_sem 靠近。作用在不同维度。

**MSE 部分** (L_RC): 这就是 FPL 式 anchor。我们 mode 4/6 已经试过, 确实有效 (EXP 82.2%)。可复用。

### 8.3 JSD 机制能加吗?

**能加, low risk, 建议试**。

- JSD 只在 **prediction (softmax)** 层面对齐, 不碰 feature 层, 所以不会和 L_orth / L_HSIC / InfoNCE 冲突
- 它等于给 augmented 样本的 CE 加了个 consistency 约束, 相当于 semi-supervised 里的 consistency regularization
- **比 SupCon 安全**, 值得作为 "low-hanging fruit" 加到 baseline 上

### 8.4 Pseudo code (20 行内, 给我们 FedDA 的适配版)

```python
# FedDA + FedAlign 借鉴, 加到 AlexNet backbone
def feddsa_fedalign_client_step(x, y, model, feat_dim, tau=0.1, lam1=0.5, lam2=0.3):
    # original + 2 augmented (MixStyle in conv2-conv3, not deepest)
    x_aug1 = mixstyle_early_layer(x)  # hook inside conv2/conv3
    x_aug2 = mixstyle_early_layer(x)

    z_sem, z_sty = model.encoder_with_decouple(x)   # 原 FedDA 双头
    z_sem1, _ = model.encoder_with_decouple(x_aug1)
    z_sem2, _ = model.encoder_with_decouple(x_aug2)
    yh, yh1, yh2 = model.cls(z_sem), model.cls(z_sem1), model.cls(z_sem2)

    L_CLS = F.cross_entropy(yh, y)
    L_orth = cos_sim(z_sem, z_sty) ** 2               # 我们原本的
    L_HSIC = hsic(z_sem, z_sty)                       # 我们原本的

    # 借鉴 FedAlign 的三个对齐 loss
    L_SC = 0.5 * (supcon_loss(z_sem1, z_sem, y, tau) + supcon_loss(z_sem2, z_sem, y, tau))
    L_RC = 0.5 * (F.mse_loss(z_sem1, z_sem.detach()) + F.mse_loss(z_sem2, z_sem.detach()))
    Y_bar = (yh.softmax(-1) + yh1.softmax(-1) + yh2.softmax(-1)) / 3
    L_JS = (kl(yh, Y_bar) + kl(yh1, Y_bar) + kl(yh2, Y_bar)) / 3

    L = L_CLS + lam_orth*L_orth + lam_hsic*L_HSIC + lam1*(L_SC + L_RC) + lam2*L_JS
    return L
```

**建议先加 L_JS (低风险) + L_RC (MSE anchor), 暂缓 L_SC (SupCon 风险大)**。先单独消融, 不要一次加全部。

---

## 9. 一句话总结

FedAlign 在 R=10 MobileNetV3 的 **极短通信 edge FL** 下, 用 MixStyle 跨样本增强 + SupCon/MSE/JSD 三重对齐, 在 PACS LOO 达到 82.96 Avg, 但 "cross-client feature sharing" 的 novelty claim 在 Algorithm 1 里并未体现, 实际主要贡献来自 **augmentation + alignment 的经验性组合**, 其 JSD 和 MSE anchor 机制可低风险借鉴到我们 R=200 ResNet-18/AlexNet FedDA 设定。

---

## 附录: 关键数字 vs 我们 FedDA

| 指标 | FedAlign (R=10 MobV3, PACS LOO) | 我们 FedDA (R=200 ResNet-18, PACS 4-client 1-domain) |
|------|:--------------------------------:|:-----------------------------------------------------:|
| Setting | Leave-One-Out, unseen target | Federated training, 所有 domain 参训, 各 domain 测 best |
| Backbone | MobileNetV3-Large | ResNet-18 |
| R (rounds) | 10 | 200 |
| FedAvg baseline | 76.10 | ~72.10 (FDSE paper) |
| FedAlign / orth_only | 82.96 | 80.64 (orth_only 3-seed mean) |
| FDSE (对照) | N/A (不在它 baseline 里) | 79.91 (本地复现) |

**结论**: 两个 setting 完全不可比。FedAlign 是 DG 任务 (测 unseen), 我们是 FL 跨域训练任务 (测各 source domain)。但**方法组件 (MixStyle / SupCon / MSE / JSD) 可跨 setting 借鉴**。
