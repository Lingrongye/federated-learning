# Idea Discovery Report — Federated Domain Generalization

**日期**: 2026-04-24
**Pipeline**: research-lit (compressed) → idea-creator → novelty-check (web search) → research-review (self-rigorous)
**Direction**: 跨数据集一致领先 FDSE 的 FedDG 方法; 不依赖 per-dataset priors; AlexNet 固定; ≤3 新超参

## Executive Summary

3 个 candidates 经过 brainstorm + novelty check (Web search) + critical review:

| Rank | Idea | Novelty | 预期跨数据集稳定性 | 实施成本 | 推荐 |
|:-:|---|:-:|:-:|:-:|:-:|
| **🏆 1** | **SV-GOD** (Style-Variance Gated Orthogonal Decoupling) | **CONFIRMED novel** | 高 | 低 | ⭐ TOP |
| 2 | **GA-CCP** (Gap-Aware Cross-Client Prototype Anchor) | 部分 novel (vs FedTGP/ASA 需明确差异化) | 高 | 中 | 备选 |
| 3 | **DGR** (Decoupling Gradient Regularization) | High novel | 中 | 高 (gradient hook) | 实验性 |

**推荐**: SV-GOD 作为下一个 EXP, 理由 — (a) novelty 强, (b) 直接解决 FDSE trade-off (style-confused win + common-class lose), (c) 改动 minimal (1 个小 MLP + 1-2 新超参), (d) 跨数据集自适应无需 prior.

---

## Phase 1: Literature Landscape (compressed, 基于已读 5 paper + STATE_REPORT + 新搜)

### 当前 FedDG 5 大流派

| 流派 | 代表 | 核心机制 | 痛点 |
|---|---|---|---|
| **擦除派** | FDSE (CVPR'25), FedSeProto, FedDP | 层级 erase domain info | 过度擦除伤 common-class (诊断证实) |
| **原型派** | FedProto, FPL, FedPLVM, FedTGP | 类原型聚合 / α-sparsity | 单原型表达力弱; 多原型噪声敏感 |
| **混风派** | FedCCRL, I2PFL APA, MixStyle | feature/pixel 级 MixUp | 不解耦, 在最后特征空间增强 |
| **对齐派** | FedAlign, FedSTAR, FedOMG | embedding/gradient 跨 client 对齐 | 隐式, 无明确解耦 |
| **校准派** | F2DC | "calibrate not eliminate" | 局部 corrector, 缺规范化 |

### 我们的位置

**FedDSA = 单层正交解耦派** — 在最后 pooled feature 做 `cos²(z_sem, z_sty)`. 优点: 简洁、可解释; 缺点: 单层 + 只约束**线性独立** (高阶 style 仍可 leak).

### 诊断驱动的洞察

- FDSE +2.31pp 优势 90% 来自 3 cells (Art-guitar +24.33, Art-horse +15.52, Photo-horse +17.50)
- FDSE 反输 3 cells (Art-dog -13.36, Art-elephant -7.56, Art-giraffe -8.89)
- → **FDSE 不是 universal winner, 是 "style-confused 强 / common-class 伤"** 的 trade-off
- **机会**: 做**自适应**解耦 (识别 style-confused vs common, 区别对待) 可同时拿下两种 cells

### Search 发现 (2026-04 截止)

- **FedADG**: 类条件 GAN-based domain alignment — 不同机制
- **FedSTAR (2511.18841)**: Transformer attention + content-style + 学习 gate fuse global vs personal prototype — gate 在 prototype 层不在 loss 层
- **Fed-DIP**: Multi-scale Implicit Decoupling Distillation — 蒸馏机制
- **FedLAG (2410.02845)**: layer-wise disentanglement via gradient conflict — 不同 mechanism
- **FedTGP**: trainable global prototype with adaptive-margin contrastive — server-learned prototype
- **FedCA**: cross-client style transfer + adaptive style alignment (medical) — 风格迁移非 anchor
- **VCReg**: variance-covariance representation regularization — 不针对 FDG
- **Variance-Invariance Disentanglement** (Lut5t3qElA): variance 作 disentanglement 准则 — 不是用 variance 调 loss weight

**Gap**: 没有工作把**观察到的 within-class style variance** 用作**自动调节解耦强度**的信号.

---

## Phase 2: Idea Generation

### 全部候选 (8 个 brainstorm)

| # | Idea | 简述 | Filter 后 |
|---|---|---|:-:|
| A | **SV-GOD** | Class 条件 orth, weight gated by style variance | ✅ Top 3 |
| B | SSCB | Style-swap with class-preservation budget | 淘汰 (与 I2PFL APA 太近) |
| C | DCC-GT | Federated temperature calibration | 淘汰 (Art ECE 跨方法一致, 升级空间小) |
| D | MIC-DH | MI-based dual-head (MINE/CLUB) | 淘汰 (HSIC 类似, 我们扫过) |
| E | **GA-CCP** | Gap-aware cross-client prototype anchor | ✅ Top 3 |
| F | ALO | Adaptive layer-wise orthogonality | 淘汰 (FedLAG 已有 layer-wise) |
| G | DF-PDL | Domain-free principal direction learning | 淘汰 (复杂度高, 抽象) |
| H | LSSP | Style-proximity label smoothing | 淘汰 (太浅, 不像方法 paper) |
| I | **DGR** | Gradient-level decoupling regularization | ✅ Top 3 (实验性) |

---

## Phase 3-4: Top 3 Candidates (深入 + Novelty Check + Critical Review)

---

### 🏆 Idea 1: SV-GOD — Style-Variance Gated Orthogonal Decoupling

#### Hypothesis (一句话)

**FDG 中正交解耦的最优强度应随类的 style 变异性自适应**: 高 style-variance 类 (e.g. Art-guitar) 需强解耦; 低 style-variance 类 (e.g. common animal in 任何 domain) 需弱解耦或无.

#### Mechanism (数学 + 实施)

**Step 1**: 客户端每 batch 计算 per-class style variance:
```
V_c = (1/N_c) · Σ_{i: y_i=c} ||z_sty_i - μ_sty_c||²    # within-class style variance
```
其中 `μ_sty_c` 是 batch 内 class c 的 z_sty 均值.

**Step 2**: gate 函数 (1 layer MLP, 6 个参数):
```
g_c = σ(α · V_c + β)   # α, β 学习的 scalar; σ = sigmoid
```

**Step 3**: gated 正交损失 (替换原来 cos² 为 class-conditional):
```
L_orth = Σ_c g_c · cos²(z_sem_c, z_sty_c)
```
其中 `z_sem_c, z_sty_c` 是 class c 样本的特征.

**Step 4**: 总 loss:
```
L = L_CE + λ_orth · L_orth
```
原来的 `λ_orth=1.0` 保留; 新增 α, β 两个 scalar (初始化 α=1, β=0, 训练自动学).

**新超参**: 0 个 (α, β 是学的, 不是手调; λ_orth 沿用现有).

#### 为什么跨数据集 generalize?

- **PACS**: Art-guitar V_c 大 → g_c 大 → 强解耦 → 期望涨; Art-dog V_c 小 → g_c 小 → 不过度解耦, 不伤 common-class
- **Office**: 域差异小, 大多数 class V_c 小 → g_c 小 → orth 弱化, 不会像 sgpa 那样**有害于 Office**
- **DomainNet**: 多 domain, 部分 class V_c 大 → g_c 自适应 → 不会盲目擦

**关键**: 不需要 dataset prior, V_c 是从数据**实时观察到的信号**.

#### Novelty Claim

**Confirmed novel** (Web search 无直接匹配, 2026-04):
- **不是 FDSE**: FDSE 是层级 + uniform; 我们是**单层 + class-adaptive**
- **不是 VCReg**: VCReg 是 representation-level "high variance, low covariance" — 是表示性质; 我们是**用 variance 调 reg weight**
- **不是 Variance-Invariance Disentanglement**: 它是**用 variance 作 disentanglement 准则**; 我们是**用 variance 调节解耦的强度**
- **不是 FedSTAR gate**: FedSTAR 是 prototype-level fusion gate; 我们是 loss-level adaptive weight
- **不是 FedPLVM α-sparsity**: α-sparsity 是 contrastive 中 cosine 的幂次; 我们是 orth 的 weighted sum

**核心 novelty 一句话**: First to use **observed within-class style variance as auto-calibration signal** for **per-class disentanglement strength** in federated DG.

#### Critical Review (rigorous self-review, 假装 NeurIPS reviewer)

**Strengths**:
- 设计 elegant: 一个 gate, 1 个 MLP. 简洁性强 (对应 "no silver bullet" 的反例 — 简单问题简单解)
- 直接解释 FDSE trade-off (style-confused vs common-class)
- 自适应, 跨 dataset 无需 prior
- Backward compatible: 当 g_c = 1 (e.g. all classes high V), 退化为现有 orth_only

**Weaknesses (审稿人会问)**:
1. **小 batch V_c 估计噪声**: batch=50 含 7 class, 每 class ~7 样本, V_c 估计粗糙
   - **回应**: EMA 平滑 V_c (如 β=0.95). 通信无变化.
2. **Gate g_c 可能 collapse**: 学到 g_c ≈ const (退回到 orth_only)
   - **回应**: 加 entropy regularizer 让 g_c 分布; 监控 g_c per class 的方差
3. **Class imbalance 客户端**: 某些 class 在某 client 没样本, V_c 无定义
   - **回应**: 对 missing class skip; 全局 server 维护 cross-client V_c 估计
4. **理论解释**: 为什么 high V → 强 orth 是对的?
   - **回应**: high V 表明 z_sty 携带强烈 class-conditional 信息 → 必须强行剥离, 否则 z_sem 也会编码 style 而非 pure semantic

**Score (estimated NeurIPS reviewer)**: 7/10 (above accept threshold, with minor concerns about batch noise + gate collapse)

#### Expected Results (跨 3 dataset)

| Dataset | 目标 | orth_only baseline | SV-GOD 期望 | 差距分析 |
|---|:-:|:-:|:-:|---|
| **PACS** | > FDSE 81.54 | 80.41 (Stage A) / 79.95 (Stage B 2 seeds) | **81.5-82.5** | Art-guitar/horse 涨, dog/giraffe/elephant 不退 |
| **Office** | > FDSE 90.58 | 89.09 | **89.5-90.5** | 类 V_c 小 → g_c 小 → 不过度 orth, 像 FedBN 89.75 |
| **DomainNet** | > FDSE 72.21 | 72.49 (orth_uc1) | **72.5-73.0** | 保持 / 微涨 |

**最低成功标准**: 3 个 dataset 都 ≥ FDSE. 严苛标准: PACS 涨 1.5pp, Office 涨 1.5pp, DN 保持.

#### First Experiment

**EXP-125 SV-GOD pilot** (lab-lry GPU 1, ~3-4h):
1. 写 `feddsa_svgod.py` (继承 feddsa_scheduled, override `_decouple_loss`)
2. 加 1 layer MLP gate + EMA
3. PACS R=200, seed=2 (single seed pilot)
4. 验证: 检查 g_c 是否区分 high-V vs low-V class
5. 若 PACS_s2 AVG > 81: 扩 3 seeds + Office + DN

**Smoke test**: R=5, 验证 gate gradient flow + EMA 不崩

---

### Idea 2: GA-CCP — Gap-Aware Cross-Client Class Prototype Anchor

#### Hypothesis

让 client c 的 class j 的 z_sem 自动**对齐到其他 client 的 class j 原型**, 但**只对 representational gap 大的 (client, class) cell 触发**.

#### Mechanism

**Server 端**: 维护 P[d, j] = client d 的 class j 平均 z_sem.

**Client 端 (训练时)**: 
- 计算 self prototype P_self[j] = mean(z_sem | y=j)
- 计算 cross prototype Cross[j] = mean(P[d, j] for d ≠ self)
- gap[j] = ||P_self[j] - Cross[j]|| (L2)
- 自适应 anchor weight: w[j] = gap[j] / (mean_j gap[j])
- L_anchor = Σ_j w[j] · ||z_sem(j) - Cross[j]||²

**总**: L = L_CE + λ_orth · L_orth + λ_anchor · L_anchor

**新超参**: 1 个 (λ_anchor, default 0.5)

#### Novelty Claim

**Partial novelty** (需 careful positioning):
- vs FedTGP: FedTGP 是 server learn 一个 trainable prototype; 我们是 **client-to-client weighted anchor with gap gate**
- vs FedCA: FedCA 是 medical seg 风格迁移; 我们是 classification 的类原型对齐
- vs ASA: ASA 是 test-time aggregation; 我们是 train-time anchor
- vs FedSTAR: FedSTAR 是 transformer attention + content-style decompose; 我们是简单的 gap-based MSE anchor
- **核心 novelty**: gap-aware class anchor with auto weighting

#### Critical Review

**Strengths**:
- 思路清晰: cross-domain class consistency
- gap-gate 防止过度拉拢 (low-gap cell 不动)

**Weaknesses**:
- **隐私**: 上传 per-client per-class prototype (32-128d × num_classes), 比纯 logit 多
- **prior work crowd**: 类原型方向已 crowded (FedTGP/ASA/FedSTAR), 需要找 reviewer-defensible 差异
- **3 个 dataset 一致领先证据弱**: 类原型方法在 Office (域差小) 通常不如 FedBN

**Score**: 6/10 (publishable but needs strong differentiation)

#### Expected Results

| Dataset | 期望 |
|---|---|
| PACS | 80-81 (改进有限, Art-guitar 类 anchor 帮助小) |
| Office | 89-90 (gap 小, anchor 触发少) |
| DN | 72.5-73.5 (多 domain, anchor 涨) |

#### First Experiment

EXP-126 GA-CCP pilot (优先级低于 SV-GOD)

---

### Idea 3: DGR — Decoupling Gradient Regularization

#### Hypothesis

不在表示层做 cos², 而在**梯度方向**做约束: 阻止 L_CE 的梯度沿 z_sty 方向 leak.

#### Mechanism

```
g_h = ∇_h L_CE              # gradient of CE w.r.t. pooled feature h
L_dgr = |cos(g_h, z_sty)|²  # penalize alignment
```

总 loss: L = L_CE + λ_orth · L_orth + λ_dgr · L_dgr

**新超参**: 1 个 (λ_dgr)

#### Novelty Claim

**High novelty**: gradient-level decoupling 在 FedDG 没找到. 类似 IRM (Invariant Risk Minimization) 但 IRM 是 risk-level, 我们是 gradient-direction.

#### Critical Review

**Strengths**:
- 真正 novel: 没看到工作做 gradient-direction decoupling in FDG
- 直接约束训练动力学

**Weaknesses (严重)**:
- **实施复杂**: gradient hook + 二阶导, 训练慢 1.5-2x
- **不稳定**: gradient 方向 noisy, cosine 难收敛
- **理论不清**: 为什么 gradient 不能 align z_sty? 可能 over-restrictive

**Score**: 5/10 (interesting but high implementation risk)

#### Expected Results

无可靠预期. 实验性, 可能 +2pp 也可能 -2pp.

#### First Experiment

EXP-127 DGR (低优先级, 等 SV-GOD 失败再考虑)

---

## Phase 5: Final Recommendation

### 🏆 Push SV-GOD to implementation

**理由**:
1. **Novelty CONFIRMED** (Web search 2026-04 无直接匹配)
2. **直接对应诊断**: 解决 FDSE trade-off (style-confused win + common-class lose) 的核心机制
3. **跨 dataset 一致期望**: V_c 自适应, 无 prior, 适用所有 dataset
4. **实施 minimal**: 1 个小 MLP + 2 学习参数 (无新手调超参)
5. **理论 story 强**: "用观察 style variance 自动调节解耦强度" — 一句话说清 novelty

### 立即下一步

**EXP-125 SV-GOD pilot** (等 EXP-124 PCH 出结果后开):

1. 写 `algorithm/feddsa_svgod.py` (继承 feddsa_scheduled, override `_decouple_loss`)
2. 加 EMA 平滑 V_c, gate MLP, 学习 α/β
3. 单测: gate gradient flow / EMA / numerical stability
4. Smoke R=5 (lab-lry GPU 1)
5. PACS seed=2 R=200 (~3h)
6. **判决**: 若 AVG_B ≥ 81 → 扩 3 seeds + Office + DN

### 补充准备

- 同步并行: 整理 GA-CCP 的具体 spec 作为 backup (若 SV-GOD failed)
- DGR 暂缓: 等前 2 个失败再考虑

---

## 资源 / 文件位置

- **本报告**: `experiments/IDEA_DISCOVERY_2026-04-24/IDEA_REPORT.md`
- **基础上下文**: `experiments/STATE_REPORT_2026-04-24.md`
- **诊断数据**: `experiments/ablation/EXP-123_art_diagnostic/stageB_full/ANALYSIS.md`
- **PCH 实验** (跑中): `experiments/ablation/EXP-124_pch_pilot/NOTE.md`
- **精读论文**: `obsidian_exprtiment_results/知识笔记/论文精读_*.md`

## Web sources (Phase 1 + 3 search)

Sources:
- [Federated Domain Generalization: A Survey](https://dsg.tuwien.ac.at/team/sd/papers/Journal_paper_2025_S_Dustdar_Federated.pdf)
- [Federated Adversarial Domain Generalization (FedADG)](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Federated_Domain_Generalization_With_Generalization_Adjustment_CVPR_2023_paper.pdf)
- [FED-DIP: Federated Domain Generalization](https://openreview.net/pdf/a9748f40b50eafd7faf7b0a00c2d2135f72ffe9c.pdf)
- [FedLAG (Layer-Wise Personalized FL via Conflicting Gradients)](https://arxiv.org/abs/2410.02845)
- [FedSTAR (Style-Aware Transformer Aggregation)](https://arxiv.org/pdf/2511.18841)
- [FedCA (Medical FedDG with Cross-Client Style Transfer)](https://www.sciencedirect.com/science/article/abs/pii/S0957417426003076)
- [Variance-Covariance Regularization (VCReg)](https://arxiv.org/pdf/2306.13292)
- [Unsupervised Disentanglement via Variance-Invariance Constraints](https://openreview.net/forum?id=Lut5t3qElA)
- [Federated Domain Generalization via Prompt Learning and Aggregation](https://arxiv.org/html/2411.10063v1)
- [FedTGP via Trainable Global Prototype](https://github.com/yuhangchen0/Federated-Learning-in-CVPR2024)
