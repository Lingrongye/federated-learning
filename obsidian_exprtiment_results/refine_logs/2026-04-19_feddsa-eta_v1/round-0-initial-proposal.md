# Research Proposal: FedDSA-ETA (ETF-Head + Style-Aware TTA)

## Problem Anchor
- **Bottom-line problem**: 在跨域联邦学习 (feature-skew FL, PACS/Office-Caltech10) 场景下,客户端来自不同视觉域 (photo/art/cartoon/sketch),如何在已稳定的"正交双头解耦"基线 (Plan A: z_sem ⊥ z_sty + LR=0.05 + 可选 SAS) 之上**再取得 ≥1% AVG 提升**,同时 **PACS 与 Office 双数据集通用**,不引入任何会长期崩溃的机制。
- **Must-solve bottleneck**: Plan A 已把 backbone/解耦几乎用到极限。当前瓶颈有两个:
  - **训练端**: sem_classifier 是普通 Linear+CE,与正交化的 z_sem 几何不匹配;同时 classifier 参加 FedAvg 聚合,是每轮一个可训练的漂移源。
  - **推理端**: z_sty 和 style_bank 在训练时已经被收集/正交约束,**推理时完全不用**,这是免费资源。domain shift 在测试样本上的 style signature 也没有被利用。
- **Non-goals**:
  - 不再引入任何持续性对齐损失 (InfoNCE/Triplet/Supcon — 已被 21/25 runs 证伪)
  - 不再做风格共享跨客户端注入 (EXP-059/078d/095 三连败)
  - 不引入 CLIP/VLM/frozen backbone (留给 FedDEAP 等,我们坚守 ResNet-18/AlexNet from-scratch)
  - 不主张"FL+ETF+decouple 第一"的 headline (FedDEAP 抢了)
- **Constraints**:
  - Backbone: AlexNet (PACS, E=5) / ResNet-18 (Office, E=1) from scratch
  - 单卡 24GB,3-seed {2,15,333} 对齐 FDSE/Plan A baseline
  - R200 训练预算 (PACS ~8h × 3 seeds, Office ~1h × 3 seeds)
  - FedBN 原则: BN 层本地私有; FedAvg backbone + classifier; SAS 只替换 sem_head 聚合
- **Success condition**:
  - **PACS**: AVG Best ≥ 81.5% (Plan A = 80.41); ALL Best ≥ 83% (Plan A = 82.31)
  - **Office**: AVG Best ≥ 90.5% (SAS = 89.82, FDSE = 90.58); ALL Best ≥ 85%
  - **稳定性**: drop ≤ 1%, 3-seed std ≤ 1.5% (Plan A AVG drop = 0.99%)
  - **消融显示**: ETF / TTA 各自独立 +0.5~1%,叠加 +1~2%

---

## Technical Gap

**Plan A 已经做对的事**:
- 正交解耦把语义 z_sem 和风格 z_sty 分开,避免跨域混淆
- LR=0.05 + decay 让长期训练稳定

**Plan A 没做的事 (= 本 proposal 的切入点)**:

### Gap 1: 分类器 (sem_classifier) 与 z_sem 几何不一致
- 正交约束让 z_sem 内部彼此接近"类间等角"的配置 (Neural Collapse 的 NC2 性质)
- 但下游的 Linear + CE 分类器权重是**自由学习**的 — 每轮被不同 client 的本地梯度推向不同方向
- 再经 FedAvg 粗平均 → classifier 成为"漂移源"
- 即使 z_sem 已经接近 ETF vertices,classifier 也没对齐这些顶点
- **缺失的机制**:一个**固定的、理论最优的分类器**让 z_sem 直接对齐已知几何

### Gap 2: z_sty 和 style_bank 推理时被浪费
- 训练结束后,每个 client 有自己的 style (μ_k, σ_k)
- style_bank 里有所有其他 client 的 (μ_j, σ_j)
- test 样本过 backbone + 双头后也能提取 (μ_test, σ_test)
- 这些风格签名**应该告诉我们**:
  - test 样本是否和训练时见过的风格差太远 → 不可靠预测
  - test 样本语义特征 z_sem 是否在受风格污染 → 需要去风格化
- 当前 Plan A 推理直接吃 argmax(linear(z_sem)),**完全忽略**这些免费信号

### Naive 修复为什么不够
- "再加一个分类器"或"再训一个 style-aware head" → 重引入 classifier 漂移
- "在训练时加 InfoNCE 拉近同类 z_sem" → 已证伪 (EXP-076/077/078 全崩)
- "把 style_bank 当特征 augmentation" → 已证伪 (EXP-059 PACS -2.54%)
- "test-time fine-tune 用 entropy" (TENT) → 涉及反向传播,触发 BN 不稳定

### 最小充分干预
- **Training side**: 用 **Fixed Simplex ETF** 替代 sem_classifier (零可训参数,零 aggregation 噪声,与 NC 理论自洽)
- **Inference side**: **完全 backprop-free** 的测试时机制 — SATA style-exchange 做可靠性 gate,T3A 在线 proto 做可靠样本的二次分类。两者都只读 z_sem/z_sty,**不更新任何参数**

### Frontier-native check
- Neural Collapse (Papyan NAS'20) + ETF 分类器 = 2023-2025 一直是 FL 领域的活跃迁移源 (FedETF, FedLSA, FedDEAP)
- TTA 方向 SATA/T3A/DeYO/EATA = 2023-2024 顶会主线
- 两者都是"foundation-model 时代的 prior 感"方法:**几何一致性 + test-time 利用**
- 我们**拒绝**使用 CLIP/VLM 作为 moat — 这是刻意选择,与 FedDEAP 区分,保住 ResNet-18 from-scratch 的公平比较赛道

### Core technical claim
**"在 feature-skew FL 下,Plan A 的正交解耦 + Fixed ETF 分类头构成训练端的'类-特征几何一致性',而推理端的 SATA 可靠性 gate + T3A 在线原型构成'风格感知的测试时校正',两者协同在 PACS/Office 双数据集上稳定超越 Plan A。"**

### Required evidence
- **E1**: ETF 替换 Linear,**单独**在 PACS/Office 都 +0.5~1% (三 seed mean)
- **E2**: SATA + T3A TTA 加在 Plan A checkpoint 上,**单独**免费 +0.5~2%
- **E3**: ETF + TTA 叠加后,PACS AVG Best ≥ 81.5%, Office ≥ 90.5%
- **E4**: 与 FedDEAP (CLIP) 不同:ResNet-18 scratch 下,我们 ≥ FedDEAP reproduced
- **E5**: 所有三 seed std ≤ 1.5%,drop ≤ 1%

---

## Method Thesis
- **One-sentence thesis**: 把 Plan A 的 sem_classifier 换成**固定单纯形 ETF** (训练端几何一致性) + 在推理时用 **SATA 风格交换 gate + T3A 在线原型** (推理端几何一致性),构成"训练-推理几何自洽"的 feature-skew FL 框架。
- **Why this is the smallest adequate intervention**:
  - Fixed ETF = **0** 可训参数 (只是个 buffer,不参加聚合) → 比 Linear 更小
  - SATA+T3A TTA = **0** 训练修改 (推理时才激活) → 完全增量
  - 不引入任何持续性对齐损失,不碰 backbone,不改训练循环
- **Why this is timely**:
  - Neural Collapse 在 FL 的 label-skew 已有 FedETF 验证,但在 feature-skew DG 场景尚无定论
  - TTA 在 centralized DG 已经成熟 (SATA'25, DeYO ICLR'24),但**federated TTA 只做过基于 entropy 的 FedCTTA**,从未利用 disentangled features 作为 gate
  - "几何一致性 across training-inference" 是一条新的 FL+DG 叙事

---

## Contribution Focus
- **Dominant contribution**: **Style-Aware Federated Test-Time Adaptation** (SATA-gate + T3A-proto 融合,完全 backprop-free,复用 decoupled features)
- **Supporting contribution**: **Fixed Simplex ETF 分类头** + 正交解耦的几何互补性论证 (含简短 Neural Collapse 理论分析)
- **Explicit non-contributions**:
  - 不是 "第一个 FL+ETF+decouple" (FedDEAP 抢先)
  - 不是 "新的损失函数/增强机制/聚合策略"
  - 不是 "FL 下的 domain generalization 新 baseline" — 是现有 Plan A 的增强

---

## Proposed Method

### Complexity Budget
- **Frozen / reused**: Plan A 全部 (backbone, 双头解耦, L_orth, LR schedule, style_bank 收集流程, SAS 可选)
- **New trainable**: **无** — Fixed ETF 是固定 buffer,TTA 完全 backprop-free
- **Intentionally excluded**:
  - 不加 contrastive loss / InfoNCE (已证伪)
  - 不做 cross-client style fusion (已证伪)
  - 不改 BN / classifier aggregation (只是换了 classifier 的几何形状)
  - 不引入 prompt/adapter/VLM (保住 ResNet 赛道)

### System Overview
```
[Training phase]
  x → backbone → double_head
       ↓
  z_sem (128d)  ⊥  z_sty (128d)    [L_orth 正交约束]
       ↓               ↓
  Fixed-ETF M     style_bank 收集 (μ,σ)
       ↓
  logits = F.normalize(z_sem) @ M / τ
       ↓
  CE loss (同时用原样本和 style-augmented 样本,保持 Plan A 的 CE+CE_aug)

[Server aggregation]
  FedAvg: backbone + sem_head + sty_head (Plan A 不变)
  FedBN: BN 本地
  SAS (Office only, opt-in): sem_head 风格加权聚合
  Fixed ETF M: 不聚合 (所有 client 用同一个 seed 构造的固定 M)

[Inference phase — NEW]
  for x in test_loader:
    with torch.no_grad():
      z_sem, z_sty = backbone + double_head(x)
      logits_orig = F.normalize(z_sem) @ M / τ
      H = entropy(softmax(logits_orig))

      # SATA gate
      mu_ext, sig_ext = sample_from_style_bank(size=B, exclude_self=True)
      z_sem_swap = AdaIN(z_sem, mu_ext, sig_ext)  # 在 z_sem 特征空间做 AdaIN
      logits_swap = F.normalize(z_sem_swap) @ M / τ
      reliable = (logits_orig.argmax(1) == logits_swap.argmax(1)) & (H < τ_H)

      # T3A proto (EMA 更新 + fallback)
      for c in range(K):
        mask = reliable & (logits_orig.argmax(1) == c)
        if mask.any():
          proto[c] = γ · proto[c] + (1-γ) · z_sem[mask].mean(0)

      # 分类 (cold-start fallback: proto 未被更新时用 ETF vertex)
      pred_proto = argmax_c cos(z_sem, proto[c])
      pred_etf = logits_orig.argmax(1)
      pred = where(reliable & proto_initialized[pred_proto], pred_proto, pred_etf)
```

### Core Mechanism

#### (1) Fixed Simplex ETF Classifier (训练端)
- **Input**: z_sem ∈ ℝ^{B×d}, d=128
- **Construction** (server init,所有 client 复制):
  ```
  U = torch.linalg.qr(torch.randn(d, K, generator=seeded_rng))[0]  # [d, K] 正交
  M = β · √(K/(K-1)) · U @ (I_K - (1/K)·1_K·1_K^T)                 # [d, K] 单纯形 ETF
  # M 是 buffer,不 requires_grad
  ```
- **Forward**:
  ```
  logits = F.normalize(z_sem, dim=-1) @ M / τ_etf    # τ_etf=0.1
  loss = CE(logits, y)
  ```
- **Why**: ETF vertices 是 K 个等角最大间隔向量,与 L_orth 产生的类间等角特征天然对齐。FedAvg 不需要聚合分类器 (所有 client 同一个 M),消除漂移源。

#### (2) SATA Style-Exchange Reliability Gate (推理端)
- **Input**: z_sem, z_sty, style_bank {(μ_k, σ_k)}_{k=1..N_domains}
- **Procedure**:
  ```
  # 从 style_bank 随机采样 (排除当前 client 的 style)
  (μ_ext, σ_ext) ← sample_uniform(style_bank, exclude=self_idx)   # [B, d]

  # AdaIN: 把 z_sem 的 channel-wise (μ, σ) 替换为外部域的
  μ_sem, σ_sem = z_sem.mean(0, keepdim=True), z_sem.std(0, keepdim=True)  # per-channel
  z_sem_swap = σ_ext · (z_sem - μ_sem) / (σ_sem + ε) + μ_ext

  # 过 ETF head,比较预测
  logits_swap = F.normalize(z_sem_swap) @ M / τ_etf
  H = -Σ_c p_c · log p_c   where p = softmax(logits_orig)
  reliable = (argmax(logits_orig) == argmax(logits_swap)) & (H < τ_H)   # τ_H=0.4
  ```
- **Why**: 若 test 样本预测对风格扰动鲁棒 (style-swap 不改变 argmax) 且原预测确定 (低熵),则它处于"clean semantic region",可安全用于 proto 更新。反之则可能被风格污染,回退到 ETF 原始预测。

#### (3) T3A On-the-fly Prototype Classifier (推理端)
- **Input**: reliable mask, z_sem
- **Bank initialization**: proto ∈ ℝ^{K×d},初始化为 M.T (ETF vertices 方向,归一化)
- **Update** (EMA):
  ```
  for c in range(K):
    mask_c = reliable & (logits_orig.argmax(1) == c)
    if mask_c.any():
      proto[c] = γ · proto[c] + (1-γ) · F.normalize(z_sem[mask_c].mean(0), dim=-1)
  ```
- **Classification** (cold-start safe):
  ```
  pred_proto = argmax_c cos(F.normalize(z_sem), proto[c])
  pred_etf = logits_orig.argmax(1)
  # fallback: 如果该类 proto 还没被 reliable 样本激活过,用 ETF
  pred = where(updated[pred_proto], pred_proto, pred_etf)
  ```
- **Why**: reliable 样本构成 domain-specific pseudo-prototype,对 test domain 的实际 class means 做 online estimate,比固定 ETF vertices 更贴近 test distribution。初始化为 ETF vertices 保证 cold-start 不退化。

### Modern Primitive Usage
- **Neural Collapse (NC)**: 作为 "为什么 ETF 是 z_sem 最优分类器" 的理论 backing
- **Test-Time Adaptation (TTA)**: 作为 "训练后仍可免费提升" 的现代范式
- **Disentangled Features + Style-exchange**: 不是简单 style transfer,是用来做 reliability signal
- **不用** CLIP/DINOv2/VLM (刻意与 FedDEAP 差异化)

### Integration
- `PFLlib/system/flcore/clients/clientdsa.py`:
  - `__init__`: 构造 Fixed ETF buffer M,替换 self.sem_classifier
  - `train()`: logits = F.normalize(z_sem) @ M / τ,其余不变
  - `test()`: 新增 `test_with_tta()` — SATA gate + T3A proto + fallback
- `PFLlib/system/flcore/servers/serverdsa.py`:
  - `aggregate_parameters()`: 排除 self.sem_classifier (ETF 是共享 buffer,不聚合)
  - 其余 FedAvg + FedBN + 可选 SAS 保持不变

### Training Plan
- **Stage 1** (R0-R200): Plan A 训练,唯一改动是 sem_classifier → Fixed ETF
  - 损失: CE(logits_sem) + CE(logits_sem_aug) + λ_orth · L_orth
  - λ_orth = 1.0, τ_etf = 0.1 (默认 NC 温度),β = 1.0
- **Stage 2** (推理时,无训练): 加载 R200 checkpoint,test_with_tta 激活 SATA+T3A

### Failure Modes and Diagnostics

| 失败模式 | 检测方法 | Fallback |
|---------|---------|---------|
| ETF 在 PACS 4-outlier 下 hurt | `per-seed ETF vs Linear` 实验,若 PACS 3-seed mean 掉 >0.5% → 回退 | 切回 Linear classifier,只用 TTA 部分 |
| SATA reliable mask 永远 True/False | 记录 `reliable.float().mean()` 每 client,应在 [0.3, 0.9] | τ_H 自适应 (per-client,第一批 test 样本校准) |
| T3A proto 冷启动 NaN / 少样本崩 | proto_initialized mask,未激活用 ETF fallback | ETF fallback 默认永远可用 |
| style_bank 只有 4 个域 (PACS) 导致 swap 多样性不足 | 记录 SATA agreement rate,若 <0.2 或 >0.95 → 报警 | 引入 per-sample shuffle style mixing |
| 与 FedDEAP (CLIP) 比较时赛道不公平 | 明确声明 "ResNet-18 from-scratch 赛道" | Related Work 中分开报告 |

### Novelty and Elegance Argument

**最接近的工作**:
- **FedDEAP** (arXiv 2510.18837, 2025-10): CLIP prompt-tuning 下用 ETF-constrained transformation decouple semantic/domain。**差异**: 我们 ResNet-18 from-scratch,单侧 ETF (只在 z_sem), z_sty 保持 Gaussian bank,完全不同技术栈。
- **FedETF** (ICCV'23): label-skew FL 的 fixed ETF head。**差异**: 我们是 feature-skew DG 场景,ETF 与正交解耦组合是新的。
- **FediOS** (ML'25): 正交投影 generic/personalized。**差异**: 我们是 semantic/style,用 ETF 代替可训 head。
- **SATA** (IVC'25): centralized DG 的 style-exchange TTA。**差异**: 我们是 FL 下,且用 z_sty+style_bank 而非图像空间。
- **T3A** (NeurIPS'21 Spot): centralized DG 的 proto TTA。**差异**: 我们在 ETF vertices 上初始化 proto,且与 SATA gate 融合。
- **FedDG-MoE** (CVPRW'25): feature-stat cosine routing + MoE。**差异**: 我们不用 MoE,直接复用 style_bank。
- **FedCTTA** (arXiv'25): pure entropy TTA for FL。**差异**: 我们利用 disentangled features 做 gate,而非 entropy only。

**核心 moat**:
- "Style-exchange reliability gate + ETF-initialized T3A proto, 基于 disentangled features, 完全 backprop-free, 在 federated setting 下" 这个完整组合在 2024-2026 文献中**无占坑**
- 与 Plan A 完全兼容,所有消融 (ETF-only / TTA-only / 组合) 都可做

---

## Claim-Driven Validation Sketch

### Claim 1 (Dominant): Style-Aware Federated TTA 在 Plan A checkpoint 上免费 +0.5~2%
- **Minimal experiment**: Plan A checkpoint × {w/o TTA, w/ SATA-only, w/ T3A-only, w/ SATA+T3A} × {PACS, Office} × 3 seeds
- **Baselines/ablations**:
  - Plan A (orth_only + LR=0.05) — 已有
  - Plan A + TENT (entropy min TTA baseline)
  - Plan A + FedCTTA (FL-TTA baseline)
- **Metric**: AVG Best, AVG Last, ALL Best, ALL Last, drop, 3-seed std
- **Expected evidence**: PACS TTA 免费 +0.5~1.5%, Office +0.3~1.5%, std 不增加

### Claim 2 (Supporting): Fixed ETF 与正交解耦几何互补
- **Minimal experiment**: 统一训练 {Linear+CE, Fixed ETF+CE, Learnable Proto+CE} × {PACS, Office} × 3 seeds
- **Baselines/ablations**:
  - Linear (Plan A 原版)
  - Learnable Prototype Classifier (IJCAI'23)
  - Fixed ETF (本方案)
- **Metric**: AVG Best / ALL Best / 3-seed std
- **Expected evidence**: Fixed ETF ≈ Plan A 或 +0.3~1%, 但 std 明显减小 (classifier 无漂移)

### Claim 3 (Integration): ETF + TTA 叠加 PACS ≥ 81.5%, Office ≥ 90.5%
- **Minimal experiment**: Fixed ETF 训练 + SATA+T3A 推理 × {PACS, Office} × 3 seeds
- **Baselines**:
  - FDSE R200 复现 (既有)
  - Plan A (既有)
  - FedDEAP adapted to ResNet-18 (次要,若可复现)
- **Expected evidence**:
  - PACS AVG Best ≥ 81.5%, ALL Best ≥ 83% (Plan A = 80.41/82.31)
  - Office AVG Best ≥ 90.5%, ALL Best ≥ 85% (Plan A = 80.41/82.55)
  - 3-seed std ≤ 1.5%, drop ≤ 1%

---

## Experiment Handoff Inputs
- **Must-prove claims**: C1 (TTA 免费 +), C2 (ETF ≥ Linear), C3 (叠加达 success condition)
- **Must-run ablations**: ETF-only / TTA-only / 组合; SATA-only / T3A-only; τ_H ∈ {0.2, 0.4, 0.6}; γ ∈ {0.8, 0.9, 0.95}
- **Critical datasets/metrics**: PACS AVG+ALL Best/Last, Office AVG+ALL Best/Last, 3-seed std
- **Highest-risk assumptions**:
  - PACS 4-outlier 下 style_bank 多样性 (N=4) 可能不够 SATA gate 激活
  - AlexNet 128d z_sem 与 ETF vertices 几何一致性实际强度
  - Fixed ETF 在 4-client FedAvg 聚合的兼容性 (理论上无需聚合但实现细节要对)

## Compute & Timeline Estimate
- **ETF 训练**: 3-seed × 2 datasets × R200 = PACS ~24h + Office ~3h ≈ **27 GPU-hours**
- **TTA 推理**: 复用 Plan A/ETF checkpoint, 每 checkpoint 几分钟 × 10 配置 × 3 seeds ≈ **3 GPU-hours**
- **Ablation**: Learnable Proto classifier × 3 seeds × 2 datasets ≈ 27 GPU-hours
- **Total**: ~60 GPU-hours (3-4 天 on 单卡 24GB)
