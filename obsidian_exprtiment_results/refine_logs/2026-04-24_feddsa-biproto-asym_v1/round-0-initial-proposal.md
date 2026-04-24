# Research Proposal: FedDSA-BiProto — 非对称双编码器与联邦双原型互斥解耦

**Date**: 2026-04-24
**Stage**: Round-0 Initial Proposal
**Target**: 在 FDSE_CVPR25 框架下, 3-seed mean AVG Best 严格胜过 FDSE 本地复现 baseline (PACS ≥ 80.91, Office-Caltech10 ≥ 91.08), 同时交付 t-SNE / probe ladder / feature metrics 三套可视化 evidence.

---

## Problem Anchor (不可变, 每轮必须逐字复刻)

- **Bottom-line problem**: 在 cross-domain 联邦学习 (PACS / Office-Caltech10) 下, 用 AlexNet from scratch, 3-seed {2, 15, 333} × R200 mean AVG Best **必须同时严格超过 FDSE 本地复现 baseline** (PACS > 79.91 / Office > 90.58). 当前 orth_only 战况: PACS 80.64 ✅ (+0.73), Office 89.09 ❌ (−1.49).
- **Must-solve bottleneck**: Office-Caltech10 必须补回 −1.49 pp 且至少再涨 +0.5 pp, 同时 PACS 不得退 (≥ 80.91). **核心技术悬念**: Office gap 的真实原因是 "shared trunk 被 CE 污染, class 弥散到 z_sty" 还是 "trunk capacity 已饱和 / 聚合不对齐 / Caltech outlier"? EXP-110 / EXP-113 / EXP-120 的证据链倾向后者, 但 trunk-freeze diagnostic **从未做过**, 必须先做诊断再决定架构级投入.
- **Non-goals**:
  1. **不换数据集** — 禁止 pivot 到 FEMNIST/Rotated MNIST/Camelyon17 等"容易打"的场景
  2. **不做诊断论文** — 不接受 "accuracy 没胜但 probe 好看 / 方法论更严谨" 的妥协叙事
  3. **不堆模块凑 novelty** — 至多 1 个 dominant contribution + 1 个 supporting contribution
  4. **不预训练** — AlexNet from scratch, 和 FDSE 公平对比
  5. **不换骨干** — 保持 FDSE_CVPR25 框架与 AlexNet (ResNet-18 留作后续扩展)
- **Constraints**:
  - Compute: seetacloud / seetacloud2 (AutoDL RTX 4090 24GB) + lab-lry (双 RTX 3090 共享). 单 run ~2-4h Office / ~7h PACS. 预算 ≤ 50 GPU-h 总计
  - Data: PACS (4 clients, 每 client ~1K), Office-Caltech10 (4 clients, A/W/D/C, DSLR 仅 157 样本)
  - Framework: FDSE_CVPR25 (flgo), AlexNet, FedBN 原则 (BN 本地), E=5 (PACS) / E=1 (Office)
  - Timeline: 1 周内完成 refine + diagnostic + pilot 判决
- **Success condition** (必须全部满足):
  1. 3-seed {2, 15, 333} × R200 mean AVG Best: PACS ≥ 80.91 **且** Office ≥ 91.08
  2. 3-seed AVG Last 不退 (PACS ≥ 79.0, Office ≥ 88.5), 无 R50 之后的明显崩盘
  3. **三套可视化 evidence** 齐备且支撑"解耦做成了"的叙事:
     - t-SNE 双面板 (z_sem by class / z_sty by domain) 结构清晰
     - Probe ladder (linear / MLP-64 / MLP-256) 四向量 (z_sem→class, z_sem→domain, z_sty→class, z_sty→domain) 呈预期方向
     - Prototype quality + feature norm + orth 轨迹均在健康区间
  4. novelty 在 FL × cross-domain × prototype × disentanglement 四维交叉下有明确差异化 (和 FedProto/FPL/FDSE/FedSTAR/FedDP/FedSeProto/FedFSL-CFRD/I2PFL/FISC/FedPall 都有 ≥1 个维度不重合)

---

## Technical Gap

### 当前方法的失败点

1. **"擦除派" (FDSE/FedDP/FedSeProto)**: 将 domain 当噪声擦除. Office 上 FDSE 90.58 / PACS 82.17 是论文数, 但我们的 FDSE 本地复现 PACS 79.91 (被 orth_only 胜 +0.73) 说明 "擦除" 在 PACS 不是上限. Office 确实强, 但机制是层分解 + QP 聚合, 参数量 0.65× 于基线, **擦除本质上不为 domain 建模**, 无法解释跨 domain 的结构化迁移.
2. **"私有派" (FedSTAR/FedSDAF/FedBN)**: 风格严格本地. FedBN 是 orth_only 的前身 (BN 本地), 已经是我们的基线组件. FedSTAR 用 FiLM 分离但 style 不跨 client. **domain 被降级为 nuisance, 不能作为联邦共享对象**.
3. **"共享不解耦派" (FISC/StyleDDG/FedCCRL/MixStyle)**: 混合空间 AdaIN. 这是我们已有 style bank 在做的事, EXP-060/061 已经充分扫过, 在 Office 反而 −1.97. **原因: 不解耦 → 风格迁移顺带搬运 class, 引入负迁移**.
4. **"解耦且 feature-level 对抗派" (CDANN/FedSeProto 的对抗变体)**: EXP-108 已 3-seed 证伪, probe 0.96 但 accuracy 零增益, 设计缺陷: 把 class 挤进 z_sty.
5. **我们当前 orth_only 的局限**: L_orth cos² 只在输出向量层面正交, EXP-109/111 已测出 MLP-256 probe 可读出 z_sty 的 class 信号到 0.81. 诊断上是"线性解耦成功, 非线性仍混淆". 但 EXP-110/113 显示在 Office 上**这个非线性泄漏未必是 accuracy 瓶颈** —— gap 的真实来源可能在聚合 / outlier / capacity 一侧.

### 为什么朴素扩展不够

- **直接加 L_CE on z_sty**: = CDANN, 已证伪
- **直接加 HSIC / MI 最小化**: EXP-017 HSIC 被证有害, EXP-095 SCPR M3 在 PACS 全 outlier 证伪
- **直接加 InfoNCE 多原型**: EXP-076/077/078 无安全阀崩盘, 必须 warmup + MSE anchor + α-sparsity
- **直接加第二个 AlexNet encoder (原 DualEncoder 方案)**: 参数 2×, 通信 2×, 且 "不让 CE 梯度流" 的 probe 下降是 tautology, 不构成真正的 disentanglement claim

### 缺失的机制

目前所有 FL cross-domain 解耦工作对 "domain 信息" 的处置只有三种: 擦除 / 私有 / 本地对抗. 从未有工作**把 domain 升级为一阶 federated 对象** —— 即像 class prototype 一样在 server 端维护跨 client 的 domain prototype bank, 并在 prototype 级 (而非 feature 级) 强制与 class prototype 几何互斥. 这个机制缺口是本文的主 attack surface.

同时, 目前 FL 里**独立 style encoder 几乎都是对称双 trunk** (2× AlexNet), 忽视了风格迁移文献里 "style 信息 ≈ (μ, σ) 统计" 的强先验 (AdaIN/MixStyle). 一个专门捕捉统计量的**轻量非对称 encoder** 可以在 <2% 参数膨胀下提供 proper inductive bias.

---

## Method Thesis

- **One-sentence thesis**:  
  **把 domain 从"要擦除的噪声"升级为"联邦共享的一阶原型 (Pd)", 与 class prototype (Pc) 通过 proto-level 几何互斥并列存在; 用非对称的统计 encoder 给 Pd 提供 inductive bias; 整套机制在 loss 数目和参数量上保持极简.**
- **Why this is the smallest adequate intervention**:  
  相较 CDANN 需要引入 GRL + dom_head + 对抗平衡 (3 个 loss, 已证伪), BiProto 只在 orth_only 基础上加 1 个轻量 encoder (~1M) + 2 条 proto-level InfoNCE + 1 条 proto-cos 排斥 = 总新增 loss 3 个. 所有 loss 都在 proto 空间 (低维 K × d, K≤10), 梯度冲突半径极小, 避开 EXP-017/095/108 的全部已知坑. 聚合层面, domain prototype 天然每 client 只有一个 (因为每 client 一个 domain), 直接 concat 无聚合冲突.
- **Why this is timely in the foundation-model era**:  
  - AdaIN (2017) + FedSTAR (2024) 已经验证 (μ, σ) 足以表征 style, 但没人把 style 统计升级为 federated primitive
  - Neural Collapse / ETF prototype (FedETF ICCV'23, 我们 EXP-096 已复现) 证明了 prototype 作为一阶对象在 FL 里的价值 — 但只用在 class, 从未用在 domain
  - 2025 年 FedDP / FedSeProto 都走"MI 最小化擦 domain"路线, 本方案是**反向补集** (保留并共享 domain 原型) 的逻辑自然继承
  - 本方案不引入 LLM/VLM/Diffusion, 因为数据规模 (Office ~2K, PACS ~9K) 不适合 foundation model scale — **intentionally conservative on frontier leverage**

---

## Contribution Focus

- **Dominant contribution (Claim C1, accuracy-level)**:  
  **Federated Domain Prototype (Pd) as a first-class shared object**, orthogonal-to-class at the prototype level via cosine exclusion. 这是首个在 FL cross-domain 场景将 domain 作为与 class 对称的一阶联邦对象建模的工作. 机制层: Pd 由非对称统计 encoder 生成, 跨 client 直接 concat (无聚合歧义, 因为每 client 一个 domain). 监督层: Pd 通过 sty-proto InfoNCE 驱动 encoder_sty 学习 "same-domain attraction, cross-domain repulsion", 同时 Pd ⊥ Pc 的 proto-level cosine 损失在低维空间强制互斥.
- **Supporting contribution (Claim C2, mechanism-level)**:  
  **Asymmetric statistic encoder** — 风格 encoder 只做 (μ, σ) 统计提取 + 轻量 MLP (~1M params, <2% of AlexNet 60M), 和已有 style bank (μ, σ) 做 AdaIN 增强在语义上天然对齐, 解决原 DualEncoder 方案 "参数/通信 2×" 的死结. 同时这个设计让 encoder_sty 物理上无法编码 class-level 判别信息 (统计量丢弃空间结构), 从架构上切断 "sty 承载 class" 的泄漏路径, 无需依赖对抗 loss.
- **Explicit non-contributions** (禁止审稿中扩展):
  - 不 claim "我们的 disentanglement 比 FedSeProto 更 MI-optimal"
  - 不 claim "我们的 FL 通信效率超过 FDSE"
  - 不 claim generalization bound / 收敛性定理 (AlexNet 小场景无必要)
  - 不 claim 多 backbone 通用性 (仅 AlexNet 证明, ResNet 留作 future work)
  - 不 claim 工业级 privacy (proto bank 的 DP 分析不是本文 scope)

---

## Proposed Method

### Complexity Budget

| 项目 | 处理 | 说明 |
|---|---|---|
| AlexNet encoder_sem | **继承 orth_only**, 不改 | ~60M, 只被 L_CE + L_orth + L_sem_proto 训 |
| semantic_head | **继承 orth_only**, 不改 | 1024 → 128 (Linear + BN + ReLU + Linear) |
| sem_classifier | **继承 orth_only**, 不改 | Linear 128 → K |
| BN running stats | **继承 FedBN**, 本地 | 聚合时 private_keys |
| style bank (μ, σ) AdaIN 增强 | **继承 FedDSA 原方案**, 不改 | channel-wise 1st/2nd stats, 跨 client 共享, AdaIN 注入 encoder_sem 中间层 |
| encoder_sty | **新增 (唯一架构改动)** | 统计 encoder: 从 encoder_sem 3 个 intermediate layer 抽 (μ, σ) → concat → 2-layer MLP → z_sty [128], **~1M params** |
| style_head | **删除** | 不再需要, 因为 encoder_sty 已是专用路径 |
| class prototype bank Pc | **新增** | [K, 128], EMA 聚合 (类 FedProto) |
| domain prototype bank Pd | **新增 (核心)** | [num_clients, 128], per-round concat (每 client 一个 domain, 天然无冲突) |
| L_orth | **继承**, cos²(z_sem, z_sty) 保留 | output-space 几何正交 |
| L_sem_proto | **新增** | z_sem → Pc InfoNCE (+ MSE anchor + α-sparsity) |
| L_sty_proto | **新增** | z_sty → Pd InfoNCE (+ MSE anchor + α-sparsity) |
| L_proto_excl | **新增** | cos²(Pc[k], Pd[d]) over all (k,d), proto 级互斥 |

**净增**: ~1M 参数 (+1.7%) + 3 条 loss (全部有安全阀) + 2 个 bank (低维存储 ≤ 2KB/round 通信).

### System Overview

```
         ┌─────────── x (input image) ───────────┐
         │                                        │
         ▼                                        │
  AlexNet encoder_sem (60M)                       │
  ├── conv1 → BN₁ → ReLU → pool ──┐               │
  │                                ├─── stat tap  │
  ├── conv2 → BN₂ → ReLU → pool ──┤   (μ, σ)      │
  │                                │   per-layer  │
  ├── conv3 → BN₃ → ReLU ────────┐ │              │
  │                              │ │              │
  └── ...deeper layers + pool → 1024d             │
                      │                           │
                      ├──► semantic_head → z_sem [128]
                      │                    │
                      │                    ├──► sem_classifier → logits ──► L_CE
                      │                    │
                      │                    └──► (z_sem, Pc) InfoNCE + MSE anchor ─► L_sem_proto
                      │
                      │   (stat tap: concat (μ_1,σ_1,μ_2,σ_2,μ_3,σ_3))
                      │       │
                      │       ▼
                      │   statistic encoder_sty (MLP ~1M) ──► z_sty [128]
                      │       │
                      │       ├──► (z_sem, z_sty) cos² ────► L_orth
                      │       └──► (z_sty, Pd) InfoNCE + MSE anchor ─► L_sty_proto
                      │
                      │   (style bank (μ,σ) AdaIN @ conv2) ─► z_sem_aug ── L_CE_aug (继承原方案)
                      │
                      └──► (Pc, Pd) cos² pairwise ─────────────► L_proto_excl
```

### Core Mechanism — Federated Domain Prototype (Pd)

**Input**: pooled intermediate stats (μ, σ) ∈ ℝ^(2×Σ Cᵢ) from 3 conv layers
**Output**: z_sty ∈ ℝ^128
**Architecture**: 2-layer MLP with LayerNorm
```python
class StatisticEncoder(nn.Module):
    def __init__(self, stat_dims: List[int]):  # e.g. [96, 256, 384] for 3 conv layers
        self.in_dim = 2 * sum(stat_dims)  # 2 for μ + σ
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, taps):  # taps: List[Tensor(B, C, H, W)]
        feats = []
        for t in taps:
            mu = t.mean(dim=[2, 3])         # [B, C]
            sigma = t.std(dim=[2, 3]) + 1e-5  # [B, C]
            feats.extend([mu, sigma])
        return self.net(torch.cat(feats, dim=-1))  # [B, 128]
```

**Pd update (server-side)**:
- 每 round 每 client k 上传自己所有样本的 z_sty 均值 p_k^sty ∈ ℝ^128
- Server 直接 concat: `Pd = stack([p_1, ..., p_K])` (无聚合冲突, 因为 num_clients × 1 domain/client)
- 下发给所有 client

**Pd training signal**: L_sty_proto 用 InfoNCE:
```
L_sty_proto = −log [ exp(cos(z_sty, Pd[d_i]) / τ) / Σ_d exp(cos(z_sty, Pd[d]) / τ) ]
```
其中 d_i 是样本 i 的 domain label (= client_id). 附加 **MSE anchor** (抗 late-round drift, 来自 FPL):
```
L_sty_MSE = ||z_sty − stopgrad(Pd[d_i])||²
```
最终: `L_sty_proto_total = L_sty_proto + 0.5 · L_sty_MSE`, 带 α-sparsity cosine^0.25 (FedPLVM-style) 弱化正例梯度.

**Why this is the main novelty**:
- **每 client 只有一个 domain** 意味着 Pd 的聚合是 **trivial concat 而非 FedAvg**. 这在 FL 里是个常被忽视的事实 — 所有 "domain-aware FL" 工作 (MP-FedCL, I2PFL, FedPLVM) 都把 domain 当 class 的 variance, 没有把 "one client = one domain" 的结构直接用到 prototype 设计
- **Pd 作为联邦一阶对象**: 跨 client 共享 Pd 意味着 client k 在本地训练时能"看到"其他 domain 的风格 prototype, 这是 FedSTAR (风格严格本地) 和 FDSE (风格被擦除) 都做不到的
- **和 class proto Pc 的几何互斥** (L_proto_excl = mean_(k,d) cos²(Pc[k], Pd[d])) 在低维 (最多 7×4 = 28 对) 上操作, 梯度冲突半径小, 不会像 feature-level HSIC (EXP-017) 那样全局污染

### Supporting Component — Asymmetric Statistic Encoder

- **Why statistics not deeper CNN**: AdaIN/MixStyle 证明 (μ, σ) 足够 express style. 深 CNN 多出来的空间判别力对 style 任务反而是累赘 (容易把 class 编进来)
- **Why 3-layer taps not 5**: conv1-3 是 "low-level texture + mid-level pattern", 对应 style 的主要承载层. conv4-5 的抽象语义不属于 style. 3 层是经验选择, 计划在 ablation 里扫描
- **Why LayerNorm not BatchNorm**: 统计向量 batch 维已经是对 mini-batch 聚合, 再 BN 会双重归一. LayerNorm 在 feature 维更稳
- **Why shared taps not separate forward**: encoder_sty 共享 encoder_sem 的 forward 以省计算 (~30% FLOPs saving vs 对称双 forward). **关键**: 我们只从 encoder_sem 抽 (μ, σ), **不反传 L_sty_proto 到 encoder_sem** (通过 `taps = [t.detach() for t in ...]`). 这样 encoder_sem 依然只被 L_CE + L_orth + L_sem_proto 训, 与 orth_only 的梯度路径完全一致
  - 这个 detach 细节是架构级 class-style separation 的关键: encoder_sty 的所有梯度到此为止, 不污染 encoder_sem

### Training Plan — Loss 组合与安全阀 Schedule

#### Loss 总表 (6 条, 其中 3 条继承, 3 条新增)

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{CE,aug}}}_{\text{continued from FedDSA-orig}} + \lambda_1 \mathcal{L}_{\text{orth}} + \lambda_2 \mathcal{L}_{\text{sem\_proto}} + \lambda_3 \mathcal{L}_{\text{sty\_proto}} + \lambda_4 \mathcal{L}_{\text{proto\_excl}}
$$

- **L_CE, L_CE,aug**: 继承 FedDSA 原方案, L_CE_aug 是 AdaIN-augmented z_sem 的 CE, 由 style bank (μ,σ) 跨 client dispatch
- **L_orth**: 继承, λ₁ = 1.0, 全程 on (EXP-080 已扫过, 1.0 最优)
- **L_sem_proto**: 新增, Bell curve schedule (warmup 0→50→0 at R150), MSE anchor coefficient 0.5, α-sparsity α=0.25. peak λ₂ = 0.5
- **L_sty_proto**: 新增, 与 L_sem_proto 同 schedule, peak λ₃ = 0.5
- **L_proto_excl**: 新增, 全程 on, λ₄ = 0.3 (proto 级小约束, 不需要 schedule 因为梯度半径极小)

#### 安全阀 Schedule (必须遵守, 对应 EXP-076/077 血泪经验)

| Round | λ_orth | λ_sem_proto | λ_sty_proto | λ_proto_excl | MSE coef | 说明 |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 0-49 | 1.0 | 0 | 0 | 0 | 0 | **Warmup**: 只开 orth_only + AdaIN aug, 等 encoder 稳定 (复刻 EXP-080 前 50 round) |
| 50-80 | 1.0 | 0→0.5 | 0→0.5 | 0→0.3 | 0.5 | **Ramp-up**: 线性升起, 原型 bank 初始化并开始更新 |
| 80-150 | 1.0 | 0.5 | 0.5 | 0.3 | 0.5 | **Peak phase**: 所有 loss 满状态 |
| 150-200 | 1.0 | 0.5→0 | 0.5→0 | 0.3 | 0.5 | **Ramp-down**: InfoNCE 退出, 让 CE 主导收尾 (避免 EXP-076 late-round 梯度冲突崩盘) |

**关键安全设计**:
- Bell schedule 复刻 EXP-076 bell_60_30 (R200 3-seed PACS mean last 79.29, 已验证最稳)
- MSE anchor 复刻 EXP-077 mode 4 (FPL-style, PACS R50 达 82.2%)
- α-sparsity 复刻 EXP-077 mode 5 (FedPLVM-style, 弱化正例梯度避免与 CE 打架)
- **L_proto_excl 全程 on** 因为它只作用在 proto 空间 [K, 128] 和 [num_clients, 128], K≤10, num_clients=4, 实际是 K × num_clients ≤ 40 对 cosine, 梯度极稀疏, 不需要 schedule

### FL 聚合协议

| 参数组 | 聚合策略 | 理由 |
|---|---|---|
| encoder_sem (conv + FC) | FedAvg | 继承 orth_only |
| semantic_head | FedAvg | 继承 orth_only |
| sem_classifier | FedAvg | 继承 orth_only |
| encoder_sty (MLP ~1M) | FedAvg | 轻量, 通信压力 ≈ 1.7%, 可 aggregate |
| BN running stats (sem 侧) | Local (FedBN) | 继承 FedBN 原则 |
| LayerNorm (sty 侧) | FedAvg | LN 无 running stats, 直接聚合 |
| **class prototype bank Pc** | **EMA** (类 FedProto): `Pc[k] ← 0.9·Pc[k] + 0.1·mean_{i:y_i=k}(z_sem_i)` over clients | 每 client 可能覆盖多 class, 用 EMA 避免 bank 噪声过大 |
| **domain prototype bank Pd** | **Concat** (per-round rebuild): `Pd = stack([mean(z_sty^(k))])` over clients | 每 client 一个 domain, 天然无歧义 |
| style bank (μ, σ) AdaIN | 继承 FedDSA 原方案, 跨 client pool | 用于 L_CE_aug |

**通信开销分析**:
- encoder_sty: ~1M params × 4 bytes = 4MB, per round per client 上下行 = 8MB. 相对 AlexNet 60M (240MB) 膨胀 +3.3%
- Pc: K × 128 × 4 bytes ≤ 10 × 128 × 4 = 5.1KB
- Pd: K_client × 128 × 4 bytes = 4 × 128 × 4 = 2KB
- **总膨胀 ≤ 4% of baseline** (vs 原 DualEncoder 方案 100% 膨胀)

### Failure Modes and Diagnostics

| Failure Mode | Detection | Fallback |
|---|---|---|
| **Office 瓶颈不在架构** (Diag A 触发) | Stage 0 Diag A 结果: trunk-freeze + 重训 head 后 Office accuracy 涨幅 < 0.5 | **Kill 整个 BiProto 方案**, 回到 Calibrator 兜底或者聚合侧改造 |
| **z_sty 坍缩** (norm 塌到 ~0) | z_sty_norm 轨迹每 10 round 监测, < 0.3 触发警告 | 增加 L_sty_proto 权重, 或加 L_sty_norm_reg (norm ≥ 1 的 ReLU) |
| **L_sem_proto / L_sty_proto 后期崩盘** (EXP-076 重演) | R100 后 AVG last 掉 > 2pp | Bell schedule R150 ramp-down 已内置; 若仍崩, kill InfoNCE 只保留 orth + proto_excl |
| **Pd prototype 被某 client dominant** (大 client memorize) | 监测 Pd pairwise cosine, 若 > 0.9 说明 domain prototype 没区分 | 增大 L_sty_proto, 或 per-client Pd 取 class-conditional mean (2D bank) |
| **L_proto_excl 与 L_CE 冲突** | R50 之后 L_CE 上涨 | λ_proto_excl 从 0.3 → 0.1 |

---

## Novelty and Elegance Argument

### Closest Works 对比 (4 维差异化矩阵)

| 方法 | 解耦? | Prototype? | Domain as shared federated object? | Feature-level vs Proto-level exclusion? | 参数膨胀 |
|---|:-:|:-:|:-:|:-:|:-:|
| FedProto (AAAI'22) | ❌ | ✅ class only | ❌ | — | 1× |
| FPL (CVPR'23) | ❌ | ✅ class multi-cluster | ❌ | — | 1× |
| FedPLVM (NeurIPS'24) | ❌ | ✅ class 2-level | ❌ | — | 1× |
| MP-FedCL (IoT'24) | ❌ | ✅ class k-means | ❌ | — | 1× |
| I2PFL (2025) | ❌ | ✅ class intra/inter-domain | ❌ (domain 是 class variance 源) | — | 1× |
| FedProto variants | ❌ | ✅ class | ❌ | — | 1× |
| FedSTAR (2024) | ✅ FiLM | ✅ content proto only | ❌ (style **本地私有**) | Feature-level (FiLM) | ~1.1× |
| FDSE (CVPR'25) | ✅ 层分解 | ❌ | ❌ (**擦除**) | Feature-level (DSE layer) | 0.65× |
| FedDP / FedSeProto (2024-25) | ✅ MI min | ❌ | ❌ (**擦除**) | Feature-level (IB/MI) | ~1× |
| FedFSL-CFRD (AAAI'25) | ✅ 双层重构 | ❌ | ❌ (共性/个性, 非 class/domain) | Feature-level (reconstruction) | ~1.2× |
| FedPall (ICCV'25) | ❌ (对抗 + 协作) | ✅ class only | ❌ | Feature-level (GAN-like) | ~1.1× |
| FISC/PARDON (ICDCS'25) | ❌ (混合空间) | ❌ (style statistics) | ⚠️ 共享但不解耦 | — | 1× |
| CDANN 变体 (EXP-108 复现) | ✅ GRL 对抗 | ❌ | ❌ | Feature-level (GRL) | ~1× |
| **FedDSA-BiProto (本文)** | ✅ **非对称统计 encoder** | ✅ **class + domain 双 bank** | ✅ **首次** (Pd concat 跨 client 共享) | **Proto-level (cos² on Pc, Pd)** | **~1.02×** |

### Claim 差异化句子

> 据我们所知, 这是**第一个**在联邦 cross-domain 学习中将 domain 作为与 class 对等的**一阶联邦共享原型** (Pd) 建模的工作. 相较 FedProto 系只管 class proto, 相较 FDSE/FedDP 把 domain 当噪声擦除, 相较 FedSTAR 把 style 锁在本地, BiProto 通过 "non-symmetric statistic encoder + prototype-level exclusion" 机制在不引入对抗训练、不依赖 feature-level MI 最小化 (EXP-017 HSIC / EXP-108 CDANN 已证这两条路在本 setup 下无效) 的前提下, 实现 **class / domain 并列的联邦对偶原型共享**.

### Why the mechanism stays small

- encoder_sty 只 +1.7% 参数, 共享 encoder_sem forward (+30% FLOPs saving vs 对称 trunk)
- 新增 3 条 loss 全部在 proto 空间 (低维, K ≤ 10, num_clients = 4), 梯度稀疏, 不会重演 HSIC 的 feature-level 全局污染
- 所有 loss 都有 EXP-076/077 验证过的安全阀 (Bell + MSE + α-sparsity)
- Pd 聚合策略利用 "每 client 一个 domain" 结构, 避免 FedAvg 原型漂移

---

## Claim-Driven Validation Sketch

**核心原则**: 方案胜负由 accuracy 判决 (C1), 但必须配 t-SNE / probe / metric 三套可视化 evidence 支撑方法叙事 (C2). C0 是启动前置 diagnostic (必须过 gate 才走 C1/C2).

### Claim 0 (PRE-REQUISITE): Office 瓶颈诊断 Gate

- **Hypothesis**: Office -1.49 是 shared trunk 被 CE 污染导致的 feature capacity 瓶颈
- **Minimal experiment (Diag A)**: 用 EXP-105 已有的 orth_only Office R200 seed=2 checkpoint, 冻结 encoder_sem, 只重训 semantic_head + sem_classifier 30 rounds, 观察 AVG 是否能从 88-89 涨到 > 91
- **Baseline**: Plan A orth_only raw checkpoint accuracy
- **Decision rule**:
  - 涨 ≥ +1.5pp → encoder capacity 有空间, BiProto 方案继续
  - 涨 +0.5 ~ +1.5pp → 部分相关, BiProto 方案继续但预期增益降档
  - 涨 < +0.5pp → **瓶颈不在 trunk, kill BiProto, 改投资聚合侧 (SAS τ tune) 或 outlier 侧 (Caltech-specific weighting)**
- **Estimated cost**: 2 GPU-h

### Claim 1 (DOMINANT): Accuracy 直接胜 FDSE (3-seed mean AVG Best)

- **Minimal experiment**: BiProto 完整方法 × 3-seed {2, 15, 333} × R200 × {PACS, Office}
- **Baselines (必跑)**:
  - FDSE 3-seed 本地复现 (已有, EXP-043/049/081)
  - orth_only 3-seed (已有, EXP-080)
  - **Ablation A**: BiProto - 非对称 encoder (用对称 AlexNet encoder_sty, 验证统计 encoder 的必要性)
  - **Ablation B**: BiProto - Pd (只保留 Pc, 验证 domain prototype 的贡献)
  - **Ablation C**: BiProto - L_proto_excl (验证 proto 级互斥的贡献)
- **Decisive metric**: 3-seed mean AVG Best
- **Expected directional outcome**:
  - PACS ≥ 81.0 (baseline 80.64 + 至少 0.36 涨幅), 严格 > 79.91
  - Office ≥ 91.1 (必须严格 > 90.58, 预期涨幅 +2.0~3.0 pp 来自 Diag A 假设成立)
  - Ablation A (对称 encoder): 预期持平或略降 (证明统计 encoder 是 "enough, not more")
  - Ablation B (无 Pd): 预期退回 FedProto 水平 (~ orth_only + 0.3)
  - Ablation C (无 proto_excl): 预期轻微退步 (Pd 与 Pc 可能有角度耦合)
- **Kill criteria**:
  - R50 任一 seed AVG < 80 (PACS) / < 86 (Office) → 立即 kill, 检查 Diag A 是否该触发
  - R150 AVG Last 崩盘 > 3pp → 立即 kill, 重演 EXP-076
- **Estimated cost**: 6 runs × ~2.5h (Office) + 6 runs × ~7h (PACS) + 3 ablations × 6 runs × 平均 ~4h = 大约 ~100 GPU-h; **必须 GPU 并行 (greedy scheduler, CLAUDE.md section 17.8 规范)** 以控制 wall time ≤ 3 天

### Claim 2 (SUPPORTING): 解耦可视化 Evidence Chain

构成三联画, 作为 C1 成功后在 paper 里 "mechanism 真起作用" 的支撑. C2 不是判决指标, 但**必须同时交付** (用户明确要求).

#### Vis-1: t-SNE 双面板 (核心可视化)
- **Input**: 3-seed × 2 dataset × 2 feature (z_sem, z_sty) × 每 seed 1000 test samples = 12 panels
- **Color coding**:
  - Panel A (z_sem): color by **class label** → 预期 class-wise 清晰聚类, domain 混合
  - Panel B (z_sty): color by **domain (client id)** → 预期 domain-wise 清晰聚类, class 混合
  - Panel C (z_sty by class) + Panel D (z_sem by domain): 预期"乱团", 证明 cross-disentanglement
- **Comparison**: 对比 orth_only / FDSE / CDANN / BiProto, 预期 BiProto 最清晰
- **Quantification**: 对每 t-SNE 投影算 silhouette score (by class for z_sem, by domain for z_sty), 建立定量比较

#### Vis-2: Per-Layer Probe Ladder
- **设置**: 训好的 BiProto checkpoint, 冻结全部参数
- **Probe 4 方向** × **3 linearity 级别** (linear, MLP-64, MLP-256) = 12 probe 值
  - (z_sem → class, z_sem → domain, z_sty → class, z_sty → domain)
- **预期方向矩阵** (hypothetical target):

| | Linear | MLP-64 | MLP-256 |
|---|:-:|:-:|:-:|
| z_sem → class | **high** (>0.85) | **high** (>0.90) | **high** (>0.92) |
| z_sem → domain | **low** (< 0.35) | **low** (< 0.45) | medium (< 0.55) |
| z_sty → class | **low** (< 0.30) | **low** (< 0.40) | medium (< 0.50) ← 关键进步 |
| z_sty → domain | **high** (>0.90) | **high** (>0.95) | **high** (>0.97) |

- **Baseline 对比**: orth_only (z_sty→class MLP-256 = 0.81, 对照 BiProto 目标 < 0.50), CDANN (z_sty→class linear = 0.96 反例), FDSE (预期 z_sty→domain 低)
- **这张表是本文 mechanism claim 的可视化核心**

#### Vis-3: Prototype Quality Matrix
- **Pc class 分离度**: pairwise cosine(Pc[i], Pc[j]) off-diagonal mean, 目标 < 0.3
- **Pd domain 分离度**: pairwise cosine(Pd[i], Pd[j]) off-diagonal mean, 目标 < 0.4
- **Pc ⊥ Pd matrix**: K × num_clients cosine matrix, 目标 mean |cos| < 0.2
- **Trajectory**: 每 10 round 快照, 绘制三个指标随 round 演化

#### Vis-4: Feature Health (防止 z_sty 坍缩)
- **z_sem_norm, z_sty_norm** trajectory (每 10 round)
- **effective rank** (via SVD on minibatch features)
- **mean pairwise sample similarity** (防止 feature collapse)

#### Vis-5: Orth Quality
- **cos(z_sem, z_sty)** trajectory, 目标收敛到 ~0
- **HSIC(z_sem, z_sty)** (计算但不作为 loss) 作为非线性依赖指标

**Expected evidence**: 综合 Vis-1~5, 如果 BiProto accuracy 胜但 Vis 表现退化 → 说明 mechanism 不是真正 responsible, claim 不成立. 反之, accuracy + Vis 双胜才叫"真 disentanglement".

---

## Experiment Handoff Inputs (for /experiment-plan)

- **Must-prove claims**: C0 (Diag A gate, must pass), C1 (accuracy win, dominant), C2 (visual evidence, supporting)
- **Must-run ablations (in order of importance)**:
  1. BiProto full vs orth_only (main comparison)
  2. BiProto vs BiProto - Pd (核心 Pd contribution)
  3. BiProto vs BiProto - 非对称 encoder (encoder asymmetry necessity)
  4. BiProto vs BiProto - L_proto_excl (proto-level exclusion value)
  5. τ sensitivity on L_sem_proto / L_sty_proto (0.05, 0.1, 0.2, 0.5)
  6. Warmup length sensitivity (30, 50, 80)
- **Critical datasets / metrics**: PACS + Office-Caltech10, AVG Best (主), AVG Last (稳定性), per-domain Best (outlier 域诊断)
- **Highest-risk assumptions**:
  - Diag A 假设 (Office 瓶颈在 trunk) — **必须先验证**
  - Pd 跨 client concat 假设 (每 client 严格一个 domain) — PACS/Office 都成立, DomainNet 需修正
  - 统计 encoder 能表达足够 style — 来自 AdaIN 文献的强先验, 但 FL 场景未直接验证

---

## Compute & Timeline Estimate

| Phase | 内容 | GPU-h | Wall |
|---|---|:-:|:-:|
| Stage 0 — Diag A | trunk-freeze 诊断 | 2 | 2h |
| Stage 1 — Implementation | 写 feddsa_biproto.py (~250 行) + unit tests + codex review | 0 (本地) | 0.5 day |
| Stage 2 — smoke | seed=2, R50, Office + PACS | 4 | 4h |
| Stage 3 — Main pilot | 3-seed × 2 dataset × R200 BiProto full | 30 | 1 day (6 并行) |
| Stage 4 — Ablations (×3) | Pd / 对称 encoder / proto_excl | 50 | 2 day (greedy) |
| Stage 5 — Visualization | t-SNE + probe + metrics pipeline | 2 | 0.5 day |
| **总计** | | **≤ 90 GPU-h** | **~4-5 days wall** |

---

## Appendix A: Intentionally Excluded (避免 scope creep)

- ❌ DomainNet 扩展 (留 future work, 与 CLAUDE.md 硬约束 "PACS + Office 为主战场"一致)
- ❌ ResNet-18 backbone (保持 AlexNet 公平对比)
- ❌ DP / privacy 分析 (venue 方向是 method, 不是 privacy)
- ❌ 可学习 temperature / learnable alpha (复杂度收益不成比例)
- ❌ Pc 的 FINCH 多聚类 (EXP-095 SCPR 已证伪)
- ❌ Adversarial domain classifier / GRL (EXP-108 已证伪)
- ❌ HSIC / MI 最小化 (EXP-017 已证伪)
- ❌ 可学习 λ (Kendall 不确定性加权, EXP-028 已证伪)
