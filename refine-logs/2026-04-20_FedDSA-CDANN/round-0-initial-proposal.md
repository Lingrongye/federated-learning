# Research Proposal: FedDSA-CDANN — Constrained Dual-Directional DANN for Federated Domain Generalization

## Problem Anchor

- **Bottom-line problem**: 联邦学习中跨域客户端(Feature-skew FL)的语义-风格解耦对不同数据集性质(风格是语义核心 vs. 纯 nuisance)的自适应处理. 当前方法 FedDSA-SGPA 的 Linear+whitening 广播在 Office(风格弱)上 +6.20pp, 在 PACS(风格强,油画/素描)上 -1.49pp.
- **Must-solve bottleneck**: 现有统计解耦约束(cos²=0 + HSIC)**没有告诉模型什么是"风格"什么是"语义"**, 导致模型可能把类别判别信号错误分到 z_sty, 被后续 whitening 磨掉. 诊断数据: PACS Linear+whitening 后 z_sty_norm 从 R10=3.12 塌到 R200=0.15(磨掉 95%); Office 仅磨 2%.
- **Non-goals**:
  - 不追求"一个机制万能解决所有数据集"(诚实承认数据集边界)
  - 不引入预训练大模型(VGG/CLIP)作风格 teacher, 保持 AlexNet from scratch
  - 不做 label noise / 少样本等正交问题
- **Constraints**:
  - 计算: seetacloud2 单 4090 24GB, R200 Office ~1h, PACS ~3h
  - 数据: Office-Caltech10(4 client 10 类) + PACS(4 client 7 类); 每 client 一个域, client id = domain id bijection
  - 代码基: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (940 行), 单测 41/41 已绿
  - 通信: 保持 FedBN 轻量化原则, 新增通信 ≤ 几 KB/round
  - Venue 目标: CVPR/ICCV 2026/2027 跨域联邦学习 track
- **Success condition**: 在 PACS R200 3-seed mean AVG Best ≥ Plan A 81.69 (从 Linear+whitening 80.20 反弹), 同时 Office 保持 ≥ 88.0 (不牺牲原 88.75). z_sty_norm R200 ≥ 1.5 (从 0.15 提升 10×). 且有 z_sty-only classification probe 实验证明 z_sty 在 PACS 上有 class 判别力.

---

## Technical Gap

**当前方法失败点**:
- FedDSA-SGPA 的解耦损失 `L_orth = cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)` 只要求两者**统计独立**, 没指定方向
- Whitening 广播 `Σ_inv_sqrt` 无差别归一化所有通道, 把跨 client 的 z_sty 差异全磨平
- 诊断指标 `z_sty_norm` 轨迹证明: PACS 上 whitening 磨掉 95% 风格信息, 但 PACS 风格(素描线条结构/油画纹理)本身携带类别判别信号 → 信息丢失 → -1.49pp

**Naive fixes 为什么不够**:
- 只关 whitening (Plan A): Office 损失 -6.20pp gain, 得不偿失
- 换 BN 策略: FedBN 已用, 边际收益饱和
- 加更多 InfoNCE: 已试过 EXP-076/085, 梯度冲突经常崩
- MixStyle / AdaIN: FISC/PARDON 等 7+ 篇已做, 同质且不解耦

**Smallest adequate intervention**: 给模型一个**监督信号**告诉它什么是风格什么是语义, 让解耦有方向性. 最自然的监督来自 **domain label**, 每 client 知道自己的 domain id (天然标签, 不泄露). z_sem 应与 domain 无关, z_sty 应与 domain 强相关. 这两个约束合起来能把风格信号"钉"在 z_sty, 避免误杀.

**Frontier-native alternative**: 用 VLM (CLIP) 文本 embedding 作语义 prior → 不符合 "AlexNet from scratch" 约束, 且 Fed-DIP 已占类似方向. 不采用.

**Core technical claim**: Constrained Dual-Directional DANN — z_sem 过 Gradient Reversal 到 dom_head (负梯度, 标准 DANN), z_sty 正向进 dom_head (正梯度, 强制保留 domain 信息). 这个**非对称双向**设计让解耦有了清晰的语义方向, 同时保留 z_sty 作为"域资产".

**Required evidence**:
- (C-main) PACS R200 3-seed AVG Best ≥ 82.2 且 Office ≥ 88.0 (即 CDANN 修好 PACS 不伤 Office)
- (C-probe) z_sty-only linear probe on PACS ≥ 50% accuracy (证明 z_sty 有 class 判别信号, 监督信号 align 假设成立)
- (C-ablation) 只保留反向 dom_head_sem (标准 DANN) vs 完整双向 CDANN 的对比, 看正向 dom_head_sty 的必要性

---

## Method Thesis

- **One-sentence thesis**: 用 **"非对称双向 DANN"** (反向 z_sem + 正向 z_sty) 把 domain 作为解耦监督信号, 让风格在不同数据集下都被正确归位, 不再被 whitening 误擦.
- **Why this is the smallest adequate intervention**: 只加 2 个小 MLP(各 ~2K params), 1 个 GRL 层(无参数), 改 3 行 loss 代码; 不改 encoder / 不改聚合 / 不增新 modality.
- **Why this route is timely**: FL + domain supervision 处于交集空白 (ADCOL 擦除派, FedPall 混合对抗不解耦, Deep Feature SCL 2025 非 FL). 2024-2026 30 篇综述中无直接 prior.

---

## Contribution Focus

- **Dominant contribution**: **Asymmetric dual-directional DANN for federated style-semantic disentanglement** — 首次把"z_sem 反向 + z_sty 正向"的双向 domain 监督引入 FL 解耦, 解决了统计约束下的"风格归位不确定"问题.
- **Optional supporting contribution**: **Dataset boundary diagnosis via z_sty classification probe** — 诊断方法论, 证明"风格是否携带类别信号"可用 z_sty-only probe 量化, 为"whitening 是否有益"提供 data-driven 判据.
- **Explicit non-contributions**:
  - 不 claim 新的聚合算法
  - 不 claim 新的 whitening 方案(继续用 pooled Σ_inv_sqrt 广播)
  - 不做 theoretical convergence proof(只做 empirical)
  - 不改 ETF classifier(EXP-097/098 已证伪其价值, 仍用 Linear)

---

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: AlexNet encoder, 双头 (sem_head/sty_head 各 1 层 Linear 128→128), Linear classifier (sem_classifier 128→K), pooled whitening broadcast, L_orth + HSIC 正交解耦, FedBN 本地 BN
- **New trainable components** (2 个, 符合 MAX_NEW_TRAINABLE_COMPONENTS=2):
  - `dom_head_sem`: 2 层 MLP 128→64→N_clients (预测域标签), 梯度从 z_sem 反向
  - `dom_head_sty`: 2 层 MLP 128→64→N_clients (预测域标签), 梯度从 z_sty 正向
  - 各 ~9K params (64×128 + 64×N), 合计新增 18K/client, 通信开销可忽略
- **Tempting additions intentionally not used**:
  - 不加 MI estimator (EXP-012 HSIC 已够, 避免不稳定)
  - 不加 class-conditional 扩展 (I5 CC-Style, 作为 supporting feature 但不写进主 claim)
  - 不加 selective whitening (I1 SASW, 保留作 future work)

### System Overview

```
                                                  ┌── sem_classifier(z_sem) ──→ L_CE(y)  [正向]
                                                  │
                            ┌── sem_head → z_sem ──┼── dom_head_sem(GRL(z_sem)) ──→ L_CE(d) [反向]
  x ── encoder ── feature ──┤                       │
                            └── sty_head → z_sty ──── dom_head_sty(z_sty) ──→ L_CE(d)      [正向]
                                      │
                                      └── cos²(z_sem,z_sty) + HSIC(z_sem,z_sty) ──→ L_dec  [正交解耦]

  z_sem, z_sty 还通过 pooled whitening 更新 source style bank (FedDSA-SGPA 保留)
```

### Core Mechanism

- **Input / output**: x (AlexNet 输入 [B,3,224,224]) → encoder feature [B, 1024] → (z_sem [B,128], z_sty [B,128])
- **Architecture or policy**:
  - `dom_head_sem = Linear(128,64) → ReLU → Dropout(0.1) → Linear(64, N_clients)`
  - `dom_head_sty`: 同结构
  - `GRL(x, λ)`: forward 恒等, backward 乘 -λ (Gradient Reversal Layer)
- **Training signal / loss**:

```python
L_task = CE(y, sem_classifier(z_sem))
L_dec  = λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)
L_adv_sem = CE(d, dom_head_sem(GRL(z_sem, λ_adv)))    # GRL 使 encoder 对此项反向更新
L_adv_sty = CE(d, dom_head_sty(z_sty))                 # 正向, dom_head 和 sty_head 都更新
L = L_task + L_dec + L_adv_sem + L_adv_sty
```

  - λ_adv schedule: `λ_adv(r) = min(1.0, r / warmup_adv)`, warmup_adv = 20 (DANN 标准做法)
  - λ_orth=1.0, λ_hsic=0.1 保持现有值
- **Why this is the main novelty**: 非对称双向 (z_sem 反向, z_sty 正向) + FL (domain label = client id, 无全局暴露) + 保留域信息不擦除. 文献综述无直接 prior.

### Domain head 聚合策略 (关键设计决策)

- **选择: 全局聚合** (dom_head_sem 和 dom_head_sty 都参与 FedAvg)
- **理由**:
  - 若本地保留: 每 client 的 data 全部来自一个 domain, 本地 dom_head 退化为 "输出 constant domain id" 的 degenerate solution, GRL 无意义
  - 全局聚合: 所有 client 共享 dom_head, 每 client 本地训练贡献梯度, 聚合后的 head 能区分 N_clients 个 domain, GRL 才真正有信号
  - 隐私: 聚合的是 head 参数, 不是 data; client id 作为 label 不敏感 (已在 FL header 公开)

### Optional Supporting Component — (故意不启用)

考虑过 class-conditional HSIC (I5, 要求 HSIC(z_sty, class)=0), 但为保持 dominant contribution 聚焦, 作 future work.

### Modern Primitive Usage

- **None by design**: 本方法不使用 LLM/VLM/Diffusion/RL 等大模型 primitive, 坚持 ResNet-18/AlexNet from scratch 传统 FL 范式. 理由:
  - 加 CLIP 语义 prior → 和 Fed-DIP (OpenReview 2025) 同质, 失去 novelty
  - 加 Diffusion 风格生成 → FedDAG (2501.13967) 已做, 稳定性差
  - 本方法 bottleneck 不是表达能力(AlexNet 够用), 是监督信号; LLM/VLM 不 address 监督信号问题
- **Frontier leverage 判断**: 跳过 = 诚实选择. 如果 reviewer 要求加, 需反驳: "加大模型会 inflation 并不解决 domain disentanglement 监督信号的根本问题".

### Integration into Base Pipeline

- 代码改动位置: `FDSE_CVPR25/algorithm/feddsa_sgpa.py`
- **新增**:
  - `class GradientReverseLayer(torch.autograd.Function)` (~15 行)
  - `FedDSASGPAModel` 构造函数加 `dom_head_sem` 和 `dom_head_sty` 两个 MLP (~10 行)
  - `Client.train` loss 新增 `L_adv_sem + L_adv_sty` 和 λ_adv schedule (~20 行)
- **聚合**: `dom_head_sem` / `dom_head_sty` 参与 FedAvg (在 aggregate_keys 白名单里)
- **关闭开关**: 加 `ue_cdann` algo_para 控制 on/off (默认 0, 为当前 baseline 兼容)

### Training Plan

- **Stagewise vs joint**:
  - **Joint training**, 不做 stagewise
  - **λ_adv warmup (R=0..20)**: λ_adv=0 → 等价于 Plan A + whitening baseline, 避免冷启动时 dom_heads 未训好导致 z_sem/z_sty 乱跑
  - **R=20..200**: λ_adv=1.0 全开, dom_heads 和 encoder 对抗
- **Data source**: 复用 flgo 框架 Office-Caltech10 / PACS_c4
- **Supervision signals**:
  - y: 原 class label
  - d: client id (0..N-1), 作 domain label
- **Loss weighting**:
  - λ_task = 1.0
  - λ_orth = 1.0 (同 Plan A)
  - λ_hsic = 0.1 (同)
  - λ_adv = [0 → 1.0] schedule
- **Pseudo-labeling / curriculum**: 无需要

### Failure Modes and Diagnostics

- **FM1: 对抗训练发散** (DANN 常见问题, loss 冲突导致 z_sem 不收敛)
  - 检测: R10 后 L_task 不降反升 / z_sem_norm 暴跌到 < 1
  - 缓解: λ_adv schedule 从 0 线性起步 (已设计); 若仍崩, 加梯度 clip (`torch.nn.utils.clip_grad_norm_(params, 10)`)
- **FM2: dom_head 欠拟合** (4 client 只有 4 domain, MLP 容易 overfit 到 trivial 分类器)
  - 检测: dom_head_sty 训练 acc > 0.95 (过拟合信号)
  - 缓解: dom_head 加 Dropout(0.1), 已设计
- **FM3: z_sty 过度携带信息** (λ_adv 太大导致 z_sty 把 class 信息也吸走)
  - 检测: z_sty-only classification probe acc 过高 (> 80% PACS)
  - 缓解: λ_adv 降到 0.5, 或加 HSIC(z_sty, y) = 0 约束
- **FM4: 数据集边界自适应** (Office 可能本身就不需要 CDANN, 开了反而 overkill)
  - 检测: Office CDANN 比 Linear+whitening 差
  - 缓解: 作为 ablation 记录, 不强改方法; claim 限定 "CDANN 特别有利于风格携带类信息的数据集"

### Novelty and Elegance Argument

**Closest work**: Deep Feature Disentanglement for Supervised Contrastive Learning (Cognitive Computation 2025)
- 他们: 用 class label 监督 disentangle z_content vs z_style, **单机 (non-FL)**, **对称双头** (两个头都正向监督)
- 我们: **FL 扩展** + **非对称双向** (反向 z_sem + 正向 z_sty) + **保留风格作资产**
- Delta: (a) FL 中 domain label = client id 的 novel mapping; (b) asymmetric directions; (c) 保留 z_sty 参与下游 Plan A 风格 bank

**Vs FedPall (ICCV 2025) Amplifier**:
- FedPall: 服务器 Amplifier 放大异构信息, 客户端 KL **抑制**. 混合 feature 空间. 擦除 domain.
- 我们: 解耦后**保留** z_sty 的 domain 信息. 空间分离. 不擦除.

**Vs FediOS (ML 2025) orthogonal subspace**:
- FediOS: generic/personalized 正交子空间. 没有 domain 监督. 纯几何.
- 我们: 有 domain 监督的双头. 几何 + 监督混合.

**为什么是 mechanism-level 而不是 module pile-up**:
- 核心新增就**一个 GRL + 两个 MLP** (各 64 dim 隐层), 不加 modality, 不加新聚合算法. 其他都复用.
- 新 trainable 参数 18K/client, 通信增量 < 100KB/round, 符合 FedBN 轻量化原则.

---

## Claim-Driven Validation Sketch

### Claim C-main (Primary): PACS CDANN ≥ Plan A 81.69, Office ≥ 88.0

- **Minimal experiment**: PACS + Office × 3 seeds × R200, 对比
  - Baseline A: Linear+whitening (EXP-102 Office 88.75 / EXP-098 PACS 80.20)
  - Baseline B: Plan A orth_only (EXP-083 Office 82.55 / EXP-080 PACS 81.69)
  - Ours: FedDSA-CDANN
- **Baselines / ablations**: 无需额外, 已有 baseline record
- **Metric**: 3-seed mean AVG Best, z_sty_norm R200
- **Expected evidence**:
  - PACS AVG Best 82-84 (回归 Plan A 基础, 可能略胜)
  - Office ≥ 88 (维持 baseline, 新监督不伤)
  - z_sty_norm R200 ≥ 1.5 (从 0.15 提升)

### Claim C-probe (Supporting): z_sty-only linear probe 证明 PACS z_sty 有类判别信号

- **Minimal experiment**: 在 CDANN 训练完的 checkpoint 上, 冻结 encoder+heads, 只训一个 Linear(128→K) 在 z_sty 上. 看 PACS / Office 的 probe accuracy.
- **Baselines**: Linear+whitening baseline 的 z_sty-only probe
- **Metric**: probe accuracy on holdout test set
- **Expected evidence**:
  - PACS CDANN z_sty probe > 40% (远超 random K=7 → 14%)
  - Office CDANN z_sty probe ~ 20-30% (仅比 random 稍高, Office 风格信号弱)
  - 对比: Baseline (Linear+whitening) z_sty probe 接近 random (因为 whitening 磨干净)
- **作用**: 直接支持 "CDANN 保留了 domain-discriminative z_sty" 的机制 claim

### Claim C-ablation (必做): 单向 DANN vs 双向 CDANN

- **Minimal experiment**: PACS R200 2 seeds
  - Variant 1: 只 dom_head_sem (反向, 标准 DANN) — 验证反向是否必要
  - Variant 2: 只 dom_head_sty (正向) — 验证正向单独够不够
  - Variant 3: 双向 CDANN — ours
- **Expected evidence**: 双向 > 各单向, Δ 至少 0.5pp, 证明非对称设计的必要性

---

## Experiment Handoff Inputs

- **Must-prove claims**: C-main (pilot 2h + full 12h), C-probe (probe 30min), C-ablation (4 runs 12h)
- **Must-run ablations**: 单向 DANN / 双向 CDANN / λ_adv 扫描 (0.1/0.5/1.0)
- **Critical datasets / metrics**: Office-Caltech10 AVG Best/Last, PACS_c4 AVG Best/Last, z_sty_norm R200, dom_head accuracy trajectory
- **Highest-risk assumptions**:
  - A1: dom_head 聚合不会泄露 domain 敏感信息 (低风险, client id 本就公开)
  - A2: 4 client / 4 domain bijection 假设在 PACS/Office 成立, 扩展到 DomainNet 需 cluster pseudo-label (future work)
  - A3: AlexNet from scratch 能学到足够解耦的 z_sem/z_sty (已在 EXP-097/100/102 验证过)

---

## Compute & Timeline Estimate

- **Estimated GPU·hours**:
  - Pilot: Office R100 1 seed (1h) + PACS R100 1 seed (1.5h) = 2.5h
  - Full: Office R200 3 seeds (3h) + PACS R200 3 seeds (9h) = 12h
  - Ablation: 单向 DANN × 2 variants × 2 seeds × R200 = 8h
  - Probe: 4 dataset×config × 30min = 2h
  - **Total: ~24 GPU·h single 4090**
- **Data / annotation cost**: 0 (复用现有 Office/PACS)
- **Timeline**:
  - Day 1: 代码实现 + 单测 (~3h 本地)
  - Day 1 评 Codex review + 修 (2h)
  - Day 2-3: Pilot (2.5h seetacloud2) → 决策
  - Day 3-5: Full + Ablation (~20h 并行跑 2-3 runs)
  - Day 5-6: Probe + NOTE + Obsidian 文档
  - Day 7: Codex re-review + refine 第二轮
