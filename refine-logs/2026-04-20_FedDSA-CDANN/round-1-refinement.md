# Round 1 Refinement — FedDSA-CDANN v1

## Problem Anchor (verbatim from round 0)

- **Bottom-line problem**: 联邦学习中跨域客户端(Feature-skew FL)的语义-风格解耦对不同数据集性质(风格是语义核心 vs. 纯 nuisance)的自适应处理. 当前方法 FedDSA-SGPA 的 Linear+whitening 广播在 Office(风格弱)上 +6.20pp, 在 PACS(风格强,油画/素描)上 -1.49pp.
- **Must-solve bottleneck**: 现有统计解耦约束(cos²=0 + HSIC)**没有告诉模型什么是"风格"什么是"语义"**, 导致模型可能把类别判别信号错误分到 z_sty, 被后续 whitening 磨掉. 诊断数据: PACS Linear+whitening 后 z_sty_norm 从 R10=3.12 塌到 R200=0.15(磨掉 95%); Office 仅磨 2%.
- **Non-goals**: 不追求"一个机制万能解决所有数据集"(诚实承认数据集边界); 不引入预训练大模型作风格 teacher; 不做 label noise / 少样本.
- **Constraints**: seetacloud2 单 4090; Office-Caltech10 + PACS_c4 (client=domain bijection); AlexNet baseline; Venue CVPR/ICCV 2026/2027.
- **Success condition**: PACS 3-seed AVG Best ≥ 82.2, Office ≥ 88.0; z_sty_norm R200 ≥ 1.5; z_sty-only probe 证 PACS 有类信号.

## Anchor Check

- **原 bottleneck**: 统计解耦缺方向性, whitening 误擦 PACS 风格里的类信号.
- **V1 方法是否仍解决**: **仍解决**. Core mechanism 不变 (domain supervision + GRL + asymmetry); 只是 head 结构简化和 claim 范围窄化.
- **Reviewer suggestions rejected as drift**: 无. 所有 reviewer 反馈都在强化 focus, 未偏离 anchor.

## Simplicity Check

- **V1 Dominant contribution**: `domain-supervised asymmetric disentanglement prevents whitening-induced style collapse` (收敛为单一 mechanism claim).
- **Components removed or merged**:
  - **合并 dom_heads** `dom_head_sem + dom_head_sty` → **一个共享 dom_head**, 不对称只在 GRL 梯度方向 (见 Changes #1)
  - **删** "Dataset Boundary Diagnosis" 作 supporting contribution → 降级为 supporting analysis, 不写进 main thesis (Changes #2)
  - **删** fallback "HSIC(z_sty, y)=0" 约束 (reviewer 指出它和 anchor 自相矛盾) (Changes #3)
  - **删** "style as asset" 叙事 (这是 FedDSA-SGPA 原 claim, 不是本 paper 的 contribution)
- **Reviewer suggestions rejected as unnecessary complexity**:
  - 拒绝 "加 shared vs split dom head 的 ablation" — 改为直接用 shared (已是 R1 建议, 无需 ablation)
  - 拒绝 "加 LLM/RL primitive" — reviewer 自己也说 "NONE otherwise", 无需
- **为什么 V1 仍是最小充分**:
  - 新增可训练仅 1 个 MLP (~9K params, 原来是 2 个共 18K) + GRL (无参)
  - 不改 encoder / 聚合 / whitening 机制
  - 1 个 dominant claim + 至多 1 个 supporting (portability check)

## Changes Made

### 1. 合并 dom_heads 为单一共享 head (R1 CRITICAL)

- **Reviewer said**: "Strongly consider replacing `dom_head_sem` + `dom_head_sty` with one shared `dom_head`, applying it to `GRL(z_sem)` and `z_sty`; that makes the asymmetry the contribution, not the existence of two extra modules."
- **Action**:
  ```python
  # V0: 两个独立 dom_head
  dom_head_sem = MLP(128, 64, N_clients)
  dom_head_sty = MLP(128, 64, N_clients)
  L_adv_sem = CE(d, dom_head_sem(GRL(z_sem)))
  L_adv_sty = CE(d, dom_head_sty(z_sty))

  # V1: 一个共享 dom_head, 对称性只在 GRL 方向
  dom_head = MLP(128, 64, N_clients)          # 唯一新 trainable 组件
  L_adv_sem = CE(d, dom_head(GRL(z_sem, λ)))  # 反向
  L_adv_sty = CE(d, dom_head(z_sty))          # 正向
  ```
- **Reasoning**: 两个独立 head 会给 reviewer "加模块" 的印象; 共享 head 让方法的 asymmetry 明确锁在"梯度方向"这一件事, 更干净.
- **Impact**: 可训练参数减半 (18K→9K); 梯度方向对抗更直接 (两路梯度在同一 head 参数上竞争, 更纯粹的 DANN 对抗).

### 2. 降级 "Dataset Boundary Diagnosis" 为 supporting analysis (R1 CRITICAL)

- **Reviewer said**: "Specific weakness: the paper currently overpackages the idea. 'Dataset boundary diagnosis' makes it feel less focused and more combinational."
- **Action**:
  - 从 "Optional supporting contribution" 段删除
  - 将 z_sty-only probe 从 claim 降级为**诊断证据**
  - 加**domain confusion readout** 作为 mechanism claim 的直接验证 (reviewer 建议 "domain probe or domain-confusion readout is more directly aligned")
- **Reasoning**: 一个 paper 一个 dominant claim; 诊断方法论单独讲太弱, 作支持证据够了.
- **Impact**: Contribution quality 收敛, reviewer 的 "combinational" 指控消除.

### 3. 窄化 claim 到 `client=domain FedDG where style carries class signal` (R1 CRITICAL)

- **Reviewer said**: "Narrow the claim. Say explicitly that the method is for `client=domain` federated DG where domain-discriminative style may also carry class signal. Remove generalized language about adaptive handling across all datasets."
- **Action**:
  - 删 "自适应处理所有数据集" 表述
  - 明确定义 applicability: "FedDG with (i) client ≈ domain bijection, (ii) evidence that style is semantic rather than nuisance (measured by z_sty probe accuracy ≫ random)"
  - 扩展到多域每 client (如 DomainNet) 明确列为 future work
- **Reasoning**: Honest scope 比 overclaim 审稿更有利; reviewer "novelty ceiling modest" 的担忧来自过宽 claim.
- **Impact**: Venue Readiness 提升, 没有 overclaim 风险.

### 4. 加 DINOv2 frozen backbone portability 实验 (R1 IMPORTANT)

- **Reviewer said**: "Keep CDANN unchanged, but add one portability check with a frozen stronger visual encoder such as DINOv2 or CLIP-visual-only, training only the heads."
- **Action**: 加 Claim C-port: 把 encoder 从 AlexNet 换成 frozen DINOv2-S/14 提取的 features (patch-avg pooled 到 384d, 接 projection 到 128d), 其他组件不变. 跑 PACS R100 1 seed.
- **Reasoning**: 证明 CDANN mechanism 不依赖弱 encoder; 同时提供 "方法 portable 到 modern pretrain" 的 credibility.
- **Impact**: Frontier Leverage 从 6→8, 破除 "dated 2026" 印象; 但需额外 ~2h GPU.

### 5. 冻结几个 R1 提到的"minor but worth freezing now"

- **Reviewer said**: "whether the domain losses are on pre- or post-whitening features, whether inference uses only `z_sem` for prediction, whether new heads share optimizer/LR"
- **Action** (都写进 method section):
  - domain losses **on post-whitening** (whitening 后的 z_sem/z_sty, 和 sem_classifier 同 feature space)
  - inference **只用 z_sem** → sem_classifier; z_sty **只**用于下游 style bank 广播 (保持 Plan A 原设计)
  - dom_head 用**相同 optimizer/LR** (SGD lr=0.05 同 encoder), 不分组
- **Impact**: Method Specificity 从 8→9.

### 6. 删 fallback "HSIC(z_sty, y)=0" (R1 Simplification)

- **Reviewer said**: "Remove the proposed fallback `HSIC(z_sty, y)=0`; it directly conflicts with your anchor that style can be class-relevant on PACS."
- **Action**: 删 FM3 的 fallback. FM3 仅保留检测, 不加 mitigation (若 z_sty probe 过高, 调 λ_adv 而非加 HSIC).
- **Reasoning**: 正确. anchor 明说 PACS 风格携带类信号, 然后又 HSIC(z_sty, y)=0 矛盾.
- **Impact**: 概念一致性提升.

---

## Revised Proposal (V1, full anchored)

# Research Proposal: FedDSA-CDANN — Domain-Supervised Asymmetric Disentanglement for Federated Domain Generalization

## Problem Anchor (same as R0, verbatim above)

## Technical Gap

**当前方法失败点**:
- FedDSA-SGPA 的 `L_orth = cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)` 只要求统计独立, 没方向
- Whitening 广播 `Σ_inv_sqrt` 无差别归一化, 磨平跨 client z_sty 差异
- 诊断: PACS whitening 后 z_sty_norm R10→R200 塌 95% → 当风格携带类信号时, 信息丢失

**Smallest adequate intervention**: 给解耦加**方向性监督**. Domain label (= client id in `client=domain` FedDG) 是天然可用的**零成本标签**: z_sem 应与 domain 无关, z_sty 应与 domain 强相关. 一个共享 `dom_head` + GRL 即可实现"反向 + 正向"双路对抗.

**Core technical claim**: Domain-supervised asymmetric disentanglement (z_sem 反向 + z_sty 正向 on **shared** dom_head) 防止 whitening 擦掉承载类信号的风格, 在 `client=domain where style carries class signal` 的 FedDG 场景有效.

## Method Thesis

- **One-sentence thesis**: 用**一个共享 domain head** 上的**不对称梯度方向** (z_sem 走 GRL 反向, z_sty 走正向) 为联邦解耦加监督, 使 whitening 不再误擦含类信号的风格.
- **Why smallest adequate**: 新增 1 个 MLP (9K params) + 1 个 GRL (无参) + 3 行 loss; 不改 encoder/聚合/whitening.
- **Why timely**: FL + constrained DANN + 保留域信息的三重交集在 2024-2026 综述 30 篇中空白.

## Contribution Focus

- **Dominant contribution** (one): **Domain-supervised asymmetric disentanglement prevents whitening-induced style collapse in client=domain FedDG**. 具体: shared dom_head + GRL-driven asymmetry + 在 PACS (风格是类信号) 上恢复精度.
- **Optional supporting contribution**: **Portability**: 机制在 frozen DINOv2 backbone 下依然有效 (C-port 实验).
- **Explicit non-contributions**:
  - 不 claim 新聚合算法
  - 不 claim 新 whitening 方案
  - 不 claim "dataset boundary diagnosis" (降级为 supporting analysis)
  - 不 claim 改 ETF classifier (EXP-097/098 已证伪, 仍用 Linear)
  - 不做 theoretical convergence proof

## Proposed Method

### Complexity Budget

- **Frozen / reused**: AlexNet encoder, 双头 sem_head/sty_head, sem_classifier, pooled whitening, L_orth + HSIC, FedBN
- **New trainable** (1 component, 符合 MAX_NEW_TRAINABLE_COMPONENTS ≤ 2):
  - `dom_head`: MLP 128→64→N_clients, 约 9K params, 被 sem/sty 两路共享
- **GRL** (no params): `forward(x) = x`, `backward(grad) = -λ · grad`
- **Tempting additions intentionally excluded**:
  - 不加 MI estimator (HSIC 已够)
  - 不加 CC-Style HSIC 扩展 (I5, 留作 future work)
  - 不加 selective whitening (I1 SASW, 留作 future work)
  - 不加 VGG/CLIP 风格 teacher

### System Overview

```
                                                     ┌──── sem_classifier(z_sem) ──→ L_CE(y)          [正向]
  x ─ encoder ─ feature ─┬─ sem_head → z_sem ─ (wh) ─┤
                         │                            └──── dom_head(GRL(z_sem, λ)) ──→ L_CE(d)       [反向]
                         └─ sty_head → z_sty ─ (wh) ──────── dom_head(z_sty) ──→ L_CE(d)              [正向]

  z_sem ⊥ z_sty:  λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)                          [正交解耦]

  Domain losses 都在 post-whitening features 上计算 (同 sem_classifier 的 feature space).
  Inference: 只走 z_sem → sem_classifier 预测 y. z_sty 只用于下游 pooled style bank 广播 (保持 Plan A).
```

### Core Mechanism

- **Input / output**: x ∈ [B, 3, 224, 224] → feature [B, 1024] → (z_sem, z_sty) ∈ [B, 128]² (post-whitening)
- **Architecture**:
  - `dom_head = Linear(128,64) → ReLU → Dropout(0.1) → Linear(64, N_clients)`
  - `GRL(x, λ)`: forward 恒等, backward 乘 -λ
- **Training signal / loss**:

```python
L_task    = CE(y, sem_classifier(z_sem))
L_dec     = λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)
L_adv_sem = CE(d, dom_head(GRL(z_sem, λ_adv)))    # GRL 反向梯度
L_adv_sty = CE(d, dom_head(z_sty))                 # 正向梯度
L         = L_task + L_dec + L_adv_sem + L_adv_sty
```

- **λ_adv schedule**: `λ_adv(r) = min(1.0, max(0, (r - 20) / 20))`, R=0..20 完全关闭 (baseline 暖启), R=20..40 线性起步, R≥40 全开
- **Why main novelty**: 共享 dom_head 被正反两路梯度对抗 → 最纯粹的"非对称监督"机制; FL 实现零通信成本 (dom_head 同 FedAvg 聚合)

### Aggregation Strategy

- `dom_head` 通过 **FedAvg 聚合** (和 encoder/sem_classifier 一致)
- 理由: 本地 data 单域, 本地 dom_head 会 degenerate. 聚合后跨 client 共享能区分 N domain 的 head, GRL 才有信号
- 隐私: 聚合 head 参数 (不是 data), domain label = client id 本就公开

### Optional Supporting Component — Portability Check

不引入新 trainable, 只把 encoder 从 AlexNet 换成 **frozen DINOv2-S/14** (ViT-Small, 21M params, 冻结), 输出 patch-avg pool 到 384d, 接 **新** Linear(384, 128) 做 projection (当作"new sem_head base", **不视为新 component**, 因为它只是替代 AlexNet 最后几层的抽取), sty_head 从 same projection 分叉. 其他组件不变.

此 check **只在 PACS 跑 R100 1 seed** (~2h), 看 AVG Best 是否 ≥ CDANN AlexNet 版本. 预期 gap < 2pp (说明机制 portable).

### Modern Primitive Usage

- **Core 方法不用 LLM/Diffusion/RL primitive**: bottleneck 是监督信号, 不是表达能力
- **Supporting portability check** 用 frozen DINOv2: 提供 credibility, 不改 mechanism
- **Stance**: 定位为 "focused low-capacity mechanism paper with portability check", 不假装 frontier

### Integration

- 代码位置: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (本方案 new config `feddsa_cdann_*.yml`)
- 新增: `GradientReverseLayer` (15 行) + `dom_head` MLP (10 行) + L_adv 计算 (20 行)
- 聚合: `dom_head` 参与 FedAvg
- 开关: algo_para 加 `ca` (cdann active, 0/1), 兼容 baseline

### Training Plan

- **Joint training**, λ_adv schedule 三段 (0→linear→1.0)
- 其他同 EXP-102 config: LR=0.05, E=1 (Office) / E=5 (PACS), R=200
- Loss weights: λ_orth=1.0, λ_hsic=0.1, λ_adv 按 schedule

### Failure Modes and Diagnostics

- **FM1: 对抗训练发散** — 检测 L_task 不降 / z_sem_norm 暴跌; 缓解 λ_adv schedule + grad clip=10
- **FM2: dom_head 欠拟合** — 检测 dom_head acc < 30% at R50; 缓解 Dropout 已加
- **FM3: z_sty 过度携带信息** (λ_adv 太大) — 检测 z_sty-only probe > 80%; 缓解**只降 λ_adv 到 0.5** (不加 HSIC(z_sty,y) 约束, 见 Change #6)
- **FM4: Office 不需要 CDANN** — 检测 Office CDANN < Linear+whitening; mitigate: 这是预期 scope, 报告为 "method targets PACS-like settings; Office 保持 parity not primary goal"

### Novelty and Elegance Argument

- **Closest**: Deep Feature Disentanglement for Supervised Contrastive (Cog. Comp. 2025), non-FL, symmetric dual-head (我们: FL + **asymmetric** + **shared head**)
- **Vs FedPall (ICCV 2025)**: 他们混合空间对抗擦除, 我们解耦后保留
- **Vs ADCOL / Federated Adversarial DA**: 擦除派, 我们保留 z_sty
- **Mechanism-level 而非 module stacking**: 唯一新 trainable = 1 个 MLP; asymmetry 封装在 GRL 方向选择; 其他复用

---

## Claim-Driven Validation Sketch

### Claim C-main (Primary): PACS CDANN 从 -1.49 回到 ≥ 0 (vs Plan A), Office 不掉 >0.75

- **Minimal experiment**: PACS R200 3 seeds + Office R200 3 seeds, vs Linear+whitening baseline
- **Metric**: AVG Best, AVG Last, **domain confusion readout** (dom_head 对 z_sem 的 accuracy 应接近 1/N_clients, 对 z_sty 应接近 1.0), z_sty_norm R200
- **Expected**:
  - PACS AVG Best 82-84 (≥ Plan A 81.69)
  - Office AVG Best ≥ 88.0
  - z_sem domain acc → ~25% (GRL 有效)
  - z_sty domain acc → >95% (监督起效)

### Claim C-port (Supporting): frozen DINOv2 下 CDANN 仍有效

- **Minimal experiment**: PACS R100 1 seed, encoder 换 DINOv2-S/14 frozen, CDANN on
- **Metric**: AVG Best ≥ 85 (希望 DINOv2 + CDANN 比 AlexNet + CDANN 更好)
- **Expected**: portable 到 modern backbone, 证明 mechanism 不依赖弱 encoder

### Claim C-ablate (必做): domain supervision 单独必要性

- **Minimal experiment**: PACS R200 2 seeds × 3 variants:
  - (V1) baseline: Linear+whitening (无 CDANN)
  - (V2) 只反向: L_adv_sem only (标准 DANN)
  - (V3) 只正向: L_adv_sty only
  - (V4) 双向: CDANN (ours)
- **Expected**: V4 > V2 ≥ V1, V3 效果不确定; **关键是 V4 vs V2 显著, 证明 z_sty 正向监督的必要性**

## Experiment Handoff Inputs

- **Must-prove**: C-main (3 seeds × 2 datasets × R200 = 12h), C-port (2h), C-ablate (4 variants × 2 seeds × R200 = 24h)
- **Critical metrics**: AVG Best, domain confusion readout (z_sem vs z_sty domain acc), z_sty_norm R200, z_sty-only probe acc
- **Highest-risk assumptions**:
  - A1: `client=domain` bijection (PACS/Office 成立)
  - A2: Shared dom_head 不会让 z_sty 正向梯度和 z_sem 反向梯度互相抵消 (待 pilot 验证)
  - A3: λ_adv schedule 稳定 (DANN 经典问题, 有成熟 mitigation)

## Compute & Timeline

- **Pilot** (Office + PACS R100 × 1 seed): 2.5h
- **Full C-main**: 12h
- **C-ablate**: 24h
- **C-port (DINOv2)**: 2h
- **Probes + NOTE**: 2h
- **Total**: ~42 GPU·h single 4090
- **Timeline**: Day 1 code+test, Day 2 pilot, Day 3-5 full runs, Day 6 probes+docs, Day 7 Codex re-review
