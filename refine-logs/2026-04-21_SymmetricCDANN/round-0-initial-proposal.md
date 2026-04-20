# Research Proposal: Symmetric CDANN — Dual-Adversarial Disentanglement for FedDG

## Problem Anchor (based on EXP-108 empirical falsification, 2026-04-21)

- **Bottom-line problem**: 我们跑完 EXP-108 FedDSA-CDANN 的 Office+PACS 3-seed 实验和 frozen post-hoc probes 后, 发现**原 CDANN proposal 的 anchor claim 有严重漏洞**:
  - **Office baseline Linear+whitening (EXP-105)** probe_sty_class = **95.8%**
  - **Office CDANN** probe_sty_class = **96.2%**
  - **差距仅 0.4pp** — CDANN 几乎没有额外贡献
  - 意味着 "CDANN 保住 z_sty 的 class-relevant 信号" 这个 claim 不成立. 因为 baseline 本来就"保留"着 (z_sty 虽然 norm 塌到 0.146, 但方向仍线性可分 class). CDANN 的 probe 0.96 是结构副作用, 不是机制贡献.
- **Must-solve bottleneck**: **z_sem 和 z_sty 从来没真的解耦**. 现有约束 `cos²(z_sem, z_sty) = 0 + HSIC(z_sem, z_sty) = 0` 只要求统计独立, 但两个头都从同一个 encoder linear project 出来, **两者都保留了线性可分的类信号**. probe_sem_class 和 probe_sty_class 都 ≈ 96% 就是证据. 解耦的核心目标 "让 z_sem 装类信号, z_sty 装风格信号, 二者分工" 没实现.
- **Historical failure (7 次)**: 2026-04-08 到 2026-04-19 我们试过 7 次风格共享 (EXP-059/067/076/078/085/086/095), 全部在 PACS 上失败 (-0.71 到 -2.54pp). 统一失败模式: z_sty 不纯, 共享 z_sty = class 信号一起共享 = 破坏 z_sem.
- **Non-goals**:
  - 不换 backbone (保持 AlexNet from scratch)
  - 不加大模型 (VGG/CLIP teacher 已排除)
  - 不追求 headline SOTA (诚实报告为主)
- **Constraints**:
  - seetacloud2 单 4090 24GB (共享)
  - Office-Caltech10 (4 client, 10 类) + PACS_c4 (4 client, 7 类), 均 client=domain bijection
  - R200, E=1 (Office) / E=5 (PACS)
  - 保持现有 CDANN 代码基础 (feddsa_sgpa.py + ca=1 flag), 最小增量改动
- **Success condition**:
  - (C-disent) **PACS probe_sty_class 从 baseline 预期 80%+ 掉到 ≤ 25%** (接近 random K=7=14.3%). 这是唯一能证明"真解耦"的指标
  - (C-preserve) 同时 probe_sem_class **保持 80%+**, 即 encoder 把 class 信号**集中**到 z_sem
  - (C-parity) PACS AVG Best **≥ 80.0** (不低于当前 CDANN 80.08), 最好 **≥ 82.0** (超 Plan A 81.69)

## Technical Gap

**Current state of the art in our pipeline**:
- FedDSA-SGPA 双头解耦仅用 cos⊥ + HSIC → 无方向性, 无法强制"谁装什么"
- CDANN 加了 z_sem → domain 反向 + z_sty → domain 正向 → **只约束 domain 信号分工**, 没约束 class 信号分工
- 结果: z_sem 装满 class (正向 CE 逼的) + z_sty 也装满 class (cos⊥ 只换方向不分工) + 双方都染上 domain (FedBN + 双路 dom_head 训练)

**Why naive fixes are insufficient**:
- 加大 λ_orth: 会破坏分类精度 (EXP-017 已试, ±0pp)
- 换更严格的 HSIC variant: 计算量大, 且 EXP-012 证 HSIC=0 效果有限
- 用 MI estimator (CLUB/InfoNCE): 小 batch FL 不稳 (EXP-078 经验)
- 冻结 encoder 只训头: 无法利用监督信号, 与 FL 冲突

**Smallest adequate intervention**: 对 z_sty 加**反向 class 梯度** (即 "z_sty 不能预测 class"). 这是 CDANN 设计的**对称扩展**. 类似 GRL 原理, z_sty 路径通过一个 GRL 接 cls_head_on_sty, 训练 cls_head 正向预测 class (让它聪明), 但 encoder 端收到反向梯度 (让 z_sty 变 class-blind).

**Frontier-native alternative considered**:
- Flow-matching 可逆解耦 (SCFlow ICCV 2025): 太重, 不适合 FL
- Information Bottleneck on z_sty: 需要 VAE-like 机制 (EXP-041 已试失败)
- Contrastive style learning: 需要正负样本定义, 小 batch FL 不友好

**Core technical claim**: **Dual-adversarial disentanglement via gradient reversal on both semantic→domain and style→class paths is the minimal repair for bi-directional information leakage in statistically-constrained decoupling.**

**Required evidence**:
- Frozen post-hoc probe_sty_class < 25% (class 信息真被清出 z_sty)
- probe_sem_class > 80% (z_sem 承担全部 class)
- 不引入训练崩溃 (loss_task 保持 < 0.01, z_sem_norm 不塌)

## Method Thesis

- **One-sentence thesis**: **对称 GRL 对抗 — 在 CDANN 基础上同时反向 z_sem→domain (已有) 和 z_sty→class (新) — 强制 z_sem 专学类、z_sty 专学域, 是在 FedBN + cos⊥ + HSIC 不足以解耦时的最小必要补丁**.
- **Why this is the smallest adequate intervention**: 复用 GRL (无参), 复用已有 CDANN 架构 (单个 cls_head_on_sty MLP ~3K 参数), 1 行 loss, 1 个 flag `ce` (class-exclude).
- **Why this route is timely**: dual-adversarial disentanglement 在 single-machine 已被反复验证 (DANN 2015, Ganin 2016, Zhao 2019 "Invariant Representations", HEX 2019). **FL 下 + 对称双向**是空白.

## Contribution Focus

- **Dominant contribution**: **对称 dual-adversarial 对 FedDG 解耦的 empirical validation** — 第一次用 **frozen post-hoc probes** 作为 pre-registered 成功标准, 定量测 z_sem/z_sty 的真分工. 不 claim 新方法 family, claim "我们第一个**诚实量化**了 FL 解耦实际效果并通过 symmetric GRL 纠正它".
- **Optional supporting contribution**: 若对称 CDANN 证实**真解耦**, 则 **风格共享 (AdaIN injection)** 可重启, 前 7 次失败的根因 (z_sty 不纯) 被修复. 可能激活历史失败的 EXP-059/067 路线.
- **Explicit non-contributions**:
  - 不 claim SOTA on PACS/Office
  - 不 claim 新 FL 聚合算法
  - 不 claim 任何 whitening 相关机制 (EXP-107 证伪 whitening-induced collapse)
  - 不 claim 理论 disentanglement guarantee (empirical only)

## Proposed Method

### Complexity Budget

- **Frozen / reused**: AlexNet encoder, 双头 sem/sty, sem_classifier, pooled whitening (保留 for Office gain), FedBN, L_orth + HSIC, existing CDANN (dom_head + GRL on z_sem)
- **New trainable**: **1** — `cls_head_on_sty`: MLP 128→64→K (K=7 PACS / 10 Office, ~9K params for PACS)
- **GRL**: 无参, 复用 existing `GradientReverseLayer` class
- **Frozen probes** (evaluation, not training): 已有
- **Excluded tempting additions**: 不加 HSIC(z_sty, y), 不加 MI estimator, 不换 encoder, 不加新 config flag 链

### System Overview

```
                                                     ┌── sem_classifier(z_sem) ── L_task = CE(y)       [encoder forward, class 主任务]
  x ─ encoder ─ feat ──┬── sem_head → z_sem ────────┤
                       │                              ├── dom_head(GRL(z_sem, λ_d)) ── L_dom_sem        [CDANN: encoder via GRL 反向, domain-blind z_sem]
                       │                              └── (inference only) no other path
                       │
                       └── sty_head → z_sty ─────────┬── dom_head(z_sty) ── L_dom_sty                   [CDANN: dom_head 正向 CE, encoder 正向]
                                                     └── cls_head_on_sty(GRL(z_sty, λ_c)) ── L_cls_sty  [NEW: encoder via GRL 反向, class-blind z_sty]

  Decoupling: cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)  [保留]
```

### Core Mechanism

- **Existing (CDANN)**: 
  - `L_dom_sem = CE(d, dom_head(GRL(z_sem, λ_d)))` — encoder 被推成产 domain-indistinguishable z_sem
  - `L_dom_sty = CE(d, dom_head(z_sty))` — encoder 被推成产 domain-discriminative z_sty
- **New (对称)**:
  - `L_cls_sty = CE(y, cls_head_on_sty(GRL(z_sty, λ_c)))` — encoder 被推成产 class-indistinguishable z_sty
  - `cls_head_on_sty` 自身通过正向梯度更新, 学成 "从 z_sty 猜 class" 的 discriminator
  - encoder 端接收反向梯度, 被推成让 z_sty 不可分 class
- **Total loss**:
  ```
  L = L_task + λ_orth · cos² + λ_hsic · HSIC + L_dom_sem + L_dom_sty + L_cls_sty
  ```
- **Schedule**:
  - λ_d warmup: R0-20 off, R20-40 ramp, R40+ = 1.0 (same as CDANN)
  - **λ_c warmup: R0-40 off, R40-60 ramp, R60+ = 1.0** (比 λ_d 晚, 让 encoder 先学好 class, 再约束 z_sty 不能偷 class)

### Aggregation

- `cls_head_on_sty`: FedAvg 聚合 (同 dom_head 策略)
- 其他聚合策略保持不变

### Integration

- `FDSE_CVPR25/algorithm/feddsa_sgpa.py`:
  - `FedDSASGPAModel`: ca=1 时增加 `cls_head_on_sty`
  - `Client.train`: 加 L_cls_sty 计算 + λ_c schedule + warmup gate
  - algo_para 加第 14 位 `ce` (class-exclude, 0/1 default 0)
- 预估改动 ~50 行, 单测加 5-6 个 (对称测试)

### Training Plan

- Joint training, 两段 schedule (λ_d 和 λ_c 错开)
- Seeds: {2, 15, 333} 严格对齐
- 其他不变

### Failure Modes

| Mode | Detection | Mitigation |
|------|-----------|------------|
| FM1: λ_c 太大导致 z_sty_norm 塌 (反向压过狠) | z_sty_norm R200 < 1.0 + probe_sty_class 掉但 probe_sty_domain 也掉 | λ_c 减半 |
| FM2: λ_c + λ_d 共同作用让 encoder 无法学 class | loss_task 不降 | λ_c schedule 推后到 R60-80 |
| FM3: probe_sty_class 没掉 (机制仍失效) | probe_sty_class > 50% | 说明 AlexNet linear 空间解耦不够, 需要 nonlinear decoder 或换 backbone |
| FM4: accuracy 崩 | PACS AVG Best < 78 | CDANN + 对称对抗太激进, 退回原 CDANN |

### Novelty and Elegance Argument

**Closest prior work**:
- **DANN (Ganin 2016)**: 单向 adversarial. 我们双向对称.
- **Zhao et al. 2019 "Invariant Representations"**: 单机, 双头但对称正向 supervision. 我们 FL + asymmetric GRL.
- **HEX (Wang 2019)**: Projecting superficial statistics out. 非 FL.
- **Factor VAE / β-TCVAE**: VAE-based disentanglement, 不适合 from-scratch CNN + FL.
- **Deep Feature Disentanglement SCL (2025 Cog. Comp.)**: symmetric dual-head positive. 我们 FL + asymmetric GRL.
- **FedPall (2025)**: adversarial amplifier. 我们 dual-path on decoupled features.

**Key delta**: **FL + 双向 GRL on both paths + 基于 frozen probe 的 pre-registered 验证**, zero direct prior.

## Claim-Driven Validation Sketch

### Claim C-disent (Primary, novel)

- **Statement**: Symmetric CDANN achieves genuine disentanglement (probe_sty_class drops from baseline ~85% to ≤ 25%).
- **Experiment**: PACS R200 seed=2 pilot, then 3 seeds if pilot succeeds
- **Metric**: frozen post-hoc `probe_sty_class` trained on frozen z_sty → class label
- **Baseline**: EXP-108 CDANN probe_sty_class = 0.962 (待跑 PACS baseline Linear+whitening 对照, 预期 ~0.85+)

### Claim C-parity (Secondary)

- **Statement**: PACS AVG Best 与 CDANN (80.08) 同级或更好, Office ≥ 88.0
- **Metric**: 3-seed mean AVG Best
- **Expected**: 80.0-82.0 (不差于 CDANN, 不超 Plan A 81.69)

### Claim C-style-share (Speculative, if C-disent succeeds)

- **Statement**: 真解耦后, AdaIN 风格共享 (EXP-059 路线) 不再破坏 z_sem → PACS 有机会涨 2-3pp
- **Experiment**: 阶段 2, 若阶段 1 成功才做

## Experiment Handoff Inputs

- Must-prove: C-disent (probe_sty_class ≤ 25%), C-parity (accuracy 不崩)
- Must-run ablations:
  - V1: CDANN (ca=1, ce=0) baseline — 已有 EXP-108
  - V2: CDANN + L_cls_sty (ca=1, ce=1) ours
  - V3: **只 L_cls_sty** (ca=0, ce=1) — 验证对称性必要
- Critical datasets: PACS (anchor 真考验), Office (style-weak 对照)
- Highest-risk assumption: **Linear probe 能检测 nonlinear 泄漏** (不行就需要 nonlinear probe)

## Compute & Timeline Estimate

- **Pilot** (PACS R100 seed=2 只 V2): 1.5h GPU
- **Full C-disent + C-parity**: PACS R200 × 3 seeds × V2 = 12h, Office 如果 pilot 不崩再跑 = 3h
- **Ablation V3**: PACS R200 × 2 seeds = 6h
- **Probes**: 30min
- **Baseline PACS probe 对照** (currently deploying): 2-3h
- **Total**: ~25 GPU·h
- **Timeline**: Day 1 pilot + baseline probe 结果 → Day 2 full 3 seeds → Day 3 ablation + 文档

## Remaining Risk / Open Question (自知不足, 请 reviewer 重点审查)

1. **Linear probe 检测不到 nonlinear 信号泄漏**: 如果 encoder 把 class 信号编码到 z_sty 的高阶流形上 (nonlinear), 线性 probe 查不出, GRL 也压不掉. 需要 nonlinear probe (2-layer MLP) 作为验证.
2. **λ_c schedule 调参复杂**: 两个 GRL 同时训, 互相干扰, 稳定性未知.
3. **FedBN 副作用**: EXP-108 证明 FedBN 让 feature 天然带 client-specific shift, 这可能让 GRL 永远压不掉 domain 信号 (probe_sem_domain = 1.0 就是证据). 对称 CDANN 的 class 反向方向也可能被类似的 BN artifact 阻挠.
4. **PACS accuracy 上限**: Plan A 81.69 已是历史最高. 对称 CDANN 希望 ≥ 82 可能是奢望.
5. **风格共享 (阶段 2)**: 如果 C-disent 成立但 C-style-share 失败, 那真解耦也没用 — 白忙一场.
