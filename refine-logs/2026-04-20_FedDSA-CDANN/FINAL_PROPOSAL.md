# FedDSA-CDANN: Minimal Repair for Whitening-Induced Style Collapse in client=domain FedDG

**Final Version (R5, proposal-complete)**
**Score**: 8.75/10, Verdict: REVISE (near review-time ceiling, novelty ceiling is intrinsic)
**Date**: 2026-04-20

---

## Problem Anchor

- **Bottom-line problem**: 联邦学习中跨域客户端 (Feature-skew FL) 的语义-风格解耦对不同数据集性质 (风格是语义核心 vs nuisance) 的自适应处理. 当前方法 FedDSA-SGPA 的 Linear+whitening 在 Office +6.20pp, 在 PACS -1.49pp.
- **Must-solve bottleneck**: 现有统计解耦约束 (cos²=0 + HSIC) **没告诉模型什么是风格**, 导致类别判别信号被错归到 z_sty, 被 whitening 磨掉. 诊断: PACS Linear+whitening 后 z_sty_norm R10=3.12 → R200=0.15 (-95%); Office 仅磨 2%.
- **Non-goals**: 不追求"一个机制万能解决所有数据集" (诚实承认 scope); 不引入 VGG/CLIP teacher; 不做 label noise / 少样本.
- **Constraints**: seetacloud2 4090; Office-Caltech10 + PACS_c4 (client=domain bijection); AlexNet baseline + DINOv2 appendix sanity; Venue CVPR/ICCV 2026/2027.
- **Success condition**: PACS 3-seed AVG Best ≥ 82.2, Office ≥ 88.0, PACS `probe_sty_class` ≥ 40% (baseline ≈ 15%).

## Technical Gap

**Observed failure mode** (measurable on real diagnostic data from 2026-04-20 实验):
- Office AVG Best 88.75 vs Plan A 82.55 (**+6.20pp**)
- PACS AVG Best 80.20 vs Plan A 81.69 (**-1.49pp**)
- PACS z_sty_norm R10=3.12 → R200=0.15 (**塌 95%**)
- Office z_sty_norm R200=2.21 (**稳定 -2%**)

**Root cause**: `L_orth = cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)` 只要求统计独立, 没方向性监督. 当 PACS 风格 (油画/素描) 携带 class 判别信号时, 特征可能错归到 z_sty 并被 whitening 磨掉.

**Smallest adequate intervention**: 给解耦加方向性 supervision. Domain label (= client id in client=domain FedDG) 是**零成本 zero-annotation** 可用的 supervision signal. 一个 **shared non-adversarial** `dom_head` (两路都 minimize standard CE) + **GRL 只作用在 z_sem encoder 路径** = 修这个 failure mode 的最小改动.

## Method Thesis

**Novelty (one sentence, 锁定)**: **A shared non-adversarial domain discriminator plus asymmetric encoder-gradient supervision (GRL on z_sem path only) is the minimal repair for whitening-induced style collapse in client=domain FedDG when style carries class signal.**

**Why smallest adequate**:
- 1 MLP (9K params) + 1 GRL (无参) + 3 行 loss
- encoder/whitening/聚合不动
- 3 个 frozen probe 作 evaluation (not new trainable)

## Contribution Focus

- **Dominant contribution (sole)**: 上面 one-sentence 的 minimal repair, 配 frozen post-hoc class probe on z_sty 作 representation-level evidence consistent with anchor.
- **Appendix sanity check (not a contribution)**: Frozen DINOv2-S/14 encoder 替换 AlexNet, 验证 mechanism not AlexNet-specific.
- **Non-contributions (explicit)**: 不 claim 新聚合 / 新 whitening / 新分类器 / "all datasets" / theoretical convergence / dataset boundary diagnosis (降级为 supporting analysis).

## Proposed Method

### Complexity Budget

- **Frozen / reused**: AlexNet encoder, 双头 sem_head/sty_head, sem_classifier, pooled whitening, L_orth + HSIC, FedBN
- **New trainable**: **1** — `dom_head`: MLP 128→64→N_clients (~9K params)
- **GRL**: 无参
- **Post-hoc evaluation probes** (not new trainable, evaluation-only): 3 个 Linear probes (sem→domain, sty→domain, sty→class)
- **Excluded**: MI estimator, CC-Style HSIC, selective whitening, VGG/CLIP teacher, HSIC(z_sty, y)=0 (冲突 anchor)

### System Overview

```
                                                     ┌── sem_classifier(z_sem) ── L_CE(y)          [encoder 正向 task]
  x ─ encoder ─ feature ─┬─ sem_head → z_sem ─ (wh) ─┤
                         │                            └── dom_head(GRL(z_sem, λ)) ── L_CE(d)        [dom_head 正向 CE; encoder via GRL 反向]
                         └─ sty_head → z_sty ─ (wh) ──── dom_head(z_sty) ── L_CE(d)                 [dom_head 正向 CE; encoder 正向]

  z_sem ⊥ z_sty:  λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)

  dom_head 自身 non-adversarial (两路都 minimize standard CE).
  Asymmetry 只在 encoder 的两条 upstream path 梯度方向上.
```

### Core Mechanism (精确表述)

- **dom_head objective**: standard Cross-Entropy discriminator, 两路 branch 都 minimize (non-adversarial head training):
  ```
  L_dom_head_update = CE(d, dom_head(GRL(z_sem))) + CE(d, dom_head(z_sty))
  ```
- **Encoder gradient flows**:
  - `z_sem → sem_classifier`: 正向 task supervision
  - `z_sem → GRL → dom_head`: encoder 路径梯度被反转 → z_sem 被推成 domain-indistinguishable
  - `z_sty → dom_head`: encoder 路径梯度不反转 → z_sty 被推成 domain-discriminative
- **Asymmetry location**: 不在 head objective, 在 encoder 两条 upstream path 的梯度方向.

### Architecture & Loss

```python
dom_head = Linear(128, 64) → ReLU → Dropout(0.1) → Linear(64, N_clients)
GRL(x, λ): forward(x) = x, backward(grad) = -λ · grad

L_task    = CE(y, sem_classifier(z_sem))
L_dec     = λ_orth · cos²(z_sem, z_sty) + λ_hsic · HSIC(z_sem, z_sty)
L_dom_sem = CE(d, dom_head(GRL(z_sem, λ_adv)))
L_dom_sty = CE(d, dom_head(z_sty))
L_total   = L_task + L_dec + L_dom_sem + L_dom_sty

# Scheduling
λ_adv(r) = min(1.0, max(0, (r - 20) / 20))   # R=0..20 off, R=20..40 linear, R≥40 full
λ_orth = 1.0, λ_hsic = 0.1 (inherited from Plan A)
```

**Critical detail**: Features entering dom_head 都是 **post-whitening** z_sem / z_sty (same feature space as sem_classifier).

**Inference**: 只走 z_sem → sem_classifier, 预测 y. z_sty 用于下游 pooled style bank 广播 (保持 Plan A).

### Aggregation Strategy

- `dom_head` 通过 **FedAvg 聚合** (和 encoder/sem_classifier 一致)
- 理由: 本地 data 单域, 本地 dom_head 会 degenerate. 聚合后跨 client 共享能区分 N domain 的 head, GRL 才有信号
- 隐私: 聚合 head 参数 (不是 data); domain label = client id 本就公开

### Integration into Codebase

- File: `FDSE_CVPR25/algorithm/feddsa_sgpa.py`
- 新增: `GradientReverseLayer` 类 (~15 行) + `dom_head` MLP (~10 行) + L_dom_sem/sty 计算 (~20 行) + FedAvg 白名单 + frozen probe eval script (~50 行)
- 开关: algo_para 加 `ca` (cdann active, 0/1 default 0), 保持 backward compat

### Training Plan

- Joint training, λ_adv 三段 schedule
- 其他同 EXP-102 config: LR=0.05, E=1 (Office) / E=5 (PACS), R=200
- Loss weights: λ_orth=1.0, λ_hsic=0.1, λ_adv scheduled

### Failure Modes and Diagnostics

- **FM1: 对抗训练发散** — 检测 L_task 不降 / z_sem_norm 暴跌; 缓解 λ_adv schedule + grad clip=10
- **FM2: dom_head 欠拟合** — 检测 head acc 过低; 缓解 Dropout=0.1 + FedAvg 聚合
- **FM3: z_sty 过度携带 class 信息** (λ_adv 过大) — 检测 `probe_sty_class` 过高同时 sem_classifier 掉; 缓解**只降 λ_adv 到 0.5** (不加 HSIC(z_sty, y) 约束, 冲突 anchor)
- **FM4: Office 不需 CDANN** — scope 已限定 "style carries class signal" 场景, Office parity 即可, 不视为失败

### Probe Protocol (R3 修正, post-whitening + 无 leak)

**Critical detail**: 所有三个 frozen probe (`probe_sem_domain`, `probe_sty_domain`, `probe_sty_class`) **输入均为 post-whitening features** (同 sem_classifier 训练时的 feature space). 因为 claimed failure mode 是 whitening-induced collapse, probe 必须在 whitening 之后验证, 才能反映 whitening 的影响.

```python
# After R200, freeze encoder / heads / whitening.
# Train probes on FROZEN TRAIN features, report test accuracy on HELD-OUT TEST features.
Z_sem_train, Z_sty_train, D_train, Y_train = aggregate_across_clients(train_loader)
Z_sem_test,  Z_sty_test,  D_test,  Y_test  = aggregate_across_clients(test_loader)

probe_sem_dom = LogReg().fit(Z_sem_train, D_train)   # → domain
probe_sty_dom = LogReg().fit(Z_sty_train, D_train)   # → domain
probe_sty_cls = LogReg().fit(Z_sty_train, Y_train)   # → class (KEY anchor-aligned evidence)

report_test_accs(probe_sem_dom, probe_sty_dom, probe_sty_cls)
```

## Novelty Argument

**锁定 one sentence (verbatim in Method Thesis / Novelty / Abstract)**:

> A shared non-adversarial domain discriminator plus asymmetric encoder-gradient supervision (GRL on z_sem path only) is the minimal repair for whitening-induced style collapse in client=domain FedDG when style carries class signal.

**Delta vs closest prior**:
- **Deep Feature Disentanglement SCL** (Cog. Comp. 2025): non-FL + symmetric positive supervision + separate heads. 我们: FL + asymmetric encoder-gradient + shared non-adversarial head.
- **FedPall** (ICCV 2025): adversarial in mixed feature space, erase domain. 我们: preserve domain info in z_sty after disentanglement.
- **ADCOL / Federated Adversarial DA**: erase-all DANN. 我们: asymmetric preserve.

**Defense against "pseudo-novelty"**: 同一个 dom_head 参数被两路更新 (both standard CE, non-adversarial), asymmetry **不在 head objective**, 在 **encoder 两条 upstream path 的梯度方向**. 这个精确表述防止被读成 "两个独立 DANN head". 关键 delta: **head 非对抗 + encoder 梯度方向非对称**.

## Claim-Driven Validation Sketch

### Claim C-main (Primary): CDANN 修 PACS 不伤 Office

- **Experiment**: PACS R200 3 seeds + Office R200 3 seeds, vs Linear+whitening baseline
- **Headline metrics** (主文 table, R4 收缩):
  - AVG Best
  - `probe_sem_domain` (expect ≈ random 1/N), `probe_sty_domain` (expect ≈ 1.0)
  - **PACS `probe_sty_class`** (KEY: CDANN ≥40%, baseline ≈15%) — Office 版本进 appendix
  - z_sty_norm R200
- **Expected evidence**:
  - PACS AVG Best 82-84 (from -1.49pp 回到 ≥0 vs Plan A 81.69)
  - Office AVG Best ≥ 88.0 (maintain baseline)
  - PACS `probe_sty_class` gap ≥ 25pp (CDANN vs baseline)
  - z_sty_norm R200 ≥ 1.5 (from 0.15)

### Claim C-ablate (必做): z_sty 正向监督的必要性

- **Experiment**: PACS R200 2 seeds × 3 variants:
  - (V1) baseline: Linear+whitening (no CDANN)
  - (V2) z_sem-only: 只 L_dom_sem (标准 DANN)
  - (V3) full CDANN (ours)
- **Expected**: V3 > V2 ≥ V1 on AVG Best **AND** PACS `probe_sty_class`
- **Key observation**: `probe_sty_class` V3 - V2 gap **consistent with** "z_sty 正向监督保留了 class-relevant style" 的 anchor claim. 这是 representation-level evidence aligned with failure mode, **not** formal causal proof (后者需要 counterfactual 分析, 本文未做).

### Appendix sanity check C-port (not a contribution)

- **Experiment**: PACS R100 1 seed, encoder 换 frozen DINOv2-S/14 (其他不变)
- **Goal**: mechanism not AlexNet-specific
- **Expected**: AVG Best ≥ AlexNet CDANN

## Experiment Handoff Inputs

- **Must-prove claims**: C-main (PACS+Office × 3 seeds × R200 = 12h), C-ablate (PACS × 3 variants × 2 seeds × R200 = 24h)
- **Appendix**: C-port (2h), 3 probes (30min)
- **Critical metrics**: AVG Best, PACS `probe_sty_class`, `probe_sem_domain`, z_sty_norm R200
- **Highest-risk assumptions**:
  - A1: `client=domain` bijection (PACS/Office 成立)
  - A2: 同一个 dom_head 同时服务两路不互相抵消 (pilot 验证)
  - A3: λ_adv schedule 稳 (DANN 经典问题, 有成熟 mitigation)

## Compute & Timeline Estimate

- **Pilot**: Office + PACS R100 × 1 seed = 2.5h
- **Full C-main**: 12h
- **C-ablate**: 24h
- **C-port**: 2h
- **Probes + NOTE**: 2h
- **Total: ~41 GPU·h** on single 4090

**Timeline**:
- Day 1: 代码实现 + 单测 (3h 本地)
- Day 2: Pilot (2.5h seetacloud2) → 决策
- Day 3-5: Full C-main + C-ablate (~36h 并行跑 2-3 runs)
- Day 6: C-port + probes + NOTE + Obsidian 文档
- Day 7: Codex re-review on results + refine paper section

## R5 Reviewer Action Items (apply during paper draft)

- **IMPORTANT**: 不再 broaden claim. 保持 exact scoped setting: `client=domain` FedDG where style carries class signal.
- **IMPORTANT**: Results section lead with failure-mode chain in one line: PACS regression → z_sty_norm collapse → domain/class probes → recovery under CDANN.
- **IMPORTANT**: Ablation table (baseline / z_sem-only / full CDANN) 做 novelty 主要工作, 不要埋在其他 diagnostics 下.

## Remaining Risk (Honest Disclosure)

- **Novelty ceiling is intrinsic**: R5 reviewer 明确指出 "proposal is near its review-time ceiling. Further proposal-side refinement is unlikely to materially change the score. The only path upward is execution quality"
- **Venue readiness 停在 8/10**: 即使完美执行, 这篇仍然会被强 reviewer 读成 "a very clean asymmetric DANN-style repair inside an existing FedDG pipeline"
- **Mitigation**: 依靠实验结果 overperform; 如果 PACS AVG Best 大幅超越 Plan A (> +2pp) 且 `probe_sty_class` 差距 > 30pp, 就能 empirical 支撑 top-venue pitch.
