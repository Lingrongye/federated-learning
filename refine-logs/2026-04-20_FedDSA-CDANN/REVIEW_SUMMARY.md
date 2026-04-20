# Review Summary — FedDSA-CDANN

**Problem**: 联邦学习跨域客户端的风格-语义解耦对不同数据集 (Office vs PACS) 的自适应处理. FedDSA-SGPA Linear+whitening 在 Office +6.20pp, PACS -1.49pp. 诊断: PACS z_sty_norm 被 whitening 磨掉 95%.
**Initial Approach**: 双向 DANN 头, z_sem 反向梯度 + z_sty 正向, 保留风格中的 class 判别信号.
**Date**: 2026-04-20
**Rounds**: 5 / 5 (MAX_ROUNDS reached)
**Final Score**: 8.75 / 10
**Final Verdict**: **REVISE (proposal-complete, near ceiling)**

## Problem Anchor (verbatim, 5 轮不变)

- **Bottom-line problem**: 跨域 FL 解耦在不同数据集性质下的自适应处理. Linear+whitening Office +6.20pp / PACS -1.49pp.
- **Must-solve bottleneck**: 统计解耦 (cos²+HSIC) 无方向性; class 信号被错归到 z_sty 并被 whitening 磨掉 (PACS z_sty_norm 塌 95%).
- **Non-goals**: 不追求万能机制; 不用 LLM teacher; 不做 label noise/少样本.
- **Constraints**: seetacloud2 4090; Office/PACS client=domain bijection; AlexNet baseline.
- **Success**: PACS 3-seed AVG Best ≥ 82.2, Office ≥ 88.0, PACS `probe_sty_class` ≥ 40%.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|-----------------------|----------------------------------------|---------|----------------|
| 1 | Contribution sprawl (6/10), Venue 6, Frontier 6. 2 个 dom_heads 看似两模块; dataset diagnosis 让 paper 发散; AlexNet-only 看起来 dated. | **合并 2 heads 为 1 shared**; **降级** dataset diagnosis 为 supporting analysis; **窄化** scope 到 `client=domain FedDG where style carries class signal`; **加 DINOv2** portability check; **删** fallback `HSIC(z_sty, y)=0`. | **Mostly** (6 → 8) | dom_head 机制描述可能被读成"对抗" |
| 2 | Mechanism 表述 "shared head with opposing gradients" 误导 (head 实际 non-adversarial). Training head acc 不 clean 作 diagnostic. C-port 可能变 second contribution. | **精确重写** dom_head non-adversarial (两路 minimize CE) + asymmetry 在 encoder 梯度方向; **替换** diagnostic 为 frozen post-hoc probe on z_sem/z_sty; **降级** C-port 明确为 appendix sanity. | **Mostly** (8.35 → 8.4) | Probe 协议可能有 leak; 缺 class-level probe |
| 3 | Probe 协议 fit on test 是 leaked; 缺 class probe on z_sty 直接证 anchor; 不应用 "30-paper gap" novelty defense. | **修** probe 协议 train on train / test on held-out; **加** `probe_sty_class` on z_sty → y; **删** 综述 gap 话术, **改** 窄化 novelty 为 "minimal repair for whitening-induced style collapse". | **Mostly** (8.4 → 8.75) | Framing 3 处 IMPORTANT 未锁 |
| 4 | Probe 描述过 "formal proof"; 没明确 probe 都在 post-whitening; novelty 句子不统一. | **改** "evidence consistent with anchor" (非 formal proof); **明确** 三 probe 都在 post-whitening features; **锁定** one-sentence novelty verbatim 在三处出现. | **Mostly** (8.75, 7 维度中 4 个 9 分) | Novelty ceiling moderate |
| 5 | **Novelty ceiling 内在限制, proposal 已到 review-time 上限. 不是 framing 问题**. | 无 proposal 改动, 仅确认 framing discipline. | **Ceiling reached** (8.75) | 靠实验结果 overperform 提分 |

## Overall Evolution

- **方法变 concrete**: R0 的 "add 2 MLPs + GRL" → R4/R5 精确表述 "**shared non-adversarial** discriminator + GRL on z_sem **encoder path** only" (asymmetry 锁在 encoder gradient direction, 非 head objective)
- **Contribution 收敛**: 从 R0 "三个并列 contribution" → R4/R5 **单一 dominant contribution** ("minimal repair for whitening-induced style collapse")
- **Complexity 减半**: 2 dom_heads (R0) → 1 shared dom_head (R1+), new trainable params 18K→9K
- **Frontier leverage 到位**: 加了 DINOv2 frozen encoder appendix 作为 portability sanity check (非 second contribution)
- **Anchor 零 drift**: 5 轮反馈无一 drift warning, 诚实保留 scope (`client=domain where style carries class signal`) 未被 reviewer 推广

## Final Status

- **Anchor status**: **Preserved** (5 轮零 drift warning)
- **Focus status**: **Tight** (1 dominant contribution, 1 appendix sanity, novelty 一句话)
- **Modernity status**: **Appropriately frontier-aware** (DINOv2 sanity check, 不做 LLM/VLM 灌水)
- **Strongest parts**:
  - Mechanism precise: shared non-adversarial head + encoder-gradient asymmetry (GRL on z_sem only)
  - Evidence aligned to anchor: `probe_sty_class` on z_sty 直接验证 "class-relevant style preserved"
  - Scope disciplined: `client=domain FedDG where style carries class signal`
- **Remaining weaknesses**:
  - Novelty ceiling intrinsic, 不是 fixable framing 问题
  - 可能仍被读成 "asymmetric DANN repair", 不是 new method family
  - 唯一提升路径: 实验结果 overperform (PACS AVG Best +2pp, `probe_sty_class` gap +30pp)

## Decision for Next Step

- **Proposal-complete, proceed to implementation + experiment** (/experiment-plan)
- **Do not continue proposal refinement**: reviewer 明确指出已到 ceiling
- **Rely on empirical execution**: 如果结果超出 expected evidence, paper 可以 overperform proposal score
