# Review Summary — FedDSA-BiProto

**Problem**: 在 FL cross-domain (PACS + Office-Caltech10) 下严格胜过 FDSE 本地复现 baseline, 当前 orth_only PACS ✅ (+0.73) 但 Office ❌ (−1.49), 必须补 Office 同时保 PACS.

**Initial Approach (user intuition)**: 解耦 + 双原型 + 双 InfoNCE 对比 + 反向约束 + 非对称 encoder + t-SNE/probe 验证性实验.

**Date**: 2026-04-24
**Rounds**: 4 / 5 (用户选择 R4 停止)
**Final Score**: **8.75 / 10**
**Final Verdict**: **REVISE** (near-READY, 距 9.0 bar 0.25)

## Problem Anchor (immutable, 逐字复刻 across 4 rounds)

- Bottom-line: AlexNet from scratch, 3-seed × R200 mean AVG Best 必须严格超 FDSE (PACS > 79.91, Office > 90.58)
- Must-solve: Office 补 −1.49 pp + 至少 +0.5 pp; PACS ≥ 80.91
- Non-goals: 不换数据集 / 不做诊断论文 / 不堆模块 / 不预训练 / 不换骨干
- Constraints: 4090/3090; Pilot ≤ 50 GPU-h hard; AlexNet + FedBN; PACS E=5 / Office E=1; 1 周
- Success: 3-seed mean AVG Best + 3 套可视化 evidence

## Round-by-Round Resolution Log

| Round | Overall | Main Reviewer Concerns | What This Round Changed | Solved? | Remaining Risk |
|:-:|:-:|---|---|:-:|---|
| 1 | 6.5 REVISE | Contribution sprawl (dual Pc+Pd), Budget 违反 (90 vs 50 GPU-h), "trunk decontamination" 叙事与 taps.detach() 矛盾, C0 诊断不干净 | 删 L_sem_proto; asym encoder 降为 enabling; Pd 改 domain-indexed; C0 matched intervention; Budget stage-gated; Visual 5→3 suites; Interface Spec 新增 gradient-flow table | partial | C0 inference path 冻死; headline-impl 错配; Pc pseudo-duality |
| 2 | 7.8 REVISE | C0 invalid (冻 head 不能动 prediction), headline vs impl 错配 (说 federated Pd 但用 batch-local), Pc pseudo-duality, sparse-batch rules 缺 | C0 只冻 encoder_sem (head trainable); L_proto_excl 改 hybrid ST axis (forward=Pd, grad via batch centroid); Pc 完全降级为 monitor; sparse-batch present-classes rule; scope tightening D=K=4 | partial | ST normalization point; −Pd 必要性; C0 role 措辞 |
| 3 | 8.25 REVISE | ST normalization 未定; absent-class fallback 不明; "federated" claim empirically 受限于 D=K=4; −Pd 必须 MANDATORY; C0 role wording | ST = `F.normalize(Pd.detach()+bc-bc.detach())` 最后 renorm; L_proto_excl pilot 阶段 present-classes-only 无 fallback; "to our knowledge" + Limitations §; −Pd MANDATORY; C0 = "pruning test, necessary but not sufficient" | partial | Pd 初始化; −Pd claim downgrade pre-register; −encoder_sty 单 factor; ST bias transparency |
| 4 | 8.75 REVISE | 4 细节 AI: Pd init / −Pd claim downgrade / −encoder_sty 单 factor / ST estimator transparency | 全部 4 AI 合并入 FINAL_PROPOSAL. Reviewer quick calls: Anchor ✅ / ST spec ✅ / C0 wording ✅ / −Pd right call ✅ / "to our knowledge" ✅ sufficient for honesty | fully design-complete | **Empirical 问题**: Pd 在 D=K=4 下是否真有 federated 增量 (只能跑 −Pd 回答) |

## Overall Evolution (dimensional)

| Dimension | R1 | R4 | Total Δ | Main Driver |
|---|:-:|:-:|:-:|---|
| Problem Fidelity | 8 | 9 | +1 | anchor 保持不变, 贯穿 4 rounds |
| Method Specificity | 6 | 9 | **+3** | gradient-flow table + ST form + Pd EMA + present-classes rule |
| Contribution Quality | 6 | 8.5 | **+2.5** | L_sem_proto 删除; Pc 降级; "to our knowledge" narrowing |
| Frontier Leverage | 8 | 9 | +1 | appropriate conservatism 接受 (no LLM/VLM/Diffusion in AlexNet setup) |
| Feasibility | 5 | 8.5 | **+3.5** | stage-gated pilot ≤ 26 GPU-h; present-classes-only 简化 |
| Validation Focus | 6 | 9 | **+3** | C0 fix (freeze encoder_sem only); −Pd MANDATORY; 3-suite visual |
| Venue Readiness | 6 | 8 | +2 | sharper 叙事 (单一 dominant), 但 D=K=4 empirical ceiling |

## Final Status

- **Anchor status**: ✅ Preserved verbatim across all 4 rounds, 无 drift
- **Focus status**: ✅ Tight — 单一 dominant contribution (Federated Pd with hybrid ST exclusion)
- **Modernity status**: ✅ Intentionally conservative — 不强加 FM-era components (reviewer 明确接受)
- **Strongest parts of final method**:
  1. 单一 crisp thesis: "domain should be a first-class federated prototype object"
  2. Straight-through hybrid axis: forward = federated Pd (headline 对齐), grad = batch centroid (真实训练信号)
  3. C0 matched-intervention pruning test: 真正可测 BiProto 增量的 low-cost gate
  4. 全部 loss 有 safety valve (Bell + MSE), 避开 EXP-076/077/095/108/017 所有已知坑
  5. 完整 3-suite visual evidence 直接回应"必须有可视化验证"的 user requirement
- **Remaining weaknesses** (R4 reviewer 确认均为 empirical, 非 design):
  1. D=K=4 下 Pd 的 federated value 需要 −Pd ablation 实证
  2. ST estimator bias vanishing 需要 Vis-C trajectory 实证
  3. Venue readiness 由 empirical 结果决定, 不由 refine 决定

## Stop Decision Rationale

用户于 Round 4 (8.75/10 REVISE) 后选择停止. Reviewer 在 R4 review 明确:

> "v4 resolves the main mechanism and interface objections. The proposal is now coherent, implementable, and appropriately scoped. The remaining reason it is not READY is not a design flaw; it is that the venue case still depends on a narrow but crucial empirical question: whether Pd shows real value beyond a local domain-axis surrogate on D=K=4 benchmarks."

**换言之**: 继续 Round 5 的边际收益为零 (所有 design 维度已至饱和), 只有实际跑 −Pd MANDATORY ablation 才能推分过 9.0. 停在 R4 进入 implementation 阶段是正确决策.

## Next Steps

1. **/experiment-plan**: 把 FINAL_PROPOSAL 转详细 experiment roadmap (S0 C0 gate 最先)
2. **/run-experiment S0** (2 GPU-h): 跑 C0 matched-intervention, 判决 BiProto 是否启动
3. 根据 C0 结果决定进 S1-S4 full pipeline 或 kill 回 Calibrator 兜底
