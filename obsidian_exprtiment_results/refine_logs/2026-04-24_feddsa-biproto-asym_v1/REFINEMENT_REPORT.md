# Refinement Report — FedDSA-BiProto

**Problem**: 在 FL cross-domain (PACS + Office-Caltech10) 下严格胜过 FDSE 本地复现 baseline (PACS > 79.91, Office > 90.58). 当前 orth_only PACS ✅ (+0.73) / Office ❌ (−1.49).
**Initial Approach**: 用户直觉 — 解耦 + 双原型 + 双 InfoNCE + 反向约束 + 非对称 encoder + t-SNE/probe 验证性实验.
**Date**: 2026-04-24
**Rounds**: 4 / 5 (用户选择 R4 stop)
**Final Score**: **8.75 / 10**
**Final Verdict**: **REVISE (near-READY)**

## Problem Anchor (逐字跨 4 rounds)

- Bottom-line: AlexNet from scratch, 3-seed × R200 mean AVG Best 严格超 FDSE 本地复现
- Bottleneck: Office 补 −1.49 pp + 至少 +0.5 pp; PACS ≥ 80.91
- Non-goals: 不换数据集 / 不做诊断论文 / 不堆模块 / 不预训练 / 不换骨干
- Constraints: 4090/3090; Pilot ≤ 50 GPU-h; AlexNet + FedBN
- Success: accuracy 目标 + 3 套可视化 evidence

## Output Files

| File | Purpose |
|---|---|
| `FINAL_PROPOSAL.md` | Paper-ready 干净 final method (v4) |
| `REVIEW_SUMMARY.md` | Round-by-round resolution log + evolution |
| `REFINEMENT_REPORT.md` | 本文件, 含 pushback log + next steps |
| `score-history.md` | 7 维分数跨 4 rounds |
| `round-N-initial-proposal.md` / `round-N-review.md` / `round-N-refinement.md` | 每轮原始 artifact |
| `round-N-review-raw.txt` | Raw codex stdout (verbatim) |
| `review_prompt_rN.txt` | 每轮发送给 reviewer 的 prompt |
| `REFINE_STATE.json` | 最终 completed state |

## Score Evolution

| Round | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 8 | 6 | 6 | 8 | 5 | 6 | 6 | **6.5** | REVISE |
| 2 | 9 | 8 | 7 | 9 | 7 | 6 | 7 | **7.8** | REVISE |
| 3 | 9 | 8 | 8 | 9 | 8 | 8 | 7 | **8.25** | REVISE |
| 4 | 9.0 | 9.0 | 8.5 | 9.0 | 8.5 | 9.0 | 8.0 | **8.75** | REVISE |

**Cumulative Δ**: +2.25 (6.5 → 8.75). 最大涨幅维度: Feasibility (+3.5), Method Specificity (+3.0), Validation Focus (+3.0).

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | Key Changes | Result |
|:-:|---|---|:-:|
| 1 | Contribution sprawl (Pc+Pd dual), Budget 违反 (90 vs 50), "trunk decontamination" 叙事矛盾, C0 不干净 | 删 L_sem_proto, demote asym encoder, Pd domain-indexed, stage-gated budget, Visual 5→3, gradient-flow table | Partial |
| 2 | C0 冻死 inference path, headline vs impl 错配, Pc pseudo-duality, sparse-batch 缺 | C0 只冻 encoder_sem, hybrid ST axis, Pc 降为 monitor, present-classes rule, D=K=4 scope | Partial |
| 3 | ST norm point / absent-class fallback / D=K=4 empirical 弱 / −Pd 必做 / C0 role wording | F.normalize final renorm, present-classes-only pilot, "to our knowledge" + Limitations §, −Pd MANDATORY, C0 = pruning test | Partial |
| 4 | 4 个细节 AI (Pd init / pre-register downgrade / −encoder_sty 精确 / ST bias transparency) | 全部合并入 FINAL_PROPOSAL. Reviewer: "design-complete, remaining is empirical only" | Full |

## Final Proposal Snapshot (3-5 bullets)

1. **Dominant**: Federated Domain Prototype Pd ∈ ℝ^{D×d_z} 作为一阶联邦共享对象, server-side EMA 跨 client 聚合, 作为 forward anchor of class-domain geometric exclusion in prototype space.
2. **Straight-Through Hybrid Axis**: `domain_axis[d] = F.normalize(Pd[d].detach() + bc_d - bc_d.detach())`, forward value = federated Pd (headline 对齐), gradient via batch-local centroid (训练信号到 encoder_sty).
3. **Enabling infra**: 非对称统计 encoder_sty (~1M MLP on (μ,σ) taps, detached), structurally 无法承载 class 判别信号.
4. **Safety valves**: Bell schedule + MSE anchor (α-sparsity default off). 继承 EXP-076/077 验证. 避开 EXP-017 HSIC / EXP-108 CDANN 全部已知坑.
5. **Stage-gated pilot**: S0 C0 matched-intervention (2 GPU-h pruning test) → S1 Office seed=2 smoke (4 GPU-h) → S2 Office 3-seed (20 GPU-h) → S3 PACS 3-seed (30 GPU-h) → S4 ablations (40 GPU-h, -Pd MANDATORY). Pilot budget ≤ 26 GPU-h fits 50 anchor.

## Method Evolution Highlights

1. **Most important simplification**: 删除 L_sem_proto + Pc 降级为 monitor (R1→R2). 把方案从"dual prototype bank" 收缩为"single federated Pd against online class centroid", contribution 单一化.
2. **Most important mechanism upgrade**: Hybrid Straight-Through Axis (R2→R3). 让 headline (federated Pd) 和 implementation (gradient path) 真正对齐, 修复 R2 reviewer 指出的 critical mismatch.
3. **Most important validation upgrade**: C0 Gate 从 "freeze all → head-only retrain" 改为 "freeze encoder_sem only → full add-on branch R20-30" (R2). 让 gate 真正可测 add-on 对 Office 的实际增量.
4. **Most important scope correction**: "to our knowledge" + Limitations 小节 (R3→R4). 诚实 scope 到 D=K=4 benchmark, 把 "federated domain object" 和 "per-client bank" 的 empirical 区分留给 −Pd MANDATORY ablation.

## Pushback / Drift Log

本次 4 轮 review 中, **reviewer 未触发任何 drift warning**. 全部建议均 accepted 或进一步 narrow (没有需要 pushback 的 case).

关键 non-drift 但被拒绝的路线:
- Frontier leverage upgrade (VLM/Diffusion): reviewer 在 R1 开放 option, 作者 argue "anchor 禁止 pretraining + 数据规模太小", reviewer 在 R2-R4 明确接受 "appropriate conservatism"
- α-sparsity 作为 default: R1 reviewer 建议删除第三层 safety valve, 作者 accept, 留作 fallback

## Remaining Weaknesses (R4 reviewer 确认均为 empirical, 非 design)

1. **D=K=4 empirical ceiling** — Venue Readiness 8.0 的硬天花板. Pd 的 federated novelty 只能通过 −Pd ablation 实证, 无法通过 refine 进一步推高
2. **Straight-Through estimator bias** — 已在 Limitations § 声明, 通过 Vis-C Pd ⊥ bc_d cosine trajectory 监测; 实证 bias vanishing 需要 R100+ 数据
3. **AlexNet-only** — ResNet-18/ViT transferability 留 future work (anchor 明确规定)

## Raw Reviewer Responses

<details>
<summary>Round 1 Review (6.5/10 REVISE)</summary>

Session ID: `019dc056-e5fa-7fd2-8721-994c42c8b3ca`. Full content in `round-1-review-raw.txt`.

Key verbatim quote:
> "The strongest issue is not drift; it is mismatch between the claimed causal story and the actual intervention. With `taps.detach()`, `L_sty_proto` does not repair the shared trunk directly, so the method is more honestly a prototype-space regularizer than a trunk decontamination mechanism."
</details>

<details>
<summary>Round 2 Review (7.8/10 REVISE)</summary>

Full content in `round-2-review-raw.txt`.

Key verbatim quote:
> "The proposal says the key object is federated Pd, but the actual exclusion loss is still driven by batch-local centroids, and the new C0 gate is currently invalid as written because it freezes the only prediction path that could improve accuracy."
</details>

<details>
<summary>Round 3 Review (8.25/10 REVISE)</summary>

Full content in `round-3-review-raw.txt`.

Key verbatim quote:
> "The hybrid straight-through axis closes the main mismatch from v2: forward semantics now use federated Pd, while gradients still reach encoder_sty. ... C0 gate is now valid as a pruning gate."
</details>

<details>
<summary>Round 4 Review (8.75/10 REVISE — stopped here)</summary>

Full content in `round-4-review-raw.txt`.

Key verbatim quote:
> "v4 resolves the main mechanism and interface objections. The proposal is now coherent, implementable, and appropriately scoped. The remaining reason it is not READY is not a design flaw; it is that the venue case still depends on a narrow but crucial empirical question: whether Pd shows real value beyond a local domain-axis surrogate on D=K=4 benchmarks."
</details>

## Next Steps

1. **`/experiment-plan`**: 把 FINAL_PROPOSAL 转为详细 experiment roadmap (每 stage 具体命令 + kill criteria + NOTE.md 模板)
2. **`/run-experiment S0`** (2 GPU-h): 跑 C0 matched-intervention pruning test, 判决 BiProto 是否启动
3. **根据 C0 结果分叉**:
   - Δ ≥ +1.0pp → S1-S4 full pipeline
   - Δ +0.3~+1.0pp → S1-S4 但降档预期
   - Δ < +0.3pp → **Kill BiProto**, 启动 Calibrator 兜底或 SAS τ tune / Caltech 权重聚合改造
4. **实现 feddsa_biproto.py** (~250 行) + 单元测试 + codex review → S1 smoke (若 S0 通过)
5. **−Pd MANDATORY ablation** (S4 阶段): 这是 paper venue-readiness 的最终实证 gate. 若 −Pd 与 full BiProto 差 < 0.5pp, 触发 pre-registered claim downgrade.
