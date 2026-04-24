# Round 1 Review (GPT-5.4 xhigh)

**Date**: 2026-04-24
**Reviewer model**: gpt-5.4 via codex exec (xhigh reasoning)
**Session ID**: 019dc056-e5fa-7fd2-8721-994c42c8b3ca

## Score Table

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 8 | Still attacks Office/PACS anchored bottleneck, no drift |
| Method Specificity | **6** | Pd/Pc update rules, normalization, missing-class handling, partial participation, gradient flow, inference usage 均不够紧 |
| Contribution Quality | **6** | 真正 novel 只有 Pd + proto-level exclusion, 但 Pc + L_sem_proto 让 paper 看起来是 dual-prototype system, 扩散了焦点 |
| Frontier Leverage | 8 | 在 no-pretraining + 小数据约束下, 不用 VLM/Diffusion/LLM 合适 |
| Feasibility | **5** | 90-100 GPU-h 违反 50 GPU-h 预算; Pc+Pd+两条 proto loss + exclusion + Bell + MSE + α-sparsity 一次上太多旋钮, InfoNCE 已知不稳 |
| Validation Focus | **6** | C0 gate 方向对但 head-only retrain 不是干净的因果测试; 5 个 Vis block 超出 paper 所需 |
| Venue Readiness | **6** | 在 PACS/Office + AlexNet 想投 top venue 必须极端 sharp, 当前 reads like moderate disentanglement variant 而非 1 crisp mechanism |

## OVERALL SCORE: 6.5 / 10 — **REVISE**

## Main Findings (from reviewer)

1. ✅ **Anchor 保持**: 数据集 / backbone / 约束 / 成功判决没变, 还有 C0 kill gate
2. ❌ **核心因果故事与实际机制不符**: 用了 `taps.detach()` → `L_sty_proto` 不反传到 trunk → 方法其实是 "proto-space regularizer", 不是 "修复被 CE 污染的 trunk"
3. ✅ **Dominant idea 好**: "make domain a federated prototype object + class-domain 在 proto 级互斥" 是正确的 mechanism novelty
4. ❌ **方案太大 + 太 under-specified**: budget 超预算, 接口级细节没锁死

## CRITICAL Fixes Required

### [CRITICAL 1] Method Specificity
- **Weakness**: Pd/Pc update rules、normalization、缺类处理、partial participation、gradient flow、inference 用法都不够紧
- **Fix**: 显式定义 Pd ∈ ℝ^{D×128}, Pc ∈ ℝ^{C×128}; 写出 update equations, L2-norm, 缺类/缺 domain 的 no-op 规则; 明确 statistic encoder 输入是 pre/post-activation; 给出 **1 行 gradient-flow 表** 列出哪条 loss 打哪个模块
- **Action**: Refinement 里新增 "Interface Specification" 小节

### [CRITICAL 2] Feasibility
- **Weakness**: 预算违反 (90 vs 50 GPU-h); 一次上 Pc+Pd+2 proto loss+exclusion+Bell+MSE+α-sparsity 太多旋钮
- **Fix**: Stage-gate aggressive. 第一 pilot = Office only + seed 2, 从 orth_only 开始, 只加 Pd + asymmetric encoder + proto_excl (不加 L_sem_proto). 正向才升 3 seeds + PACS
- **Action**: 调整 Compute 表, 删 L_sem_proto, 阶段化预算

## IMPORTANT Fixes

### [IMPORTANT 3] Contribution Quality
- **Weakness**: Pc + L_sem_proto 让真正的 dominant (Pd + exclusion) 被稀释
- **Fix**: Asym encoder 保留为 enabling infra (不升到第 2 contribution); **删除 L_sem_proto**, Pc 只作为 running class centroid (EMA from z_sem) 用于 L_proto_excl 输入, 不再是 separate loss 对象
- **Action**: Refinement 大幅简化 Loss 组合

### [IMPORTANT 4] Validation Focus
- **Weakness**: C0 head-only retrain 不能 cleanly 测试 "CE 污染 trunk vs 聚合/outlier"; 5 个 Vis block 过载
- **Fix**:
  - **C0 改为 matched intervention test**: freeze encoder_sem, 加 encoder_sty + Pd + proto_excl, Office R20-30 → 若无法推动 Office 就 kill (相当于直接试做小号 BiProto 上限)
  - **Visual 从 Vis-1~5 压缩为 3 套**: (i) t-SNE (ii) probe ladder (iii) merged prototype/feature metrics
- **Action**: 重写 Claim 0 + 改写 Claim 2

### [IMPORTANT 5] Venue Readiness
- **Weakness**: 当前 reads like moderate disentanglement variant
- **Fix**:
  - 整篇文章一句话 thesis: "**domain should be a shared federated prototype object**"
  - Pd 改为 **domain-indexed** (非 client-concat); 其他组件全部降级为 support
- **Action**: 重写 thesis/title/对比矩阵

## Simplification Opportunities (reviewer 提供, 全部接受)

1. **删 L_sem_proto** — Pc 只作为 EMA class centroid 用于 L_proto_excl 输入, 不单独做 InfoNCE
2. **Pd 改 domain-indexed bank** — PACS/Office 实现上等价, 但更 clean, 更 benchmark-agnostic
3. **只用 Bell + 1 个 anchor 项** — 不默认发 Bell + MSE + α-sparsity 三件套, 只在 pilot 真崩时加第三件

## Modernization Opportunities

**NONE** — under current anchor, 冻结 VLM / diffusion prior 违反 no-pretraining, 且扩张 paper 不利于 crisp mechanism. 接受 reviewer 判断.

## Drift Warning

**NONE** at proposal level. 唯一风险: 若 C0 负结果但仍继续 BiProto, 方案会退化成"更干净的 disentanglement"叙事而非解决 Office bottleneck.

## Verdict

**REVISE** — 方向对, Pd 是对的 mechanism novelty 层次. 但需要 cleaner causal gate / 更少组件 / 更严接口 / 单一 contribution 叙事.

<details>
<summary>Raw reviewer output (click to expand)</summary>

See `round-1-review-raw.txt` for full reviewer output including scoring rationales and fixes.

</details>
