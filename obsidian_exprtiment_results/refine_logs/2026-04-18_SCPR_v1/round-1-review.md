# Round 1 Review — GPT-5.4 xhigh

**Reviewer**: GPT-5.4 via codex exec (model_reasoning_effort=xhigh)
**Session ID**: 019da01c-b0bf-7ff2-8739-f9278b9664d2
**日期**: 2026-04-18

---

## 总体评分

| 维度 | 分数 | 权重 |
|------|------|------|
| Problem Fidelity | **9/10** | 15% |
| Method Specificity | **6/10** ⚠️ | 25% |
| Contribution Quality | **6/10** ⚠️ | 25% |
| Frontier Leverage | **8/10** | 15% |
| Feasibility | **8/10** | 10% |
| Validation Focus | **8/10** | 5% |
| Venue Readiness | **6/10** ⚠️ | 5% |
| **OVERALL** | **7.1/10** | — |

**Verdict**: **REVISE**
**Drift Warning**: NONE(方向仍然对准锚定问题)

---

## 关键批评(按优先级)

### CRITICAL - 两个变体在数学上不一致

- **问题**:初稿同时把"单原型 retrieved mean"和"weighted multi-positive"当作主方法的两个变体,并声称"worst case = M3 (+5.09%)"
- **事实**:
  - 单原型 retrieved mean 路径在 uniform weight 下**退化为 FedProto global mean**,**不是 M3**
  - 只有 weighted multi-positive 路径在 uniform weight 下才退化为 M3
- **后果**:lower-bound story 只对其中一个变体成立;初稿的"严格泛化 FedProto 和 M3"claim 同时成立是不可能的

### IMPORTANT - Share 可能塌缩为 no-share

- **问题**:4-5 个 client 下,`cos(s_i, s_i) = 1` 是最大值,w_{i→i} 会主导 softmax → 每个 client 完全只用自己原型 → 重现 FedBN-like 本地化
- **修复方向**:默认 **self-mask**(j ≠ i)或对 self-weight 显式 cap

### IMPORTANT - 接口没有冻结

- **问题**:
  1. Style key 用 `z_sty proto` 还是 `(μ, σ)`?初稿混用
  2. Style bank 什么时候更新?(每轮 pack 时、每轮 iterate 时、EMA?)
  3. 某客户端某类**没有**类原型时怎么处理?(renormalize over available clients)
- **修复方向**:三者都要在接口层冻结

---

## 各维度详评

### Problem Fidelity (9/10) — 仍然对准锚定问题

- 方法仍然攻击"global mean prototype 稀释 outlier"这一 bottleneck
- SCPR 把对齐目标从 mean 换成 retrieval,正好命中

### Method Specificity (6/10, < 7) — CRITICAL fix 需要

- **具体弱点**:两条主线数学上不一致
- **Concrete fix**:
  > 选**唯一** canonical algorithm。推荐:**基于已有 M3 代码路径,把 equal positive weights 替换为 style-conditioned weights w_{i→j},默认 self-mask(j ≠ i),对没有该类原型的 client 做 renormalize**。这样 uniform weight 真正退化为 M3、τ→0 退化为 nearest-style sharing。
- **Priority**: CRITICAL

### Contribution Quality (6/10, < 7) — IMPORTANT fix 需要

- **具体弱点**:初稿感觉像 3 个 contributions 共存(retrieval、multi-pos 泛化、SCPR+SAS composability)
- **Concrete fix**:
  > 定义 SCPR 为**一个**机制。最佳选项:**self-masked style-weighted M3** 作为主方法。retrieved-mean 和 SCPR+SAS 移到附录或删除。若坚持 retrieved-mean 为主线,则删掉所有"M3 lower-bound/generalization"语言。
- **Priority**: IMPORTANT

### Frontier Leverage (8/10) — 已经正确

- attention-based retrieval 对这个问题是自然 primitive
- 强行加 CLIP/VLM/RL 反而是 drift

### Feasibility (8/10) — 工程可行

- 代码增量小,无新 trainable 组件,60 GPU·h 预算合理

### Validation Focus (8/10) — 基本 tight

- 唯一 bloat:SCPR+SAS 不应进主表格,仅 appendix composability check

### Venue Readiness (6/10, < 7) — IMPORTANT fix 需要

- **具体弱点**:reviewers 会先攻击 formulation 不一致,才会讨论 gain
- **Concrete fix**:
  > 把主 claim 收窄为 3 点:
  > 1. Style-conditioned sharing 在 PACS 上严格优于 uniform sharing
  > 2. 不依赖参数个性化(保留 PACS 跨域知识共享)
  > 3. 补齐 FedDSA Share 章节缺口
  >
  > 不要在 headline 里写 "SCPR+SAS" 或 "严格泛化 FedProto 和 M3" 除非最终 canonical formulation 真的支持这些声明。
- **Priority**: IMPORTANT

---

## Simplification Opportunities

1. `self-mask` 默认设计,不作 ablation
2. 只用已有 SAS/M3 的 style key 和 bank,不引入 z_sty proto 和 (μ, σ) 两种竞争 key
3. 从核心论文里砍掉 SCPR+SAS 和 grad_CE vs grad_InfoNCE 监控(不是证明主 claim 所需)

## Modernization Opportunities

NONE — 已经是正确的 modernity 水平

---

## Reviewer Summary(原文)

> If you want the sharpest version of this paper: make SCPR one clean algorithm, self-masked by construction, and build it directly on top of the already-verified `M3` path rather than carrying both a retrieved-mean version and a weighted multi-positive version.

---

<details>
<summary>Raw codex output (tokens=29126)</summary>

**Key Findings**
- `CRITICAL`: the current "default" algorithm and the claimed lower-bound story do not match. The single retrieved-prototype path collapses to `FedProto/global mean` under uniform weights, not to `M3`. The `M3` reduction is only true for the weighted multi-positive branch.
- `IMPORTANT`: `Share` can collapse into no-share. With 4-5 clients, `w_{i->i}` will likely dominate unless self is excluded or explicitly capped, which recreates the FedBN-like local behavior you already diagnosed.
- `IMPORTANT`: the interface is close, but not frozen. The proposal still leaves unresolved which style key is queried (`z_sty` proto vs `(mu, sigma)`), when the bank is updated, and how missing class prototypes are handled.

**Simplification Opportunities**
- Make `self-mask` the default design, not an ablation.
- Reuse the existing SAS/M3 style key and bank exactly; do not introduce both `z_sty`-proto and `(mu, sigma)` as competing query interfaces.
- Drop `SCPR+SAS` and `grad_CE` vs `grad_InfoNCE` from the core paper; they are not needed to prove the main claim.

**Verdict**: REVISE

(Full raw response stored in `round-1-review-raw.txt`.)

</details>
