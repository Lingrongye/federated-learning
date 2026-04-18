# 第 2 轮 Review —— sas-FH 修订后的提案

**Reviewer**：GPT-5.4 via codex (reasoning=high)
**Session ID**：`019d9ebd-235a-7ae3-9dbc-739cc40b9995`（与 R1 同一线程）
**日期**：2026-04-18
**结论**：**REVISE**（比 READY 低一档）

## 各维度评分

| Dimension | Score | Rationale |
| --- | --- | --- |
| 1. Problem Fidelity | 10/10 | 仍紧扣锚点；对 bottleneck 的 framing 更好。 |
| 2. Method Specificity | 9/10 | 具体、最小、工程师可直接实现。剩余差距：C2 update rule 还需给出精确的操作形式。 |
| 3. Contribution Quality | 9/10 | 一个干净的 mechanism-level contribution，配以问对了问题的 counterfactual。 |
| 4. Frontier Leverage | 9/10 | 克制得当；没有生搬 FM-era primitives。 |
| 5. Feasibility | 9/10 | 在声明的代码 / 计算预算内高度可行。 |
| 6. Validation Focus | 8/10 | Validation 现在基本 minimal 且对症。剩余问题：A2 vs C2 的 statistical strength（计划里只有 1-seed）。 |
| 7. Venue Readiness | 8/10 | 从 workshop 跃升到 conference 很锐利。离 top-venue 更近，但核心 novelty 的实证严谨度需再提。 |

**Overall: 9.0/10**（算术 8.95 四舍五入）

## Anchor Check
**已保留。** 仍是同一个问题：style heterogeneity 下共享 classifier boundary 导致 outlier 域失败。修订没有漂移到更宽泛的 personalization paper 或更大的 FedDG 系统。

**Contribution 更锐利**：从 "full-head personalization + side ideas" 变为 **"style similarity is the routing signal for classifier-boundary sharing."**

**Method 更简单、不臃肿**：移除 encoder-last-block ablation、multi-centroid fallback 和 parallel contribution track 是对的。

**Frontier leverage 合适**：克制，不是那种坏的 old-school。

## 剩余 critique（全 MINOR，无 blocking CRITICAL）

### Validation / novelty isolation (VF=8)
- C1、C2、swap diagnostic 正确回应了 R1 的弱点 ✓
- **剩余担忧**：除非 A2 成功，否则 C1/C2 只跑 1-seed。若 A2 − C2 margin 小，1-seed 对 top-venue 不足。
- Paper 的 novelty 不再是 "classifier personalization helps"，而是 "**style-conditioned sharing > generic classifier personalization**"。A2 vs C2 现在是最重要的实验，不是 side check。

### Method specificity (MS=9)
- C2 "FedROD-style uniform per-client classifier" 概念清楚，但 server update rule 必须无歧义：
  - Option α：每轮 `head_i ← mean_j head_j`（uniform mean = global FedAvg）
  - Option β：`head 跨轮保持仅本地`（= C1）
  - 这是**不同**的 counterfactual；必须锁定其中一个。

### Pseudo-novelty risk (VR=8)
- Top-venue reviewer 仍可能说 "just move `head.*` into personalized key set."
- 修订后的 framing 有帮助，但 paper 必须让 routing claim 承担工作量：*style 不仅是另一个 personalization scope，它是 selective classifier sharing 的 criterion.*
- A2 vs C2 是 paper 的主定理；它需要 statistical strength。

## Drift 警告
**无。**

## 简化机会
1. PACS 严格作为 negative-control appendix；不要让它变成 paper 的第二条 thread。
2. 若预算紧张，砍掉 C1 —— C2 是必要 counterfactual，C1 次要。
3. 若 PACS 实际不跑，就从 success criteria 移除 art-painting。Office 才是真正的 main path。

## 现代化机会
**无。**

## 第 3 轮 Action Items
1. **在 update-rule 层面精确定义 C2** —— 在本 codebase 里给出无歧义的 server-side 逻辑 "uniform per-client classifier"。
2. **预承诺：若 A2 − C2 margin 在 modest band（例如 < 1.5%），把 C2 从 1 seed 升到 3 seeds。** 这是核心 novelty 测试。
3. **Claim 2 澄清**："global FedAvg head" 必须取自与 A2 **同一 round/checkpoint** 以避免 confound。
4. **Paper framing（更严格的 thesis）**："**The key question is not whether to personalize the classifier, but how to route classifier sharing.**"
5. **Mechanism evidence**：为 Office 报告一次 style-similarity 矩阵和由此得到的 routing weights。轻量、直接支撑 claim。

## Raw Response

<details>
<summary>Full verbatim GPT-5.4 response</summary>

See `round-2-review-raw.txt` lines 172-244 for full text.

Key passages:

> Problem Anchor is preserved. This is still the same problem: outlier-domain failure caused by a shared classifier boundary under style heterogeneity. The revision does not drift into a broader personalization paper or a larger FedDG system.
>
> The dominant contribution is now sharper. The paper is no longer "full-head personalization plus several side ideas"; it is now clearly "style similarity is the routing signal for classifier-boundary sharing."
>
> The method is simpler, not overbuilt. Removing the encoder-last-block ablation, multi-centroid fallback, and parallel contribution track was the right move.
>
> **Overall Score: 9.0/10**
>
> Using the stated weights, the arithmetic lands at 8.95, which rounds to 9.0. I still would not mark it `READY`, because one blocking issue remains around proof strength for the central novelty claim.
>
> The remaining top-venue risk is still pseudo-novelty: a reviewer may say this is just moving `head.*` into an existing personalized key set. The revised framing helps, but the paper must make the routing claim do the work: style is not just another personalization scope, it is the criterion for selective classifier sharing. That means the A2 vs C2 result is now the most important experiment in the paper, not a side check.
>
> **Verdict: REVISE**
>
> This is a much stronger proposal than Round 1. The anchor is preserved, the contribution is sharper, the method is simpler, and the validation is far more on-point. The only reason it is not `READY` is that the core novelty now lives almost entirely in `A2 > C2`, so that comparison needs to be specified and, if promising, validated with enough statistical strength to survive top-venue scrutiny.

</details>
