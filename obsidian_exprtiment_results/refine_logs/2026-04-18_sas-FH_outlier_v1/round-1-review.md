# Round 1 Review — sas-FH Proposal

**Reviewer**：GPT-5.4 via codex (reasoning=high)
**Session ID**：`019d9ebd-235a-7ae3-9dbc-739cc40b9995`
**日期**：2026-04-18
**Verdict**：**REVISE**

## 评分（Scores）

| 维度 | 分数 | 理由 |
| --- | --- | --- |
| 1. Problem Fidelity | 9/10 | 紧扣锚定瓶颈（shared classifier hurts outliers）。无 drift。 |
| 2. Method Specificity | 9/10 | 具体到可立即实现（精确的 keys、aggregation rule）。 |
| 3. Contribution Quality | 8/10 | 单一主导机制，简洁。局限：novelty 偏窄（scope extension + conditioning choice）。 |
| 4. Frontier Leverage | 8/10 | 恰当 — 未强行引入 FM-era 原语。 |
| 5. Feasibility | 9/10 | 服务器端小改动；稳定；compute plan 可信。 |
| 6. Validation Focus | **6/10** | 缺少隔离 novelty 的关键对比。 |
| 7. Venue Readiness | **6/10** | 仍像 "scope tweak"。 |

**overall score：8.3/10**（加权：PF 15% + MS 25% + CQ 25% + FL 15% + Feas 10% + VF 5% + VR 5%）

## 关键发现（Critical Findings）

### `6. Validation Focus: 6/10`（CRITICAL）
- **弱点**：实验无法隔离增益是来自 *style-conditioned classifier aggregation* 还是 *任何通用的 classifier personalization*。`sem-only` vs `full-head` 不够。
- **Fix 1（必改）**：删除 A3（encoder last block）ablation。替换为一个 matched counterfactual：
  - `sem_head sas + local classifier`（完全不聚合 head），或
  - `sem_head sas + uniformly personalized/local head`（FedROD style：per-client classifier 但无 style conditioning）
- **Fix 2**：加一个 mechanism diagnostic：在某个 checkpoint freeze 特征，评估 client i，仅在 personalized 和 shared 之间 swap classifier，报告 outlier 域的 delta。直接测试 "feature right, boundary wrong"。

### `7. Venue Readiness: 6/10`（IMPORTANT）
- **弱点**：论文仍像 scope tweak。顶会 reviewer 会问 "为什么不就是 Plan A 应该做的一个 ablation？"
- **Fix 1**：把 thesis 收紧为：*"classifier personalization is the minimal sufficient personalization granularity under style-driven domain outliers, and style similarity is the right routing signal for sharing classifier boundaries."*
- **Fix 2**：从 proposal body 移除 Top-2 (multi-centroid) fallback（不同 method direction，稀释 elegance）。
- **Fix 3**：把 PACS 从 co-equal claim 降级为 **negative control / boundary condition**。按 anchor，Office 才是真正的 target。

## 简化机会（Simplification Opportunities）

1. 删除 `A3: sem-head + classifier + encoder last block` — 削弱 "smallest adequate" 故事。
2. 移除 Top-2 multi-centroid fallback。
3. 把 PACS 从 must-run 降级为 negative-control（仅当 Office 成功才跑）。

## 现代化机会（Modernization Opportunities）

**无**。对于这个锚定的问题，FM-era machinery 会改变论文本身，而不是在可比条件下解决所述瓶颈。

## Drift Warning

**无**。主方法仍攻击锚定问题。

**轻微注意**：encoder-last-block 和 multi-centroid fallback 若升级为主 story 会造成 drift。

## Raw Response

<details>
<summary>Full verbatim GPT-5.4 response</summary>

```
Reviewing against the anchor first: whether full-head SAS is still solving the shared-boundary bottleneck, and whether the proposal stays minimal rather than turning into another multi-module FedDG paper.

The proposal is well anchored and unusually implementable, but it is not yet at top-venue bar. The main issue is not method drift. It is that the current validation plan does not cleanly isolate the claimed novelty, and the paper would still read as a very small server-side scope change unless the mechanism evidence is sharpened.

**Scores**
| Dimension | Score | Rationale |
| --- | --- | --- |
| 1. Problem Fidelity | 9/10 | This stays on the anchored bottleneck: shared classifier boundaries hurting outlier domains. Extending personalization from `semantic_head` to `head` is directly aligned, not drift. |
| 2. Method Specificity | 9/10 | The interface is concrete enough to implement immediately: exact keys, exact aggregation rule, unchanged losses, unchanged client path, explicit config switch, explicit file/method. |
| 3. Contribution Quality | 8/10 | The proposal has one dominant mechanism and is admirably parsimonious. The only limitation is that the novelty is narrow: it is a scope extension plus a conditioning choice, not a fundamentally new algorithmic object. |
| 4. Frontier Leverage | 8/10 | Not using FM-era primitives is appropriate here. A VLM/LLM add-on would mostly change the problem, not solve the anchored boundary-sharing issue under the stated comparability constraints. |
| 5. Feasibility | 9/10 | High. The method is a small server-side change, training remains stable in expectation, and the compute plan is credible. |
| 6. Validation Focus | 6/10 | The plan includes some unnecessary breadth while missing one critical comparison that would prove the actual novelty. |
| 7. Venue Readiness | 6/10 | If results land, this could be a strong, clean result. But as currently framed, it still risks reading like "move `head.*` into the personalized key set" rather than a sharp top-venue contribution. |

**Overall Score: 8.3/10**

**Weaknesses For Scores < 7**

`6. Validation Focus: 6/10`
- Specific weakness: The experiments do not isolate whether gains come from `style-conditioned classifier aggregation` versus any generic form of classifier personalization.
- Concrete fix 1: Delete A3 ablation. Replace with matched counterfactual:
  - `sem_head sas + local classifier` (no head aggregation), or
  - `sem_head sas + uniformly personalized/local head` in FedROD style.
- Concrete fix 2: Add mechanism diagnostic — freeze features, swap only classifier between personalized and shared, report outlier-domain delta.
- Priority: CRITICAL

`7. Venue Readiness: 6/10`
- Specific weakness: Paper story still too close to scope tweak.
- Fix 1: Tighten claim to "classifier personalization is minimal sufficient granularity under style-driven outliers, and style similarity is right routing signal".
- Fix 2: Remove Top-2 multi-centroid fallback from body.
- Fix 3: Treat PACS as negative control.
- Priority: IMPORTANT

**Simplification Opportunities**
- Delete A3 (encoder last block).
- Remove Top-2 (multi-centroid) fallback.
- Demote PACS from must-run to boundary condition.

**Modernization Opportunities**
- NONE.

**Drift Warning**
- NONE. Minor caution: encoder-last-block and multi-centroid would drift if elevated.

**Verdict**: REVISE
```

</details>
