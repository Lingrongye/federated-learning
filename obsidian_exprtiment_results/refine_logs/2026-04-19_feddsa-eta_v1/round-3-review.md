# Round 3 Review — Codex GPT-5.4 xhigh

## 评分汇总

| 维度 | R1 | R2 | **R3** | Δ(R2→R3) |
|------|----|----|------|---|
| Problem Fidelity | 8 | 8.6 | **9.0** | +0.4 |
| Method Specificity | 8 | 8.7 | **9.1** | +0.4 |
| Contribution Quality | 6 | 9.0 | **9.2** | +0.2 |
| Frontier Leverage | 8 | 8.7 | **8.9** | +0.2 |
| Feasibility | 6 | 8.1 | **8.7** | +0.6 |
| Validation Focus | 8 | 8.3 | **9.0** | +0.7 |
| Venue Readiness | 6 | 8.5 | **8.9** | +0.4 |
| **OVERALL** | **7.2** | **8.5** | **9.0** | **+0.5** |
| Verdict | REVISE | REVISE | **REVISE** (one small revision away from READY) | — |

## Reviewer 明确表态
> "This is one small revision away from READY, but not READY yet."

## 4 个剩余 Action Items (全部 wording/pseudocode)

### 1. 文字 overreach
- **原文**: "provably invariant to cross-client private BN affine distortion"
- **改为**: "provides a shared Mahalanobis metric under a pooled second-order approximation"
- **理由**: pooled whitening 在 shared affine transform 下 invariant,但不是任意 client-specific 变换

### 2. Warm-up pseudocode bug
- **原文** (r3 prompt line 71-73): `if batch_idx < 5: ... continue` → 前 5 batch 无预测输出
- **改为**: warm-up 期 `reliable = False`,但**仍 output ETF fallback prediction** (round-2-refinement 的写法其实是对的)

### 3. 加 diagnostic 图/表
- Distance 分布 before/after whitening (sanity)
- Ablation: whitening on/off
- Ablation: Σ_within only vs Σ_within + Σ_between

### 4. Paper 正文 wording
- 显式写: "no raw sample or per-instance feature sharing; only client-level summary statistics (μ, Σ) are broadcast"
- symeig → torch.linalg.eigh (API 更新)
- 量化比较: "每轮 SGPA 广播 = 66KB,AlexNet FedBN γ/β 广播 = ~X KB"

## 结论
Round 3 Score 9.0 已达 threshold。4 个 action items 是 wording-level,apply 后直接进 FINAL。不再需要第 5 轮 codex review (reviewer 已明确 "one revision away")。

<details>
<summary>Raw</summary>
见 round-3-review-raw.txt
</details>
