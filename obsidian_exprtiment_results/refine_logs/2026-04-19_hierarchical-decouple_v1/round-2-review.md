# Round 2 Review — 8.0/10 REVISE

| 维度 | 分数 |
|------|------|
| Problem Fidelity | 9 |
| Method Specificity | 7 |
| Contribution Quality | 8 |
| Frontier Leverage | 9 |
| Feasibility | 8 |
| Validation Focus | 8 |
| Venue Readiness | 6 ⚠️ |

## 关键 CRITICAL

**Venue Readiness 6/10**:Loss 公式里 `F.normalize + /HW` 重复(normalize 已单位化,再除 HW 让数值过小,阈值不可解读)。

**Fix**:选一种
- 要么 cosine decorrelation(keep F.normalize,**去 /HW**)
- 要么 true cross-correlation(centered/standardized + /HW,**不 L2-normalize**)

## Simplification
- 去 "hierarchical"(只用 2 层不算 depth stages)
- "no added trainable parameters" 代替 "0 new trainable"
- 主表报 final + best,不只 best

## Verdict: REVISE
