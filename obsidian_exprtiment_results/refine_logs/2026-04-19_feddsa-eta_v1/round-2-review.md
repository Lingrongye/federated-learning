# Round 2 Review — Codex GPT-5.4 xhigh

## 评分汇总 (跳升)

| 维度 | R1 | **R2** | Δ |
|------|-----|------|---|
| Problem Fidelity | 8 | 8.6 | +0.6 |
| Method Specificity | 8 | 8.7 | +0.7 |
| **Contribution Quality** | 6 | **9.0** | **+3.0 ★ RESOLVED** |
| Frontier Leverage | 8 | 8.7 | +0.7 |
| **Feasibility** | 6 | 8.1 | +2.1 (PARTIAL) |
| Validation Focus | 8 | 8.3 | +0.3 |
| **Venue Readiness** | 6 | **8.5** | **+2.5 ★ RESOLVED** |
| **OVERALL** | 7.2 | **8.5** | **+1.3** |
| Verdict | REVISE | **REVISE** (逼近 READY) | — |

## Round 1 CRITICAL 处理状态

| Item | R2 Status |
|------|----------|
| 1. Contribution sprawl | ✅ **RESOLVED** — SGPA 单点贡献 |
| 2. AdaIN 数学 bug | 🟡 **PARTIAL** — AdaIN 删掉了,但新问题:FedBN 下 z_sty 跨 client 可比性 |
| 3. Venue 叙事 | ✅ **RESOLVED** — one-liner 清晰 |

## 新 CRITICAL (仅 1 个)

### FedBN 私有 BN 下,跨客户端 z_sty 不在同一坐标系
- **细节**: 每 client 私有 BN γ/β/running stats → 客户端 A 的 z_sty 和 B 的 μ_k 受不同 affine distortion 影响
- **L2 normalize 能消尺度,但消不掉 affine**
- **影响**: `dist_min = min_k ||normalize(z_sty) - normalize(μ_k)||` 是 heuristic 而非 principled
- **Fix**: 需要 **whitening** 或让 z_sty 头的 BN 走 FedAvg (牺牲 FedBN 原则)

## IMPORTANT 剩余

1. **删除 `bn.track_running_stats=False`** — 实际上会把 BN 推向 batch-stat,weakens 稳定性声明。保留 `model.eval() + torch.no_grad()` 即可
2. **first-batch threshold calibration 改为 running quantile** — 当前 order-sensitive
3. **σ 决断**: σ 在 bank 中广播但未真正用 → 要么删除 σ,要么用 Mahalanobis 距离 `(z_sty-μ)^T Σ^{-1} (z_sty-μ)`
4. **ETF 在 abstract/intro 保持 subordinate** — 不要重新浮到 headline
5. **Non-goal narrowing**: 明确说 "no raw feature/sample sharing, only lightweight source-domain summary statistics (μ, Σ) for test-time shift detection" — 把 source_style_bank 广播正当化

## Drift Warning

一个 procedural drift:source_style_bank 广播本身是"lightweight cross-client summary statistics sharing",与原 non-goal "不跨客户端风格共享" 有轻微冲突 → 需要 narrow non-goal 表述

## Checks (a-d)

- (a) SGPA 是真 dominant ✅
- (b) z_sty gate 用到了 z_sty + style_bank ✅,但 τ_S 从 first-batch calibrate 是 relative 而非 absolute
- (c) FedDEAP 差异化正确 ✅
- (d) 跨 client z_sty 可比性 numerically yes,statistically partial — **需要 whitening**

## Action Items → Round 2 Refinement

1. **CRITICAL**: SGPA gate 改为 **Mahalanobis distance in whitening space** — 既用 σ 又解决跨 client 可比
2. **IMPORTANT**: 删除 bn.track_running_stats=False
3. **IMPORTANT**: first-batch calibration → running quantile (每 batch 更新 p50/p30)
4. **IMPORTANT**: non-goal 重写,正当化 summary statistics broadcast
5. **MINOR**: notation 统一 (L2 vs 1-cosine)

<details>
<summary>Raw Codex response</summary>

见 `round-2-review-raw.txt`
</details>
