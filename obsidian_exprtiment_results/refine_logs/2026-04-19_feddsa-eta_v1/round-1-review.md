# Round 1 Review — Codex GPT-5.4 xhigh

## 评分汇总

| 维度 | Score | Verdict |
|------|-------|---------|
| Problem Fidelity (15%) | 8/10 | Office success condition 数字混乱 (82.55/89.82/90.58) |
| Method Specificity (25%) | 8/10 | exclude_self / BN eval / std correction 未说明 |
| **Contribution Quality (25%)** | **6/10** | **CRITICAL — 2.5 个贡献散乱** |
| Frontier Leverage (15%) | 8/10 | 适合 2025-era,拒绝 CLIP 合理 |
| **Feasibility (10%)** | **6/10** | **CRITICAL — AdaIN(z_sem) 数学 bug** |
| Validation Focus (5%) | 8/10 | 去掉 Learnable Proto / FedDEAP 次要 baseline |
| **Venue Readiness (5%)** | **6/10** | **IMPORTANT — story 不够单点** |
| **OVERALL** | **7.2/10** | **REVISE** |

## 核心反馈 (3 个 CRITICAL)

### 1. Contribution sprawl (CRITICAL)
- **问题**: ETF / SATA-gate / T3A-proto 同时抢 novelty headline,读起来像 "borrowed parts + new combination"
- **修复**: 把 novelty 收拢到 **一个**命名模块 "**Style-Gated Prototype Adjustment (SGPA)**"
  - Dominant contribution = 用 disentangled style 做 **test-time reliability signal** 指导 prototype 更新
  - Fixed ETF 降级为 **supporting stabilizer + prototype prior**
  - C2 要求放宽:ETF "不伤 + 稳定方差 + 给 proto 提供好的冷启动" 即可,不再要求 beat Linear

### 2. AdaIN 在 z_sem 空间的数学 bug (CRITICAL)
- **问题**:
  - `z_sem.std(0)` B=1 时 NaN
  - `μ,σ` 从 `z_sty` 拿,但应用到 `z_sem` 空间不 well-justified
  - `exclude_self` 在未见 target domain 下无意义
  - `torch.no_grad()` 不够,需要 `model.eval()` 否则 BN running stats 会污染
  - ETF 构造 `β` 与 `τ` 冗余,应固定 β=1
- **修复**: 完全删除 AdaIN(z_sem),改用 **z_sty-space distance gate**:
  ```
  # 训练时额外维护 source_style_bank ∈ ℝ^{N×d_sty}
  reliable = (H < τ_H) & (min_k ||z_sty - μ_k|| < τ_S)
  ```
  - 更小 (没有 swap forward pass)
  - 更符合 disentanglement 叙事 (直接用分离出的 z_sty)
  - 数值稳定 (两个标量比较)
- **其他必须修**:
  - `model.eval()` + `torch.no_grad()` 双保险
  - `torch.std(..., correction=0)` 或 `var(unbiased=False).sqrt()`
  - `assert d >= K` for ETF 构造
  - prototype EMA 换成 "top-m reliable supports per class" (避免脏样本污染)

### 3. Venue 叙事不够单点 (IMPORTANT)
- **问题**: reviewer 会一句话归纳为 "FedETF-ish classifier + SATA/T3A on disentangled FL backbone"
- **修复**: paper one-liner =
  > "Disentangled source-domain style statistics serve as a **federated, backprop-free reliability signal** for test-time prototype correction."
- **vs FedDEAP 新 rephrase**:
  > "FedDEAP improves multi-domain FL through CLIP prompt tuning and semantic/domain transformations **during training**. Our contribution is orthogonal: in scratch federated DG, we use disentangled style statistics **only at inference** to decide which unlabeled test samples may update a prototype memory; the fixed ETF head is a stable prior, not the main novelty."

## Simplification Opportunities (按 reviewer)
- ✂️ 删除 AdaIN(z_sem) → 换 z_sty distance gate
- ✂️ 删除 "training-inference geometry coherence" headline,ETF 降级为 prototype prior
- ✂️ 删除 Learnable Proto+CE 和 FedDEAP-adapted-to-ResNet 次要 baseline (不是证 core claim 必需)

## Modernization Opportunities
- NONE (scratch-CNN 已足够现代,CLIP 会变成 FedDEAP paper)

## Drift Warning
- NONE,但警告:若 ETF-only 实验失败,不要变成 "pure TTA on Plan A",那会丢失训练端几何叙事

## Mathematical Soundness 剩余 Issues
- [x] ETF d >= K assert
- [x] β=1 固定
- [x] std correction=0
- [x] model.eval() + no_grad
- [x] AdaIN space mismatch (通过切换到 z_sty distance gate 消除)

## Action Items → Round 1 Refinement
1. 改名: **FedDSA-ETA → FedDSA-SGPA (Style-Gated Prototype Adjustment)**
2. 重写 Contribution Focus: 1 dominant + 1 support (ETF 降级)
3. 重写 Core Mechanism (2): 从 AdaIN(z_sem) → **z_sty distance gate + entropy**
4. 新增 Mathematical Correctness 小节 (assert d>=K, eval mode, std correction, top-m proto)
5. 清理 baseline: 去掉 Learnable Proto 和 FedDEAP 核心 compare
6. 重写 one-liner + FedDEAP 差异化段落

<details>
<summary>Raw Codex response (完整原文)</summary>

见 `round-1-review-raw.txt`
</details>
