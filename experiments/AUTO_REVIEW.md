# FedDSA-Adaptive 自动审稿记录

## Round 1 (2026-04-14T08:31)

### Assessment (Summary)
- Score: 5/10
- Verdict: Not ready
- Key criticisms: M2丢失Share claim; gap度量用h混合信息; 4客户端归一化脆弱; 缺固定α基线和M3-only

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score: 5/10.** Good debugging instinct, but this is **not yet paper-grade**.

1. **Technical soundness**
- **M1:** reasonable patch, but weak theory. It assumes client-level (μ,σ) on backbone h measures **domain gap**; it may also reflect **class mix / semantics**.
  - **Min fix:** compute gap on **style head / BN stats**, use **EMA + robust normalization**.
- **M2:** weakest. This is mostly **DSU-style stochastic perturbation**, not real cross-client style transfer. Diagonal Gaussian from 4 clients is noisy and can sample unrealistic styles.
  - **Min fix:** sample around **shared-bank distribution excluding self**, not local stats only.
- **M3:** strongest and most justified. It attacks the real bottleneck: **prototype dilution**.

2. **Novelty**
- **M1/M2:** incremental engineering.
- **M3:** still borrowed, but meaningful.
- Overall this reads more like **repairing FedDSA** than adding a new contribution.

3. **Experiment design**
- **Not sufficient.** Choosing direction from **seed=2 only** is not credible.
- Missing:
  - **M3-only**
  - **fixed α baselines** (0.2/0.5/0.8)
  - **DSU/local stochastic augmentation baseline**
  - **"style bank off for low-gap clients"** sanity baseline
  - **3 seeds before branching**

4. **Feasibility of M1+M2**
- They can coexist, but **currently they double-count augmentation strength**.
- M2 already scales perturbation via variance; M1 scales again via gap.
  - **Min fix:** let **M2 choose style source**, **M1 choose mixing weight only**.

5. **Potential issues**
- **Mixed semantic+style gap metric:** yes, real problem.
  - **Fix:** use **style-only features** or class-balanced stats.
- **M2 loses "Share":** yes. Then it is no longer really **Decouple-Share-Align**.
  - **Fix:** keep shared external statistics in the sampling distribution.
- **Deterministic aug_strength:** too rigid.
  - **Fix:** use it as a **mean**, add small stochastic jitter.
- **4-client normalization:** weak and unstable.
  - **Fix:** use **rank/percentile/EMA z-score**, not divide-by-max.

6. **Missing baselines**
- **M3-only**
- **DSU**
- **FedFA-style feature perturbation**
- **Fixed-alpha AdaIN**
- **Disable augmentation only on low-gap domains**

7. **Verdict**
**Not READY** for full implementation.
**READY only for:** **M3-first**, plus a **simpler M1 sanity test**.
**Do not invest in M2** until you prove it still preserves the paper's **"Share"** claim.
If I were reviewing the paper, I would say: **promising fix path, but currently under-controlled and conceptually drifting.**

</details>

### Actions Taken
1. **砍掉M2** — 审稿指出它丢失"Share"核心claim，不是真正的跨客户端迁移
2. **Gap度量改用z_sty** — 不再用backbone h（混合语义+域），改用style head输出的统计量
3. **归一化改为EMA z-score** — 不用divide-by-max（4客户端不稳定）
4. **aug_strength加随机抖动** — alpha_mean + N(0, 0.05)
5. **补充固定α基线** — EXP-072a/b/c (α=0.2/0.5/0.8)
6. **补充M3-only基线** — EXP-072d
7. **所有实验改为3 seeds** — 2, 333, 42

### Status
- 方案已修订为v2.0，提交Round 2审稿

---

## Round 2 (2026-04-14T09:15)

### Assessment (Summary)
- Score: **6/10** ✅ (≥6 threshold met)
- Verdict: **Almost, not ready** (contains "almost" → threshold met)
- Key criticisms:
  1. 空间不匹配：z_sty用于gap度量，但AdaIN仍作用于h → 需两套bank
  2. z_sty作为域信号未验证 → 需domain-ID线性探针
  3. EMA z-score对4客户端偏弱 → 可对比rank/percentile
  4. 缺DSU/FedFA式non-share扰动基线
  5. M1+M3可能不可加 → 需严格2×2 factorial

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score: 6/10. Verdict: Almost, not ready.**

**Round-1 concerns:** mostly addressed, but not fully. M2 removal, fixed-α, M3-only, 3 seeds, and low-gap sanity are good. Biggest remaining issue is **new conceptual inconsistency**.

**Critical weaknesses (ranked):**
1. **Coherence bug:** plan says gap/style stats come from **z_sty** (lines 77–95), but AdaIN is still applied on **h** using dispatched style stats (99–113, 289–310). That is a space mismatch, likely dimension mismatch too.  
   **Min fix:** use **two banks**: `z_sty` only for gap scoring; `h`/BN stats only for AdaIN transfer.

2. **z_sty is not yet proven a reliable domain signal.** Orthogonality vs `z_sem` does **not** guarantee domain purity.  
   **Min fix:** add a cheap **domain-ID linear probe / class-balanced analysis** on `z_sty`.

3. **EMA z-score is acceptable but weak for 4 clients.** Better than divide-by-max, still coarse/arbitrary.  
   **Min fix:** compare with **rank/percentile mapping**.

4. **Baselines still not enough for paper-grade claim.** Missing **DSU/FedFA-style non-share perturbation**.  
   **Min fix:** add one such baseline on PACS.

5. **M1+M3 may not be additive.** M3 keys prototypes by **client_id**, which can overfit "client/domain identity."  
   **Min fix:** run strict **2×2 factorial** and report interaction/domain-wise effects.

**Answers:**  
- **z_sty for gap?** Better than `h`, but only **conditionally sound** after validation.  
- **EMA z-score?** **Okay heuristic**, not strong evidence.  
- **Matrix sufficient?** **Almost** for implementation triage, **not yet** for top-venue support.  
- **M1+M3 concern?** Yes: interaction and client/domain overfitting.

</details>

### Actions Planned (for implementation)
1. **双bank设计** — z_sty bank (gap度量) + h bank (AdaIN增强), 解决空间不匹配
2. **域信号验证** — 实现后先跑domain-ID linear probe验证z_sty确实编码域信息
3. **归一化对比** — 实现时同时支持EMA z-score和rank两种模式
4. **DSU基线** — EXP-072f: 本地DSU扰动（无跨客户端共享），作为non-share对照
5. **严格2×2 factorial** — M1单独/M3单独/M1+M3/baseline，完整交互分析

### Status
- **Score 6/10 ≥ 阈值6, verdict含"almost" → Review Loop通过** ✅
- 进入Phase 3: 实现 feddsa_adaptive.py
- 5个实现层面的改进纳入编码计划
