# Refinement Report — Style-Conditioned Prototype Retrieval (SCPR)

**Problem**:跨域联邦学习中的 FedDSA "Share" 章节从未落地;85 次实验验证的 M3 多原型对齐和 SAS 风格感知参数聚合各自在 PACS / Office 上有效但不普适;需要一个真正能同时 work 在两种 regime 下的 Share 机制
**Initial approach**:Style-Conditioned Prototype Retrieval(SCPR),用客户端风格对原型 bank 做 attention 检索
**Date**:2026-04-18 → 2026-04-19
**Rounds**:5 / 5
**Final Score**:**9.1 / 10**
**Final Verdict**:**READY ✅**
**Reviewer**:GPT-5.4 via codex exec xhigh(session `019da01c-b0bf-7ff2-8739-f9278b9664d2`)

---

## Problem Anchor(5 轮 verbatim)

- **Bottom-line problem**:跨域 FL 中,global-mean prototype 稀释 outlier 域客户端
- **Must-solve bottleneck**:(1) global-mean washes out style;(2) SAS 参数路由 PACS 全 outlier 退化(EXP-086);(3) FedDSA Share 章节从未落地(EXP-059 / EXP-078d 均失败)
- **Non-goals**:风格作训练数据、分类器个性化、辅助损失、架构改动、新 trainable 组件
- **Constraints**:ResNet-18 / AlexNet、FedBN、R=200、3-seed、正交解耦保留、只改 InfoNCE target 权重
- **Success**:PACS AVG Best ≥ 81.5%;Office AVG Best ≥ 90.5%;drop ≤ 2%;< 100 LOC;0 新 trainable

---

## Output Files

- **Review summary**:`REVIEW_SUMMARY.md`
- **Final proposal**:`FINAL_PROPOSAL.md`
- **Score history**:`score-history.md`
- **Round raw reviews**:`round-{1..5}-review-raw.txt`
- **Round parsed reviews**:`round-{1..5}-review.md`
- **Round refinements**:`round-{1..4}-refinement.md`
- **Round prompts**:`review_prompt_r{1..5}.txt`

---

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 9                | 6 ⚠️               | 6 ⚠️                 | 8                 | 8           | 8                | 6 ⚠️            | 7.1     | REVISE  |
| 2     | 9                | 8                  | 8                    | 9                 | 9           | 9                | 7               | 8.4     | REVISE  |
| 3     | 10               | 9                  | 8                    | 9                 | 9           | 9                | 8               | 8.9     | REVISE  |
| 4     | 10               | 9                  | 8                    | 9                 | 9           | 8 ⚠️             | 8               | 8.8     | REVISE  |
| **5** | **10**           | **9**              | **9**                | **9**             | **9**       | **8**            | **9**           | **9.1** | **READY ✅** |

**轨迹**:7.1 → 8.4 (+1.3) → 8.9 (+0.5) → 8.8 (−0.1) → **9.1 (+0.3) ✅**

---

## Round-by-Round Review Record

| Round | 主要 reviewer 关切 | 本轮改动 | 结果 |
|-------|---------------------|---------|------|
| **1** | (a) retrieved-mean 和 multi-pos 两条主线数学不一致;(b) Share 会塌缩为 no-share(w_ii 主导);(c) 接口未冻结 | 仅 initial proposal,未 refine | ⚠️ CRITICAL,触发 Round 1 refinement |
| **2** | 检验 Round 1 refinement 是否正确收缩 | 收敛为 self-masked style-weighted M3;self-mask 默认;style key / bank update / missing class 接口冻结;SCPR+SAS 移到附录;grad 监控删除 | ✅ 两 CRITICAL 解决;5 分升到 8 分 |
| **3** | (a) warmup 降级为 impl detail;(b) partial-class fallback 移到附录;(c) 符号统一;(d) Venue Readiness 的 "reweighting trick" 感知 | 符号统一 `s_k`;warmup / fallback 移到 implementation note;加"decouple imperfection"机制 section | ✅ 3 simplification 全吃下;Venue Readiness +1 |
| **4** | (a) SNR equivalent 过强,claimed 而非 derived;(b) `ρ(w, -style_dist)` tautological(w 本来就是 cos 的函数) | 升级 "Minimal Derivation" 文字 + 加 tautological ρ 诊断 | ⚠️ Validation Focus -1(诊断有问题);derivation 仍未到 formal |
| **5** | R5 明确给出关键标准:要么 entropy-regularized explicit objective,要么非 tautological mechanism check | 两者都吃下:**Formal entropy-regularized MaxEnt 推导**(softmax 是唯一 Boltzmann 最优);删 tautological ρ,换 **outlier-ness correlation ρ(iso_k, gain_k)** | ✅ **READY 9.1/10** |

---

## Final Proposal Snapshot

(canonical 版本:`FINAL_PROPOSAL.md`)

**5 个 bullet 概括 SCPR 最终方案**:

1. **单一 mechanism**:Self-Masked Style-Weighted Multi-Positive InfoNCE —— 在 M3 代码路径上,把等权 positives 替换为按客户端风格相似度加权,自掩码(j ≠ k)阻断 local-only 退化
2. **Formal Derivation**:softmax-over-cosine 是 entropy-regularized noise-minimization 目标的**唯一 Boltzmann 最优解**,不是设计选择(条件:imperfect decouple + 线性噪声近似)
3. **严格退化**:τ_SCPR → ∞ 时 SCPR = M3(已验证 PACS +5.09% 下界),内置安全网
4. **复杂度预算**:0 新 trainable、1 新超参(τ_SCPR = 0.3 继承 SAS)、~30 行代码、60 GPU·h
5. **Non-tautological 机制诊断**:ρ(iso_k, gain_k) > 0(outlier 客户端从 SCPR 受益更多,gain_k 是 downstream accuracy 不是 cos 的函数)

---

## Method Evolution Highlights

1. **最重要的简化**(Round 2):从"两个变体并存(retrieved-mean + multi-pos)+ SCPR+SAS composability"的 sprawl → **唯一 mechanism**。保留 self-masked style-weighted M3,删除所有竞争分支。这个决定让数学性质自洽(uniform → M3)、公式单一、论文 claim 可控
2. **最重要的机制升级**(Round 5):从 "SCPR is like a reasonable reweighting" 的 heuristic → **entropy-regularized MaxEnt 推导**,softmax 是唯一最优解。这从根本上反驳了 "pseudo-novelty / just reweighting" 的 venue reviewer 攻击
3. **最重要的现代化/坚持**(全程):拒绝 VLM/CLIP/LLM/RL bolt-on,坚持 attention-based retrieval over shared prototype bank 作为唯一 modern primitive。Reviewer 5 轮均认可 "appropriate, not forced"

---

## Pushback / Drift Log

| Round | Reviewer 说 | 作者响应 | 结果 |
|-------|-------------|---------|------|
| 1 | retrieved-mean 和 multi-pos 同时当 contribution 会 sprawl;建议选一个 | 同意,砍掉 retrieved-mean,只留 multi-pos | ✅ 接受 |
| 1 | Self-mask 作 fallback variant | 反向接受:self-mask 改为默认(reviewer 建议) | ✅ 接受 |
| 2 | warmup 是 named component 还是 impl detail | 采纳:移到 implementation note | ✅ 接受 |
| 3 | 加 SNR 诊断 ρ(w, -style_dist) | 采纳但…… | ⚠️ Round 4 被 reviewer 指出是 tautological |
| 4 | 删 tautological ρ,换非 tautological | 采纳:换成 outlier-ness correlation ρ(iso_k, gain_k) | ✅ 接受 |
| 4 | derivation claimed not derived,建议 explicit entropy-regularized objective 或 minimal diagnostic | 采纳:写出 Formal Lagrangian + 一阶条件 + 线性近似 | ✅ 接受 |
| 5 | derivation 要限定在 residual-noise 模型,不作 unconditional theorem | 采纳 at writing stage,在 FINAL_PROPOSAL 的 Formal Derivation 段标注"注意范围" | ✅ 接受 |

**Drift 记录**:**NONE**(5 轮所有 review 均 "Problem Anchor PRESERVED")

---

## Remaining Weaknesses(诚实记录)

1. **Formal Derivation 的线性近似依赖**:`l_j ≈ c · style_dist(k, j)` 是 approximation,实际可能有高阶项。写论文时要明确"在该 residual-noise 模型下"的 scope
2. **Outlier-ness correlation 样本量小**:K = 4-5 客户端下,`ρ(iso_k, gain_k)` 的统计显著性有限,应视为 supporting evidence 而非独立证明
3. **Validation Focus 停在 8/10**:主因为 diagnostic 在小 K 下的有效性有限,不是方法问题
4. **PACS AlexNet-from-scratch 目标略紧**:81.5% threshold 高于 M3 孤立值 81.91%(但严格期望 +0.5% 风格加权增益才能通过)
5. **Office 90.5% 目标**:比 SAS 的 89.82% 要高 +0.68%,存在 SCPR 与 SAS 增益非叠加的风险(附录 composability check 的 motivation)

---

## Raw Reviewer Responses

<details>
<summary>Round 1 Review(7.1/10 REVISE)</summary>

Overall Score: **7.1 / 10**
Verdict: **REVISE**

Key Findings:
- **CRITICAL**:default algorithm 和 claimed lower-bound 不匹配(retrieved-mean 路径 uniform → FedProto,不是 M3)
- **IMPORTANT**:Share 可能塌缩为 no-share(w_{i→i} 主导)
- **IMPORTANT**:接口未冻结(style key / bank / missing class)

Simplification:
- self-mask 改为默认
- 统一 style key(不引入 z_sty 和 (μ, σ) 两套)
- 砍掉 SCPR+SAS 主 claim 和 grad 监控

(Full raw in `round-1-review-raw.txt`)

</details>

<details>
<summary>Round 2 Review(8.4/10 REVISE)</summary>

Overall Score: **8.4 / 10**
Verdict: **REVISE**

Reviewer summary:
> This is materially stronger than Round 1. The anchor is preserved, the method is now singular and implementable, and the use of a modern attention-style primitive is appropriate rather than trendy. The remaining issue is not complexity but sharpness: top-venue success will depend on whether style-weighted positives produce a clear, repeatable win over uniform M3, not merely a tidy reformulation.

Simplification:
- warmup 降级为 implementation guard
- Partial-class fallback 移 implementation note
- 用一个符号 `s_k` 贯穿全文

(Full raw in `round-2-review-raw.txt`)

</details>

<details>
<summary>Round 3 Review(8.9/10 REVISE)</summary>

Overall Score: **8.9 / 10**
Verdict: **REVISE**

Signal-to-Noise Motivation:
> Improved materially, but one more pass needed. Weakest when saying "equivalent to SNR importance weighting" — more claimed than derived.

Two options for READY:
- weaken to "can be interpreted as a bias-control / SNR-aware weighting"
- add minimal mechanism diagnostic showing marginal usefulness decays with style distance

Simplification / Modernization / Drift / <7 fix:ALL NONE

(Full raw in `round-3-review-raw.txt`)

</details>

<details>
<summary>Round 4 Review(8.8/10 REVISE)</summary>

Overall Score: **8.8 / 10**(Validation Focus -1)
Verdict: **REVISE**

Key points:
- Pseudo-novelty partially resolved:derivation justifies direction but not exact softmax form
- `ρ(w, -style_dist)` is close to tautological(w 构造上就是 cos 的函数)

Simplification: drop tautological diagnostic or replace with non-tautological check

(Full raw in `round-4-review-raw.txt`)

</details>

<details>
<summary>Round 5 Review(9.1/10 ✅ READY)</summary>

Overall Score: **9.1 / 10**
Verdict: **READY** ✅

Reviewer summary:
> This is now a focused, elegant method proposal with one dominant contribution, a concrete implementation path, and an appropriately modern primitive. The formal derivation and the non-tautological mechanism check are enough to move SCPR out of the "nice reweighting trick" bucket and into a defensible mechanism paper, provided the final writeup keeps the derivation scoped to the stated residual-noise model.

Formal Derivation vs. "Just Reweighting":
> Yes. This is the first revision that adequately resolves the pseudo-novelty concern.

Mechanism Evidence from Outlier-ness Correlation:
> Yes, as supporting evidence. ρ(iso_k, gain_k) is non-tautological because gain_k is downstream accuracy improvement over a style-agnostic baseline.

Simplification / Modernization / Drift / <7 fix:ALL NONE

(Full raw in `round-5-review-raw.txt`)

</details>

---

## Next Steps

- **推荐**:`/experiment-plan` 将 Claim A / B / C 细化成完整 claim-driven 实验路线图(runs、ablations、metrics、fallbacks、exact compute)
- **实现**:`FDSE_CVPR25/algorithm/feddsa_scheduled.py` 新增 ~30 行 SCPR attention;单测覆盖 self-mask + τ 极限 + renormalize
- **代码 review**:codex exec 对实现做 gpt-5.4 审核
- **跑实验**:单卡 60 GPU·h,5 天 end-to-end 完成主表 + 附录
- **论文写作 caveat**(reviewer R5 提醒):Formal Derivation 段需明确限定"under imperfect decoupling with linear style-dependent noise approximation",不作 unconditional theorem
