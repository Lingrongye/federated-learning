# Review Summary — SCPR Refine Session v1

**Problem**:跨域联邦学习中的 FedDSA "Share" 章节从未落地,需要一个真正能 work 的 Share 机制
**Initial approach**:Style-Conditioned Prototype Retrieval(SCPR),用客户端风格对原型 bank 做 attention 检索
**Date**:2026-04-18 → 2026-04-19
**Rounds**:5 / 5(达到 MAX_ROUNDS 的同时拿到 READY)
**Final Score**:**9.1 / 10**
**Final Verdict**:**READY** ✅

---

## Problem Anchor(每轮复制,不变)

- **Bottom-line**:跨域 FL 中,global-mean prototype 稀释 outlier;SAS 在 PACS 全 outlier 退化;FedDSA Share 章节从未成功
- **Must-solve**:
  1. global-mean washes out style
  2. SAS 参数路由退化(EXP-086)
  3. Share 章节缺口
- **Non-goals**:风格作训练数据、分类器个性化、辅助损失、架构改动、新 trainable 组件
- **Constraints**:ResNet-18 / AlexNet,FedBN,R=200,3-seed,正交解耦保留,只改 InfoNCE target 权重
- **Success**:PACS ≥ 81.5% / Office ≥ 90.5% / drop ≤ 2% / <100 LOC / 0 新 trainable

---

## Round-by-Round Resolution Log

| Round | 主要 reviewer 关切                                                                                                                    | 本轮简化/现代化                                                                                                    | 解决?            | 剩余风险                                              |
| ----- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------- |
| **1** | ① 两个变体(retrieved-mean + multi-pos)数学不一致;② Share 可能退化为 no-share;③ 接口未冻结                                                            | —                                                                                                           | ⚠️ CRITICAL 待修 | 主 mechanism 含糊                                    |
| **2** | 收敛为单一 "self-masked style-weighted M3";self-mask 默认;style key / bank update / missing class 三个接口冻结;SCPR+SAS 移到附录;删 grad 监控         | 砍掉 retrieved-mean 分支;统一公式                                                                                   | ✅ CRITICAL 解决  | 数学泛化 claim 仍过宽                                    |
| **3** | ① style_proto_k 精确定义;② warmup 是否需要;③ partial-class fallback;④ Venue Readiness "just reweighting" 感知                               | 符号统一 `s_k`;warmup 降级为 implementation detail;fallback 移到附录;加 SNR 机制论证小节                                      | ✅ IMPORTANT 解决 | SNR 是 claimed 而非 derived                          |
| **4** | ① SNR 论证 claimed not derived;② `ρ(w, -style_dist)` 诊断 tautological                                                                | 删除 tautological 诊断;补"Minimal Derivation"                                                                    | ⚠️ 部分          | Validation Focus 下降;derivation 未完全 derive softmax |
| **5** | ① Formal derivation(entropy-regularized MaxEnt);② 换非 tautological 诊断(outlier-ness correlation);③ derivation 限定在 residual-noise 模型 | MaxEnt 拉格朗日一阶条件 + 线性近似 → softmax-over-cosine 是 **unique Boltzmann 最优**;`ρ(iso_k, gain_k)` 替换 tautological ρ | ✅ READY        | 无阻塞;写作时 derivation 需限定 scope                      |

---

## Overall Evolution

- **方法如何变得更具体**:
  - Round 0:两个变体(retrieved-mean + multi-pos)并存,数学性质混乱
  - Round 1:统一为 **Self-Masked Style-Weighted M3**,self-mask 默认
  - Round 2:符号、bank 时机、missing class 全部冻结
  - Round 3:mechanism 从 heuristic SNR 论证升级为"decouple imperfection"推导
  - Round 4:**Formal entropy-regularized MaxEnt 推导**,softmax-over-cosine 是唯一最优

- **主 contribution 如何变得更聚焦**:
  - Round 0:3 个 contributions 共存(retrieval + multi-pos 泛化 + SCPR+SAS composability)
  - Round 1 起:唯一主 contribution;SCPR+SAS 移到附录;双重泛化 claim 删除

- **不必要复杂度如何删除**:
  - 删除:retrieved-mean 变体、grad 监控、tautological ρ 诊断、双重泛化 claim、warmup 命名组件
  - 保留:单一 loss 公式、self-mask、per-class renormalize、τ 敏感性 + outlier-ness 诊断

- **Modern leverage 如何成型**:
  - 始终保持 attention-based retrieval(Transformer 原语)作为核心;拒绝 VLM / CLIP / RL bolt-on
  - 最终 reviewer 评:"Attention/Boltzmann-style retrieval primitive is the right modern move here"

- **Drift 如何避免**:所有 5 轮均 `Drift Warning: NONE`

---

## Final Status

| 维度 | 最终状态 |
|------|---------|
| **Anchor status** | **PRESERVED**(从未漂移) |
| **Focus status** | **TIGHT**(单一 mechanism,从 Round 2 起) |
| **Modernity status** | **APPROPRIATELY FRONTIER-AWARE**(attention 原语足够,无需 VLM) |
| **Derivation status** | **DERIVED**(Formal MaxEnt,不是 heuristic) |
| **Diagnostic status** | **NON-TAUTOLOGICAL**(outlier-ness correlation) |

**最强部分(final method)**:
1. softmax-over-cosine 权重是**推导结果**,不是设计选择(entropy-regularized MaxEnt)
2. uniform-w 严格退化为 **M3(+5.09% 已验证下界)**,内置安全网
3. self-mask 阻断 local-only 退化(PACS SAS 失败的根因)
4. 参数共享 → 跨域共识不被破坏(PACS 普适);风格加权 → 邻域先验(Office 增益)
5. 0 新 trainable、~30 行代码、1 新超参(继承 SAS τ=0.3)

**Remaining weaknesses**:
- Formal derivation 依赖"风格噪声随 style_dist 线性增长"的近似假设,实际情形可能有高阶项
- `ρ(iso_k, gain_k)` 在 K=4-5 客户端下是 supporting evidence 而非独立证明
- Validation Focus 8(仍低于 9),原因是诊断有效性在小 K 下有限
- Venue submission 时 derivation 段落需明确限定 "在 residual-noise 线性模型下"

---

## Next Steps

1. **推荐**:运行 `/experiment-plan` 将 Claim A/B/C 细化为完整实验路线图(runs、ablations、metrics、fallback、compute)
2. **实现**:在 `FDSE_CVPR25/algorithm/feddsa_scheduled.py` 上 ~30 行代码;单测覆盖 self-mask + τ 极限 + renormalize
3. **审核**:codex exec 做代码 review
4. **跑实验**:5 天 end-to-end,单卡 60 GPU·h
5. **写作 caveat**:论文 mechanism section 的 derivation 要明确限定 "under imperfect decoupling with linear style-dependent noise approximation"
