# Review Summary — ES-IBND Refine Session v1

**Problem**:FedDSA Plan A 的 cos² 解耦太 shallow(只在末梢),85+ 下游方案(含 SCPR v1/v2)都失败
**Initial Approach**:Hierarchical IBN-Augmented Decoupling — 前 2 BN 换 IBN + 层级 orth loss
**Date**:2026-04-19
**Rounds**:4 / 5
**Final Score**:**9.0 / 10**
**Final Verdict**:**READY ✅**
**Session**:`2026-04-19_hierarchical-decouple_v1/`

---

## Problem Anchor(每轮 verbatim)

(详见 FINAL_PROPOSAL.md)

---

## Round-by-Round 演化

| Round | 核心问题 | 本轮解决 | 解决? |
|-------|---------|---------|-------|
| **1** | (a) L_orth_layer 在 IN+GAP 下**数学退化**!(b) 过度声称"层级 style/semantic 分离" (c) 缺 IBN-only ablation | 仅 initial proposal | ⚠️ CRITICAL 待修 |
| **2** | 检验 R1 fix | 改 loss 为 spatial cross-correlation;收窄 claim;加 IBN-only | ✅ 数学 bug 修 |
| **3** | Loss 公式里 `F.normalize + /HW` 重复;"hierarchical" framing 过大;metrics 太 soft | 去 /HW;改名 ES-IBND;4 指标(Best/Last × ALL/AVG) | ✅ 数学正确 |
| **4** | 主决策指标 AVG Last 不是 Best;mean±std;ResNet-18 scope 界定;Plan A HSIC 说明 | 所有 exact fixes 吃下 | ✅ **READY** |

---

## Method 收敛轨迹

| Round | Method | 关键词 |
|-------|--------|-------|
| R0 Init | Hierarchical IBN + L_orth_layer(GAP cos)| "hierarchical"/broad |
| R1 | + math fix(spatial cross-correlation)+ IBN-only ablation | "fix degeneracy" |
| R2 | 锁定 spatial cosine decorrelation(无 /HW)+ 改名 ES-IBND | "cosine decorr" |
| R3 | AVG Last 主决策 + mean±std + ResNet appendix | "presentation rigor" |
| **R4** | **Final confirmation — READY** | **proposal-ready** |

---

## 主贡献最终版

> **Early-Split IN-BN Decorrelation for FedBN (ES-IBND)**:在 FedBN 框架下,将 AlexNet 前 2 个 BN 替换为 IBN(IN affine=False + BN on half channels parallel),并施加 pairwise spatial cosine decorrelation 损失(L_orth_layer = normalize + bmm + square.mean)。**无新可训练参数,~80 LOC**,作为 Plan A 末梢 cos² 的前端正则。

**核心差异化**:
- vs FDSE(CVPR 2025):FDSE 用 DSE 新参数分解,ES-IBND **无新参数**,只改 normalization
- vs IBN-Net(ECCV 2018):IBN-Net 是单机 DG,ES-IBND 加**显式正交 loss + FedBN 共存规则**
- vs Plan A:前端 2 层直接施加解耦梯度,减轻末梢层压力

---

## Final Status

| 维度 | 最终状态 |
|------|---------|
| **Anchor status** | PRESERVED(4 轮 drift warning = NONE)|
| **Focus status** | TIGHT(唯一主贡献,无 sprawl)|
| **Modernity status** | APPROPRIATELY CONSERVATIVE(reviewer 确认无需 VLM/CLIP,IBN 是正确 primitive)|
| **Math status** | VALID(R1 修 GAP 退化,R2 修 normalize + /HW,R3 锁公式)|
| **Validation status** | 3-way ablation(Plan A / IBN-only / ES-IBND)+ AVG Last 主决策 + mean±std + worst-domain |

**Remaining empirical risks(非设计问题)**:
- Plan A 可能已经是 ceiling(R4 reviewer 明说:"if gain marginal or only in Best, treat as evidence ceiling; do NOT add modules")
- IN 通道 batch-1 可能不稳(fallback:GroupNorm affine=False)
- K=4 下 IN 信号可能弱于 IBN-Net 单机 DG 场景

---

## Next Steps

1. **实现**:`FDSE_CVPR25/algorithm/feddsa_scheduled.py` 新增 IBN 类 + L_orth_layer 计算 + config 开关
2. **单测**:IBN 前向、decorrelation loss 数值、num_ibn=2 / λ=0.5 默认
3. **Codex code review** + 修 CRITICAL/IMPORTANT
4. **M0 Sanity**(10 min R=20):看 L_orth_layer 下降 + AVG Best 不崩
5. **M1 Full**(PACS R=200 × 3 configs × 3 seeds)
6. **结果判决**:
   - A.3 AVG Last ≥ 81.67 且 A.3 > A.2 → 方法成立
   - 否则 → 接受 Plan A ceiling 证据,**不加新模块**(reviewer R4 明确指示)
