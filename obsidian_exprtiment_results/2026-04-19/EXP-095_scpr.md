# EXP-095: SCPR — Self-Masked Style-Weighted Multi-Positive InfoNCE

**日期**:2026-04-18 启动 / 2026-04-19 完成
**算法**:`feddsa_scheduled` (scpr=1/2)
**服务器**:seetacloud2(单卡 24GB)
**状态**:❌ **SCPR 主 claim 全部证伪**

## 这个实验做什么(大白话)

> 把 M3 "所有同类域原型等权拉近"改成"按风格相似度加权拉近",自己不算(self-mask)。
>
> - **scpr=0**:关闭,等价 Plan A orth_only(无 InfoNCE)
> - **scpr=1**:uniform 多原型拉近(M3 风格,不用 style)
> - **scpr=2**:按风格相似度加权(SCPR 主方法,OURS)
>
> 数学保证:`scpr_tau → ∞` 时 scpr=2 严格等于 scpr=1。

## 一句话结论(诚实)

> **Claim A/B/C 全部证伪**。Plan A orth_only(无 InfoNCE)仍然是最强配置。在 feddsa_scheduled 1024d z_sem 路径下,**任何形式的 domain-indexed multi-positive InfoNCE(包括 uniform M3 和 style-weighted SCPR)都不如 Plan A**。整个 "Share 章节"(EXP-059 z_sem AdaIN / EXP-078d h-AdaIN→InfoNCE / EXP-095 SCPR)的所有尝试都失败了。

## 🏆 完整结果(3-seed mean {2, 15, 333}+ 对照行 + Δ 行)

### Claim A(主):PACS 全 outlier,SCPR > M3 uniform

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Art | Cart | Photo | Sketch |
|------|------|---------|---------|---------|---------|-----|------|-------|--------|
| **Plan A orth_only**(EXP-080 对照) | **mean** | 80.41 | 79.42 | **82.31** | 81.17 | 65.2/62.1 | 87.3/86.3 | 82.6/81.2 | 90.5/88.0 |
|  | 2 | 81.87 | 80.37 | 83.35 | 82.04 | 70.6/65.2 | 86.8/86.3 | 82.6/81.4 | 90.3/88.5 |
|  | 15 | 80.08 | 79.65 | 82.14 | 81.24 | 62.3/61.3 | 88.9/87.2 | 83.2/82.6 | 90.3/87.5 |
|  | 333 | 79.28 | 78.23 | 81.44 | 80.23 | 62.7/59.8 | 86.3/85.5 | 82.0/79.6 | 90.8/88.0 |
| **A.2 M3 uniform**(scpr=1, OURS) | **mean** | 79.99 | 78.97 | **81.84** | 80.84 | 65.4/— | 85.8/— | 82.8/— | 90.2/— |
|  | 2 | 81.42 | 79.80 | 82.84 | 81.34 | 70.1 | 85.5 | 83.2 | 90.1 |
|  | 15 | 78.58 | 78.39 | 80.73 | 80.43 | 60.8 | 87.2 | 80.8 | 90.3 |
|  | 333 | 79.95 | 78.73 | 81.94 | 80.74 | 65.2 | 84.6 | 84.4 | 90.3 |
| **A.3 SCPR τ=0.3**(scpr=2, OURS 主) | **mean** | 79.62 | 78.44 | **81.60** | 80.50 | 63.1/— | 86.5/— | 82.6/— | 90.2/— |
|  | 2 | 80.97 | 79.42 | 82.84 | 81.54 | 65.7 | 85.5 | 85.6 | 91.3 |
|  | 15 | 79.42 | 79.27 | 81.24 | 81.14 | 61.8 | 88.9 | 81.4 | 89.3 |
|  | 333 | 78.48 | 76.64 | 80.74 | 78.83 | 61.8 | 85.0 | 80.8 | 90.1 |
| **Δ A.3 − Plan A** | — | **−0.79** | −0.98 | **−0.71** ❌ | −0.67 | −2.1 | −0.8 | ±0 | −0.3 |
| **Δ A.3 − A.2(核心 claim!)** | — | **−0.37** | −0.53 | **−0.24** ❌ | −0.34 | −2.3 | +0.7 | −0.2 | ±0 |
| **Δ A.2 − Plan A**(M3 自身是否有效?) | — | **−0.42** | −0.45 | **−0.47** ❌ | −0.33 | +0.2 | −1.5 | +0.2 | −0.3 |

**Claim A verdict**:
- ❌ **SCPR τ=0.3 < M3 uniform**(Δ = −0.24% AVG Best)—— 主 claim 证伪
- ❌ **M3 uniform 本身也不如 Plan A**(Δ = −0.47%)—— 整个 "multi-positive InfoNCE" 路径证伪
- ❌ **所有 SCPR 变体都不如 Plan A**

### Claim B(主):Office 单 outlier,SCPR ≥ SAS

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Caltech | Amazon | DSLR | Webcam |
|------|------|---------|---------|---------|---------|---------|--------|------|--------|
| **Plan A orth_only**(EXP-083) | **mean** | 88.61 | 87.30 | **82.55** | 81.35 | 72.6/70.2 | 90.9/88.1 | 100.0/97.8 | 94.3/93.1 |
|  | 2 | 86.45 | 86.45 | 78.19 | 78.19 | 67.0/66.1 | 86.3/83.2 | 100.0/100.0 | 96.6/96.6 |
|  | 15 | 89.59 | 87.75 | 83.74 | 81.36 | 74.1/70.5 | 91.6/87.4 | 100.0/100.0 | 96.6/93.1 |
|  | 333 | 89.81 | 87.69 | 85.72 | 84.52 | 76.8/74.1 | 94.7/93.7 | 100.0/93.3 | 89.7/89.7 |
| **SAS τ=0.3**(EXP-084) | **mean** | 89.82 | 88.28 | **84.40** | 83.07 | 75.0/73.8 | 91.6/88.4 | 100.0/97.8 | 95.4/93.1 |
|  | 2 | 88.61 | 87.64 | 81.76 | 80.17 | 71.4/68.8 | 87.4/85.3 | 100.0/100.0 | 96.6/96.6 |
|  | 15 | 90.11 | 88.16 | 84.14 | 82.15 | 73.2/73.2 | 91.6/86.3 | 100.0/100.0 | 100.0/93.1 |
|  | 333 | 90.74 | 89.03 | 87.31 | 86.89 | 80.4/79.5 | 95.8/93.7 | 100.0/93.3 | 89.7/89.7 |
| **B.3 SCPR τ=0.3**(OURS) | **mean** | 88.76 | 87.14 | **83.21** | 81.88 | 74.4/72.6 | 90.9/87.4 | 100.0/97.8 | 94.3/90.8 |
|  | 2 | 87.38 | 86.63 | 80.16 | 78.58 | 71.4/67.9 | 87.4/82.1 | 100.0/100.0 | 96.6/96.6 |
|  | 15 | 88.64 | 86.43 | 82.94 | 81.35 | 74.1/73.2 | 89.5/86.3 | 100.0/100.0 | 96.6/86.2 |
|  | 333 | 90.25 | 88.36 | 86.51 | 85.70 | 77.7/76.8 | 95.8/93.7 | 100.0/93.3 | 89.7/89.7 |
| **Δ B.3 − SAS(核心 claim!)** | — | **−1.06** | −1.14 | **−1.19** ❌ | −1.19 | −0.6 | −0.7 | ±0 | −1.1 |
| **Δ B.3 − Plan A** | — | +0.15 | −0.16 | +0.66 | +0.53 | +1.8 | ±0 | ±0 | ±0 |

**Claim B verdict**:
- ❌ **SCPR AVG Best 83.21 < SAS 84.40**(Δ = **−1.19%**)—— 主 claim 证伪
- ⚠️ SCPR 比 Plan A 微弱 +0.66,但不如 SAS → **没超过现有最强 Office 方案**

### Claim C(机制):τ 敏感性 + 非 tautological 诊断

| τ_SCPR | PACS AVG Best 3-seed mean | Δ vs Plan A 82.31 | 评价 |
|--------|---------------------------|-------------------|------|
| 0.1 | 82.07 | −0.24 | 最优但仍不 beat Plan A |
| 0.3 | 81.60 | −0.71 | |
| 1.0 | 81.27 | −1.04 | 谷底 |
| 3.0 | 81.91 | −0.40 | |

**τ 扫形状**:双峰(τ=0.1 和 τ=3.0 都高,中间低)—— **不符合"居中 τ 最优"预期**,违反 Formal Derivation 的线性噪声模型假设。说明 style weighting 本身的机制在 4-client 下不 active。

**Outlier-ness correlation ρ(iso_k, gain_k)** 诊断**未计算**(因为 Claim A/B 均已证伪,诊断价值不再)。

### 附录:SCPR + SAS Composability(Office)

| 配置 | AVG Best 3-seed mean | Δ vs SAS only | Δ vs SCPR only |
|------|---------------------|---------------|----------------|
| SAS only (B.2) | 84.40 | — | +1.19 |
| SCPR only (B.3) | 83.21 | −1.19 | — |
| **SCPR + SAS (附录)** | **82.68** | **−1.72** ❌ | **−0.53** ❌ |

SCPR+SAS **比单独任一个都差**,说明两机制**不正交**,叠加有害。

## 🔍 根因分析

### 为什么 SCPR 全面失败?

1. **客户端数 K=4 太少**:
   - self-mask 后 attention key 只剩 K−1=3 个
   - style_proto 的 cos 相似度区分度极低(高维 1024d 下 4 客户端 cos 几乎都 0.5-0.7)
   - R5 refine 时 reviewer 已预警 `H(w_k) → log(K-1)` attention collapse 风险,实际发生

2. **Style bank 用 pool5 μ 不是 z_sty**:
   - FINAL_PROPOSAL 约定 `s_k := normalize(E[z_sty])`,实际代码复用 SAS/EXP-084 的 style_bank(1024d pool5 均值)
   - 如果真用 z_sty(128d 解耦后风格),区分度可能更好,但会引入新 bank,打破"复用现有接口"原则

3. **Formal Derivation 的线性噪声近似失效**:
   - 假设 `l_j ≈ c · style_dist(k, j)` 在 4 clients 下不成立(style_dist 分布不够均匀,线性拟合极差)
   - 正交解耦 `cos²(z_sem, z_sty) → 0` 的残余噪声分布也未按理论

4. **M3 uniform 本身也不 work**:
   - EXP-072 的 M3 +5.09% 在 feddsa_adaptive.py 128d z_sem 成立
   - 换到 feddsa_scheduled.py 1024d z_sem,**M3 uniform 反而 −0.47%**
   - 说明 "M3 +5.09%" **不可跨架构泛化**,之前 cite 该数字作为 SCPR 下界是 **误导**

5. **Plan A sm=0 w_aux=0 的优势**:
   - 在 feddsa_scheduled 里 Plan A 根本不跑 InfoNCE
   - 任何加上 InfoNCE 的配置(M3 / SCPR)都是**新增 noise**
   - Refine 时把 Plan A 写成 baseline 是假设"加 InfoNCE 会 help",实际相反

### 整个 FedDSA "Share 章节" 的累计失败

| 方法 | 结果 | 失败模式 |
|------|------|---------|
| EXP-059 z_sem AdaIN + CE + InfoNCE | PACS −2.54% | 破坏 z_sem 语义 |
| EXP-078d h-AdaIN → InfoNCE only | NaN 崩溃 | 梯度爆炸 |
| **EXP-095 SCPR(domain-indexed multi-pos + style weight)** | **PACS −0.71, Office −1.19** | **Plan A 本身就最优** |

**结论**:FedDSA 的 "Share" 章节 **不存在**。任何跨客户端的原型共享机制都伤害性能。**Plan A orth_only 是真实最强配置**。

## 📋 论文叙事调整

### 原 "Decouple-Share-Align" 三章叙事 → 调整为 "Decouple-Only" 单章

| 原章节 | 状态 | 建议 |
|--------|------|------|
| **Decouple**(正交 cos² + HSIC)| ✅ 有效 | 保留作主贡献 |
| **Share**(风格共享) | ❌ 所有尝试失败 | **删除章节**,叙事改为"我们尝试了多种跨客户端 Share 机制,全部失败;说明在 4-client FedBN 下 orth_only 即最优" |
| **Align**(InfoNCE) | ❌ 加 InfoNCE 反而 hurt | 删除 |

### 可选 salvage:

1. **EXP-084 SAS(Office 专属)**:仍是目前 Office 唯一正向贡献(+1.21% over Plan A 单 outlier 场景)
2. **EXP-080 Plan A 正交解耦**:PACS 主贡献(+2% 以上 over FedAvg/FedBN)
3. **论文主题改为**:"Orthogonal Feature Decoupling + Optional Style-Aware Parameter Aggregation for Cross-Domain FL"
   - Decouple 是通用,适用 PACS/Office
   - SAS 是 Office 专属加分项
   - 不再提 "Share" 章节

## 📊 实验统计

- **总 runs**:21(15 PACS + 6 Office)
- **总 GPU·h**:~50(PACS 15 并行 8h ≈ 8h wall,Office 6 并行 1h)
- **启动**:2026-04-18 20:35
- **完成**:2026-04-19 10:55
- **wall-clock**:14h 20min
- **NaN / OOM / 崩溃**:0(代码非常稳定,仅结果不支持 claim)

## 变更记录

- 2026-04-18 初版设计(GPT-5.4 refine READY 9.1/10)
- 2026-04-19 实现 + codex review MUST_FIX 3 修 + 15/15 单测 + 部署
- 2026-04-19 22 runs 完成 + 回填完整数据 + verdict

## 📎 相关文件

- FINAL_PROPOSAL: `obsidian_exprtiment_results/refine_logs/2026-04-18_SCPR_v1/FINAL_PROPOSAL.md`
- EXPERIMENT_PLAN: `obsidian_exprtiment_results/refine_logs/2026-04-18_SCPR_v1/EXPERIMENT_PLAN.md`
- 代码:`FDSE_CVPR25/algorithm/feddsa_scheduled.py`(scpr / scpr_tau 参数)
- 单测:`FDSE_CVPR25/tests/test_scpr.py`(15/15 通过)
- JSON records:`seetacloud2:/root/autodl-tmp/federated-learning/FDSE_CVPR25/task/{PACS_c4,office_caltech10_c4}/record/*scpr*.json`
