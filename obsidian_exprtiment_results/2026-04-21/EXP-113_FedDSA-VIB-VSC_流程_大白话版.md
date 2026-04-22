# EXP-113 FedDSA-VIB + VSC 完整流程 — 大白话版

**对应技术版**: [EXP-113_FedDSA-VIB-VSC_流程_完整技术版](EXP-113_FedDSA-VIB-VSC_流程_完整技术版.md)
**一句话**: 用"信息瓶颈 + 原型引力"给 z_sem 瘦身,强迫它只保留 class,风格被挤到 z_sty,解决非线性 probe 泄漏.

---

## 🤔 为什么要做这个实验?

### 背景: 之前失败 3 次了

| 尝试 | 方法 | 结果 |
|------|------|------|
| #1 CDANN (EXP-108) | 给 z_sem 加 GRL 压 domain | ❌ z_sty MLP probe = 0.96 (全破),CDANN 把 class 灌进了 z_sty |
| #2 Symmetric CDANN | CDANN + z_sty 也加约束 | ❌ Codex RETHINK 5.8/10 |
| #3 Purify-Share | CLUB 互信息压 | ❌ Codex RETHINK 6.8/10 |
| #4 FedDSA-Swap | feature 空间交换 style | ❌ Codex RETHINK 6.2/10,L_swap 本质和 L_aug 一样 |

**最新数据** (EXP-111 lo=3 强正交):
- linear probe(z_sty→class) = 0.20 ✅
- **MLP-256 probe = 0.71** ❌ (非线性泄漏没解)

### 根本问题

所有 loss 级方法都压不住 **非线性 class 泄漏**:
- z_sem 里装了太多 (class + 风格细节),非线性 probe 一挖就破
- **需要一个直接"压缩 z_sem 信息量"的机制**,不是让 z_sem 和 z_sty 几何正交

---

## 💡 我们的方案 A (FedDSA-VIB) + B (FedDSA-VSC)

### 核心比喻: 瘦身 + 原型

**想象**:
- z_sem 是一个**大背包**,装了很多东西 (class + 背景 + 风格 + 细节)
- 我们想让它**只装"这是什么"** (class)
- **方案**: 给每个 class 在空间里画个"家" (**语义原型**),要求:
  1. z_sem 必须落在本类"家"附近 (InfoNCE 已有)
  2. **z_sem 的"背包"要瘦身** — 不准装太多信息 (VIB 新加)

**VIB 瘦身怎么做**:
- 每个样本的 z_sem 变成一个**模糊范围** (z_sem ~ N(μ, σ²))
- 这个模糊范围要**靠近本类的家** `N(prototype[y], σ²_prior)`
- 模糊得太清晰 → 装信息多 → KL 大 → 惩罚
- 模糊得刚好能分类就行 → KL 小 → 奖励

**结果**:
- class 本质信息留在 `μ_sem ≈ prototype[y]`
- 风格等多余信息被压成噪声,消失在 `σ_sem`
- 风格信息无处可去 → 被迫进 z_sty (真解耦)

### 方案 A vs B 区别 (就差一个 loss)

| Loss | 方案 A (VIB) | 方案 B (VSC) |
|------|:------------:|:------------:|
| 原主任务 CE | ✅ | ✅ |
| 原 L_aug (h-space AdaIN) | ✅ | ✅ |
| 原 L_orth + L_HSIC | ✅ | ✅ |
| **新 L_VIB** (瘦身) | ✅ | ✅ |
| Prototype pull | InfoNCE (原) | **SupCon** (升级) |

- **A**: 最小改动,只加 VIB
- **B**: A + InfoNCE→SupCon (多正样本对比)

### 为什么两个都做

- A 告诉我们 **VIB 单独够不够**
- B-A 差距告诉我们 **SupCon 升级到底有没有用**
- 这是个 ablation,不是两个方案竞争

---

## 🛠️ 4 个"大坑修复" (逻辑自洽的关键)

Codex 评审指出 VIB 天真实现会踩 4 个坑,我们都修了:

### ① 鸡和蛋问题 (Prior 跟着 z_sem 一起学 → 崩)
**问题**: "家"(prototype) 自己还在被 VIB 压着学,同时又被当"锚点" → 自己追自己,塌缩成一个点
**修**: **EMA 滞后 1 步 + 梯度断开**
- prototype 每轮平滑更新 (β=0.99)
- 用作 prior 时 `.detach()` 不让梯度回流
- 比喻: "家人搬家,但不告诉路标 — 路标慢慢跟上,不跟着你瞎跑"

### ② 不确定性 FedAvg 污染
**问题**: σ 参数被跨客户端平均 → 混淆了各 domain 自己的不确定性
**修**: **σ_head 本地化** (类似 FedBN 做法)
- `log_var_head` 参数**不聚合**,每客户端本地维护
- 比喻: "你的疑惑程度别人不懂,别强求一致"

### ③ 退化到 lookup 表
**问题**: 如果 z_sem 直接等于 prototype[y] 就能满足 KL → 退化成查表,失去 class 内区分力
**修**: **每 class 可学习 σ_prior**
- `log_sigma_prior[c]` 对每个 class 独立学习
- 监控 `intra_class_z_std`: 如果太小 → 预警 collapse

### ④ HSIC 和 VIB 可能冗余
**问题**: HSIC 和 VIB 都在压缩依赖,可能做重复工作
**修**: 加 **ablation** (去 HSIC 看 VIB 是否独立 work)

---

## 🔬 诊断指标 — 总共 50+

### 第一层: 训练时每 5 轮 (原 21 + 新 30+)

**VIB 专属** (A/B 共用):
- `KL_mean` — 压缩强度,健康 0.5-2.0
- `σ_sem_mean` / `σ_sem_max` — 瘦身程度,应 0.1-1.0
- `z_sem_to_prior_cos` — z_sem 靠家程度,应 > 0.7
- `rate_R` / `distortion_D` — 信息论 Pareto

**SupCon 专属** (只 B):
- `pos_sim_mean` — 同类相似度,应 > 0.7
- `neg_sim_mean` — 异类相似度,应 < 0.2
- `alignment` / `uniformity` — Wang-Isola 理论指标
- `n_positive_avg` — batch 内正样本数

**Prototype**:
- `prototype_drift_L2` — 每轮原型位移,应衰减
- `intra_class_z_std` 🔥 **KL-collapse 预警** (< 0.1 报警)
- `inter_class_proto_min_cos` — 类间分离度

**新 Round-2 诊断** (从 Codex review 加):
- `R_per_domain` — 每个 domain 的 KL (domain-conditional rate)
- `R_std_across_domains` — domain 间 KL 差异
- `irm_grad_var` — IRM 式跨域梯度方差 (越低越 domain-invariant)
- `kl_collapse_alert` (bool) — 综合预警

**梯度诊断**:
- `grad_cos(CE, VIB)` — CE 和 VIB 是否打架
- `grad_norm_{CE, VIB, SupCon}` — 相对量级

### 第二层: 训练时每 20 轮的 proxy probe ⭐ 最重要

冻结当前 encoder,用 **500 样本子集**跑 probe:

| Probe | 意思 | 健康预期 |
|-------|------|:-------:|
| `proxy_probe_sty_class_linear` | z_sty 能否线性预测 class | ↓ < 0.3 |
| `proxy_probe_sty_class_MLP64` | MLP-64 非线性能否挖出 | ↓ < 0.4 |
| **`proxy_probe_sty_class_MLP256`** | MLP-256 能否挖出 🔥 | **↓ < 0.5** (核心) |
| `proxy_probe_sem_class_linear` | z_sem 分类能力 | ≈ train acc |
| `proxy_probe_sty_domain_linear` | z_sty 有没有 domain 信息 | > 0.8 (不是灭活) |
| `proxy_probe_sem_domain_linear` | z_sem 漏不漏 domain | < 0.5 |

**关键**: 训练时就看 MLP-256 probe 下降,**不用等 R200 训练完才知道**。

### 第三层: 训练后 Moyer sweep (FL 领域首次)

冻结 encoder 后跑 0/1/2/3 层 adversary (Moyer 2018 黄金标准):
- 每层 hidden=64, BN, Adam lr=0.001, absolute error loss
- + Hewitt-Liang **selectivity** (防 probe 过拟合作弊)

---

## 📊 实验矩阵 (2×2 设计)

| | 无 VIB | 有 VIB |
|---|:---:|:---:|
| InfoNCE | M0 orth_only (已跑) | **M4 FedDSA-VIB (A)** 🆕 |
| SupCon | **M6 orth+SupCon** 🆕 | **M5 FedDSA-VSC (B)** 🆕 |

**这个 2×2 能回答**:
- VIB 独立贡献 = (A − orth_only) + (B − M6) / 2
- SupCon 独立贡献 = (M6 − orth_only) + (B − A) / 2
- 交互效应 = B − (orth + VIB_solo + SupCon_solo)

### 完整 Run 数

- **Datasets**: PACS + Office (每个 dataset 2 倍)
- **Seeds**: 10 seeds (2, 15, 333, 42, 7, 100, 201, 500, 777, 999)
- **新 Method**: M4, M5, M6 × PACS + Office × 10 seeds = **60 runs**
- 外加 baseline 补齐: M0 补 7 seeds × 2 datasets = **14 runs**
- **总: ~74 runs**

### GPU 预算
- 74 run × 3h × 1/6 并发 = **~37h wall**
- 约 1.5 天完成

---

## 🎯 预期结果

| Metric | orth_only | CDANN (坏) | A (VIB) | B (VSC) |
|--------|:--------:|:----------:|:-------:|:-------:|
| linear probe (sty→class) | 0.24 | 0.96 | **0.10-0.18** | 0.08-0.15 |
| MLP-256 probe 🔥 | 0.71 | 0.96 | **0.45-0.55** | **0.40-0.50** |
| 3-layer Moyer probe | ~0.75 | ~0.90 | 0.50-0.60 | 0.45-0.55 |
| PACS AVG Best | 80.64 | 80.08 | 80.5-81.5 | **81.0-82.0** |
| Office AVG Best | 89.09 | 89.54 | 89.0-89.5 | 89.3-89.8 |

**注**: orth_only std=1.46,单纯看 accuracy 可能**统计不显著**。**主卖点是 probe 大幅下降**,accuracy 只要 non-inferior 即可。

---

## 🎯 成败判定

```
(A) MLP-256 probe ≤ 0.50 + PACS acc ≥ 80.0 + 10-seed std < 1.5
    → ✅ 全胜,probe-validated 解耦 + accuracy non-inferior,写 paper

(B) MLP-256 probe ≤ 0.55 但 acc 掉到 79-80
    → ⚠️ 机制 work 但伤 accuracy,调 λ_IB 降低
    
(C) MLP-256 probe > 0.65 (没压住)
    → ❌ 方案再失败,走诊断论文路线

(D) KL-collapse 预警触发 (intra_class_z_std < 0.1)
    → ⚠️ Prior 退化 lookup,σ_prior 要 ablate
```

---

## 📂 当前进展

### ✅ 已完成 (Phase 5c-1/2/3/4)
- 文献调研 5 论文精读 (DSN / MUNIT / von Kügelgen / Moyer / FediOS)
- 50+ 诊断指标综合调研
- Round-2 Codex review (RETHINK 6.1/6.0) + 4 个 fix 全纳入
- Round-3 修订方案 (+ 2×2 设计 + Moyer sweep)
- **代码完成** 3 个核心模块:
  - `algorithm/common/vib.py` (VIBSemanticHead + closed-form KL + EMA prototype)
  - `algorithm/common/supcon.py` (SupCon loss + 诊断)
  - `algorithm/common/diagnostic_ext.py` (KL-collapse / R_d / IRM grad var)
- **单元测试 35/35 全绿** ✅

### 🔄 进行中 (Phase 5c-剩余)
- algorithm/feddsa_vib.py (方案 A 集成)
- algorithm/feddsa_vsc.py (方案 B 集成)
- algorithm/feddsa_supcon.py (M6 集成)
- Server 改动 (log_var_head 加入 private_keys)
- 6 个 config 文件 (PACS + Office × 3 method)
- smoke test (8 样本 1 round)

### 📋 待做
- **Phase 5d**: Codex code review (代码审核)
- **Phase 5e**: 本地 smoke test (跑通 8 样本)
- **Phase 5f**: 部署到服务器 (需用户最终确认)

---

## 📂 相关文件

- **技术版**: [EXP-113_FedDSA-VIB-VSC_流程_完整技术版](EXP-113_FedDSA-VIB-VSC_流程_完整技术版.md)
- Round-3 修订方案: `refine-logs/2026-04-22_FedDSA-Swap/round-3-revised-proposal.md`
- Round-2 Codex review: `refine-logs/2026-04-22_FedDSA-Swap/round-2-review.md`
- Self-review: `refine-logs/2026-04-22_FedDSA-Swap/round-2-self-review.md`
- 文献调研: `.planning/disentangle_research/DISENTANGLEMENT_METRICS_SURVEY.md`
- VIB 代码: `FDSE_CVPR25/algorithm/common/vib.py`
- SupCon 代码: `FDSE_CVPR25/algorithm/common/supcon.py`
- 诊断扩展: `FDSE_CVPR25/algorithm/common/diagnostic_ext.py`
- 单测: `FDSE_CVPR25/tests/test_{vib,supcon,diagnostic_ext}.py` (35/35 ✅)

---

## 一句话总结当前状态

**4 次 RETHINK 后定型为 FedDSA-VIB/VSC. 用信息瓶颈替代 swap 方向, 理论背书 Moyer 2018, 2×2 ablation 设计, 50+ 诊断指标, 10 seeds × 3 new methods × 2 datasets = 60 runs. 核心 3 模块代码就位 + 35 测试全绿, 还差 3 个 algorithm 集成文件 + 6 configs + smoke test + 最终 Codex code review, 用户点头后部署.**
