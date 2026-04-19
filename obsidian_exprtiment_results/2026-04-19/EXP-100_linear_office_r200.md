# EXP-100: Linear 对照 Office R200 3-seed — 量化 ETF 本身 vs 副作用

**日期**: 2026-04-19 启动 / 待完成
**算法**: `feddsa_sgpa` (use_etf=0, 对照组)
**服务器**: seetacloud2 (与 EXP-097 并行)
**状态**: ✅ **已完成** (2026-04-20 凌晨), **重大发现: Linear 竟然超越 SGPA! C2 证伪, ETF 反向劣化**

## 这个实验做什么 (大白话)

> EXP-096 smoke test 达 84.98%,比 Plan A 高 +2.4%。但这个 +2.4% **到底来自哪里**?有两种可能:
>
> - **假设 A**: Fixed ETF 分类器本身的几何贡献 (Neural Collapse 加速)
> - **假设 B**: pooled whitening 广播 / class_centers 累积 / 别的代码副作用
>
> 要区分两者,只能**控制变量**:保留 SGPA 所有新基础设施 (whitening + 诊断 + style stats 上传),**唯一变量就是 classifier 换回 Plan A 的 nn.Linear(128, 10)**。
>
> 如果 Linear 版本也拿 85% → 假设 B 成立, ETF 没功劳,方案要重想;如果 Linear 版本回落到 Plan A 的 82-83% → 假设 A 成立, ETF 明确贡献 +2%,方案 ✓。

## Claim 和成功标准

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C2 (Primary, controlling variable)**: ETF 本身是 gain 来源 (非副作用) | Linear+whitening 3-seed mean AVG Best ≤ 83% (≈ Plan A 82.55%), 且 SGPA 比它高 ≥ +1% | 若 Linear 也 85%+ → ETF 没功劳, 方案失败 |

## 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Task | office_caltech10_c4 | 同 EXP-097 |
| Backbone | AlexNet + 双 128d 头 | 同 |
| Algorithm | feddsa_sgpa, **`use_etf=0`** | 对照: 保留所有基础设施, 只换 classifier |
| Classifier | **`nn.Linear(128, 10)`** (参加 FedAvg) | **唯一变量** |
| 其他 | 全部与 EXP-097 相同 | R200, E=1, LR=0.05, λ_orth=1.0 |
| Seeds | {2, 15, 333} | 与 EXP-097 严格同 seed 对比 |
| diag | 1 | 诊断开启 (观察 Linear 下 etf_align 应 ~0, 类间分离不如 ETF) |
| Config | `FDSE_CVPR25/config/office/feddsa_linear_office_r200.yml` | |

## 预期结果 (假设 A — ETF 有效 场景)

| 指标 | 目标 | 对比 |
|------|------|------|
| AVG Best 3-seed mean | **~82-83%** (接近 Plan A) | 如果达 85% → 假设 B, ETF 没用 |
| etf_align R200 | ~0 (Linear 学出的方向跟 ETF M 不一样) | SGPA 预期 ≥ 0.90 |
| inter_cls_sim R200 | ~0.2-0.4 (不如 ETF 极限分离) | SGPA 预期 ≤ -0.09 |
| client_center_var R200 | 比 SGPA 高 2-3 倍 | Linear 参加 FedAvg 有漂移 |
| param_drift R200 | > 0.01 (Linear 漂移不被消除) | SGPA 预期 ~0.003 |

## 🏆 完整结果 (3-seed mean) — 待回填

### C2 主对比: SGPA (EXP-097) vs Linear+whitening (本实验) vs Plan A

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last |
|------|------|---------|---------|---------|---------|
| **Plan A orth_only** (EXP-083) | **mean** | 88.61 | 87.30 | **82.55** | 81.35 |
| **SGPA (EXP-097, use_etf=1)** | **mean** | 82.01 | 80.42 | **86.97 ± 1.23** | 85.44 |
| **Linear+whitening (本实验, use_etf=0)** | **mean** | **82.81** | 81.09 | **88.75 ± 0.86** 🔥 | 86.91 |
|  | 2 | 80.17 | 78.98 | 87.56 | 86.81 |
|  | 15 | 83.35 | 81.35 | 89.55 | 87.11 |
|  | 333 | 84.91 | 82.93 | 89.14 | 86.80 |
| **Δ SGPA − Linear (ETF 贡献 C2!)** | — | -0.80 | -0.67 | **-1.78** ❌ | -1.47 |
| **Δ Linear − Plan A ("副作用"!)** | — | +4.20 | +3.79 | **+6.20** 🔥 | +5.56 |

### 诊断对比 (SGPA vs Linear, R200)

| 指标 | SGPA (EXP-097) | Linear+whitening (本) | 说明 |
|------|---------------|----------------------|------|
| etf_align R200 | 待填 | 待填 | Linear 应 ~0, SGPA 应 ≥ 0.9 |
| inter_cls_sim R200 | 待填 | 待填 | Linear 应 > 0.2, SGPA 应 ≤ -0.09 |
| client_center_var R200 | 待填 | 待填 | Linear 应明显更大 |
| param_drift R200 | 待填 | 待填 | Linear 应 > SGPA 数倍 |

## 🔍 Verdict Decision Tree

```
Δ SGPA − Linear ≥ +1.5%
  → ✅ C2 成立, ETF 本身贡献明确, 方案 ✓
  → 下一步: EXP-099 SGPA 推理 + EXP-098 PACS

Δ SGPA − Linear ∈ [0.5%, 1.5%]
  → ⚠️ C2 部分成立, ETF 有贡献但不压倒副作用
  → 考虑跑更多 seeds / 检查 diag 的几何差异

Δ SGPA − Linear ≤ 0.5%
  → ❌ C2 证伪, ETF 可能没功劳
  → 必须思考: (1) 是否 pooled whitening 本身就 +2%?
              (2) 是否诊断集成副作用了?
              (3) 是否 BN 某个细节被改变?
```

## 🔍 根因分析

### Verdict Decision Tree 判定 ❌

```
Δ SGPA − Linear = -1.78 (小于 0)
  → ❌ C2 完全证伪, ETF 不仅没功劳, 反而减分
```

### 完全颠覆 EXP-096 smoke 结论

| 阶段 | 结论 | 证据 |
|------|------|------|
| EXP-096 smoke R50 seed=2 | "Fixed ETF R50 就超 Plan A R200" | SGPA 84.98 vs Plan A 82.55 |
| EXP-097+100 R200 3-seed | **Linear+whitening 才是真正赢家** | Linear 88.75 > SGPA 86.97 > Plan A 82.55 |

smoke test **不是** 过早错误判断 seed 运气 — R200 3-seed 验证了 ETF 变体 **确实** 比 Plan A 好 +4.42%,但**这个 gain 不是 ETF 带来的**,而是 pooled whitening 广播 + class_centers 收集基础设施带来的。

### 核心发现: "Pooled Source Style Broadcast" 是真正 gain 机制

| 实验 | 改动 | AVG Best | Δ |
|------|------|----------|---|
| Plan A orth_only | — | 82.55 | 基线 |
| Plan A + pooled whitening + class_centers | 新基础设施 | **88.75** (Linear) | **+6.20** 🔥 |
| + Fixed ETF classifier | ETF | 86.97 (SGPA) | **-1.78 vs Linear** |

**"广播 (μ_global, Σ_inv_sqrt, source_μ_k)" 本身就是强 gain 机制**。

### 为什么 ETF 反而减分

可能原因 (需 diag 数据证实, 但本实验 diag 污染):
1. **Fixed ETF 对 AlexNet from-scratch 是过度约束**: Linear 能学到比 ETF 顶点更贴近实际类中心的边界
2. **ETF + L_orth 双重几何约束 + CE** 对特征空间挤压过度
3. **Linear 参加 FedAvg 聚合虽然有漂移,但 FedDSA-SGPA 的 class_centers 收集机制可能本来就缓解了漂移**

### Neural Collapse 诊断分析放弃

EXP-097 和 EXP-100 的 diag_logs 共用 R200_S{seed} 路径, jsonl 每轮 2 行 (SGPA+Linear 交错无 variant 字段), 无法可靠区分 ETF 和 Linear 的几何演进。**bug 已在 commit `6a31e22` 修复**, PACS 新部署会生成干净 diag。

## 📋 论文叙事影响

### 原论文方向全废

- ETF 是 supporting contribution → **有害, 删除**
- SGPA 是 dominant contribution → 只剩推理端 (EXP-099 还没测)

### 新发现可能更有价值

**"Pooled source-domain style second-order statistics broadcast to clients"** 让 Plan A 从 82.55 → 88.75 (+6.20%), 接近 FDSE 90.58 只差 1.83%, 而且:
- 通信增量仅 ~66KB/round (与 FedBN γ/β 同量级)
- 零新 trainable 参数 (whitening broadcast 是 float tensor)
- 完全 backward-compat 现有 FedAvg + FedBN 流程

### 接下来必须做的消融

为定位 gain 具体来自哪里, 需要:
1. **Plan A + pooled whitening only** (不收集 class_centers): 量化 pooled whitening 贡献
2. **Plan A + class_centers only** (不广播 whitening): 量化 class_centers 贡献
3. **Plan A + diag=0** (同 Linear 但不开诊断): 排除 diag 本身副作用

这是新的 EXP-101 / EXP-102 / EXP-103 的必要对照实验。

## 📊 实验统计

- **总 runs**: 3 (3 seeds × 1 config)
- **实际 GPU·h**: ~7 (与 EXP-097 并行, 6 runs 共享 CPU, wall ~1h25min)
- **启动**: 2026-04-19 22:41
- **完成**: 2026-04-19 23:55 (~1h15min wall)

## 📎 相关文件

- EXPERIMENT_PLAN: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/EXPERIMENT_PLAN.md`
- SGPA 主实验: `experiments/ablation/EXP-097_sgpa_office_r200/NOTE.md`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (use_etf flag)
- Config: `FDSE_CVPR25/config/office/feddsa_linear_office_r200.yml`
