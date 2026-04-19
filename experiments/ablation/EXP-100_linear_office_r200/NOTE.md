# EXP-100: Linear 对照 Office R200 3-seed — 量化 ETF 本身 vs 副作用

**日期**: 2026-04-19 启动 / 待完成
**算法**: `feddsa_sgpa` (use_etf=0, 对照组)
**服务器**: seetacloud2 (与 EXP-097 并行)
**状态**: 🟡 部署中 (3 runs 并行)

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
| **SGPA (EXP-097)** | **mean** | 待填 | 待填 | **待填** | 待填 |
| **Linear+whitening (本实验)** | **mean** | 待填 | 待填 | **待填** | 待填 |
|  | 2 | 待填 | 待填 | 待填 | 待填 |
|  | 15 | 待填 | 待填 | 待填 | 待填 |
|  | 333 | 待填 | 待填 | 待填 | 待填 |
| **Δ SGPA − Linear (ETF 贡献)** | — | 待填 | 待填 | **待填** | 待填 |
| **Δ Linear − Plan A (副作用)** | — | 待填 | 待填 | 待填 | 待填 |

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

## 🔍 根因分析 (待填)

## 📋 论文叙事影响 (待填)

## 📊 实验统计

- **总 runs**: 3 (3 seeds × 1 config)
- **预估 GPU·h**: ~3 (与 EXP-097 并行, 共享 1h wall)
- **启动**: 待填
- **完成**: 待填

## 📎 相关文件

- EXPERIMENT_PLAN: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/EXPERIMENT_PLAN.md`
- SGPA 主实验: `experiments/ablation/EXP-097_sgpa_office_r200/NOTE.md`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (use_etf flag)
- Config: `FDSE_CVPR25/config/office/feddsa_linear_office_r200.yml`
