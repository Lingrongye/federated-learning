# EXP-097: SGPA Office R200 3-seed — 验证 smoke test 不是 seed 运气

**日期**: 2026-04-19 启动 / 待完成
**算法**: `feddsa_sgpa` (use_etf=1)
**服务器**: seetacloud2 (单卡 24GB)
**状态**: 🟡 部署中 (3 runs 并行)

## 这个实验做什么 (大白话)

> EXP-096 smoke test 在 seed=2 R50 拿到 AVG 84.98% (超 Plan A R200 82.55% +2.4%),但 85+ 实验史证明单 seed 不可信 (EXP-075 曾经 81.7% 峰值后崩到 51.2%)。这次就**同配置跑 3 seeds {2, 15, 333} × 满 R200**,如果 3-seed mean ≥ 84% 且低 std,方案就**不是运气**。
>
> 跟 EXP-096 的唯一差异: R50 → R200 (全量训练), seed 2 → {2, 15, 333}.

## Claim 和成功标准

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C1 (Primary)**: SGPA R200 > Plan A R200 | 3-seed mean AVG Best ≥ 84% (Plan A = 82.55, Δ ≥ +1.5%) | smoke 84.98% 是 seed=2 运气, 回归 Plan A |
| **C4 (Anti-claim)**: ~~不是 seed 运气~~ | 3-seed std ≤ 1.5%, 无单 seed 崩 (drop > 5%) | 方案不稳, 需要调 warmup/τ |

## 配置

| 参数 | 值 |
|------|-----|
| Task | office_caltech10_c4 (10 类, 4 clients) |
| Backbone | AlexNet + 双 128d 头 |
| Algorithm | feddsa_sgpa, `use_etf=1` |
| Classifier | Fixed Simplex ETF buffer (seeded, all clients consistent) |
| R | 200 |
| E | 1 (Office 惯例) |
| LR | 0.05 |
| Decay | 0.9998 |
| λ_orth | 1.0 |
| τ_etf | 0.1 |
| warmup_r | 10 |
| eps_sigma | 1e-3 |
| Seeds | {2, 15, 333} (对齐 EXP-083/084/096) |
| diag | 1 (完整 Layer 1+2 诊断) |
| Config | `FDSE_CVPR25/config/office/feddsa_sgpa_office_r200.yml` |

## 预期结果 (设计目标)

| 指标 | 目标 | 参考 |
|------|------|------|
| AVG Best 3-seed mean | **≥ 84%** | Plan A 82.55, FDSE 90.58 |
| AVG Best std | **≤ 1.5%** | EXP-083 Plan A std ≈ 1-2% |
| drop (Best - Last) | ≤ 1% | EXP-083 Plan A drop ≈ 1% |
| etf_align R200 | ≥ 0.90 | smoke R50 0.83 |
| inter_cls_sim R200 | ≤ -0.09 (接近理论下界 -0.111) | smoke R50 -0.08 |

## 🏆 完整结果 (3-seed mean {2, 15, 333} + 对照行 + Δ 行) — 待回填

### 主结果 Office

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Caltech | Amazon | DSLR | Webcam |
|------|------|---------|---------|---------|---------|---------|--------|------|--------|
| **Plan A orth_only** (EXP-083 对照) | **mean** | 88.61 | 87.30 | **82.55** | 81.35 | 72.6 | 90.9 | 100.0 | 94.3 |
|  | 2 | 86.45 | 86.45 | 78.19 | 78.19 | 67.0 | 86.3 | 100.0 | 96.6 |
|  | 15 | 89.59 | 87.75 | 83.74 | 81.36 | 74.1 | 91.6 | 100.0 | 96.6 |
|  | 333 | 89.81 | 87.69 | 85.72 | 84.52 | 76.8 | 94.7 | 100.0 | 89.7 |
| **SAS τ=0.3** (EXP-084) | **mean** | 89.82 | 88.28 | **84.40** | 83.07 | 75.0 | 91.6 | 100.0 | 95.4 |
| **FDSE** (EXP-051) | **mean** | 86.38 | 85.05 | **90.58** | 89.22 | — | — | — | — |
| **SGPA (OURS, use_etf=1)** | **mean** | 待填 | 待填 | **待填** | 待填 | — | — | — | — |
|  | 2 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
|  | 15 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
|  | 333 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
| **Δ SGPA − Plan A (核心 claim!)** | — | 待填 | 待填 | **待填** | 待填 | — | — | — | — |
| **Δ SGPA − SAS** | — | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
| **Δ SGPA − FDSE** | — | 待填 | 待填 | 待填 | 待填 | — | — | — | — |

### Neural Collapse 诊断演进 (3-seed mean, 从 Layer 1 jsonl 提取)

| Round | etf_align mean | inter_cls_sim | intra_cls_sim | orth | client_center_var | param_drift |
|-------|---------------|---------------|---------------|------|-------------------|-------------|
| R5 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| R50 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| R100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| R200 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 参考值 (EXP-096 smoke R50 seed=2 单 client 0)

R50: etf_align=0.83 / inter_cls=-0.08 / intra_cls=0.85 / orth=0.0003 / center_var=0.0019 / param_drift=0.003

## 🔍 根因分析 (待填)

## 📋 论文叙事影响 (待填)

## 📊 实验统计

- **总 runs**: 3 (3 seeds × 1 config)
- **预估 GPU·h**: ~3 (3 并行 1h wall)
- **启动**: 待填
- **完成**: 待填

## 📎 相关文件

- EXPERIMENT_PLAN: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/EXPERIMENT_PLAN.md`
- EXP-096 smoke: `experiments/ablation/EXP-096_sgpa_smoke/NOTE.md`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py`
- Config: `FDSE_CVPR25/config/office/feddsa_sgpa_office_r200.yml`
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (113 passed)
- 诊断框架: `FDSE_CVPR25/diagnostics/sgpa_diagnostic_logger.py`
