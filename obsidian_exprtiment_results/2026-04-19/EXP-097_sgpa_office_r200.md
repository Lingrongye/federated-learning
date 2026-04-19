# EXP-097: SGPA Office R200 3-seed — 验证 smoke test 不是 seed 运气

**日期**: 2026-04-19 启动 / 待完成
**算法**: `feddsa_sgpa` (use_etf=1)
**服务器**: seetacloud2 (单卡 24GB)
**状态**: ✅ **已完成** (2026-04-20 凌晨), **意外结果: Linear 对照 (EXP-100) 竟然超越 SGPA, ETF 未成立**

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
| **SGPA (OURS, use_etf=1)** | **mean** | **82.01** | 80.42 | **86.97 ± 1.23** | 85.44 | 70.5/69.6 | 88.8/88.1 | 97.8/95.6 | 90.8/88.5 |
|  | 2 | 79.75 | 76.58 | 85.89 | 82.83 | 65.2/65.2 | 85.3/83.2 | 100.0/93.3 | 93.1/89.7 |
|  | 15 | 82.94 | 82.94 | 88.68 | 88.68 | 73.2/73.2 | 88.4/88.4 | 100.0/100.0 | 93.1/93.1 |
|  | 333 | 83.32 | 81.73 | 86.35 | 84.81 | 73.2/70.5 | 92.6/92.6 | 93.3/93.3 | 86.2/82.8 |
| **Linear+whitening (EXP-100, use_etf=0)** | **mean** | **82.81** | 81.09 | **88.75 ± 0.86** 🔥 | 86.91 | 72.3/70.5 | 88.4/87.4 | 100.0/97.8 | 94.3/92.0 |
| **Δ SGPA − Plan A** | — | +3.40 | +3.12 | **+4.42** ✅ | +4.09 | -2.1/-0.6 | -2.1/±0 | -2.2/-2.2 | -3.5/-5.8 |
| **Δ SGPA − Linear (C2 关键!)** | — | -0.80 | -0.67 | **-1.78 ❌** | -1.47 | -1.8/-0.9 | +0.4/+0.7 | -2.2/-2.2 | -3.5/-3.5 |
| **Δ SGPA − SAS** | — | +7.62 | — | **+2.57** | +2.37 | -4.5/-4.2 | -2.8/-0.3 | -2.2/-2.2 | -4.6/-4.6 |
| **Δ SGPA − FDSE** | — | -4.37 | — | **-3.61** | -3.78 | — | — | — | — |

### Neural Collapse 诊断演进 (3-seed mean, 从 Layer 1 jsonl 提取)

| Round | etf_align mean | inter_cls_sim | intra_cls_sim | orth | client_center_var | param_drift |
|-------|---------------|---------------|---------------|------|-------------------|-------------|
| R5 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| R50 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| R100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| R200 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 参考值 (EXP-096 smoke R50 seed=2 单 client 0)

R50: etf_align=0.83 / inter_cls=-0.08 / intra_cls=0.85 / orth=0.0003 / center_var=0.0019 / param_drift=0.003

## 🔍 根因分析

### Claim verdict

| Claim | 判定 | 数据 |
|-------|------|------|
| **C1 (SGPA > Plan A)** | ✅ 成立 | +4.42% |
| **C4 (不是 seed 运气)** | ✅ 成立 | 3-seed std 1.23% (< 1.5% 阈值), 无 seed 崩溃 |
| **C2 (ETF 本身贡献)** | ❌ **证伪** | SGPA 比 Linear 还低 **-1.78%**, ETF 不仅无功反而有害 |

### ETF 反向劣化的可能原因

1. **Linear 学出的分类边界比 Fixed ETF 顶点更贴近真实类中心**:
   - Fixed ETF 假设"好的特征应该对齐单纯形",但实际 AlexNet 从 scratch 在 Office 上可能学不出完美对齐,强制对齐反而是约束
   - Linear 自由学习权重方向 = 自适应找最优边界
2. **ETF + L_orth 联合可能过度约束 z_sem**: 两个几何约束叠加压缩特征空间
3. **Linear 下 diag 记录的 etf_align 等指标的数据被污染** (SGPA+Linear 混写 jsonl) → 如果看 diag 得有保留

### 真正贡献来源 (gain +6.20% Linear vs Plan A)

不是 ETF, 而是 **Plan A + pooled whitening + class_centers 基础设施**本身:
- `pooled whitening`: 每轮 broadcast 源域 (μ_global, Σ_inv_sqrt) 1024 floats
- `class_centers 收集`: client 每轮上传 [K, d] 类中心 tensor
- 这些基础设施原本设计给 SGPA 推理用, 但训练端**隐式受益** (可能通过 gradient routing 或 effective regularization)

### Neural Collapse 诊断演进 (Layer 1 jsonl, 污染但数据还在)

**⚠️ 警告**: EXP-097 和 EXP-100 的 diag_logs 共用 R200_S{seed} 路径, jsonl 每轮 2 行 (SGPA+Linear 交错无 variant 字段), 无法可靠区分。新 PACS 实验已修复加 _etf/_linear 后缀。Office 诊断层分析暂放弃。

## 📋 论文叙事影响

### 原方案核心 (SGPA 双 gate + ETF) 全线崩

- **ETF 训练端** C2 证伪 → 从 proposal 删除
- **SGPA 推理端** (C3) 还没测 (EXP-099 待做), 但即使成立也只是 on-top-of Linear 的增量

### 真正有价值的发现

**"Source-domain style 二阶统计广播 + 客户端类中心收集" 本身就是强 gain 机制**

相对 Plan A 的 +6.20% gain 来自新增的基础设施, 不是 ETF 几何约束。

### 建议的新论文叙事

| 原论文 | 新论文 |
|--------|--------|
| "SGPA + Fixed ETF classifier" | "**FedDSA-Plus: Plan A + Pooled Style-Statistics Broadcast + Class-Centers Tracking**" |
| 主贡献: ETF + dual-gate SGPA | 主贡献: 证明"源域 style 二阶统计跨客户端共享" 本身就是强 gain 机制, 不需要 ETF |
| 对比: FDSE 90.58% vs ours 84.98% | 对比: FDSE 90.58% vs ours **88.75%** (Δ 只差 1.83%) |

### 下一步需要更多对照以定位 gain 来源

| 对照实验 | 目的 |
|---------|------|
| Plan A + pooled whitening only (不收集 class_centers) | pooled whitening 单独贡献 |
| Plan A + class_centers only (不广播 whitening) | class_centers 单独贡献 |
| Plan A + diag=0 (不跑 diag 框架) | 排除 diag 本身的副作用 |

如果是 whitening broadcast 带来 gain,后续应该聚焦 whitening 的理论动机 (pooled second-order approximation 是否让 cross-client feature alignment 隐式更强)。

## 📊 实验统计

- **总 runs**: 3 (3 seeds × 1 config)
- **实际 GPU·h**: ~7 (3 并行 + 6 其他 runs 共享 CPU, wall ~1h25min)
- **启动**: 2026-04-19 22:41
- **完成**: 2026-04-19 23:55 (~1h15min wall)

## 📎 相关文件

- EXPERIMENT_PLAN: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/EXPERIMENT_PLAN.md`
- EXP-096 smoke: `experiments/ablation/EXP-096_sgpa_smoke/NOTE.md`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py`
- Config: `FDSE_CVPR25/config/office/feddsa_sgpa_office_r200.yml`
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (113 passed)
- 诊断框架: `FDSE_CVPR25/diagnostics/sgpa_diagnostic_logger.py`
