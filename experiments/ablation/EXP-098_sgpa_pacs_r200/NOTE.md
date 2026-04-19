# EXP-098: SGPA + Linear 对照 PACS R200 3-seed — 双数据集扩展

**日期**: 2026-04-19 设计 / 等部署
**算法**: `feddsa_sgpa` (use_etf=1 SGPA + use_etf=0 Linear)
**服务器**: seetacloud2 (等 SCPR v2 释放 PACS GPU 后部署)
**状态**: 🟡 WAIT — SCPR v2 PACS 还在跑, 预计 1h 后释放

## 这个实验做什么 (大白话)

> EXP-097 + EXP-100 如果在 Office 上证明 SGPA 有效 + ETF 贡献明确, 下一步就是在 **PACS 4-outlier (全域)** 上扩展。PACS 比 Office 更难:
>
> - PACS 4 clients 都是 outlier (Photo/Art/Cartoon/Sketch), Office 只有 DSLR 是 outlier
> - PACS K=7 理论单纯形下界 -1/6=-0.167, Office K=10 下界 -0.111, **PACS 下界更宽, ETF 理论收益可能更大**
> - PACS E=5 训练更慢 (6 倍于 Office E=1), 但每 seed 约 2-3h, 3 seeds 并行约 3h
>
> 跟 EXP-097/100 同样 SGPA vs Linear 对照, 只是换数据集 + E=5.

## Claim 和成功标准

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C1 PACS**: SGPA R200 PACS > Plan A R200 | 3-seed mean AVG Best ≥ 82.8% (Plan A 82.31% + 0.5%) | PACS 4-outlier ETF 理论优势未体现, 方案不通用 |
| **C2 PACS**: ETF 贡献 (对照 Linear) | Linear+whitening ≈ Plan A 82.31%, SGPA ≥ Linear + 1% | 同 EXP-100 但 PACS 场景 |
| **预期**: PACS 的 inter_cls_sim 更接近理论下界 -0.167 | R200 mean ≤ -0.12 | 若未达, ETF 几何收益有限 |

## 配置

### SGPA (use_etf=1)
| 参数 | 值 |
|------|-----|
| Task | PACS_c4 (7 类, 4 clients) |
| Backbone | AlexNet + 双 128d 头 |
| Algorithm | feddsa_sgpa, `use_etf=1` |
| R | 200 |
| **E** | **5** (PACS 惯例, 更长本地训练) |
| LR | 0.05 |
| λ_orth | 1.0 |
| τ_etf | 0.1 |
| Seeds | {2, 15, 333} |
| diag | 1 |
| Config | `FDSE_CVPR25/config/pacs/feddsa_sgpa_pacs_r200.yml` |

### Linear 对照 (use_etf=0)
同上但 `use_etf=0`, config `feddsa_linear_pacs_r200.yml`

## 🏆 完整结果 (3-seed mean) — 待回填

### Claim C1 PACS: SGPA vs Plan A

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Art | Cart | Photo | Sketch |
|------|------|---------|---------|---------|---------|-----|------|-------|--------|
| **Plan A orth_only** (EXP-080) | **mean** | 83.45 | 76.49 | **81.69** | 73.87 | — | — | — | — |
| **SGPA** | **mean** | 待填 | 待填 | **待填** | 待填 | — | — | — | — |
|  | 2 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
|  | 15 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
|  | 333 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
| **Δ SGPA − Plan A** | — | 待填 | 待填 | **待填** | 待填 | — | — | — | — |

### Claim C2 PACS: ETF 贡献

| 配置 | AVG Best 3-seed mean | Δ vs Plan A |
|------|---------------------|-------------|
| Plan A orth_only | 81.69 | — |
| **Linear+whitening (本实验)** | 待填 | 待填 |
| **SGPA (本实验)** | 待填 | 待填 |
| **Δ SGPA − Linear (ETF 贡献)** | **待填** | — |

### Neural Collapse 诊断 PACS (K=7, 理论下界 -1/6=-0.167)

| Round | etf_align | inter_cls_sim | client_center_var | param_drift |
|-------|-----------|---------------|-------------------|-------------|
| R50 | 待填 | 待填 | 待填 | 待填 |
| R100 | 待填 | 待填 | 待填 | 待填 |
| R200 | 待填 | 待填 | 待填 | 待填 |

## 📊 实验统计

- **总 runs**: 6 (SGPA 3 + Linear 3)
- **预估 GPU·h**: ~18 (PACS E=5 单 run ≈ 3h, 6 并行 ~3h wall 但 CPU 可能瓶颈)
- **启动**: 待 SCPR v2 释放 (约 1h)
- **完成**: 待填

## 📎 相关文件

- EXPERIMENT_PLAN: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/EXPERIMENT_PLAN.md`
- Office 主实验: `experiments/ablation/EXP-097_sgpa_office_r200/NOTE.md`
- Linear 对照: `experiments/ablation/EXP-100_linear_office_r200/NOTE.md`
- Configs: `FDSE_CVPR25/config/pacs/feddsa_{sgpa,linear}_pacs_r200.yml`
