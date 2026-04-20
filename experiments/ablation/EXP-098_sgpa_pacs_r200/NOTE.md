# EXP-098: SGPA + Linear 对照 PACS R200 3-seed — 双数据集扩展

**日期**: 2026-04-19 设计 / 2026-04-20 09:16 完成
**算法**: `feddsa_sgpa` (use_etf=1 SGPA + use_etf=0 Linear)
**服务器**: seetacloud2 GPU 0 (单卡 4090, 6 runs 并行 ~8h wall)
**状态**: ✅ **已完成**. **PACS 验证 Office 结论: Linear > Hard ETF, ETF 后期严重退化 (-5.59pp last)**

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

## 🏆 完整结果 (3-seed mean, 2026-04-20 回填)

### Claim C1 PACS: SGPA vs Plan A (per-domain Best/Last)

| 配置 | seed | AVG Best | AVG Last | Art | Cart | Photo | Sketch |
|------|------|---------|---------|-----|------|-------|--------|
| **Plan A orth_only** (EXP-080) | **mean** | 81.69 | 73.87 | — | — | — | — |
| **SGPA (use_etf=1)** | **mean** | **78.96 ± 0.37** | **73.77** | 62.6/54.6 | 85.0/78.9 | 80.0/74.3 | 88.2/87.3 |
|  | 2 | 79.48 | 73.56 | 64.7/53.4 | 85.5/77.8 | 78.4/76.0 | 89.3/87.0 |
|  | 15 | 78.81 | 74.30 | 61.8/53.9 | 85.5/83.3 | 80.2/72.5 | 87.8/87.5 |
|  | 333 | 78.60 | 73.44 | 61.3/56.4 | 84.2/75.6 | 81.4/74.3 | 87.5/87.5 |
| **Δ SGPA − Plan A** | — | **-2.73** ❌ | -0.10 | — | — | — | — |

### Claim C2 PACS: ETF 贡献 (Linear 对照)

| 配置 | seed | AVG Best | AVG Last | Art | Cart | Photo | Sketch |
|------|------|---------|---------|-----|------|-------|--------|
| **Linear+whitening (use_etf=0)** | **mean** | **80.20 ± 0.94** | **79.36** | 63.4/61.4 | 86.0/84.0 | 81.8/82.4 | 89.5/89.5 |
|  | 2 | 81.46 | 80.74 | 64.7/64.7 | 86.8/83.3 | 83.8/85.6 | 90.6/89.3 |
|  | 15 | 79.94 | 78.65 | 61.8/58.3 | 87.2/86.3 | 82.0/81.4 | 88.8/88.5 |
|  | 333 | 79.21 | 78.70 | 63.7/61.3 | 84.2/82.5 | 79.6/80.2 | 89.3/90.8 |
| **Δ Linear − SGPA (ETF 减分!)** | — | **+1.24** ✅ | **+5.59** 🔥 | +0.8/+6.8 | +1.0/+5.1 | +1.8/+8.1 | +1.3/+2.2 |
| **Δ Linear − Plan A** | — | **-1.49** ⚠️ | +5.49 | — | — | — | — |

### 关键观察

1. **Linear 全面胜 SGPA**: Best +1.24pp, Last +5.59pp
2. **ETF 后期严重退化**: 3 seed 都是 max 出现在 R130-160, 之后 Last 掉到 73-74%
3. **Photo/Art 受 ETF 约束影响最大**: Photo Last -8.1pp, Art Last -6.8pp
4. **但 Plan A PACS 仍是冠军**: AVG Best 81.69 > Linear 80.20 > SGPA 78.96
   - 推测: PACS E=5 下 whitening broadcast 带来的 gain 不如 Plan A 的 L_orth 有效
   - Office E=1 是另一番景象 (Linear 88.75 > Plan A 82.55)

## 📊 实验统计

- **总 runs**: 6 (SGPA 3 + Linear 3)
- **实际**: ~48 GPU·h (seetacloud2 单 4090 6 并行 wall 8h10min)
- **启动**: 2026-04-20 01:05:56
- **完成**: 2026-04-20 09:16 (所有 6 runs DONE + record JSON 保存成功)

## 📎 相关文件

- EXPERIMENT_PLAN: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/EXPERIMENT_PLAN.md`
- Office 主实验: `experiments/ablation/EXP-097_sgpa_office_r200/NOTE.md`
- Linear 对照: `experiments/ablation/EXP-100_linear_office_r200/NOTE.md`
- Configs: `FDSE_CVPR25/config/pacs/feddsa_{sgpa,linear}_pacs_r200.yml`
