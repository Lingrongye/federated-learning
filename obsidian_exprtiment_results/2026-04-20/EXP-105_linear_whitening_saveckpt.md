# EXP-105: Linear+whitening+se=1 Office R200 seed=2 — 专为 EXP-099 SGPA 推理保存 checkpoint

**日期**: 2026-04-20 02:06 启动 / 2026-04-20 04:12 完成
**算法**: `feddsa_sgpa` (use_etf=0, use_whitening=1, use_centers=0, **se=1 新增**)
**服务器**: seetacloud2 GPU 0 (R200, wall ~2h)
**状态**: ✅ **完成** — checkpoint 已保存, 被 EXP-099 使用, 训练精度符合 EXP-102 baseline

## 这个实验做什么 (大白话)

> EXP-099 要跑 SGPA 推理, 但需要一个**训好的 checkpoint** 配套 source_style_bank + whitening + client BN. flgo 默认训练不保存这些附加状态 — 所以给 feddsa_sgpa.py 加了 **`se` 新 flag (save_endpoint)**: se=1 时训练末轮保存:
>
> - `global_model.pt` — 全局模型 state_dict
> - `client_models.pt` — 每 client 私有 BN / Linear classifier
> - `whitening.pt` — μ_global, Σ_inv_sqrt
> - `source_style_bank.pt` — 每 client 的 (μ_k, σ_k)
> - `meta.json` — 任务/seed/R 等元数据
>
> 然后再跑 EXP-106 λ_pull 实验 (它需要同样的训练流程, 但不存 checkpoint).
>
> 配置是 **EXP-102 当前冠军**: Linear + pooled whitening, 只跑 seed=2 (足够 EXP-099 推理用).

## Claim 和成功标准

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C5 checkpoint**: se=1 正确保存 5 个 artifact 且能被推理脚本加载 | 5 文件在 `/root/fl_checkpoints/sgpa_*/`, EXP-099 load 成功 | 保存机制 bug |
| **训练精度符合 EXP-102**: seed=2 AVG Best ≥ 87% | Max ≥ 87% | 说明 se 机制没污染训练流程 |

## 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Task | office_caltech10_c4 | 同 EXP-102 |
| use_etf | 0 | Linear |
| use_whitening | 1 | EXP-102 winning config |
| use_centers | 0 | 不收集 class centers |
| **se** | **1** | 新增: 保存末轮 checkpoint |
| R / E / LR | 200 / 1 / 0.05 | |
| λ_orth | 1.0 | |
| Seed | 2 only (单 seed pilot) | |
| Config | `FDSE_CVPR25/config/office/feddsa_whiten_only_saveckpt_office_r200.yml` | |

## 🏆 完整结果 (seed=2, 2026-04-20)

### 训练精度 (R200)

| 指标 | 值 | 对照 EXP-102 seed=2 | Δ |
|------|-----|--------------------|---|
| AVG Best | **~87.56** | 87.56 (EXP-102 mean 89.26 的单 seed 版) | ±0 |
| AVG Last | ~86.81 | 86.74 | +0.07 |
| (per-domain 数据暂未提取, 本 checkpoint 仅为 EXP-099 服务) | | | |

**精度一致性 ✅**: se=1 机制没污染训练.

### Checkpoint 产物

存储路径: `/root/fl_checkpoints/sgpa_office_caltech10_c4_s2_R200_1776629530/`

| 文件 | 大小 | 用途 |
|------|-----|------|
| `meta.json` | 214B | task/seed/R/use_etf/use_whitening |
| `global_model.pt` | 53MB | 服务器端聚合后的全局 state_dict |
| `client_models.pt` | 212MB | 每 client 的私有 BN + Linear classifier state |
| `whitening.pt` | 70KB | μ_global, Σ_inv_sqrt, source_μ_k |
| `source_style_bank.pt` | 267KB | 每 client (μ_k, σ_k) 风格统计 |

5/5 文件齐全, EXP-099 脚本加载成功 (见 EXP-099 NOTE).

## 🔍 Verdict Decision Tree

```
se=1 保存 5 artifact + EXP-099 成功 load + 训练精度一致
  → ✅ C5 成立, checkpoint 机制通过
```

## 📋 下游实验

- **EXP-099 SGPA 推理 (已完成)**: 使用本 checkpoint, 结果 fallback_rate=1.00, C3 证伪 (与本实验无关, SGPA 推理机制本身的问题)

## 📊 实验统计

- **总 runs**: 1 (seed=2 only, 够 EXP-099 用)
- **GPU·h**: ~2h wall (单卡 4090)
- **启动**: 2026-04-20 02:06:15
- **完成**: 2026-04-20 04:12 (log End, checkpoint 保存成功)

## 📎 相关文件

- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (`_save_sgpa_checkpoint()`)
- Config: `FDSE_CVPR25/config/office/feddsa_whiten_only_saveckpt_office_r200.yml`
- Record: `FDSE_CVPR25/task/office_caltech10_c4/record/feddsa_sgpa_..._se1_...S2_...R200.json`
- Checkpoint (服务器): `/root/fl_checkpoints/sgpa_office_caltech10_c4_s2_R200_1776629530/`
- 下游 EXP-099: `experiments/ablation/EXP-099_sgpa_inference/NOTE.md`
