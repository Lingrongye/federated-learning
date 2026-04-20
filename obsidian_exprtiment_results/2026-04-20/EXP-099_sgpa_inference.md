# EXP-099: SGPA 推理独立 script — C3 推理端 claim 证伪

**日期**: 2026-04-19 设计 / 2026-04-20 10:30 完成 (seed=2)
**算法**: 独立推理 script, 加载 EXP-105 (Linear+whitening+se=1) Office R200 seed=2 checkpoint
**服务器**: seetacloud2 GPU 0 (推理 ~30s)
**状态**: 🔴 **C3 证伪** — fallback_rate=1.00, proto_vs_etf_gain=0.00, SGPA 推理机制完全未启用

## 这个实验做什么 (大白话)

> EXP-096/097/100 的 test accuracy 实际上**只测了 Linear/ETF argmax**, 因为 flgo 默认 test 走 `model.forward()`, 绕过了 `test_with_sgpa`. 但 SGPA 论文的真正 dominant contribution **是推理端的双 gate + 原型校正**, 不是训练端 ETF. 要验证这个 claim, 必须**独立写 script**, 加载 checkpoint, 跑完整 SGPA (warmup 5 batch → 双 gate 筛 reliable → top-m proto bank → cos 分类 + ETF fallback), 报 `proto_vs_etf_gain`。
>
> 这个实验**GPU 成本极低**: 复用 EXP-105 训练好的 checkpoint, 推理只需 forward pass. EXP-105 是专门为此保存的 (se=1 flag → 训练末轮保存 global_model/client_models/source_style_bank/whitening).

## Claim 和成功标准

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C3 (Primary)**: SGPA 推理比 ETF argmax 提分 ≥ +0.5% | 4 clients mean proto_vs_etf_gain ≥ +0.5% | SGPA 推理 == ETF argmax, 论文弱化推理端 |
| **诊断**: 双 gate 有效 | reliable_rate ∈ [0.3, 0.7], proto_etf_offset R200 > 0.1 | 若 gate 永远全过或全不过, gate 设计 fail |

## 配置

| 参数 | 值 |
|------|-----|
| 依赖 checkpoint | EXP-105 Office R200 seed=2 (use_etf=0 use_whitening=1 se=1) |
| 存储路径 | `/root/fl_checkpoints/sgpa_office_caltech10_c4_s2_R200_1776629530/` |
| 包含文件 | global_model.pt (53MB) + client_models.pt (212MB) + whitening.pt (70KB) + source_style_bank.pt (267KB) + meta.json |
| τ_H_q | 0.5 (p50 warmup entropy) |
| τ_S_q | 0.3 (p30 warmup dist) |
| m_top | max(K*5, 20) = 50 for Office |
| warmup | 5 batches |
| GPU | 0 (30s wall) |

## 🏆 完整结果 (seed=2, 2026-04-20)

### Claim C3 主对比

| 推理方式 | Client mean AVG | Caltech | Amazon | DSLR | Webcam |
|---------|-----------------|---------|--------|------|--------|
| **ETF fallback** (baseline) | 86.67 | 66.96 | 83.16 | 100.00 | 96.55 |
| **SGPA full** (原本的 claim) | 86.67 | 66.96 | 83.16 | 100.00 | 96.55 |
| **Δ SGPA − ETF (C3 核心)** | **+0.00** ❌ | 0 | 0 | 0 | 0 |

**🔴 C3 彻底证伪**: proto_vs_etf_gain = 0.00 全 client, 全 sample.

### 根因诊断指标

| 指标 | 值 | 说明 |
|------|-----|------|
| reliable_rate | **0.00** | 所有 query 都没进 reliable 集合 (双 gate 太严) |
| fallback_rate | **1.00** | 全部回退到 ETF fallback (或 Linear classifier) |
| proto_vs_etf_gain | **0.00** | SGPA 预测 ≡ fallback 预测 |
| proto bank 是否命中 | ❌ | 未触发 top-m proto 分类 |

**诊断**: 默认 τ_H_q=0.5 / τ_S_q=0.3 warmup 校准下, 没有一个 query 同时通过两 gate.

## 🔍 Verdict Decision Tree — 落在 "proto_vs_etf_gain = 0" 分支

```
proto_vs_etf_gain ≥ +0.5%
  → ✅ C3 成立, SGPA 推理端是论文真 dominant contribution
  → 扩到 PACS 验证

proto_vs_etf_gain ∈ [0, +0.5%]
  → ⚠️ SGPA 推理微弱 +, 算 nice-to-have

▶ proto_vs_etf_gain = 0 (**本实验落点**)
  → ❌ C3 证伪: SGPA 推理 = ETF fallback, 无任何 proto 校正
  → 决策选项:
     (A) 调阈值: τ_H_q 0.5→0.8, τ_S_q 0.3→0.5 放宽
     (B) 改 gate 设计: 双 gate AND → OR
     (C) 放弃推理端 SGPA claim, 回到训练端故事
```

**暂定**: **选 (C)** — Linear+whitening (EXP-102 89.26%) 已经很强, SGPA 推理太复杂且 0 触发.

## 📋 论文叙事影响

### "SGPA 推理端双 gate + proto 校正 ETF vertex" 故事已死

EXP-099 seed=2 = 4 个 client × ~251 query = 至少 1000 样本, 0 个通过双 gate. 即使放宽 seed 差异, 这个机制在 Office R200 checkpoint 下基本不触发.

### 可能救的方向

1. **调阈值 + 3-seed 重测**: τ_H_q=0.8, τ_S_q=0.5, 3 seeds × 4 clients, 看 reliable_rate 能否到 0.3-0.7 合理区间
2. **接受失败, 转向训练端**: Linear+whitening 广播是真正的 dominant contribution, 推理端回到朴素 argmax

### 论文新骨架 (若选 C)

- **Dominant contribution**: "Pooled source-style statistics broadcast" (whitening + μ_k) → Office +6.20pp vs Plan A
- **Subordinate**: L_orth decouple (已有)
- **删除**: Fixed ETF, SGPA 双 gate 推理, class_centers 收集 (待 EXP-103 确认)

## 📊 实验统计

- **总 runs**: 1 (seed=2 only, 作为 proof-of-concept)
- **GPU**: 30s wall, 可忽略
- **启动**: 2026-04-20 10:25
- **完成**: 2026-04-20 10:30 (~5min 含 flgo.init)

## 📎 相关文件

- 本地 NOTE: `experiments/ablation/EXP-099_sgpa_inference/NOTE.md`
- 脚本: `FDSE_CVPR25/scripts/run_sgpa_inference.py`
- 结果: `experiments/ablation/EXP-099_sgpa_inference/s2_result.json` + `terminal_s2.log`
- 依赖 checkpoint: `/root/fl_checkpoints/sgpa_office_caltech10_c4_s2_R200_1776629530/` (服务器保存, 暂未 commit)
- Config: `FDSE_CVPR25/config/office/feddsa_whiten_only_saveckpt_office_r200.yml` (se=1)
