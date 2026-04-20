# EXP-099: SGPA 推理独立 script — 测 Layer 3 完整 13 指标

**日期**: 2026-04-19 设计 / 2026-04-20 10:30 完成 seed=2 推理
**算法**: 独立推理 script, 加载 EXP-105 (whitening + se=1) R200 seed=2 checkpoint
**服务器**: seetacloud2 GPU 0 (30s)
**状态**: 🔴 **C3 证伪** — fallback_rate=1.00, proto_vs_etf_gain=0

## 这个实验做什么 (大白话)

> **当前 EXP-096/097/100 的 test accuracy 实际上只测了 ETF argmax** (或 Linear argmax), 因为 flgo 默认 test 走 `model.forward()` = 直接 argmax, **绕过了** `test_with_sgpa`。
>
> 但 SGPA 论文的真正 dominant contribution **是推理端的双 gate + 原型校正**, 不是 ETF 训练。要验证这个 claim, 必须**独立写 script**, 加载 checkpoint, 跑完整 SGPA (warmup 5 batch → gate 筛 reliable → top-m proto bank → cos 分类 + ETF fallback), 报 `proto_vs_etf_gain`。
>
> 这个实验**零 GPU 成本**: 复用 EXP-096/097 已训练好的 checkpoint, 推理只需 forward pass, CPU 就够。

## Claim 和成功标准

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C3 (Primary)**: SGPA 推理比 ETF argmax 提分 ≥ +0.5% | 4 clients mean proto_vs_etf_gain ≥ +0.5% | SGPA 推理 == ETF argmax, 论文弱化推理端, 强化 ETF 训练端 |
| **诊断**: 双 gate 有效 | reliable_rate ∈ [0.3, 0.7], proto_etf_offset R200 > 0.1 | 若 gate 永远全过或全不过, gate 设计 fail |

## 实验设计

### 独立 script: `FDSE_CVPR25/scripts/run_sgpa_inference.py`

**输入**:
- `--checkpoint`: EXP-096 或 EXP-097 任一训练好的 model state_dict
- `--task`: office_caltech10_c4 / PACS_c4
- `--source-style-bank`: 该 checkpoint 对应训练轮次的 source_style_bank (μ_k, Σ) 或从 JSON record 反推最后轮次的

**流程**:
1. Load model (FedDSASGPAModel(use_etf=1))
2. Load source μ_k + μ_global + Σ_inv_sqrt (从训练末轮 server 状态)
3. 对每个 client 的 test set 调用 `test_with_sgpa()`
4. 对比 `pred_etf.argmax()` 和 `pred_sgpa` (warmup 后) 的 accuracy
5. 输出 Layer 3 全量 13 指标 jsonl

**消融配置**:
- `ETF argmax only` (baseline, 等价 flgo 默认 test)
- `SGPA full` (双 gate + top-m proto + ETF fallback)
- `entropy-only gate` (消融 dist gate)
- `dist-only gate` (消融 entropy gate)
- `no-gate` (T3A 式, 纯 proto 更新无 reliability filtering)

## 配置

| 参数 | 值 |
|------|-----|
| 依赖 checkpoint | EXP-096 R50 smoke (先验证 pipeline) + EXP-097 R200 (真正结果) |
| τ_H | p50 of first-5-batch entropy (warmup calibration) |
| τ_S | p30 of first-5-batch dist_min |
| m_top | max(K*5, 20) = 50 for Office, 35 for PACS |
| warmup | 5 batches |
| EMA decay | 0.95 |
| GPU | 可 CPU, 但有 GPU 更快 |

## 🏆 完整结果 (2026-04-20, seed=2 only)

### Claim C3 主对比

| 推理方式 | Client mean AVG | Caltech | Amazon | DSLR | Webcam |
|---------|-----------------|---------|--------|------|--------|
| **ETF fallback** (baseline) | 86.67 | 66.96 | 83.16 | 100.00 | 96.55 |
| **SGPA full** | 86.67 | 66.96 | 83.16 | 100.00 | 96.55 |
| **Δ SGPA − ETF (C3 核心)** | **+0.00** | — | — | — | — |

**🔴 C3 证伪**: proto_vs_etf_gain = 0.00 全 client.

### 根因 (诊断指标)

| 指标 | 值 | 说明 |
|------|-----|------|
| reliable_rate | **0.00** | 所有 query 都没进 reliable 集合 (双 gate 太严) |
| fallback_rate | **1.00** | 全部回退到 ETF |
| proto_vs_etf_gain | **0.00** | SGPA 预测 ≡ ETF fallback 预测 |
| proto bank 是否命中 | ❌ | 未触发 top-m proto 分类 |

**诊断**: 默认 τ_H_q=0.5 / τ_S_q=0.3 的 warmup 校准下, 没有 query 同时通过两 gate.

### 消融

| 配置 | AVG acc | reliable_rate | proto_etf_offset |
|------|---------|---------------|------------------|
| ETF argmax | 待填 | N/A | N/A |
| entropy-only gate | 待填 | 待填 | 待填 |
| dist-only gate | 待填 | 待填 | 待填 |
| no-gate (T3A) | 待填 | 1.0 (等价无 gate) | 待填 |
| **SGPA full** | 待填 | 待填 | 待填 |

### Layer 3 诊断 (单 client 代表)

| 指标 | 值 | 说明 |
|------|-----|------|
| reliable_rate | 待填 | 目标 [0.3, 0.7] |
| entropy_rate | 待填 | 仅熵 gate 通过率 |
| dist_rate | 待填 | 仅距离 gate 通过率 |
| dist_min p50 | 待填 | 白化后距离中位数 |
| whitening_reduction ratio | 待填 | 白化前后 cross-client scatter 比 |
| sigma_cond | 待填 | Σ_global 条件数 (<1e4 健康) |
| proto_fill_mean | 待填 | per-class support 平均数 |
| proto_etf_offset_mean | 待填 | > 0.1 说明 proto 校正 ETF vertex |
| fallback_rate | 待填 | < 0.3 说明 SGPA 大部分接管 |

## 🔍 Verdict Decision Tree — 落在 "proto_vs_etf_gain = 0" 分支

```
→ ❌ C3 证伪: SGPA 推理 = ETF fallback, 无任何 proto 校正
→ 决策选项:
   (A) 调阈值: τ_H_q 从 0.5 → 0.8 (放宽), τ_S_q 从 0.3 → 0.5 (放宽)
   (B) 改 gate 设计: 双 gate AND → OR (任一通过即 reliable)
   (C) 放弃推理端 SGPA claim, 回到训练端故事:
       - Linear + whitening (EXP-102 89.26) 才是真正 dominant contribution
       - 论文叙事转向: "FedBN + pooled whitening broadcast" 足以解释 +6.20pp gain
```

**暂定**: 选 (C) — SGPA 推理太复杂, Linear+whitening 已经很强且可解释. 下一步做 EXP-106 pull 扩 seed 看 soft ETF pull 是否给训练端再提 0.5pp.

## 📊 实验统计

- **总 runs**: 1 script × 5 配置 × 4 clients = 20 inferences
- **预估成本**: ~30 min CPU (无需 GPU)
- **启动**: 待 EXP-097 至少 1 seed 跑完后启动 (约 1.5h 后)
- **完成**: 待填

## 📎 相关文件

- EXPERIMENT_PLAN: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/EXPERIMENT_PLAN.md`
- 依赖 checkpoint: `experiments/ablation/EXP-096_sgpa_smoke/results/*.json` (model state + source stats)
- 代码 (待写): `FDSE_CVPR25/scripts/run_sgpa_inference.py`
- 参考: `feddsa_sgpa.py` 里 `Client.test_with_sgpa` (已实现核心逻辑)
