---
date: 2026-04-28 启动, 2026-04-29 完成
type: 完整 4-method × N-seed 主表 + cold path 诊断 dump 实验
status: ✅ 完成 (PACS 8/8 R100 sc 高质量, Office 10 R100 V100)
exp_id: EXP-135
goal: 用诊断 hook 完整对比 4 method (F2DC vanilla / F2DC+DaA / PG-DFC / PG-DFC+DaA), 找 PG-DFC 真实优缺点
---

# EXP-135: 完整 4-method 主表 + cold path 诊断对比

## 一句话总览

**4 method (F2DC vanilla / F2DC+DaA / PG-DFC / PG-DFC+DaA) × Office (3 seed) + PACS (2 seed) 全套 R100**, 全部带诊断 hook (round_*.npz light + best/final heavy dump). 用于 cold path mine PG-DFC vs F2DC 的 7 维度差异.

## 实验配置

### 公共参数 (跟 EXP-130/131 主表对齐)

- **Backbone**: ResNet-10 (跟 F2DC 论文一致)
- **Allocation**: fixed `photo:2, art:3, cartoon:2, sketch:3` (PACS) / `caltech:3, amazon:2, webcam:2, dslr:3` (Office)
- **Data ratio**: PACS 30% / Office 20% per client (跟 F2DC paper Sec 5.1 一致)
- **R=100 communication, E=10 local epoch**
- **Optimizer**: SGD lr=0.01 momentum=0.9 wd=1e-5
- **Seed**: PACS {15, 333}, Office {2, 15, 333}

### 4 method 命令模板

```bash
# F2DC vanilla
python main_run.py --model f2dc --dataset fl_pacs --use_daa False --seed 15 --dump_diag <dir>

# F2DC + DaA
python main_run.py --model f2dc --dataset fl_pacs --use_daa True --agg_a 1.0 --agg_b 0.4 --seed 15 --dump_diag <dir>

# PG-DFC vanilla
python main_run.py --model f2dc_pg --dataset fl_pacs --use_daa False --seed 15 \
  --pg_proto_weight 0.3 --pg_attn_temperature 0.5 --pg_server_ema_beta 0.8 \
  --pg_warmup_rounds 30 --pg_ramp_rounds 20 --dump_diag <dir>

# PG-DFC + DaA (default 主方案)
python main_run.py --model f2dc_pg --dataset fl_pacs --use_daa True --agg_a 1.0 --agg_b 0.4 --seed 15 \
  --pg_proto_weight 0.3 --pg_attn_temperature 0.5 --pg_server_ema_beta 0.8 \
  --pg_warmup_rounds 30 --pg_ramp_rounds 20 --dump_diag <dir>
```

### 诊断 dump 触发设置

- **Light dump**: 每 round 一个 `round_NNN.npz` (~50KB), 存 sample_shares, daa_freqs, online_clients, domain_per_client, per_domain_acc, grad_l2, layer_l2, local_protos (PG-DFC only)
- **Heavy dump**: best round + final R100 各一个 (~14MB), 存 features, labels, preds, logits, confusion, state_dict
- **Heavy throttle**: warmup=30, gain≥1pp, 间隔≥5 round (防 dump 过频)

## 服务器分工

| Server | 实验 | 跑完情况 |
|---|---|:--:|
| **sc6** (4090 24GB) | PACS s=15: F2DC, F2DC+DaA, PG-DFC, PG-DFC+DaA + s=333: F2DC, PG-DFC+DaA | ✅ 6/6 R100 |
| **sc3** (4090 24GB) | PACS s=333: PG-DFC vanilla, F2DC+DaA | ✅ 2/2 R100 |
| **V100** (V100 32GB) | Office s=2/15/333 全套 (4 method × 3 seed = 12) + PACS s=2 副本 | ✅ Office 10 完整 (5 method × 2 seed) + PACS s=2 R5 死掉 |

注: V100 PACS 副本是因为 V100 4 并发太慢, sc6 跑完更快, V100 PACS 数据是双保险 (跨服务器一致性问题, 不进主表).

## 完成情况

### PACS 8/8 R100 (sc 主表)

| Method | s=15 | s=333 |
|---|:--:|:--:|
| F2DC vanilla | sc6 ✅ | sc6 ✅ |
| F2DC+DaA | sc6 ✅ | sc3 ✅ |
| PG-DFC vanilla | sc6 ✅ | sc3 ✅ |
| PG-DFC+DaA | sc6 ✅ | sc6 ✅ |

### Office 10 R100 (V100, 跨服务器一致性问题, 不进主表)

| Method | s=2 | s=15 | s=333 |
|---|:--:|:--:|:--:|
| F2DC vanilla | ✅ | ✅ | (无, 没要求) |
| F2DC+DaA | ✅ | ✅ | ✅ |
| PG-DFC vanilla | ✅ | ✅ | (无) |
| PG-DFC+DaA | ✅ | ✅ | ✅ |

## 主表数据 (PACS sc 高质量, 2-seed mean)

| Method | AVG_B | AVG_L | gap |
|---|:--:|:--:|:--:|
| 🥇 **F2DC vanilla** | **73.69** | 69.37 | 4.32 |
| 🥈 PG-DFC vanilla | 72.48 | 69.01 | 3.47 |
| 🥉 F2DC+DaA | 71.90 | 69.12 | **2.78** ⭐最稳 |
| ❌ **PG-DFC+DaA** | **71.52** ⚠️ | 68.07 ⚠️ | 3.45 |

详细数据回填到主表: `obsidian_exprtiment_results/2026-04-27/PG-DFC对比基线主表_完整结果.md` Table 1b

## 核心诊断发现

### 1. DaA 在 PACS 上是 negative (zero-sum 拉平器)

| Domain | sample/client | vanilla Last | +DaA Last | Δ |
|---|:--:|:--:|:--:|:--:|
| photo | 449 | 68.12 | 74.26 | +6.14 ⭐ |
| art | 552 | 53.31 | 60.91 | +7.60 ⭐ |
| cartoon | 631 | 75.54 | 74.36 | -1.18 |
| sketch | **1059** | **80.51** | **66.94** | **-13.57** ⚠️ |

**机制**: DaA 公式按 client sample 数拉平到 1/K=0.10. PACS sketch client sample 0.151 (主导 client) 被 DaA 降权 -31%, sketch 学习失去主力 → acc 跌 -13.57.

### 2. PG-DFC 在 PACS sketch 引发 mode collapse (s=333 严重)

| seed | PG sketch class pairwise cos | F2DC | Δ |
|---|:--:|:--:|:--:|
| s=15 | 0.570 | 0.621 | -0.051 (PG 略好) |
| **s=333** | **0.791** ⚠️ | 0.566 | +0.226 (PG 大崩) |

**机制**: PACS sketch 是黑白简笔画 outlier domain, server class_proto 被彩色 photo/art/cartoon 主导, sketch client 训练时被注入"彩色化 prototype" → 抹平 stroke 区分 → mode collapse.

### 3. PG-DFC 真实优点 (跨 dataset 公平对比)

| 优点 | Office 证据 | PACS 证据 |
|---|:--:|:--:|
| 训练稳定 (gap 小) | vanilla **-4.88pp** ⭐⭐ | vanilla -0.85 |
| dslr (罕见 domain) 救活 | +DaA last **+6.67** ⭐⭐ | (PACS 没对应稀有 domain) |
| 跨域一致性 | +DaA +0.018 | +DaA +0.030 ⭐ |
| Domain-invariance | +DaA -0.018 | +DaA -0.023 |

### 4. PACS 复现度差距 (vs 论文)

| 项 | 论文 | 我们 | Δ |
|---|:--:|:--:|:--:|
| F2DC w/o DaA | 75.33 | 73.69 | -1.64 |
| F2DC full | 76.47 | 71.90 | -4.57 |

可能原因: τ=0.06 跟 σ=0.1 等超参没 sweep (我们用默认 τ=0.1).

## 文件

- 实施代码: `F2DC/models/f2dc.py`, `f2dc_pg.py`, `models/utils/federated_model.py` (DaA + diag hook)
- 诊断 hook: `F2DC/utils/diagnostic.py`
- Logs: `experiments/ablation/EXP-135_diag_full/logs/{sc6,sc3,v100}/`
- Diagnostic dumps (~1.5GB, .gitignore 排除): `experiments/cold_path_analysis/diag_office/`, `diag_pacs/` (本地)
- 诊断分析脚本: `experiments/cold_path_analysis/*.py` (9 个)
- 主表回填: `obsidian_exprtiment_results/2026-04-27/PG-DFC对比基线主表_完整结果.md` Table 1b
- 知识笔记: `obsidian_exprtiment_results/知识笔记/大白话_F2DC原理与PG-DFC改造.md`
- 论文精读: `obsidian_exprtiment_results/知识笔记/论文精读_F2DC.md` §10 重大纠正

## 后续

- [ ] 超参 sweep (τ=0.06 / σ=0.1 / λ1=0.8 λ2=1.0) 复现到论文水平
- [ ] PG-DFC 改造: per-domain class_proto (避免 PACS sketch 被彩色 prototype 污染)
- [ ] Digits 数据集补全实验 (主表 Table 3 缺 PG-DFC+DaA)
