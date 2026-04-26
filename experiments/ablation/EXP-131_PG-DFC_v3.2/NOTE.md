# EXP-131 | PG-DFC v3.2 主实验 + ablation (PACS)

## 基本信息
- **日期**: 2026-04-26 深夜启动 / 2026-04-27 凌晨完成
- **服务器**: sc5 (westc:15576, GPU 0 RTX 4090 24GB 完全空闲)
- **目的**: 第一次实测 PG-DFC v3.2 在 F2DC PACS K=10 setup 下能否涨过 F2DC vanilla
- **状态**: 🟡 启动中

## 这个实验做什么 (大白话)

我们设计的 PG-DFC (Prototype-Guided DFC) 经过 4 轮 review 迭代:
1. v1: 初版 (基于 paper 推理)
2. v2: M1/M2 fix (实测 F2DC client 分布,确认 batch-mean EMA 噪声 + sample-weighted skew)
3. v3: NV1-NV4 fix (cosine attention + detach + 不跨 round 平滑 + server EMA)
4. v3.2: NN 训练专家第四轮关键 review — query 用 r_pooled (不是 nr_pooled), 加 NaN safety

今晚的实验回答 3 个核心问题:
- Q1: PG-DFC v3.2 在 F2DC PACS K=10 setup 下能不能涨? 涨多少?
- Q2: 各个超参 (proto_weight, attn_temperature, server_ema_β) 敏感度如何?
- Q3: NN 训练专家警告的几个隐患 (cosine attn τ=0.1 梯度悬崖 / server EMA cold start race) 是否真发生?

## 实验配置 (对齐 EXP-130 baseline)

### 数据集
- **PACS K=10** (F2DC paper 原 setup), percent=30%, 4 域 (photo/art/cartoon/sketch), 7 类
- backbone: resnet10_dc_pg (128×128, BasicBlock 1-1-1-1)

### 训练 hyperparameters
- communication_epoch = 100 (对齐 F2DC paper)
- local_epoch = 10
- local_lr = 0.01, batch_size = 64, SGD momentum=0.9 wd=1e-5
- F2DC 原超参: gum_tau=0.1, tem=0.06, λ1=0.8, λ2=1.0

### PG-DFC v3.2 超参
- proto_weight (target after warmup+ramp): grid {0, 0.3, 0.5}
- attn_temperature: grid {0.1, 0.3, 0.5}
- server_ema_beta: grid {0, 0.8}
- warmup_rounds = 30, ramp_rounds = 20

### Seeds
- 第一波 (sanity + ablation): seed=2 单 seed, 6 runs
- 第二波 (best config 3-seed 验证): seed={2, 15, 333}, 选最优 1 个 config

## 启动跑次表

### Wave 1: 6 个核心 runs (seed=2, R=100, 单 seed 探索)

| run | label | proto_weight | attn_τ | server_β | warmup/ramp | 验证什么 |
|-----|-------|:--:|:--:|:--:|:--:|---|
| **R0** | sanity_pw0 | 0.0 | (n/a) | (n/a) | 0/0 | 退化等价 F2DC vanilla (基线对齐) |
| **R1** | full_v3.2 | 0.3 | 0.3 | 0.8 | 30/20 | **主推荐配置** |
| **R2** | grid_τ0.5 | 0.3 | 0.5 | 0.8 | 30/20 | NN 专家建议从 τ=0.5 起 grid (RV3) |
| **R3** | grid_τ0.1 | 0.3 | 0.1 | 0.8 | 30/20 | NN 专家警告"梯度悬崖", 验证是否真崩 |
| **R4** | no_server_ema | 0.3 | 0.3 | 0.0 | 30/20 | NV4 ablation: 验证 server EMA 必要性 |
| **R5** | grid_pw0.5 | 0.5 | 0.3 | 0.8 | 30/20 | proto_weight grid |

### Wave 2: 3-seed 主表验证 (W1 best config)

待 Wave 1 出结果后, 选 best run 跑 seed={15, 333} 补 3-seed.

## 关键诊断指标 (训练时记录)

代码已加 hook (`backbone/ResNet_DC_PG.py: DFC_PG._diag_*`):
- `mask_sparsity_mean` — 验证 C1 (mask 是否塌缩)
- `attn_entropy_mean` — 验证 M4 (attention 是否塌缩到 one-hot)
- `proto_signal_ratio_mean` — 验证 M3 (proto vs rec_units 量级比)
- `global_proto_norm_mean` — 验证 NV4/NV5 (NaN safety)

每 round 自动写入 `models.f2dc_pg.proto_logs`, 跑完保存到 metrics.json.

## 预期 (基于 NN 训练专家第四轮 review)

| 跑次 | 预期 PACS AVG | 上下浮动原因 |
|:----:|:----:|---|
| **R0** sanity_pw0 | F2DC ±0.3pp | 应该 ≈ 76 左右 (跟 EXP-130 F2DC baseline 对齐) |
| **R1** full v3.2 | **76.5 ~ 77.5pp** | 主推荐, 期望 +0.5~1.0pp (R query 优化使期望略高) |
| **R2** τ=0.5 | **77.0 ~ 78.0pp** | NN 专家预测 best case |
| **R3** τ=0.1 | 74.5 ~ 76.0pp | 梯度悬崖警告 — 可能崩或不涨 |
| **R4** β=0 | 76.0 ~ 77.0pp | 在 K=10 setup 下 server EMA 应该有用 |
| **R5** pw=0.5 | 76.0 ~ 77.5pp | proto 主导, 可能 over-regularize |

## 必跑前 sanity (smoke test 已通过)

✅ Smoke #1 PACS pw=0 R=5 → R0:10.5 R1:11.5 R2:20.6 R3:23.2 R4:22.3 (跟 EXP-130 F2DC vanilla 一致)
✅ Smoke #2 PACS pw=0.3 R=6 → R5:21.9 (有数字, 不崩)
✅ Unit test: forward + backward + attention grad norm 正常

## 实验状态追踪

| Wave | 状态 | 启动时间 | 完成时间 | 输出 |
|------|:----:|:----:|:----:|------|
| Wave 1 (6 runs) | 🟡 启动中 | 2026-04-26 深夜 | 待定 | 待回填 |
| Wave 2 (best × 2 seed) | ⏳ | 等 Wave 1 | 待定 | 待回填 |

## 结果 (跑完后回填)

### Wave 1 数字 (PACS R=100 seed=2)

| run | proto_weight | attn_τ | server_β | AVG | photo | art | cartoon | sketch | 备注 |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|------|
| R0 sanity_pw0 | 0 | - | - | TBD | TBD | TBD | TBD | TBD | 应 ≈ F2DC vanilla |
| R1 full v3.2 | 0.3 | 0.3 | 0.8 | TBD | TBD | TBD | TBD | TBD | **主推荐** |
| R2 τ=0.5 | 0.3 | 0.5 | 0.8 | TBD | TBD | TBD | TBD | TBD | |
| R3 τ=0.1 | 0.3 | 0.1 | 0.8 | TBD | TBD | TBD | TBD | TBD | 梯度悬崖警告 |
| R4 β=0 | 0.3 | 0.3 | 0.0 | TBD | TBD | TBD | TBD | TBD | NV4 ablation |
| R5 pw=0.5 | 0.5 | 0.3 | 0.8 | TBD | TBD | TBD | TBD | TBD | |

### 诊断指标 (跑完后回填)

| run | mask_sparsity 演化 | attn_entropy | proto_signal_ratio | NaN/explosion? |
|-----|:--:|:--:|:--:|:--:|
| R1 | TBD | TBD | TBD | TBD |
| ... | TBD | TBD | TBD | TBD |

### 结论 (跑完后写)

TBD

## 时间预算

| 阶段 | 估算 |
|------|:----:|
| Wave 1 (6 runs × R=100, GPU 4 并行) | 6/4 wave × 1.5h = 2.25h |
| Wave 2 (best 3-seed 补 2 runs) | 2/2 wave × 1.5h = 1.5h |
| **总计** | **~3.75h wall** |

## 下一步 (起床后人接管)

1. 看 Wave 1 数字, 确认哪个 config 是 best
2. 如果 best > F2DC + 1pp: 进入 Wave 2 三 seed
3. 如果 best ≤ F2DC: 看诊断指标判断是哪个隐患触发, 决定 v3.3 调整
4. 同步结果到 obsidian 笔记
