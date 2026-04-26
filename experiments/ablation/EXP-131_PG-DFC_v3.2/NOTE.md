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
| Wave 1a PACS R0/R1/R2 | 🟡 跑中 (R6, ~282s/round) | 04:02-04:27 | ~12:00 | logs/ |
| Wave 1a PACS R3/R4/R5 | ❌ 跳过 (时间不够) | - | - | 起床后手动启 |
| Wave 1b Office × 4 | ⏳ 待 R0/R1/R2 完成 | - | ~12:30 | 由 auto_wave2_v2 触发 |
| Wave 2 default × seed=15+333 (PACS+Office) | ⏳ 待 Office 完成 | - | ~16:00 | 由 auto_wave2_v2 触发 |

## 实施过程踩的坑 (供回看)

### 坑 1: Stdout buffering 导致 25 min log 一片空白
- 现象: 启动后 25 min log 只有 import warning,但 GPU 占 22GB 86% util (在跑只是 print 没 flush)
- 原因: nohup ... > log 时 Python print fully-buffered (非 line-buffered for tty)
- Fix: 加 `PYTHONUNBUFFERED=1 python -u` (v2 launcher)

### 坑 2: 6 进程并行 (2 PACS + 4 Office) 让单 PACS 慢 3x
- 现象: 单 PACS round 244s (vs 单跑应 80s)
- 原因: GPU 100% 共享, PACS 计算重 (128x128 input + ResNet) 受 office 抢占
- Fix: 杀掉 office, 维持 3 PACS 并行 (但实际还是 282s/round, 没更快 — 3 个 PACS 互相抢 GPU 算力)

### 坑 3: setproctitle 改进程名,pgrep -f main_run.py 找不到
- 现象: `pgrep -af main_run.py` 显示 0 个进程,但 nvidia-smi 有 3 个 GPU 进程
- 原因: setproctitle 改成 `f2dc_pg_10_fl_pacs_100_10`
- Fix: 用 `nvidia-smi --query-compute-apps=pid` 或 `ps -ef | grep f2dc_pg`

## 真实跑次表 (已部署)

### Stage 0: smoke (已完成)
- ✅ PACS pw=0 R=5: 10.5→23.2 (退化等价 F2DC vanilla)
- ✅ PACS pw=0.3 R=6: 11.0→24.9 (full v3.2 不崩)

### Stage 1: PACS Wave 1a (跑中)
- R0 sanity_pw0 (PID 6955) — 04:02 启动
- R1 full_v32 (PID 7202) — 04:03 启动
- R2 tau05 (PID 9414) — 04:27 启动 (office 杀后 launcher 自动启)

### Stage 2: Office Wave 1 (auto 触发)
- R0/R1/R2/R4 office × seed=2 — 等 PACS R0/R1/R2 完成后启

### Stage 3: Wave 2 default × seed=15+333 (auto 触发)
- 配置: pw=0.3, τ=0.5, β=0.8 (NN 专家预测 best case)
- PACS × 2 seeds + Office × 2 seeds = 4 runs

## 早期数字 (R0-R6, 04:30 截止)

| run | R0 | R1 | R2 | R3 | R4 | R5 | R6 | trend |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| R0 sanity_pw0 | 10.31 | 20.78 | 19.43 | 19.18 | 21.58 | - | 29.57 | 单调上涨 ✓ |
| R1 full_v32 | 10.81 | 21.92 | 23.01 | 21.33 | 23.54 | - | 27.32 | 跟 R0 几乎一致 (warmup 期 pw=0 等价) |
| R2 tau05 | 10.69 | - | - | - | - | - | - | 04:27 启动, 还在 R0 |

**早期判断**:
- ✅ Sanity pw=0 跟 F2DC vanilla 完全一致 (退化路径正确)
- ✅ R1 full v3.2 在 warmup 期数字跟 R0 一致 (proto_weight=0 等价)
- ⏳ 等 R30+ (proto_weight 启动后) 才能看 PG-DFC vs vanilla 真实差异

## R7 进度更新 (05:43)

| run | R6 | R7 | trend |
|-----|:--:|:--:|---|
| R0 sanity_pw0 | 29.57 | 29.97 | 单调上涨 ✓ |
| R1 full_v32 | 27.32 | 30.22 | 单调上涨 ✓, 略高于 R0 (虽 warmup 期等价 — 可能 proto buffer 初始化噪声不同) |
| R2 tau05 | (启动晚) | (R1: 23.02) | 在 R1, 单调上涨 ✓ |

### 踩的"假停"坑 (供参考)
- 中间监控时用 `tail -20 ... | grep ... | tail -8` 显示 R0 还在 R6, 误判 1h 没新数据
- 实际 grep 全文找到 R7 已完成
- 教训: 不要用 chained tail+grep+tail, 直接 `grep 'Acc' log | tail -3` 看真实最新

## R18 进度更新 (06:33)

| run | R7 | R18 | trend | round time |
|-----|:--:|:--:|---|:--:|
| R0 sanity_pw0 | 29.97 | **52.15** | 单调上涨 ✓ | 282 → 640s/round (3并行+attention 计算后变慢) |
| R1 full_v32 | 30.22 | **50.97** | 单调上涨 ✓ (跟 R0 同步, warmup 期等价) | 同上 |
| R2 tau05 | (R1: 23.02) | **R12: 40.47** | 单调上涨 ✓ (启动晚, 进度正常) | 同上 |

### 时间表修正

实测 round time ~640s/round (慢于预期 282s),100 round 需要 17.8h。

| 时间 | 预计事件 |
|------|---|
| **08:00 用户起床** | R0/R1 ~R30 (proto_weight 刚开始 ramp), R2 ~R20 |
| 10:00 | R0/R1 R45 (proto 全开), R2 R30 — 关键观察期 |
| 13:00 | R0/R1 R65, R2 R55 |
| **17:00** | R0/R1 R85+, R2 R75 |
| **21:00** | R0/R1 完成 R100, R2 R95+ |
| 21:30 | R2 完成, auto_wave2_v2 触发 Office Wave |
| 22:00 | Office Wave 完成, Wave 2 PACS seed=15+333 启动 |
| **次日 06:00** | Wave 2 PACS 完成, Wave 2 Office 启动 |
| **次日 06:30** | 全部完成 |

### 起床后能看到的内容

- ✅ 完整 sanity 验证 (退化等价 F2DC, R0 跟 R1 在 warmup 期一致)
- ✅ 早期收敛趋势 (R0-R30 acc 数据点)
- ⏳ PG-DFC vs vanilla 真实差异 (proto_weight 启动后, ~R40 起)
- ❌ 完整 R=100 数字 (要等到 21:00)

### 起床后的 cheatsheet

```bash
# 查 PACS R0/R1/R2 当前 round 进度
ssh sc5 "for log in /root/autodl-tmp/federated-learning/experiments/ablation/EXP-131_PG-DFC_v3.2/logs/R*pacs_seed2.log; do echo \$log; grep Acc \$log | tail -3; echo; done"

# 查 GPU 状态
ssh sc5 "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader"

# 如果想加速 (可选, 起床后做): 杀 R2 让 R0/R1 加速 ~2x
# ssh sc5 "kill -9 9414"  # 注意: pid 可能变, 用 nvidia-smi 查最新

# 如果 R0/R1 已完成, 可以跑剩余 ablation R3 R4 R5
ssh sc5 "PYTHONUNBUFFERED=1 nohup /root/miniconda3/bin/python -u /root/autodl-tmp/federated-learning/F2DC/main_run.py --device_id 0 --communication_epoch 100 --local_epoch 10 --parti_num 10 --model f2dc_pg --dataset fl_pacs --pg_proto_weight 0.3 --pg_attn_temperature 0.1 --pg_server_ema_beta 0.8 --pg_warmup_rounds 30 --pg_ramp_rounds 20 --num_classes 7 --seed 2 > /root/autodl-tmp/federated-learning/experiments/ablation/EXP-131_PG-DFC_v3.2/logs/R3_tau01_pacs_seed2.log 2>&1 &"
```

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
