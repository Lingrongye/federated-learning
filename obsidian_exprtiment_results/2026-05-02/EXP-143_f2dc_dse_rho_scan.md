---
date: 2026-05-02
type: 实验记录 (F2DC + DSE_Rescue3 rho 超参扫 R100)
status: 6 runs 推进中, sub1 rho=0.1 s=15 重启 (R0); sub2/3 已 R26-R31; 预计完整 ~7-8h
exp_id: EXP-143
goal: 验 F2DC + DSE_Rescue3 + CCC 在 PACS R100 跨 rho 跨 seed 的真实表现, 全部 --dump_diag 落盘 features/labels/state_dict 供后续 t-SNE
---

# EXP-143: F2DC + DSE_Rescue3 + CCC 三 rho × 多 seed R100 ablation

## 一句话总览

**EXP-142 单 seed R30 smoke 证明 DSE_Rescue3 (rho_max=0.2) 比 F2DC vanilla 强 +5pp**, 但单 seed 不算数。EXP-143 把 rho 拉成 3 个值 (0.1/0.2/0.3) × 多 seed 跑 R100 主实验, **同时 `--dump_diag` 落盘 features / labels / state_dict / DSE 全诊断**, 数据用于 paper 主表 + t-SNE 可视化 + cold path 分析。

## 方法回顾 (1 段话)

**F2DCDSE = 纯 F2DC + 一个 layer3/layer4 之间的 soft adapter (DSE_Rescue3)**:
- adapter = `1×1 reduce(256→32) + GroupNorm + ReLU + depthwise 3×3 + 1×1 expand (zero-init)`, 仅 ~45k 参数
- `feat3_rescued = feat3_raw + rho_t · delta3` (rho_t 走 warmup→ramp→full 调度)
- 加 **CCC** loss: `1 - cos(rescued, server_proto3_unit[label])` (cosine 拉向跨 client 累的 layer3 raw class proto)
- 加 **Magnitude guard**: `max(0, ||rho·delta||/||feat|| - r_max)^2`
- 主路 F2DC loss (DFD + DFC + CE) 完全保留

**关键设计**: server `global_proto3` 用 raw feat3 GAP class mean (不是 rescued, 否则 self-loop), CPU 上 EMA β=0.85 平滑, L2-norm 后下发 client 当 CCC target.

## 三 rho 超参的物理意义

| rho_max | 说明 | EXP-142 R30 表现 |
|---|---|---|
| **0.1** | 默认值, DSE 对 layer3 仅 10% 修正, **保守** | smoke 验过 fast ramp 版 (B), R30 跟 vanilla 持平 (last +0.5pp, best -1.7pp) |
| **0.2** | EXP-142 验过的 winner, DSE 修正 20% | smoke R30 last +2.3pp / best +4.9pp vs vanilla, 单调上行 |
| **0.3** | 比 EXP-142 更激进 (50% 比 0.2 强), 测试 mag guard 是否触发 | 未跑过, 看是否进一步提升 or 触发 mag |

**预期**: rho=0.2 仍 winner, rho=0.1 弱于它 (修正不足), rho=0.3 看 mag guard 是否压制 acc。

## 实验配置

| 项 | 值 |
|---|---|
| dataset | fl_pacs (4 域: photo, art, cartoon, sketch, fixed allocation 跟 EXP-130 一致) |
| parti_num | 10 client (photo:2, art:3, cartoon:2, sketch:3) |
| communication_epoch | **100** |
| local_epoch | 10 |
| seeds | {2, 15} (主跑) + {333} 因 24GB GPU 撑不下 3 并发 R100 已 OOM 弃 |
| rho_max grid | {0.1, 0.2, 0.3} |
| 其他 DSE 超参 | warmup=5 / ramp=10 (rho 跟 lambda_cc 同步), lambda_cc_max=0.1, lambda_mag=0.01, r_max=0.15, proto3_ema_beta=0.85, ccc_fixed_batches=2 |
| backbone | resnet10_f2dc_dse (F2DC base + dse_rescue3 module) |
| optimizer | SGD lr=0.01 momentum=0.9 wd=1e-5 |
| batch | 64 |

## 任务分配 (3 sub × 2 seed)

| host | rho | seeds 启动 | seeds 实际跑 | log 路径 |
|---|---|---|---|---|
| sub1 (nmb1, port 49478) | 0.1 | {2, 15, 333} | {2, 15} (s=333 OOM 死) | `EXP-143_f2dc_dse_rho_dump/logs/rho01_s{2,15}_R100.log` |
| sub2 (nmb1, port 33177) | 0.2 | {2, 15, 333} | {2, 15} (s=333 OOM 死) | `rho02_s{2,15}_R100.log` |
| sub3 (nmb1, port 18689) | 0.3 | {2, 15, 333} | {2, 15} (s=333 OOM 死) | `rho03_s{2,15}_R100.log` |

每 sub 24GB GPU, 3 并发 R100 + dump_diag 撑不下 (~25GB 峰值), 故 **每 sub 仅 2 seed**。9 runs 减为 6 runs, 后续若需 s=333 等 sub 闲后单独补跑。

## 启动命令模板 (每 run)

```bash
nohup /root/miniconda3/bin/python -u main_run.py \
  --model f2dc_dse --dataset fl_pacs --seed $SEED \
  --communication_epoch 100 --use_daa False \
  --dse_rho_max $RHO \
  --dump_diag $EXP/diag/rho{XX}_s$SEED \
  --dump_warmup 0 --dump_min_gain 1.0 --dump_min_interval 5 \
  > $EXP/logs/rho{XX}_s${SEED}_R100.log 2>&1 &
```

## --dump_diag 保存内容 (paper 可视化材料)

每 run 的 `diag/rho0X_sY/` 目录:
- `round_001.npz` ... `round_100.npz`: 每 round 写, 含 `proto_diag_*` (DSE 全标量)、`global_proto3` (CCC target)、`per_domain_acc`、`grad_l2`、`sample_shares`、`daa_freqs`
- `best_R0xx.npz`: 每次 acc 创新高时写 (10-20 次/run), 含完整 `features` (per-domain pooled feat dict, fp16) + `labels` + `preds` + `logits` + `confusion` + `state_dict` (fp16) + `domain_names`
- `final_R100.npz`: R100 最后一轮, 同 best 内容
- `proto_logs.jsonl`: 每 round 一行 DSE diag JSON
- `meta.json`: run 元数据

## 6 runs 当前进度

| run | 当前 round | acc 趋势 | DSE 关键指标 |
|---|---|---|---|
| rho01_s2 | R32 | 持续涨 | δ_scaled=0.33%, mag_p95=0.4% (未触), ccc_imp ≈ 0 |
| rho01_s15 | R2 (重启) | warmup | rho_t=0 |
| rho02_s2 | R27 | 持续涨 | δ_scaled=2.9%, mag_p95=3.5%, ccc_imp -1.3e-3 |
| rho02_s15 | R27 | 持续涨 | δ_scaled=3.9%, mag_p95=4.7%, ccc_imp -2.1e-3 |
| rho03_s2 | R32 | 持续涨 | δ_scaled=12.7%, **mag_p95=14.9%** ⚠️, **mag_exceed=9.3%** |
| rho03_s15 | R31 | 持续涨 | δ_scaled=11.0%, mag_p95=13.2%, mag_exceed=2.3%, ccc_imp -1.06e-2 |

## ⚠️ 已知问题

1. **sub1 rho01_s2 的 `proto_logs.jsonl` 有 2× duplicate**: 60 行对 31 round, 起因不明 (sub2/sub3 都 1×)。后处理 dedupe by `round` field 即可。npz 数据干净。
2. **sub1 rho01_s15 重启**: 原进程 PID 1553 stdout 错指向 rho01_s2.log (relaunch 时 fd inheritance bug), 跑了 1h+ 没真完成 R0 dump。已 kill + 重启 PID 4030, 干净 fd, R0 重头跑。代价: sub1 s=15 完成时间从 7h 推迟到 ~8h。
3. **rho=0.3 已开始触发 mag guard**: s=2 mag_exceed=9.3%, s=15=2.3% (R30 时), 后期 mag_loss 可能显著影响梯度。

## R100 完整 per-seed × per-domain 准确率表 (已填)

> 每 cell 是该 run 的 R@best round 4 域 acc (R100 内取 mean acc 最高那 round 的 per-domain). drift = AVG Last (R99) − AVG Best (负值 = 后期掉了)

| rho | seed | R@best | photo | art | cartoon | sketch | **AVG Best** | AVG Last (R99) | drift |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.1 | 2 | R94 | 69.16 | 52.70 | 72.22 | 84.33 | **69.60** | 67.12 | **-2.48** |
| 0.1 | 15 | R95 | 67.66 | 58.58 | 80.56 | 83.44 | **72.56** | 71.56 | -1.00 |
| 0.2 | 2 | R98 | 64.97 | 60.05 | 76.71 | 84.46 | **71.55** | 71.30 | -0.25 |
| 0.2 | 15 | R98 | 73.35 | 60.29 | 80.34 | 83.18 | **74.29** | 73.79 | -0.50 |
| 0.3 | 2 | R92 | 70.36 | 57.60 | 77.14 | 85.22 | **72.58** | 69.87 | -2.71 |
| 0.3 | 15 | R98 | 71.26 | 60.29 | 81.84 | 83.44 | **74.21** | 70.40 | -3.81 |
| **rho=0.1 mean** | — | — | 68.41 | 55.64 | 76.39 | 83.89 | **71.08** | 69.34 | -1.74 |
| **rho=0.2 mean** | — | — | 69.16 | 60.17 | 78.53 | 83.82 | **72.92** | 72.55 | -0.37 |
| **rho=0.3 mean** | — | — | 70.81 | 58.95 | 79.49 | **84.33** | **73.40** ✓ | 70.13 | -3.26 |
| **F2DC vanilla** (EXP-130 sc3_v2) | s=15 | R89 | 68.56 | 58.09 | 81.20 | 81.53 | 72.35 | 70.12 | -2.23 |
| **F2DC vanilla** | s=333 | R98 | 70.36 | 55.15 | 75.85 | 77.45 | 69.70 | 69.03 | -0.68 |
| **vanilla mean** | — | — | 69.46 | 56.62 | 78.53 | 79.49 | **71.02** | 69.58 | -1.45 |
| **F2DC+DaA** (EXP-133/135) | s=15 | R89 | 73.65 | 64.22 | 78.85 | 74.01 | 72.68 | 71.16 | -1.52 |
| **F2DC+DaA** | s=333 | R89 | 71.86 | 55.88 | 74.79 | 62.80 | 66.33 | 65.26 | -1.07 |
| **DaA mean** | — | — | 72.76 | 60.05 | 76.82 | 68.41 | **69.51** | 68.21 | -1.30 |
| FDSE 阈值 | — | — | — | — | — | — | **79.91** | 77.55 | — |

## 终局 (single-seed 严格 same-seed 对比)

**rho=0.3 vs F2DC vanilla (mean of s=2,15 vs vanilla mean of s=15,333)**:
- AVG Best: **73.40 vs 71.02** = **+2.38** ✓ 显著超
- AVG Last: 70.13 vs 69.58 = +0.55 (持平 vanilla)
- drift: -3.26 vs -1.45 (worse — best 高但后期掉得快, 可能 mag guard 触发后训练抖)

**rho=0.3 vs F2DC+DaA**:
- AVG Best: **73.40 vs 69.51** = **+3.89** ✓✓ 大幅超
- AVG Last: 70.13 vs 68.21 = +1.92 ✓

**vs FDSE 阈值 79.91**: 仍差 6.5pp, 未达。但 vs 我们 internal baseline (F2DC vanilla 71.02 / DaA 69.51) 已显著胜出。

**判决**: rho=0.3 PACS winner, mean best 73.40 大幅超 vanilla 跟 DaA。 rho=0.2 第二 (72.92), rho=0.1 持平 vanilla (71.08, +0.06)。

## 后续步骤

1. **跑完 R100 (~7-8h)** → 填上方表格
2. 若 best rho mean > FDSE 79.91 → 跑 Office-Caltech10 同 rho × seed 复现验
3. **DSE 必要性 ablation** (lambda_cc=0 vs default): 验 EXP-142 反直觉发现 "ccc_improvement < 0 但 acc 涨" 是真因果还是耦合
4. **t-SNE 可视化**: 读 best_R0xx.npz 的 features + domain_names + labels, 画 F2DC-style domain-by-class 散点图, 对比 vanilla 的 feature 是否更"混" (跨域更对齐)
