---
date: 2026-05-02
type: 实验记录 (F2DC + DSE_Rescue3 Office-Caltech10 R100, rho={0.2,0.3} × s={15,333})
status: 4 runs 全 R100 完成
exp_id: EXP-145
goal: 验 EXP-143 PACS 上 rho=0.3 winner 的结论是否在 Office 上复现
---

# EXP-145: F2DC + DSE_Rescue3 Office-Caltech10 rho 扩展实验

## 一句话总览

**EXP-143 PACS 上 rho=0.3 mean best 73.40 大幅超 vanilla (71.02) 跟 DaA (69.51), 在 Office 上复现 + 看 DSE 是否仍有效**。结果:Office 上 DSE 优势平淡 — rho=0.3 mean best 60.98 vs vanilla 60.56 仅 +0.42 (持平), vs DaA 63.55 -2.57 (输 DaA)。结论: **Office 弱 stylistic shift, DaA (按域加权) 更适合**, DSE 在这个 setting 没有发挥空间。

## 实验配置

| 项 | 值 |
|---|---|
| dataset | fl_officecaltech (4 域: caltech, amazon, webcam, dslr) |
| parti_num | 10 client (caltech:3, amazon:2, webcam:2, dslr:3 fixed allocation) |
| communication_epoch | 100 |
| local_epoch | 10 |
| seeds | {15, 333} (跟主表 baseline 对齐) |
| rho_max | {0.2, 0.3} (EXP-143 PACS 的 winner 跟次优, 跳过 0.1) |
| 其他 DSE 超参 | warmup=5 / ramp=10 (rho 跟 lambda_cc 同步), lambda_cc_max=0.1, lambda_mag=0.01, r_max=0.15 |
| 服务器 | sub3 (autodl nmb1, RTX 4090 24GB) |
| 单 run wall | ~37min (Office 数据小, 比 PACS 200s/round 快得多, 仅 22s/round) |
| 4 并发显存 | ~6GB / 24GB (Office 最省显存) |

## 启动命令

```bash
EXP=experiments/ablation/EXP-145_f2dc_dse_office_rho03
PY=/root/miniconda3/bin/python
F2DC=/root/autodl-tmp/federated-learning/F2DC

for SEED in 15 333; do
  for RHO in 0.2 0.3; do
    nohup bash -c "cd $F2DC && $PY -u main_run.py \
      --model f2dc_dse --dataset fl_officecaltech --seed $SEED \
      --communication_epoch 100 --use_daa False --dse_rho_max $RHO \
      --dump_diag $EXP/diag/rho{XX}_s$SEED \
      --dump_warmup 0 --dump_min_gain 1.0 --dump_min_interval 5" \
      > $EXP/logs/office_rho{XX}_s${SEED}_R100.log 2>&1 &
  done
done
```

## R100 完整 per-seed × per-domain 准确率表

> 每 cell = R@best round 4 域 acc (R100 内 mean acc 最高那 round 的 per-domain). drift = AVG Last (R99) − AVG Best.

| rho | seed | R@best | caltech | amazon | webcam | dslr | **AVG Best** | AVG Last (R99) | drift |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.2 | 15 | R98 | 64.73 | 77.37 | 55.17 | 43.33 | **60.15** | 56.49 | **-3.66** |
| 0.2 | 333 | R85 | 67.86 | 83.68 | 43.10 | 43.33 | **59.49** | 58.67 | -0.82 |
| 0.3 | 15 | R97 | 66.96 | 74.74 | 53.45 | 46.67 | **60.46** | 56.64 | -3.82 |
| 0.3 | 333 | R63 | 65.62 | 73.68 | 50.00 | **56.67** | **61.49** | 56.94 | **-4.55** |
| **rho=0.2 mean** | — | — | 66.30 | 80.53 | 49.14 | 43.33 | **59.82** | 57.58 | -2.24 |
| **rho=0.3 mean** | — | — | 66.29 | 74.21 | 51.73 | 51.67 | **60.98** ✓ | 56.79 | -4.19 |
| **F2DC vanilla** (EXP-130 sc3_v2) | s=15 | R98 | 63.84 | 75.79 | 56.90 | 46.67 | 60.80 | 54.62 | -6.18 |
| **F2DC vanilla** | s=333 | R98 | 63.84 | 78.95 | 55.17 | 43.33 | 60.32 | 58.74 | -1.58 |
| **vanilla mean** | — | — | 63.84 | 77.37 | 56.04 | 45.00 | **60.56** | 56.68 | -3.88 |
| **F2DC+DaA** (EXP-133) | s=15 | R99 | 62.50 | 72.63 | **67.24** | 53.33 | 63.93 | 63.93 | 0 (best=last) |
| **F2DC+DaA** (EXP-135) | s=333 | R67 | 64.29 | 74.74 | 60.34 | 53.33 | 63.18 | 60.22 | -2.96 |
| **DaA mean** | — | — | 63.40 | 73.69 | 63.79 | 53.33 | **63.55** ✓ | 62.07 | -1.48 |

## 终局对比 (mean best, 2-seed)

| Method | mean best | vs vanilla 60.56 | vs DaA 63.55 |
|---|---|---|---|
| **F2DC+DaA (winner)** | **63.55** | +2.99 ✓ | — |
| F2DC vanilla | 60.56 | — | -2.99 |
| **EXP-145 rho=0.3** | 60.98 | +0.42 (持平) | **-2.57 ❌** |
| EXP-145 rho=0.2 | 59.82 | -0.74 (略输) | -3.73 ❌ |

## 关键 finding

1. **DaA 在 Office 上是真正的 winner** (63.55 vs vanilla 60.56 +2.99pp), DSE 反而输 DaA 2.6pp
2. **DSE rho=0.3 跟 vanilla 几乎持平**, 没有像 PACS 上那样的 +2.38pp 优势
3. **per-domain 看**: 
   - caltech / amazon: 跨方法持平 (~64-67% / ~73-83%, 容易类)
   - **webcam / dslr 差异最大** — DaA 在 webcam (67.24 vs vanilla 56.90 = +10pp) / dslr (53.33 vs 43.33 = +10pp) 大胜
   - DSE 在 webcam (49-51) / dslr (43-52) 跟 vanilla 几乎一样, 没补到 DaA 那种 +10pp

## 为什么 Office 上 DSE 不强?

1. **Office 域差异是"采集设备级别"** (amazon=高清产品图, webcam/dslr=低分辨率拍照, caltech=mixed网络爬), 不像 PACS (photo/sketch/cartoon/art) 是 stylistic transformation
2. **dslr 样本极少** (~157 张, 跟 amazon ~958, webcam ~295 比), DSE adapter 的 layer3 校正信号在 dslr 上太稀疏, 拉不出 lift。 DaA 是按 client 加权聚合, 给 dslr (3 client / 10) 公平权重, 反而保住 dslr
3. **DSE 拉跨域 layer3 mean** = 朝 amazon (大 client + 高质量) 方向拉, 让 webcam/dslr 反而被推到错方向

## 对比 EXP-143 PACS

| dataset | DSE 表现 | DaA 表现 | DSE 优势? |
|---|---|---|---|
| **PACS** (EXP-143) | rho=0.3 73.40 +2.38 vs vanilla | DaA mean 69.51 输 vanilla -1.51 | **DSE > DaA +3.89** ✓ DSE clear winner |
| **Office** (EXP-145) | rho=0.3 60.98 +0.42 vs vanilla | DaA mean 63.55 +2.99 vs vanilla | **DSE < DaA -2.57** ❌ DaA winner |

## paper 价值

负结果也是 paper 重点:
- **DSE 适用 stylistic domain shift (PACS)**, 不适用 acquisition-device shift (Office)
- 跟 EXP-146 Digits 上 DSE 在 high-rho 下灾难性下跌一致, **DSE 有明确适用边界**
- paper 写成: "DSE complementary to DaA — 二者各有 setting 适用"

## 后续可能 ablation

1. **DSE + DaA combo** (`--use_daa True --dse_rho_max 0.3` Office), 看能否反超 DaA-only 63.55
2. **lambda_cc=0** (关 CCC), 验是否纯 adapter 也行 — 跟 EXP-142 反直觉发现一致

## 数据保存

- **logs**: `experiments/ablation/EXP-145_f2dc_dse_office_rho03/logs/*.log` (4 个 log, 每 ~85KB)
- **diag npz** (rsync 不入 git): `EXP-145_f2dc_dse_office_rho03/diag/rho{02,03}_s{15,333}/` 各含 round_001-100.npz + best_R0xx.npz + final_R100.npz + meta.json + proto_logs.jsonl
