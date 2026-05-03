---
date: 2026-05-03
type: 实验记录 (Office-Caltech10 高 rho 扫: 0.5/0.7/1.0 × s={15,333})
status: 6 runs R100 全完成
exp_id: EXP-148
goal: EXP-145 rho=0.3 mean 仅 60.98 (vs vanilla 60.56 +0.42 持平), 试 rho 拉大看是否能突破 plateau, 找 Office DSE sweet spot
---

# EXP-148: F2DC + DSE_Rescue3 Office 高 rho 扫

## 一句话总览

**EXP-145 Office rho=0.3 仅 +0.42pp 几乎持平 vanilla, 试更大 rho={0.5, 0.7, 1.0} 找 sweet spot。结果 rho=0.5 mean 62.09 是真胜利 (+1.53 vs vanilla, 跟 DaA 差距从 -2.57 缩到 -1.46)**, rho=0.7/1.0 反而下降 (信号过强扰动 main loss)。**Office DSE sweet spot = rho=0.5**。

## 配置

| 项 | 值 |
|---|---|
| dataset | fl_officecaltech (4 域: caltech, amazon, webcam, dslr) |
| parti_num | 10 client (caltech:3, amazon:2, webcam:2, dslr:3) |
| communication_epoch | 100 |
| local_epoch | 10 |
| seeds | {15, 333} |
| rho_max grid | **{0.5, 0.7, 1.0}** (跟 EXP-145 的 {0.2, 0.3} 互补, 完整覆盖) |
| 其他 DSE 超参 | warmup=5 / ramp=10, lambda_cc=0.1, lambda_mag=0.01, r_max=0.15 |
| 服务器 | sub3 |
| 单 run wall | ~37min (Office 数据小) |
| 6 并发显存 | 9.5GB / 24GB |

## 启动命令模板

```bash
EXP=experiments/ablation/EXP-148_f2dc_dse_office_high_rho
PY=/root/miniconda3/bin/python
F2DC=/root/autodl-tmp/federated-learning/F2DC

for SEED in 15 333; do
  for RHO in 0.5 0.7 1.0; do
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

> 每 cell = R@best round 的 4 域 acc (caltech, amazon, webcam, dslr). drift = R99 last - mean_best.

| rho | seed | R@best | caltech | amazon | webcam | dslr | **mean_best** | R99_last | drift |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.5 | 15 | R62 | 63.84 | 74.21 | 56.90 | 50.00 | **61.24** | 57.50 | -3.74 |
| 0.5 | 333 | R70 | 62.05 | 79.47 | 56.90 | 53.33 | **62.94** | 53.95 | **-8.99** |
| 0.7 | 15 | R72 | 64.73 | 77.89 | 48.28 | 50.00 | 60.23 | 56.63 | -3.60 |
| 0.7 | 333 | R75 | 66.52 | 78.95 | 53.45 | 50.00 | **62.23** | 56.85 | -5.38 |
| 1.0 | 15 | R75 | 63.84 | 77.37 | 53.45 | 53.33 | 62.00 | 57.35 | -4.65 |
| 1.0 | 333 | R99 | 66.52 | 76.32 | 46.55 | 50.00 | 59.85 | 59.85 | 0 (R99=best) |
| **rho=0.5 mean** | — | — | 62.95 | 76.84 | 56.90 | 51.67 | **62.09** ✓ Office DSE winner | 55.73 | -6.36 |
| **rho=0.7 mean** | — | — | 65.63 | 78.42 | 50.87 | 50.00 | 61.23 | 56.74 | -4.49 |
| **rho=1.0 mean** | — | — | 65.18 | 76.85 | 50.00 | 51.67 | 60.92 | 58.60 | -2.32 |

## 完整 Office DSE rho 扫 (合并 EXP-145 + EXP-148)

| rho | mean_best | vs vanilla 60.56 | vs DaA 63.55 |
|---|---|---|---|
| 0.2 (EXP-145) | 59.82 | -0.74 | -3.73 |
| 0.3 (EXP-145) | 60.98 | +0.42 (持平) | -2.57 |
| **0.5 (EXP-148)** | **62.09** ✓ | **+1.53** ← Office DSE winner | -1.46 (拉近 1.1pp) |
| 0.7 (EXP-148) | 61.23 | +0.67 | -2.32 |
| 1.0 (EXP-148) | 60.92 | +0.36 | -2.63 |
| F2DC vanilla | 60.56 | — | -2.99 |
| F2DC+DaA | 63.55 ✓ | +2.99 | — |

**rho 不是单调** — 0.5 是 sweet spot, 0.7/1.0 反降。

## DSE 诊断 R99

| rho | δ_scaled | mag_p95 / r_max=0.15 | mag_exceed | ccc_imp | δ_cos_feat |
|---|---|---|---|---|---|
| 0.5 | 0.84-0.94% | 1.0-1.4% (1% 阈值) | 0% | +1.1e-3 | 0.27-0.36 |
| 0.7 | 1.5-2.8% | 1.9-3.4% (2% 阈值) | 0% | +2.2~4.3e-3 | 0.33-0.35 |
| 1.0 | 2.5-3.4% | 3.2-4.7% (3% 阈值) | 0% | +2.9~4.7e-3 | 0.30-0.32 |

✅ **DSE 信号变强了**: rho 0.5 → 1.0 时 δ_scaled 从 0.9% → 3.4% (4×), ccc_imp 从 1e-3 → 5e-3 (5×)
✅ **mag guard 全程没触发** (mag_p95 < 5% << r_max=15%) — Office delta 自然小, 不爆界
❌ **但 acc 不单调涨**: rho=0.5 sweet spot, 0.7+ 反降
🔑 **再次印证 EXP-142 反直觉**: ccc_imp 大不代表 acc 高, DSE 帮 acc 不靠 CCC alignment

## 为什么 rho>0.5 反而下降?

1. **mag guard 没触发**, 不是 mag 限制问题
2. **DSE 过强扰动 main loss 收敛**: rho=1.0 时 adapter delta 占 feat 3.4%, 主路 (CE+DFD+DFC) 还要带着这个修正学, gradient flow 被 perturbed
3. **Office 域差异 弱 (acquisition-level not stylistic)**, DSE 0.5 已经够"修正", 再大就是 over-correction

## paper 价值

- **DSE 在 Office 上真胜了** (rho=0.5 mean 62.09 vs vanilla 60.56 +1.53), 不是持平
- **rho 单 sweet spot 现象**: PACS rho=0.3 winner / Office rho=0.5 winner / Digits rho=0.1 winner — 不同 dataset 不同最佳 rho, 是 paper 重要 ablation
- 但 Office 仍输 DaA (-1.46), 可作为 follow-up: DSE+DaA combo 是否能反超

## 数据保存

- **logs**: `experiments/ablation/EXP-148_f2dc_dse_office_high_rho/logs/office_rho{05,07,10}_s{15,333}_R100.log` (6 个)
- **diag npz**: `EXP-148/diag/rho{05,07,10}_s{15,333}/` 各 ~85MB (round_001-100.npz + best_R0xx.npz + final_R100.npz + meta.json + proto_logs.jsonl)
- 总 diag size: 582MB / 6 dirs

## 后续 ablation 候选

1. **DSE + DaA combo on Office**: `--use_daa True --dse_rho_max 0.5` 看能否反超 DaA-only 63.55
2. **rho=0.4, 0.6** 微调 sweet spot 精确位置
