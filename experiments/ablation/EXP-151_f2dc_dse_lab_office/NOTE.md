# EXP-151: F2DC + DSE_Rescue3 + LAB v4.2 Office 双 sweet 结合

**date**: 2026-05-03 启动
**status**: ✅ R100 全 4 runs 完成 + rsync 回本地 (955MB)
**exp_id**: EXP-151
**关联**: 复制 EXP-149 PACS 结合范式到 Office, 用 EXP-148 + EXP-144 P4 已发现的 sweet point

---

## 一句话目标

把 EXP-148 找到的 **Office DSE sweet rho=0.5** 跟 EXP-144 P4 找到的 **Office LAB sweet (small_protect rmin=2.0/rmax=4.0)** 结合到 f2dc_dse_lab 模型, 验证 1+1 > 2 协同 (类比 EXP-149 PACS R30+ 大胜利)。

## 配置 (跟 EXP-148 + EXP-144 P4 sweet point 完全对齐)

| 项 | 值 | 来源 |
|---|---|---|
| dataset | fl_officecaltech (4 域: caltech 53.1%, amazon 30.2%, webcam 9.3%, dslr 7.4%) | 复用 |
| model | **f2dc_dse_lab** (EXP-149 新加的 model, F2DC + DSE_Rescue3 + LAB v4.2) | EXP-149 |
| parti_num | 10 (caltech:3, amazon:2, webcam:2, dslr:3 fixed allocation) | 复用 |
| communication_epoch | 100, local_epoch=10 | 复用 |
| seeds | **{15, 333}** | 用户指定 |
| **rho_max grid** | **{0.3, 0.5}** (0.5 = Office DSE sweet, 0.3 = neighbor 对照) | EXP-148 |
| 其他 DSE 超参 | warmup=5/ramp=10, lambda_cc=0.1, lambda_mag=0.01, r_max=0.15, proto3_ema_beta=0.85 | EXP-148 |
| LAB λ | 0.15 | EXP-144 共识 |
| LAB EMA α | 0.30 | EXP-144 共识 |
| **LAB projection_mode** | **office_small_protect** | EXP-144 P4 v2-C |
| **LAB small_share_threshold** | **0.125** | EXP-144 P4 v2-C |
| **LAB small_ratio_min** | **2.00** | EXP-144 P4 v2-C ⭐ |
| **LAB small_ratio_max** | **4.00** | EXP-144 P4 v2-C ⭐ |
| LAB val_size_per_dom | 50 (dslr cap 39-40 因数据不足) | EXP-144 |
| **best dump 触发条件** | `dump_warmup=0 dump_min_gain=0.01 dump_min_interval=1` (任何提升即存) | 用户要求, 区别于 EXP-149 默认 1pp gain 5 round interval |

## 任务分配 (4 runs 跨 sub1 + sub2)

| host | seed | rho | log 路径 (相对 EXP-151) | dump_diag (相对 EXP-151) |
|---|:--:|:--:|---|---|
| **sub1** (4090, PID 1550 PACS rerun 同机) | 15 | **0.5 ⭐** | `logs/office_s15_rho05.log` | `diag_office_s15_rho05_dselab/` |
| **sub1** | 15 | 0.3 | `logs/office_s15_rho03.log` | `diag_office_s15_rho03_dselab/` |
| **sub2** (4090, PID 3194 PACS rho02 同机) | 333 | **0.5 ⭐** | `logs/office_s333_rho05.log` | `diag_office_s333_rho05_dselab/` |
| **sub2** | 333 | 0.3 | `logs/office_s333_rho03.log` | `diag_office_s333_rho03_dselab/` |

**布局原因**: 按 seed 分机, 让同 seed 的 rho 对照在同 GPU 跑 (噪声对称); 每机加 2 Office (~3GB each) + 已有 PACS (~7-9GB), 总 ~13GB / 24GB 安全。

## 启动命令模板

```bash
EXP_DIR=experiments/ablation/EXP-151_f2dc_dse_lab_office
F2DC=F2DC
PY=/root/miniconda3/bin/python

setsid nohup env CUDA_VISIBLE_DEVICES=0 $PY -u main_run.py \
  --model f2dc_dse_lab --dataset fl_officecaltech --seed $SEED \
  --parti_num 10 --communication_epoch 100 --local_epoch 10 \
  --use_daa false \
  --dse_rho_max $RHO --dse_lambda_cc 0.1 --dse_lambda_mag 0.01 \
  --lab_lambda 0.15 \
  --lab_projection_mode office_small_protect \
  --lab_small_share_threshold 0.125 \
  --lab_small_ratio_min 2.0 --lab_small_ratio_max 4.0 \
  --num_classes 10 \
  --dump_diag $EXP_DIR/diag_office_s${SEED}_rho0${RHO_TAG}_dselab \
  --dump_warmup 0 --dump_min_gain 0.01 --dump_min_interval 1 \
  > $EXP_DIR/logs/office_s${SEED}_rho0${RHO_TAG}.log 2>&1 < /dev/null & disown
```

## R100 完整结果 (2026-05-03 R100 全完, rsync 回本地 955MB)

### 4 runs per-domain best/final acc

| host | seed | rho | best@R | **best AVG** | caltech | amazon | webcam | dslr | final R100 AVG | caltech | amazon | webcam | dslr |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| sub1 | 15 | **0.3 ⭐** | R86 | **64.77** | 63.84 | 71.58 | 60.34 | **63.33** ⭐ | 59.33 | 62.95 | 68.95 | 62.07 | 43.33 |
| sub1 | 15 | 0.5 | R92 | 63.57 | 61.61 | 72.11 | **67.24** ⭐ | 53.33 | 61.26 | 62.05 | 72.63 | 60.34 | 50.00 |
| sub2 | 333 | 0.5 | R93 | 62.22 | 65.18 | **78.42** | 58.62 | 46.67 | 60.40 | 68.75 | 72.63 | 56.90 | 43.33 |
| sub2 | 333 | 0.3 | R76 | 63.21 | 64.29 | **78.42** | 53.45 | 56.67 | 58.52 | 62.95 | 72.63 | 55.17 | 43.33 |

### rho 平均 (paper-grade target)

| Metric | rho=0.3 | rho=0.5 | DaA mean | rho=0.3 vs DaA | rho=0.5 vs DaA |
|---|:---:|:---:|:---:|:---:|:---:|
| **Best AVG** | **63.99** | 62.90 | 63.55 | **+0.44** ✅ | -0.65 |
| Final R100 AVG | 58.92 | 60.83 | (DaA last 未取) | drop -5.07 | drop -2.07 |

### 跟 4-baseline best 同 R0-R100 累计对比

| Run | EXP-151 best | DSE-only | DaA | vanilla | LAB(s=2) | **vs DaA** | **vs DSE** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| s=15 rho=0.3 R86 ⭐ | **64.77** | 60.45 | 63.92 | 60.80 | 65.59 | **+0.84** ✅ | **+4.31** ✅ |
| s=15 rho=0.5 R92 | 63.57 | 61.24 | 63.92 | 60.80 | 65.59 | -0.35 (平) | +2.33 ✅ |
| s=333 rho=0.5 R93 | 62.22 | 62.94 | 63.17 | 57.85 | — | -0.95 | -0.72 |
| s=333 rho=0.3 R76 | 63.21 | 61.49 | 63.17 | 57.85 | — | +0.04 (平) | +1.72 ✅ |

### 关键发现

1. **rho=0.3 mean best = 63.99 vs DaA 63.55 = +0.44 ✅** (paper-grade target 通过, 但 single-seed-pair, 需 P3 3-seed 验证)
2. **rho=0.5 不胜 DaA** (mean 62.90 vs 63.55 = -0.65) — Office DSE sweet 单跑 0.5 加 LAB 后反不胜
3. **vs DSE-only 加成存在** — s=15 双 rho 都 +2.3~+4.3pp; s=333 rho=0.3 +1.7pp; 仅 s=333 rho=0.5 -0.7
4. **vs LAB(s=2) 仍输 0.8pp** — Office LAB 单跑 s=2 R86 = 65.59, EXP-151 s=15 rho=0.3 = 64.77 略低
5. **dslr 是关键差异化域**: s=15 rho=0.3 dslr 63.33 ⭐ (跟 LAB v2-C s=2 持平), s=333 dslr 46-57 弱
6. **Final R100 vs Best 大跌 5pp** — 跟 PACS 不同, Office 后期 acc 抖动严重 (epoch loss 已 0.5-0.6 收敛但 acc oscillate 60→58→64 大幅震荡)

### 跟 EXP-149 PACS 对比 (paper 双卖点对照)

| Dataset | sweet 配置 winner Best mean | vs DaA Δ | seed 鲁棒性 |
|---|:---:|:---:|:---:|
| **PACS** (EXP-149) | rho=0.2 win, single best ~74 | s=15 +4.5 / s=333 +5.8 | 强双 seed |
| **Office** (EXP-151) | rho=0.3 win mean 63.99 | s=15 +0.84 / s=333 +0.04 | 弱 (+0.4 边缘) |

→ **PACS 是 EXP-149 主 paper finding (大胜), Office EXP-151 仅 marginal pass** (s=15 强 s=333 弱)。Office paper claim 需要 P3 3-seed 才稳。

## 预期与对比 baseline

| Method | Office AVG Best (s=2 single seed) | 来源 |
|---|:--:|---|
| F2DC vanilla | 60.56 | EXP-130 |
| F2DC + DaA | 63.55 | EXP-133 |
| F2DC + DSE rho=0.5 (仅 rho) | 62.09 (s=15+s=333 mean) | EXP-148 |
| F2DC + LAB v2-C (仅 LAB) | 65.60 (s=2 only, 单 seed) | EXP-144 P4 |
| **F2DC_DSE_LAB rho=0.5 ⭐** | ?? (paper-grade target) | **本实验** |

**期望** (类比 EXP-149 PACS):
- 如果协同 (1+1 > 2): mean_best > 65.6 (LAB only) + ~1pp = 66.5+
- 如果加性 (1+1 = 2): mean_best ≈ 62 (DSE) + 65 (LAB) - 60 (vanilla) ≈ 67
- 如果干扰 (1+1 < 2): mean_best < 65.6, 即 LAB 单跑就够

## 数据保存 (CLAUDE.md 零零零规则)

- 4 个独立 dump_diag 路径, 互不 overwrite
- 每路径: round_001-100.npz + best_R*.npz + final_R100.npz + meta.json + proto_logs.jsonl
- 单 round npz 含 DSE 22 + LAB 147 = 169 proto_diag 字段
- best/final heavy snapshot 含 state_dict_fp16 + features + labels + preds + logits + confusion + proto_diag
- best dump 触发条件 `min_gain=0.01 min_interval=1`: 跟 EXP-149 默认 (1.0/5) 不同, 任何 +0.01pp 提升 + 至少 1 round gap 就 dump → 预计 100 round 触发 ~10-30 次 dump

## R100 完成后回填

(待: 各 run 跑完后从 final_R100.npz / 最大 best_R*.npz 提取 per-domain acc 填表)

## paper 价值 (预期)

如果协同确认 (mean_best > LAB only):
- **Office 第二个 paper 卖点** (PACS 第一卖点是 EXP-149 协同, Office 第二卖点 EXP-151)
- 完整跨数据集论证 DSE+LAB 范式 (PACS + Office), Digits 后续做
- main table 加 DSE+LAB row 直接比 DaA-only / LAB-only / DSE-only / vanilla 4 个 baseline

如果不协同:
- 揭示 Office 域差异弱 (acquisition-level not stylistic), DSE rescue 跟 LAB boost 在 Office 上信号重叠 (都修 webcam/dslr 同样 underfit)
- 仍是 paper finding (negative result 也有 value)
