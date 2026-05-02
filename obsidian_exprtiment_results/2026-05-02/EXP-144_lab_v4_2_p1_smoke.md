---
date: 2026-05-02
type: 实验记录 (LAB v4.2 Loss-Aware Boost P1 主对照, 4 实验 R100)
status: PACS / Office×2 / Digits 全部 R100 完成 (1 seed P1 阶段)
exp_id: EXP-144
goal: 验 LAB v4.2 在 PACS / Office / Digits 是否过 gate (PROPOSAL §5), 替代 F2DC 的 DaA
---

# EXP-144: LAB v4.2 (Loss-Aware Boost) P1 主对照实验

## 一句话总览

**LAB v4.2 在 PACS 大胜 (74.58 翻盘 DaA 69.88 = +4.70pp), 但 Office 暴露多 underfit 域硬限制 (gate 全失败), Digits 接近 gate 差 0.04pp**. 核心 paper 卖点: PACS 翻盘成功. P4 触发条件: Office 多 underfit 域时 LAB λ-mix 公式预算分摊 cap 顾此失彼 (rmax=2.0 dslr 升不动 / rmax=4.0 webcam 反损), 需要重新设计 boost 分配机制.

## 实验配置

| 项 | 值 |
|---|---|
| dataset | fl_pacs (4 域, 7 类) / fl_officecaltech (4 域, 10 类) / fl_digits (4 域, 10 类) |
| parti_num | PACS/Office=10 (fixed allocation), Digits=20 |
| communication_epoch | 100 |
| local_epoch | 10 |
| seeds | PACS s=15, Office s=2, Digits s=15 (P1 单 seed 阶段) |
| LAB 超参 | λ=0.15, ratio_min=0.80, ratio_max=2.00 (默认), EMA α=0.30 |
| Office 对照 | 额外跑 ratio_max=4.0 一个对照, 验 cap 是否 Office gate 失败的唯一原因 |
| val partition | 每域 50/dom (Office/Digits 10 类×5), 35/dom (PACS 7 类×5), val_seed=42, deterministic eval transform |
| 服务器 | sub3 (autodl nmb1, RTX 4090 24GB) |
| 单 run wall | PACS ~5h (180s/round × 100), Office ~40min (24s/round × 100), Digits ~1.5h (50s/round × 100) |
| 4 并行显存 | ~10GB / 24GB |
| codex review | 4 轮全过 (一轮 5 个 critical/important + 二三四轮各 ~3 个 minor/important) |

## 启动命令模板

```bash
EXP=experiments/ablation/EXP-144_lab_v4_1
PY=/root/miniconda3/bin/python
F2DC=/root/autodl-tmp/federated-learning/F2DC

# PACS
$PY -u $F2DC/main_run.py --model f2dc_pg_lab --dataset fl_pacs --seed 15 \
  --parti_num 10 --communication_epoch 100 --local_epoch 10 \
  --use_daa false --lab_lambda 0.15 \
  --dump_diag $EXP/diag_p1_pacs_s15

# Office (默认 rmax=2.0)
$PY -u $F2DC/main_run.py --model f2dc_pg_lab --dataset fl_officecaltech --seed 2 \
  --parti_num 10 --communication_epoch 100 --local_epoch 10 \
  --use_daa false --lab_lambda 0.15 --num_classes 10 \
  --dump_diag $EXP/diag_p1_office_s2

# Office rmax=4.0 对照
$PY -u $F2DC/main_run.py --model f2dc_pg_lab --dataset fl_officecaltech --seed 2 \
  --parti_num 10 --communication_epoch 100 --local_epoch 10 \
  --use_daa false --lab_lambda 0.15 --lab_ratio_max 4.0 --num_classes 10 \
  --dump_diag $EXP/diag_p1_office_s2_rmax4

# Digits
$PY -u $F2DC/main_run.py --model f2dc_pg_lab --dataset fl_digits --seed 15 \
  --parti_num 20 --communication_epoch 100 --local_epoch 10 \
  --use_daa false --lab_lambda 0.15 --num_classes 10 \
  --dump_diag $EXP/diag_p1_digits_s15
```

## R100 完整 per-dataset 结果表

> 每 cell = R@best round 4 域 acc (R100 内 mean acc 最高那 round 的 per-domain). drift = AVG Last (R99) − AVG Best.

### PACS s=15 (主战场, 替代 DaA 灾难)

| Method | seed | R@best | photo | art | cartoon | sketch | **AVG Best** | AVG Last | drift |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LAB v4.2** | 15 | R97 | 74.85 | 63.97 | 79.49 | 80.00 | **74.58** ⭐ | 73.58 | -1.00 |
| LAB R96 (sketch peak) | 15 | R96 | 77.25 | 60.29 | 75.21 | **81.91** | 73.67 | 73.58 | -0.09 |
| **F2DC vanilla** (EXP-130 baseline) | 15 | — | — | — | — | — | 72.02 | — | — |
| **F2DC + DaA** (EXP-133/137 baseline) | 15 | — | — | — | — | — | 69.88 | — | — |
| **Δ LAB vs vanilla** | — | — | — | — | — | — | **+2.56** ✅ | — | — |
| **Δ LAB vs DaA** | — | — | — | — | — | — | **+4.70** ⭐⭐ | — | — |

### Office s=2 (gate 失败, 但暴露 P4 触发)

| Method | seed | R@best | caltech | amazon | webcam | dslr | **AVG Best** | AVG Last | drift |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LAB v4.2 rmax=2.0** | 2 | R95 | — | — | — | — | 62.67 | 57.16 | -5.51 |
| **LAB R99 (last)** | 2 | R99 | 66.52 | 72.11 | 50.00 | **40.00** ⚠️ | 57.16 | — | — |
| **LAB v4.2 rmax=4.0** | 2 | R96 | — | — | — | — | 61.15 | 60.07 | -1.08 |
| **LAB R100 rmax=4.0** | 2 | R100 | 68.30 | 72.11 | 46.55 | **53.33** ⭐ | 60.07 | — | — |
| **F2DC vanilla** | 2 | — | 66.07 | 76.49 | 54.60 | 50.00 | 61.79 | — | — |
| **F2DC + DaA** | 2 | — | 63.25 | 69.47 | 63.79 | 53.33 | **64.69** ✓ | — | — |
| **Δ rmax=2.0 vs vanilla** | — | — | — | — | — | — | +0.88 | — | — |
| **Δ rmax=2.0 vs DaA** | — | — | — | — | — | — | -2.02 ❌ | — | — |
| **Δ rmax=4.0 vs vanilla** | — | — | — | — | — | — | -0.64 | — | — |
| **Δ rmax=4.0 vs DaA** | — | — | — | — | — | — | -3.54 ❌ | — | — |

**关键 finding (Office)**: rmax=4.0 救起 dslr (40.0 → 53.33 = hit DaA baseline), 但 webcam 反向 -3.45pp (50.00 → 46.55), 整体 best 反而比 rmax=2.0 略低 (62.67 → 61.15). 这暴露 LAB 公式 `w = (1-λ)×share + λ×q` 的多 underfit 域硬限制 — 升 dslr 必然挤压 webcam.

### Digits s=15 (接近 gate, 差 0.04pp)

| Method | seed | R@best | mnist | usps | svhn | syn | **AVG Best** | AVG Last | drift |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **LAB v4.2** | 15 | R100 | 96.92 | 92.23 | 90.16 | 94.73 | **93.51** | 93.51 | 0 (best=last) |
| **F2DC vanilla** | 15 | — | — | — | — | — | 93.39 | — | — |
| **F2DC + DaA** | 15 | — | — | — | — | — | 93.75 | — | — |
| **Δ LAB vs vanilla** | — | — | — | — | — | — | **+0.12** ✅ | — | — |
| **Δ LAB vs DaA** | — | — | — | — | — | — | -0.24 ❌ | — | — |
| Gate 阈值 | — | — | — | — | — | — | **93.55** | — | — |
| **Gate 差距** | — | — | — | — | — | — | **-0.04 ⚠️ noise level** | — | — |

## Gate 验收（PROPOSAL §5）

### PACS — 3/3 全过 ✅✅✅

| Gate | 阈值 | LAB Best | 状态 |
|---|:---:|:---:|:---:|
| **PACS-1** AVG_B | ≥ 72.02 | **74.58 (R97)** | ✅ **+2.56pp** |
| **PACS-2** sketch best | ≥ 79.0 | **81.91 (R96)** / 80.00 (R97) | ✅ **+2.91pp** |
| **PACS-3** sketch ratio mean | ≥ 0.80 | 0.85 全程 | ✅ |

### Office — 0/2 失败 ❌

| Gate | 阈值 | rmax=2.0 | rmax=4.0 |
|---|:---:|:---:|:---:|
| Office-1 AVG_B | ≥ 64.19 | 62.67 ❌ -1.52 | 61.15 ❌ -3.04 |
| Office-2 dslr final | ≥ 53.3 | 40.0 ❌ -13.3 | 53.33 ✅ (= DaA) |

### Digits — 接近但差 0.04pp ⚠️

| Gate | 阈值 | LAB Best | 状态 |
|---|:---:|:---:|:---:|
| Digits-1 AVG_B | ≥ 93.55 | 93.51 | ⚠️ -0.04pp (single round noise level) |

## 终局对比 (mean best, 1 seed P1 阶段)

| Dataset | LAB Best | vs vanilla | vs DaA | Gate |
|---|:---:|:---:|:---:|:---:|
| **PACS** | **74.58** ⭐ | **+2.56** ✅ | **+4.70** ⭐⭐ | ✅ 3/3 全过 |
| Office rmax=2 | 62.67 | +0.88 | -2.02 | ❌ 0/2 |
| Office rmax=4 | 61.15 | -0.64 | -3.54 | ❌ dslr ✅ but webcam ❌ |
| Digits | 93.51 | +0.12 | -0.24 | ⚠️ 接近 (差 0.04) |

## 关键 finding

1. **PACS 翻盘 +4.70pp** ⭐⭐ — DaA 输 vanilla -2.14pp, LAB 赢 vanilla +2.56pp. 这是论文核心卖点: LAB 解决 DaA 在 PACS 上误砍 sketch 的灾难, 同时大幅超 vanilla.

2. **PACS sketch 持续涨**: R0 ~16 → R76 80.25 (first hit 79+) → R96 **81.91** (peak) → R100 80.51. sketch 在 LAB 下没被砍 (ratio 全程 0.85, 没碰 ratio_min 0.80), 跟 DaA 砍到 ×0.69 完全相反.

3. **Office 失败 = LAB 公式硬限制**:
   - 多 underfit 域 (webcam + dslr) 时 λ=0.15 总预算被分摊
   - rmax=2.0: dslr cap 触发 (cum_boost 727%) 但升不到 DaA 的 ×3.76 程度, dslr 卡 40
   - rmax=4.0: dslr 升到 53.33 (hit DaA baseline) 但 webcam 被挤压 (-3.45)
   - **不是 cap 参数问题, 是 `w = (1-λ)×share + λ×q` 公式结构问题**

4. **Digits svhn = saturated underdog**: cum_boost 368% but ROI -0.09 (waste warning 触发). svhn vanilla acc 87+, LAB 加 boost 但学不动. 这是正常 saturate, 不是 LAB bug.

5. **LAB waste detector 工作** ⭐ — 3 个 dataset 都触发 warning:
   - PACS art: cum_boost 436%, ROI 0.03
   - Digits svhn: cum_boost 368%, ROI -0.09
   - Office dslr (rmax=2.0): cum_boost 727%, ROI 0.45
   
   PROPOSAL §5.3 设计的检测机制证明 paper-grade 价值 — 自动识别 saturated underdog.

## LAB 设计验证 (PROPOSAL §5)

### Algorithm correctness ✅
- 4 个实验 ratio 全程在 [0.80, 2.00] (rmax=2.0) 或 [0.80, 4.0] (rmax=4.0) 内
- bisection projection: 全部收敛 (n_iter 30-33), 0 NaN

### Boost 流向语义 ✅
- PACS: art (PACS underdog 99% rounds below avg) 拿大头
- Digits: svhn (Digits underdog) 拿大头
- Office rmax=2.0: dslr (持续 cap), webcam (轻量升)
- Office rmax=4.0: dslr ratio×2.48, 真升到位 (但代价是 webcam 被挤)

### val partition 正确 ✅
- PACS: 35/dom (C=7×5, 严格 stratified 不打破均衡)
- Office: 50/50/50/39 (dslr unused 64 张限制 cap 到 39, codex 三轮 review 修)
- Digits: 50/dom (C=10×5)

## P4 修复方向 (待实施)

按 PROPOSAL §6 P4 触发条件 + Office 实测数据归纳:

| 方案 | 公式改动 | 期望效果 | 优先级 |
|---|---|---|:---:|
| **P4-B**: 强域 only 取预算 | gap=0 的"中间域" (webcam) 设 ratio_min=0.95 | webcam 不被被动稀释 → +3pp | 高 |
| **P4-C**: λ 自适应 | λ_eff = λ × num_underfit_domains | 总预算够分给两个 underfit 域 | 中 |
| **P4-D**: ratio_max 按 sample share 反比 | ratio_max[d] = max(2.0, K/share[d]) | dslr cap 自动放宽 | 中 |

**预计 P4-B + P4-D 组合解 Office 问题**.

## 后续可能 ablation

1. **P3 PACS 主对照** (3-seed × seed 2/15/333) — 验证 PACS +4.70pp 不是 single-seed luck
2. **P4 设计 + 实现** (lab_aggregation.py 改强域 only 取预算)
3. **Office P4 验证** (rmax=2.0 + ratio_min=1.0 for q=0 doms)
4. **Digits 复跑确认 0.04pp 差距是 single-seed noise** (3-seed mean 应能过 gate)

## 数据保存

- **Logs** (commit 入 git): `experiments/ablation/EXP-144_lab_v4_1/logs/p1_*.log` (4 个 R=100 log + dryrun + smoke, 每 ~40KB)
- **Round npz** (commit 入 git): `EXP-144_lab_v4_1/diag_p1_*/round_001-100.npz` 含 132-174 个 LAB 诊断字段 (lab_ratio/lab_boost/lab_clip_at_max/val_class_counts/lab_used_this_round 等)
- **Heavy snapshots** (PACS/Office 入 git, Digits 单文件 144MB 超 GitHub 100MB 限制 .gitignore 排除, 留本地+sub3): `best_*.npz`, `final_*.npz`
- **NOTE.md**: `experiments/ablation/EXP-144_lab_v4_1/NOTE.md` 完整 P1 结果回填

## 核心 codex review 历史 (4 轮)

| 轮 | 主要修复 |
|---|---|
| 1 | 5 个 critical (best_args 注入 / 3 dataset backbone alias / Digits 兼容 / is_eval=True / ROI hook 接入) |
| 2 | 4 个 critical (val pool 反推 / bisection projection / full participation / dslr 数量) |
| 3 | 3 个 important (lab_delta_acc 滞后 / PACS 35 cap / 数字用 ≈) |
| 4 | 3 个 minor (live waste warning post-eval / PROPOSAL §10 旧公式 / T13 端到端) |

Sanity test 55/55 全过, py_compile / CLI parse / model 注册全部验证通过.
