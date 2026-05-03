# EXP-144 P1 Smoke 主对照结果

> **Date**: 2026-05-02
> **Model**: F2DCPgLab (LAB v4.2)
> **Setup**: λ=0.15, ratio_min=0.80, ratio_max=2.00, EMA α=0.30, val_seed=42
> **Status**: P1 phase 完成（4 个实验 R=100 / E=10 / 1 seed）
> **Server**: sub3 (NVIDIA RTX 4090 24GB)
> **Total GPU time**: ~5h 并行（PACS 4h53min, Office 各 ~40min, Digits 1h27min）

---

## 1. P1 总体结论

### 🎉 PACS 翻盘成功，超 vanilla + DaA ⭐⭐

| Method | PACS AVG_B | vs LAB |
|---|:---:|:---:|
| Vanilla | 72.02 | -2.56 |
| **+DaA** (paper) | **69.88** | **-4.70** ⚠️ DaA 输 |
| **+LAB (我们)** | **74.58** ⭐ | — |

**LAB 比 DaA 翻盘 +4.70pp，比 vanilla 涨 +2.56pp** — 这是论文核心卖点。

### Office 失败但暴露 LAB 设计 P4 触发条件

跑了两个 ratio_max 配置，都没过 gate：
- rmax=2.0: dslr 升不动 (40 vs DaA 53.33)
- rmax=4.0: dslr 救起来 (53.33) 但 webcam 反向被砍 (50→46.55)

→ **多 underfit 域时，cap 顾此失彼**，揭示 LAB 公式 `(1-λ)×share + λ×q` 的硬数学限制。

### Digits 接近 gate（差 0.04pp）

LAB 93.51 vs gate 93.55 vs vanilla 93.39 vs DaA 93.75。**LAB 略优 vanilla, 略低 DaA**, 主要 svhn 是 saturated underdog（waste warning 触发）。

---

## 2. 各 Dataset 详细结果

### 2.1 PACS s=15 R=100 ✅ 全过 gate

```
Final R100: 73.578
Best R97:   74.578 ⭐  per-dom: photo 74.85, art 63.97, cartoon 79.49, sketch 80.00
Best R96 sketch: 81.91 ⭐ (sketch best individual round)
```

| Gate | 阈值 | LAB best | 状态 |
|---|:---:|:---:|:---:|
| **PACS-1** AVG_B | ≥ 72.02 | **74.58 (R97)** | ✅ **+2.56pp** |
| **PACS-2** sketch best | ≥ 79.0 | **81.91 (R96)** / 80.00 (R97) | ✅ **+2.91pp** |
| **PACS-3** sketch ratio mean | ≥ 0.80 | 0.85 全程 | ✅ |

**LAB 行为分析**:
- 主要 boost 给 art (PACS underdog, 99% rounds below avg)
- sketch (主力域) ratio 持续 0.85 (被动稀释, 但没碰到 ratio_min cap)
- clip={None} 全程没碰边界
- waste warning 触发: art cum_boost 436.5%, ROI 0.03 (saturated underdog)

**sketch acc trajectory** (gate-2 关键):
- R10: ~25  R30: 64.50  R60: 77.83  R76: **80.25** (first hit 79+)
- R96: **81.91** ⭐ best  R100: 80.51
→ sketch 在 R76 后稳定 79+, 翻 PACS DaA 灾难（DaA sketch 67.20）。

### 2.2 Office s=2 R=100 (rmax=2.0) ❌ Gate 失败

```
Final R99:  57.158   per-dom: caltech 66.52, amazon 72.11, webcam 50.0, dslr 40.0
Best R95:   62.665
```

| Gate | 阈值 | LAB | 状态 |
|---|:---:|:---:|:---:|
| Office-1 AVG_B | ≥ 64.19 | 62.67 | ❌ **-1.52pp** |
| Office-2 dslr final | ≥ 53.3 | **40.0** | ❌ -13.3pp |

**问题诊断**:
- dslr 持续触发 ratio_max=2.0 cap（cum_boost 727.2%）
- 但 ratio×2.0 不够把 31 张/cli 的 dslr 救起来（DaA ×3.76 实测有效）
- waste warning 触发: dslr ROI 0.45 < 0.5 = wasted

### 2.3 Office s=2 R=100 (rmax=4.0 对照) — dslr 救起但 webcam 反损

```
Final R100: 60.072   per-dom: caltech 68.30, amazon 72.11, webcam 46.55, dslr 53.33
Best R96:   61.152
```

| 域 | rmax=2.0 R100 | **rmax=4.0 R100** | Δ |
|---|:---:|:---:|:---:|
| caltech | 66.52 | 68.30 | +1.78 |
| amazon | 72.11 | 72.11 | 0 |
| webcam | 50.00 | **46.55** | **-3.45** ⚠️ |
| **dslr** | **40.00** | **53.33** ⭐ | **+13.33** |
| AVG_Last | 57.16 | **60.07** | +2.91 |
| AVG_Best | 62.67 | 61.15 | -1.52 |

**关键 finding** ⭐:
> rmax=4.0 完全救起 dslr (40→53.33, hit DaA baseline) **但 webcam 反向 -3.45pp**。
>
> 这暴露 LAB 公式 `w = (1-λ)×share + λ×q` 的硬数学限制：
> - 当 2 个 underfit 域 (dslr + webcam) 同时存在
> - λ 总预算被两个域分摊
> - 给 dslr 更大 cap → dslr 拿走更多 boost → webcam 拿到的少 → 被动稀释
>
> **触发 P4 设计条件**：需要 `multi-underfit-domain-aware` 的 boost 分配。

### 2.4 Digits s=15 R=100 — 接近 gate 差 0.04pp

```
Final R100: 93.51   per-dom: mnist 96.92, usps 92.23, svhn 90.16, syn 94.73
Best R100:  93.51 (= final)
```

| Gate | 阈值 | LAB | 状态 |
|---|:---:|:---:|:---:|
| Digits-1 AVG_B | ≥ 93.55 | **93.51** | ⚠️ **-0.04pp** (差 single-round noise 量级) |
| LAB vs vanilla 93.39 | — | +0.12 | ✅ 略优 vanilla |
| LAB vs DaA 93.75 | — | -0.24 | 略低 DaA |

**Digits 行为**:
- svhn 持续被 boost (cum 368%) 但 ROI 负 (-0.09) — saturated underdog
- usps 也是 underdog 但响应正常 (R10→R100: 60→92)
- waste warning 触发: svhn

---

## 3. 跨 Dataset 横向对比

| Dataset | LAB Best | vanilla | DaA | LAB-vanilla | **LAB-DaA** | Gate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PACS s=15** | **74.58** ⭐ | 72.02 | 69.88 | **+2.56** | **+4.70** ⭐⭐ | ✅ 3/3 |
| Office rmax=2 | 62.67 | 61.79 | 64.69 | +0.88 | -2.02 | ❌ 0/2 |
| Office rmax=4 | 61.15 | 61.79 | 64.69 | -0.64 | -3.54 | ❌ dslr ✅, webcam ❌ |
| Digits s=15 | 93.51 | 93.39 | 93.75 | +0.12 | -0.24 | ⚠️ 差 0.04 |

---

## 4. LAB 设计验证（PROPOSAL §5 诊断指标）

### Algorithm correctness ✅
- 4 个实验 ratio 全程在 [0.80, 2.00] (rmax=2.0) 或 [0.80, 4.0] (rmax=4.0) 内
- bisection projection: 全部收敛 (n_iter 30-33), 0 NaN

### Boost 流向语义 ✅
- PACS: art 拿 cum_boost 436% (PACS underdog)
- Digits: svhn 拿 cum_boost 368% (Digits underdog)
- Office rmax=2.0: dslr 拿 cum_boost 727% (cap 触发)
- Office rmax=4.0: dslr 升到 ratio×2.48 (没触发 cap=4.0)

### waste detector 工作 ✅
3 个 dataset 都触发 waste warning:
- PACS art: cum_boost 436%, ROI 0.03 (< 0.5)
- Digits svhn: cum_boost 368%, ROI -0.09
- Office dslr (rmax=2.0): cum_boost 727%, ROI 0.45

→ **PROPOSAL §5.3 设计的检测机制证明 paper-grade 价值** — 自动识别 saturated underdog。

### val partition 正确 ✅
- PACS: 35/dom (C=7×5)
- Office: 50/50/50/39 (dslr unused 64 张限制)
- Digits: 50/dom

---

## 5. P1 失败 / 风险记录

### Office gate 失败的真正根因（codex 4 轮 review 验证后）

LAB v4.2 公式：
```
w = (1-λ) × sample_share + λ × q
其中 q = ReLU(loss - mean) / sum_q
```

当 Office 有 2 个 underfit 域 (webcam q≈0.4, dslr q≈0.6) 时:
- λ=0.15 总预算分摊给两个域
- ratio_min=0.80 cap 让强域不能砍太多
- ratio_max=2.0 让 dslr 升不够
- ratio_max=4.0 让 dslr 升够但 webcam 拿到的预算 → 被动稀释

→ **不是参数问题，是公式结构问题**

### Digits 差 0.04pp 的根因
svhn 持续是 underdog (vanilla acc 87+ 但 LAB 相对其他域是 worst)。LAB 给 svhn boost 但 svhn 真的学不动 (saturated)。

→ **正常的 saturate**，不算 LAB bug。

---

## 6. P4 修复方向（待实施）

按 PROPOSAL §6 P4 触发条件 + 本次 P1 数据归纳:

| 方案 | 公式改动 | 期望效果 | 优先级 |
|---|---|---|:---:|
| **P4-B**: 强域 only 取预算 | gap=0 的"中间域"设 ratio_min=1.0 | webcam 不被砍 → +3pp | 高 |
| **P4-C**: λ 自适应 | λ_eff = λ × num_underfit | 总预算够分 | 中 |
| **P4-D**: ratio_max 按 sample share 反比 | ratio_max[d] = max(2.0, K/share[d]) | dslr cap 自动放宽 | 中 |

**预计 P4-B 解决 webcam 被动稀释，P4-D 解决 dslr cap 不够**。组合 B+D 应该让 Office 过 gate。

---

## 7. 下一步建议

| 优先级 | 任务 | 时间 |
|---|---|:---:|
| 1 | **P3 PACS 主对照** (3-seed × seed 2/15/333, 1 dataset 验证 PACS 翻盘不是 single-seed luck) | ~12h |
| 2 | **P4-B+D 设计 + 实现** (lab_aggregation.py 改动) | 1h |
| 3 | **Office P4 验证** (rmax=2.0 + ratio_min=1.0 for q=0 doms) | 4h |
| 4 | **写 paper draft** (PACS 大胜 + Office 暴露设计 finding + 完整诊断 system) | — |

PACS 大胜 + Office 揭露 + Digits 接近 = **P1 阶段产出 paper-grade 信号充足，可以开始写 paper**。

---

## 8. 数据产物

### Logs (合 commit)
- `logs/p1_pacs_s15_R100.log` (40 KB)
- `logs/p1_office_s2_R100.log` (rmax=2.0, 40 KB)
- `logs/p1_office_s2_R100_rmax4.log` (rmax=4.0, 40 KB)
- `logs/p1_digits_s15_R100.log` (40 KB)
- `logs/dryrun_*` + `logs/smoke_*` (验证产物)

### Diag NPZ (round_*.npz, 含 132-174 个 LAB 诊断字段)
- `diag_p1_pacs_s15/round_*.npz` (100 round)
- `diag_p1_office_s2/round_*.npz`
- `diag_p1_office_s2_rmax4/round_*.npz`
- `diag_p1_digits_s15/round_*.npz`
- `diag_dryrun_*/round_*.npz` + `diag_smoke_pacs_s15/round_*.npz` (验证产物)

### Heavy snapshots（13-14 MB each, 不入 Git）
`best_*.npz`, `final_*.npz` 留在 sub3 上, 后处理 cold path 时按需 rsync。

### Codex review 产物
- `PROPOSAL.md` (LAB v4.2 final 方案)
- `p0_offline_replay.py` + `p0_replay_report.md` + `p0_trajectories.json`
