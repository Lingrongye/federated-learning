---
date: 2026-05-03
type: 实验记录 (f2dc_dse_lab Office, 双 sweet 结合)
status: ✅ R100 全 4 runs 完成 + rsync 本地 (955MB)
exp_id: EXP-151
goal: 复制 EXP-149 PACS 双 sweet 结合范式到 Office, 验证 DSE+LAB 1+1>2 协同
---

# EXP-151: F2DC + DSE_Rescue3 + LAB v4.2 Office 双 sweet 结合

## 一句话

把 EXP-148 找到的 **Office DSE sweet rho=0.5** 跟 EXP-144 P4 找到的 **Office LAB sweet (small_protect rmin=2.0/rmax=4.0)** 结合到 f2dc_dse_lab, 4 runs (rho ∈ {0.3, 0.5} × s ∈ {15, 333}) 跑在 sub1 + sub2, 验证类似 EXP-149 PACS R30+ 的协同大胜利。

## 关键差异跟 EXP-149

| 项 | EXP-149 (PACS) | EXP-151 (Office) |
|---|---|---|
| dataset | fl_pacs (4 域 7 类) | fl_officecaltech (4 域 10 类) |
| rho grid | {0.2, 0.3} | {0.3, 0.5} |
| LAB projection | standard | **office_small_protect (small_share<0.125)** |
| LAB ratio cap | 0.80-2.00 | 2.0-4.0 (small) |
| best dump 触发 | warmup=30 / gain=1.0 / interval=5 | **warmup=0 / gain=0.01 / interval=1** (任何提升即存) |
| 服务器布局 | sub3 ×2 (s=15) + sub2 ×2 (s=333) | sub1 ×2 (s=15) + sub2 ×2 (s=333) |

## sweet point 来源

- **DSE rho=0.5**: EXP-148 Office 完整 rho 扫 (0.5/0.7/1.0), rho=0.5 mean_best 62.09 是 sweet (vs vanilla 60.56 +1.53)
- **LAB v2-C (rmin=2.0, rmax=4.0)**: EXP-144 P4 office_small_protect sweep (rmin ∈ {1.25, 1.5, 1.75, 2.0}), rmin=2.0 best 65.60 (vs DaA 64.66 +0.94)

## 任务分配

| host | seed | rho | dump_diag |
|---|:--:|:--:|---|
| **sub1** | 15 | **0.5 ⭐** | `diag_office_s15_rho05_dselab/` |
| sub1 | 15 | 0.3 | `diag_office_s15_rho03_dselab/` |
| **sub2** | 333 | **0.5 ⭐** | `diag_office_s333_rho05_dselab/` |
| sub2 | 333 | 0.3 | `diag_office_s333_rho03_dselab/` |

按 seed 分机, sub1 跑 s=15 对照, sub2 跑 s=333 对照。每机加 2 Office (~3GB each) + 已有 PACS (~7-9GB) → 总 ~13GB / 24GB 安全。

## 期望 (paper-grade target)

| Method | Office AVG Best | 说明 |
|---|:--:|---|
| F2DC vanilla | 60.56 | 弱 baseline |
| F2DC + DaA | 63.55 | 强 baseline (主对手) |
| F2DC + DSE rho=0.5 (单跑) | 62.09 | 输 DaA -1.46 |
| F2DC + LAB v2-C (单跑) | 65.60 | 赢 DaA +0.94 |
| **F2DC_DSE_LAB rho=0.5 ⭐** | **>=66.5? (target)** | 协同验证 |

## R100 完成结果 (2026-05-03)

### 4 runs per-domain best/final

| seed | rho | best@R | **best AVG** | caltech | amazon | webcam | dslr | final R100 AVG | drop |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 15 | **0.3 ⭐** | R86 | **64.77** | 63.84 | 71.58 | 60.34 | **63.33** ⭐ | 59.33 | -5.45 |
| 15 | 0.5 | R92 | 63.57 | 61.61 | 72.11 | **67.24** ⭐ | 53.33 | 61.26 | -2.31 |
| 333 | 0.5 | R93 | 62.22 | 65.18 | **78.42** | 58.62 | 46.67 | 60.40 | -1.82 |
| 333 | 0.3 | R76 | 63.21 | 64.29 | **78.42** | 53.45 | 56.67 | 58.52 | -4.69 |

### rho 平均 (paper-grade target)

| Metric | rho=0.3 | rho=0.5 | DaA mean | rho=0.3 Δ | rho=0.5 Δ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Best AVG** | **63.99** ✅ | 62.90 | 63.55 | **+0.44** ✅ | -0.65 |
| Final R100 | 58.92 | 60.83 | — | — | — |

### 跟 4-baseline 累计 best 对比 (R0-R100 max)

| Run | EXP-151 | DSE-only | DaA | vanilla | LAB(s=2) | vs DaA | vs DSE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| s=15 rho=0.3 R86 ⭐ | **64.77** | 60.45 | 63.92 | 60.80 | 65.59 | **+0.84** ✅ | **+4.31** ✅ |
| s=15 rho=0.5 R92 | 63.57 | 61.24 | 63.92 | 60.80 | 65.59 | -0.35 | +2.33 ✅ |
| s=333 rho=0.5 R93 | 62.22 | 62.94 | 63.17 | 57.85 | — | -0.95 | -0.72 |
| s=333 rho=0.3 R76 | 63.21 | 61.49 | 63.17 | 57.85 | — | +0.04 (平) | +1.72 ✅ |

### 关键发现

1. **rho=0.3 mean best 63.99 vs DaA 63.55 = +0.44 ✅** (paper-grade target marginally pass, single seed-pair 不算 paper claim, 需 P3 3-seed)
2. **vs DSE-only 加成普遍存在** — 4 runs 中 3 胜 1 输, LAB-on-top-of-DSE 在 Office 上有效
3. **rho=0.3 比 rho=0.5 强** (跟 PACS 同模式: sweet 下 0.1-0.2 的 rho 加 LAB 更佳, 因 LAB 已增强信号)
4. **dslr 是 winner 关键域**: s=15 rho=0.3 dslr 63.33 跟 LAB v2-C s=2 单跑持平 ⭐
5. **vs LAB(s=2) 仍输 0.8pp** — Office LAB 单跑威力强 (s=2 R86 = 65.59), DSE+LAB 暂未超越
6. **Final R100 大跌 5pp vs Best** — Office 后期 acc 抖动严重 (跟 PACS R100 持续涨形成对比)

### 跟 EXP-149 PACS 对比 (paper 双数据集对照)

| Dataset | sweet rho | best mean | vs DaA Δ | seed 鲁棒性 |
|---|:---:|:---:|:---:|:---:|
| **PACS** (EXP-149) | 0.2 | s=15 R69 = 74.39 ⭐ | +4.5 (s=15) / +5.8 (s=333) | **强** |
| **Office** (EXP-151) | 0.3 | mean = 63.99 | +0.44 边缘 | 弱 (+0.84/+0.04) |

→ **PACS 是主 paper finding**, Office EXP-151 仅 marginal pass。Office paper claim 需要 P3 3-seed 才稳。

## 数据保存 (CLAUDE.md 零零零规则)

- 4 独立 dump_diag 路径
- per-round npz 含 DSE 22 + LAB 147 = 169 proto_diag 字段
- best/final heavy snapshot 含 state_dict_fp16 + features + labels + preds + logits + confusion + proto_diag
- best dump 触发宽松 (`gain=0.01/interval=1`) → 预计 100 round 触发 ~10-30 次 dump (vs EXP-149 默认 ~3 次)

## R100 完成后

1. rsync sub1+sub2 diag 数据回本地 EXP-151
2. 提取 per-domain acc + best/final round + per-round LAB diag (ratio/boost/cum_boost)
3. 填回完整对比表 + 决策 paper 结论
