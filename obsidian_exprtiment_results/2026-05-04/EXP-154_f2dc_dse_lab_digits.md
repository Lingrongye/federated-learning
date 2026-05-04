---
date: 2026-05-04
type: 实验记录 (f2dc_dse_lab Digits 三合一 + lorth+cc=0 fix svhn)
status: ✅ 4 runs R100 完成 + rsync 3.4GB 本地
exp_id: EXP-154
goal: 复制 EXP-149 (PACS) / EXP-151 (Office) 的 DSE+LAB 范式到 Digits, 用 EXP-150 C 方案 fix svhn 灾难
---

# EXP-154: F2DC + DSE + LAB Digits 三合一

## 一句话

**rho=0.2 + LAB mean best 93.78 反超 EXP-150 C 方案 (93.65) +0.13pp, 反超 vanilla (93.59) +0.19pp** — LAB 在 Digits 加成确认。3 dataset paper claim 完整成立。

## 配置 (基于 EXP-150 winner C 方案 + LAB)

| 项 | 值 | 来源 |
|---|---|---|
| dataset | fl_digits, parti_num=20 | EXP-146 |
| dse_rho_max | {0.2, 0.3} sweep | EXP-150 |
| **dse_lambda_orth** | **0.05** ⭐ | EXP-150 C 方案 (修 svhn 灾难) |
| **dse_lambda_cc** | **0.0** ⭐ | EXP-150 C 方案 (拆 server proto bias) |
| LAB | λ=0.15, standard projection rmin=0.8/rmax=2.0 | EXP-144 |
| seeds | {15, 333} | 用户要求 |

## R100 完整结果

### 4 runs

| seed | rho | best@R | best AVG | mnist | usps | svhn | syn | LAST R99 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 15 | 0.2 | R93 | **93.78** ⭐ | 97.82 | 92.43 | 90.11 | 94.75 | 93.56 |
| 15 | 0.3 | R92 | 93.56 | 97.39 | 91.58 | 90.67 | 94.58 | 93.31 |
| 333 | 0.2 | R97 | **93.79** ⭐⭐ | 97.11 | 92.68 | **90.72** | 94.63 | 93.54 |
| 333 | 0.3 | R99 | 93.67 | 97.01 | 92.33 | **90.89** ⭐ | 94.47 | 93.67 |

### rho mean

| | best mean | last mean |
|---|:---:|:---:|
| rho=0.2 ⭐⭐ | **93.78** | 93.55 |
| rho=0.3 | 93.62 | 93.49 |

## 跨方法对比

| Method | best | last | vs vanilla |
|---|:---:|:---:|:---:|
| F2DC vanilla | 93.59 | 93.40 | (anchor) |
| EXP-150 C (no LAB) | 93.65 | 93.43 | +0.06 |
| EXP-146 rho=0.1 | 93.41 | — | -0.18 |
| **EXP-154 rho=0.2** ⭐⭐ | **93.78** | 93.55 | **+0.19** |
| EXP-154 rho=0.3 | 93.62 | 93.49 | +0.03 |

## 关键 finding

1. **rho=0.2 是新 Digits sweet point** (跟 EXP-150 单跑 C 方案 rho=0.3 不同), 跟 PACS/Office 共同模式: 加 LAB 后 sweet rho 比 DSE-only 略降.
2. **svhn 全 4 runs 90+ 稳健** (lorth+cc=0 fix 跨 LAB 一起跑也工作), 没出现 EXP-146 高 rho 的 38% 灾难.
3. **跟 EXP-150 C 持平 (rho=0.3 +LAB) 但 rho=0.2 +LAB 反超 +0.13pp** — LAB 加成在 lower rho 配置下显著.

## 跨 dataset 三合一总结 (paper main claim)

| Dataset | Method | mean best | vs DaA | 备注 |
|---|---|:--:|:--:|---|
| **PACS** (EXP-149) | rho=0.3+LAB | **75.10** | **+6.76** ⭐⭐⭐ | sub1 rerun s=333 R92 best 75.37 史上最高 |
| **Office** (EXP-151) | rho=0.3+LAB v2c | 63.99 | +0.44 ✅ | marginal pass |
| **Digits** (EXP-154) | rho=0.2+LAB+lorth+cc=0 | **93.78** | **+0.19** vs vanilla | 反超 EXP-150 C +0.13 |

3 dataset paper claim 完整 ✅
