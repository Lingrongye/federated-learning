# EXP-154: F2DC + DSE_Rescue3 + LAB v4.2 Digits 三合一 (基于 EXP-150 C 方案 + LAB)

**date**: 2026-05-04
**status**: ✅ 4 runs R100 全完成 + rsync 3.4GB 本地
**关联**: EXP-150 找到 Digits 救 svhn 的 lorth+cc=0 配方 (C 方案 winner 93.65 vs vanilla 93.59 +0.06pp), EXP-154 加 LAB 看是否再加成

---

## 一句话总览 ⭐⭐

**EXP-154 rho=0.2 mean best 93.78 反超 EXP-150 C 方案 (93.65) +0.13pp, 反超 vanilla (93.59) +0.19pp** — LAB v4.2 在 Digits 加成确认 (跟 PACS / Office 共同建立 LAB 跨 dataset 加成模式).

## 实验设计

基于 EXP-150 winner C 方案 (rho=0.3 + lambda_orth=0.05 + lambda_cc=0) 加 LAB v4.2:
- DSE: lorth=0.05 修 svhn 灾难 (横向约束 adapter delta), cc=0 拆 server proto bias (CCC 是 svhn 跟 mnist 拉对齐的祸根)
- LAB: standard projection (rmin=0.8/rmax=2.0), 4 域 share 25-29% 都 > 0.125 不需 small_protect

## 配置

| 项 | 值 | 来源 |
|---|---|---|
| dataset | fl_digits, parti_num=20 (mnist=3/usps=6/svhn=6/syn=5 fixed) | EXP-146 |
| communication_epoch | 100, local_epoch=10 | 共识 |
| **dse_rho_max** | **{0.2, 0.3}** sweep | EXP-150 C 用 0.3 |
| **dse_lambda_orth** | **0.05** ⭐ | EXP-150 C 方案 |
| **dse_lambda_cc** | **0.0** (关 CCC) ⭐ | EXP-150 C 方案 |
| dse_lambda_mag | 0.01 | EXP-150 |
| LAB λ | 0.15 | EXP-144 共识 |
| LAB projection | standard rmin=0.8/rmax=2.0 | EXP-144 (Digits 不需 small_protect) |
| seeds | {15, 333} | 用户要求 |
| 服务器 | sub2 (s=15 双 rho) + sub1 (s=333 双 rho) | |
| best dump | warmup=0/gain=0.01/interval=1 | EXP-151/152 风格 |

## R100 完整结果 ✅

### 4 runs per-seed × per-domain

| seed | rho | best@R | best AVG | mnist | usps | svhn | syn | LAST R99 | drift |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 15 | 0.2 | R93 | **93.78** ⭐ | 97.82 | 92.43 | 90.11 | 94.75 | 93.56 | -0.22 |
| 15 | 0.3 | R92 | 93.56 | 97.39 | 91.58 | 90.67 | 94.58 | 93.31 | -0.25 |
| 333 | 0.2 | R97 | **93.79** ⭐⭐ | 97.11 | 92.68 | **90.72** | 94.63 | 93.54 | -0.25 |
| 333 | 0.3 | R99 | 93.67 | 97.01 | 92.33 | **90.89** ⭐ | 94.47 | 93.67 | 0 |

### rho mean

| | s=15 | s=333 | best mean | last mean |
|---|:---:|:---:|:---:|:---:|
| rho=0.2 ⭐⭐ | 93.78 | 93.79 | **93.78** | 93.55 |
| rho=0.3 | 93.56 | 93.67 | 93.62 | 93.49 |

## 跨方法 Digits 对比

| Method | best mean | last mean | per-dom svhn (best) | vs vanilla |
|---|:---:|:---:|:---:|:---:|
| F2DC vanilla (EXP-150) | 93.59 | 93.40 | 90.18 | (anchor) |
| **EXP-150 C** (rho=0.3+lorth+cc=0, no LAB) | 93.65 | 93.43 | 90.51 | +0.06 |
| EXP-150 B (rho=0.3+lorth=0.2) | 93.55 | 93.22 | 90.98 | -0.04 |
| EXP-150 A (rho=0.3+lorth=0.05) | 93.56 | 93.13 | 90.42 | -0.03 |
| EXP-146 rho=0.1 (DSE only, no lorth) | 93.41 | — | — | -0.18 |
| **EXP-154 rho=0.2 ⭐⭐ (+LAB)** | **93.78** | 93.55 | 90.42 | **+0.19** |
| **EXP-154 rho=0.3 (+LAB)** | 93.62 | 93.49 | 90.78 | +0.03 |

**核心 finding**:
1. **rho=0.2 + LAB mean 93.78 反超 EXP-150 C 方案 (93.65) +0.13pp**, 反超 vanilla +0.19pp ⭐⭐
2. **rho=0.3 + LAB 跟 EXP-150 C 几乎持平** (93.62 vs 93.65 -0.03) — LAB 在高 rho 下加成有限
3. **rho=0.2 是新 sweet spot** (Digits 上 +LAB 后比原 C 方案 rho=0.3 更优), trade-off 解读: LAB 已经拉权重给小域, DSE rho 不需要那么大也能修 svhn
4. **svhn 全 4 runs 90+ 稳健** (lorth+cc=0 fix 工作 OK, 没出现 EXP-146 的 svhn 38 灾难)
5. **跟 PACS/Office 共同模式**: 加 LAB 后 sweet rho 比 DSE-only 稍降 (PACS 0.3→0.2, Office 0.5→0.3, Digits 0.3→0.2)

## 数据保存 (CLAUDE.md 零零零规则)

- 4 个独立 dump_diag, 1.3-3.4GB 本地 (4 runs avg ~17 best dumps + 4 finals + 400 round npz)
- proto_diag 全 169 字段 (DSE 22 + LAB 147)
- best/final heavy snapshot 含 state_dict_fp16 + features + labels + preds + logits + confusion + proto_diag

## paper 价值

**3 dataset 完整加成 (PACS / Office / Digits) 全部反超原 baseline**:
- PACS EXP-149: rho=0.3+LAB mean 75.10 vs DaA 68.34 = **+6.76pp** ⭐⭐⭐
- Office EXP-151: rho=0.3+LAB mean 63.99 vs DaA 63.55 = **+0.44pp** marginal pass
- **Digits EXP-154: rho=0.2+LAB+lorth+cc=0 mean 93.78 vs vanilla 93.59 = +0.19pp** ✅

3 dataset paper claim 完整成立 (PACS 大胜 + Office 边缘 + Digits 加成).
