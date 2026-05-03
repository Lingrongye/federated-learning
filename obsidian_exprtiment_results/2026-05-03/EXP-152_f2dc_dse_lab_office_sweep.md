---
date: 2026-05-03
type: 实验记录 (f2dc_dse_lab Office sweep, 救 webcam)
status: ✅ 6 runs R100 全完成 + rsync 1.6GB 本地 (实际 wall ~1h, 33s/round)
exp_id: EXP-152
goal: EXP-151 winner webcam 输 DaA -6.9pp, 多 path 探索找 webcam 救回点
---

# EXP-152: F2DC + DSE_Rescue3 + LAB v4.2 Office sweep

## 一句话

EXP-151 winner s=15 rho=0.3 webcam 60.34% vs DaA 67.24% **输 -6.9pp**, 诊断显示真凶是 DSE 信号扰动 webcam (LAB 权重 18.58% 已是 sample_share 2 倍)。EXP-152 跑 3 path × 2 seed = 6 runs 探索:
- **Path A** (rho=0.2 v2c): DSE 弱化能否救 webcam
- **Path B** (rho=0.1 v2c): DSE 更弱看 dslr 是否丢
- **Path C** (rho=0.3 standard): 关 small_protect 看 LAB raw 效果

## EXP-151 webcam 诊断 (来自 best_R086.npz)

| 域 | sample_share | LAB ratio | w_proj | small_protect | val_loss_ema | q | EXP-151 acc | DaA acc | Δ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| caltech | 0.5306 | 0.80 | 0.4245 | No | 3.20 | 0.476 | 63.84 | 62.50 | +1.34 |
| amazon | 0.3024 | 0.80 | 0.2419 | No | 1.42 | 0.000 | 71.58 | 72.63 | -1.05 |
| webcam | 0.0929 | **2.00** | **0.1858** ⭐ | **Yes** | 1.62 | **0.000 ❗** | 60.34 | **67.24** | **-6.90** ❌ |
| dslr | 0.0741 | 2.00 | 0.1482 | Yes | 3.29 | 0.524 | 63.33 | 53.33 | **+10.00** ⭐ |

**真凶分析** (vs DSE-only s=15 webcam = 56.90%):
- DSE-only 单跑就压 webcam -10pp (DSE 在低差异域是 negative interaction)
- LAB 用 small_protect ratio=2.0 把 webcam 权重抬到 18.58% (sample_share 2×)
- LAB 救回 +3.45pp (56.90 → 60.34) 但救不到 DaA 的 67.24
- **不是权重问题, 是 DSE 信号扰动 webcam 干净特征空间**

## 6 runs 配置

| run | rho | LAB mode | host:seed |
|---|:---:|:---:|:---:|
| **A1** rho=0.2 v2c s=15 | 0.2 | rmin=2.0/rmax=4.0 | sub1:s=15 |
| A2 rho=0.2 v2c s=333 | 0.2 | rmin=2.0/rmax=4.0 | sub2:s=333 |
| **B1** rho=0.1 v2c s=15 | 0.1 | rmin=2.0/rmax=4.0 | sub1:s=15 |
| B2 rho=0.1 v2c s=333 | 0.1 | rmin=2.0/rmax=4.0 | sub2:s=333 |
| **C1** rho=0.3 std s=15 | 0.3 | **standard rmin=0.8/rmax=2.0** | sub1:s=15 |
| C2 rho=0.3 std s=333 | 0.3 | standard | sub2:s=333 |

GPU 编排: sub1 余 14.6GB → 3 Office × 3GB = 9GB + PACS rerun 10GB = 19GB; sub2 余 13.6GB → 3 Office + PACS rho=0.2 9.5GB = 18.5GB。

## 决策树

| 假设结果 | 结论 |
|---|---|
| Path A rho=0.2 webcam ≥ 64 | **DSE 弱化救 webcam ✅** → 新 winner rho=0.2 |
| Path B rho=0.1 webcam ≥ 65 但 dslr ≤ 53 | DSE 拉太弱, rho=0.2 是 sweet |
| Path A/B 都救不回 webcam | DSE+LAB 在 Office 本质问题, paper 接受 trade-off |
| **Path C standard webcam 反而高** | small_protect 设计错了, standard mode 才对 — 推翻 EXP-144 P4 finding |
| Path C standard 跟 EXP-151 持平 | small_protect 优势仅 dslr |

## EXP-151 baseline (对照基准)

| Run | best | 同 rho DaA | Δ |
|---|:---:|:---:|:---:|
| s=15 rho=0.3 v2c | 64.77 | 63.92 | +0.84 |
| s=15 rho=0.5 v2c | 63.57 | 63.92 | -0.35 |
| s=333 rho=0.3 v2c | 63.21 | 63.17 | +0.04 (平) |
| s=333 rho=0.5 v2c | 62.22 | 63.17 | -0.95 |

EXP-152 任何 path 反超 EXP-151 winner 64.77 都是新 finding. 任何 path 把 webcam 拉到 65+ 都解决了痛点。

## R100 完整结果 ✅

### 6 runs per-domain best+final

| Run | bestR | best_avg | caltech | amazon | webcam | dslr | final | drop |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| s=15 rho=0.1 v2c | R82 | 62.09 | 62.05 | 72.63 | 60.34 | 53.33 | 57.39 | -4.70 |
| s=15 rho=0.2 v2c | R87 | 61.36 | 61.16 | 77.37 | 56.90 | 50.00 | 58.26 | -3.10 |
| s=15 rho=0.3 std | R76 | 60.94 | 61.61 | 75.26 | 56.90 | 50.00 | 57.50 | -3.44 |
| s=333 rho=0.1 v2c | R72 | 63.78 | 64.73 | 78.42 | 58.62 | 53.33 | 59.05 | -4.73 |
| **s=333 rho=0.2 v2c** ⭐⭐ | R80 | **65.43** | 65.62 | 77.37 | **62.07** ⭐ | 56.67 | 58.47 | -6.96 |
| s=333 rho=0.3 std | R80 | 61.93 | 65.62 | 78.42 | 60.34 | 43.33 | **61.35** ⭐ | **-0.58** ⭐ |

### rho mean (s=15+s=333 mean per config)

| Config | best mean | final mean | drop avg |
|---|:---:|:---:|:---:|
| rho=0.2 v2c | 63.39 | 58.37 | -5.03 |
| rho=0.1 v2c | 62.93 | 58.22 | -4.72 |
| **rho=0.3 std** | 61.43 | **59.43** ⭐ | **-2.00** ⭐⭐ |

### 跟 EXP-151 + DaA 完整对比

| Method | best mean | final mean | vs DaA best | 备注 |
|---|:---:|:---:|:---:|---|
| f2dc + DaA (Office) | 63.55 (paper avg) | — | — | baseline |
| **EXP-151 rho=0.3 v2c** ⭐ | **63.99** | 58.92 | **+0.44** ✅ | paper main winner |
| EXP-151 rho=0.5 v2c | 62.90 | 60.83 | -0.65 | webcam 67.24 single domain ⭐ |
| EXP-152 rho=0.2 v2c | 63.39 | 58.37 | -0.16 | A2 single seed 65.43 ⭐⭐ |
| EXP-152 rho=0.1 v2c | 62.93 | 58.22 | -0.62 | DSE 太弱 |
| **EXP-152 rho=0.3 std** | 61.43 | **59.43** ⭐ | -2.12 | **final 最稳 drop -0.58** ⭐ |

### 关键 finding

1. **A2 s=333 rho=0.2 v2c best 65.43 ⭐⭐ 是 Office 历史单 seed-pair 最高** (vs DaA s=333 R67 best 63.17 = **+2.26** ✅, 4 域全胜 DaA: caltech +1.33 / amazon +2.63 / webcam +1.73 / dslr +3.34)
2. **webcam 不是 rho 单调函数**:
   - s=15 rho=0.5 (EXP-151): webcam **67.24** ⭐⭐ (跟 DaA 持平!)
   - s=15 rho=0.3 (EXP-151): 60.34
   - s=15 rho=0.2 (EXP-152): **56.90** (最差)
   - s=15 rho=0.1 (EXP-152): 60.34
   - **rho=0.5 强 DSE 反而最救 webcam** (DSE adapter 强力修正 webcam, 但牺牲 dslr)
3. **rho=0.3 standard mode (关 small_protect) final 最稳** — drop 仅 -0.58, mean 59.43 paper-grade final 最高
4. **paper 主 claim 仍是 EXP-151 rho=0.3 v2c best mean +0.44 vs DaA**, EXP-152 揭示了 webcam rho 非单调反直觉 + s=333 rho=0.2 单 seed 大胜 + standard mode 稳定性优势

## 数据保存

- 本地 EXP-152 1.6GB (125 best + 6 final + 600 round npz, 6 个独立 dump_diag 路径)
- proto_diag 全 180-181 字段完整 (DSE 22 + LAB 158-159)
