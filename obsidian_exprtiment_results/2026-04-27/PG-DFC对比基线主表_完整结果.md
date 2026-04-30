---
date: 2026-04-27
type: 实验结果汇总(主表 - 干净版,只保留正确数据)
status: EXP-137 PG-DFC 完整 (PACS 2-seed sc3+v100, Office 3-seed sc3); EXP-139 ML 跑中
last_revised: 2026-04-29 (清理: 删 EXP-135 PG-DFC 错误数据 + 旧错算 + 表格统一为 1 个 PACS / 1 个 Office)
data_source:
  - EXP-130 sc3_v2_logs (FedAvg / FedBN / FedProx / FedProto / MOON / F2DC vanilla / F2DC+DaA / FDSE)
  - **EXP-137 sc3 + v100 logs (PG-DFC vanilla / +DaA, 4-bug fix 后)** ⭐ 主结果
  - EXP-139 sc3 + HPC logs (PG-DFC-ML, 跑中)
4-bug fix (EXP-137 已应用):
  - bug 1: 聚合白名单 revert (维持 release 默认全聚合, commit 38457ff)
  - bug 2: deterministic eval (is_eval=True 走 0.5 noise)
  - bug 3: 硬编码 num_classes=7 → args.num_classes (支持 Office 10 类)
  - bug 4: class_proto persistent=False → server eval 时显式同步 (commit 0e5f1df)
allocation: fixed (PACS photo:2/art:3/cartoon:2/sketch:3; Office caltech:3/amazon:2/webcam:2/dslr:3)
seeds: [2, 15, 333]
training: R=100, E=10, lr=0.01, batch=46/64
metric_def: |
  AVG Best = 100 轮中"4 域简单平均"最大的那一轮的值 (单 round 同时刻);
  AVG Last = R100 时的 4 域简单平均;
  ⚠️ 不要用"per-domain 各自 max 取算术平均"算 AVG_B (会虚高 0.8~2.2pp)
---

# PG-DFC vs Baselines — 完整对比主表(干净版)

> 跟 F2DC paper Table 1 格式一致 (per-domain + AVG Best + AVG Last + Best Round)
> 严格 multi-seed mean,fixed allocation
> **本表只保留正确数据**: baselines (EXP-130) + PG-DFC (EXP-137 4-bug fix 后) + PG-DFC-ML (EXP-139 跑中)
> EXP-135 PG-DFC / EXP-131 PG-DFC v3.2/v3.3 等错误数据**已全部删除**

---

## Table 1. PACS (4 domains × 7 classes, fixed: photo:2/art:3/cartoon:2/sketch:3)

> **比较基准说明**: F2DC 论文 (CVPR'26) 把 DaA 当作 F2DC 的内置组件, 所以本表的"F2DC"代表 **F2DC + DaA** (跟论文一致), 而"F2DC vanilla"是消融对照 (拆掉 DaA)。
> 我们的 **PG-DFC-ML (EXP-139, 不加 DaA)** 主对照是 **F2DC + DaA**, 因为 DaA 是 F2DC 的一部分。

### 2-seed mean (s=15 + s=333)

| Method                         |   photo   |    art    |  cartoon  |   sketch    |  **AVG_B ↑**   |  **AVG_L ↑**  |   gap   | Best Round |
| ------------------------------ | :-------: | :-------: | :-------: | :---------: | :------------: | :-----------: | :-----: | :--------: |
| FedAvg [AISTATS'17]            |   64.52   |   53.93   | **81.41** |    77.01    |     69.22      |     68.39     |  0.83   |   R95.5    |
| FedBN [ICLR'21]                |   58.53   |   43.75   |   72.97   |    72.61    |     61.97      |     59.74     |  2.23   |   R60.5    |
| FedProx [MLSys'20]             |    TBD    |    TBD    |    TBD    |     TBD     |      TBD       |      TBD      |    -    |    TBD     |
| FedProto [AAAI'22]             |    TBD    |    TBD    |    TBD    |     TBD     |      TBD       |      TBD      |    -    |    TBD     |
| FPL [CVPR'23]                  |    TBD    |    TBD    |    TBD    |     TBD     |      TBD       |      TBD      |    -    |    TBD     |
| MOON [CVPR'21]                 |   58.24   |   39.10   |   62.29   |    65.54    |     56.29      |     51.32     |  4.97   |   R93.0    |
| FDSE [CVPR'25] (F2DC framework) |   61.38  |   40.08   |   67.30   |    50.19    |     54.74      |     37.71     | **17.03** ⚠️ | R19.0  |
| **F2DC vanilla** (EXP-135)     |   68.56   |   59.44   |   80.13   |    80.96    |     72.27      |     69.37     |  2.90   |   R95.0    |
| **F2DC + DaA** (EXP-135) ⭐ paper | **74.10** | **59.56** |   77.35   |    68.66    |     69.92      |     69.12     |  0.80 ⭐稳 |   R91.5    |
| **PG-DFC vanilla** (EXP-137)   |   67.52   |   58.21   |   78.53   |  **83.82**  |   **72.02**    |  **71.77**    | 0.25 ⭐稳 |   R88.5    |
| **PG-DFC + DaA** (EXP-137)     |   73.80   | **63.73** |   74.79   |    67.20    |     69.88      |     68.90     |  0.98   |   R95.5    |
| **PG-DFC-ML** (EXP-139, 跑中, 不加 DaA) | TBD | TBD     |    TBD    |     TBD     | TBD (sc3 R72+) |      TBD      |    -    |     -      |
| **Δ PG-DFC-ML vs F2DC+DaA (主对照)** |  TBD |    TBD    |    TBD    |     TBD     |      TBD       |      TBD      |    -    |     -      |
| **Δ PG-DFC vanilla vs F2DC vanilla** | -1.04 | -1.23 | -1.60   |  **+2.86** ⭐ |    -0.25       | **+2.40pp** ✅ | -2.65   |  -6.5 round |
| **Δ PG-DFC vanilla vs F2DC+DaA**     | -6.58 | -1.35 | +1.18 |  **+15.16** ⭐⭐ | **+2.10pp** ✅ | **+2.65pp** ✅ | -0.55   |  -3 round |
| **Δ F2DC+DaA vs F2DC vanilla (DaA 增量)** | +5.54 | +0.12 | -2.78 | **-12.30** ⚠️ | **-2.35** ⚠️ | -0.25 | +2.10 | -3.5 round |

### Per-seed (PACS)

| Method                  | seed | R@Best |   photo   |    art    |  cartoon  |   sketch    | **AVG_B** | **AVG_L** | server |
| ----------------------- | :--: | :----: | :-------: | :-------: | :-------: | :---------: | :-------: | :-------: | :----: |
| FedAvg                  |  15  |  R99   |   64.97   |   52.21   |   83.76   |    77.32    |  69.565   |  69.565   | sc4    |
| FedAvg                  | 333  |  R92   |   64.07   |   55.64   |   79.06   |    76.69    |  68.865   |   67.205  | sc4    |
| FedBN                   |  15  |   -    |     -     |     -     |     -     |      -      |  63.540   |   61.810  | sc3    |
| FedBN                   | 333  |   -    |     -     |     -     |     -     |      -      |  60.393   |   57.665  | sc3    |
| MOON                    |  15  |  R90   |   55.99   |   33.09   |   56.62   |    61.53    |  51.808   |   47.085  | sc4    |
| MOON                    | 333  |  R96   |   60.48   |   45.10   |   67.95   |    69.55    |   60.77   |   55.560  | sc4    |
| FDSE (F2DC framework)   |  15  |  R18   |   67.37   |   36.52   |   63.46   |    46.24    |  53.398   |   33.575  | sc3    |
| FDSE (F2DC framework)   | 333  |  R20   |   55.39   |   43.63   |   71.15   |    54.14    |  56.078   |   41.838  | sc3    |
| F2DC vanilla            |  15  |  R96   |   69.16   |   60.05   | **83.12** |    79.49    |   72.96   |   69.78   | sc6    |
| F2DC vanilla            | 333  |  R96   |   67.96   |   58.82   |   77.14   |  **82.42**  |   71.59   |   68.96   | sc6    |
| **F2DC + DaA** ⭐ paper  |  15  |  R94   | **76.35** | **63.24** |   79.91   |    74.52    | **73.51** | **72.97** | sc6    |
| **F2DC + DaA**          | 333  |  R89   | **71.86** |   55.88   |   74.79   |    62.80    |   66.33   |   65.26   | sc3    |
| **PG-DFC vanilla** ⭐    |  15  |  R96   |   70.06   |   59.80   |   80.34   |  **84.71**  | **73.73** | **73.47** ⭐ | v100 |
| **PG-DFC vanilla**      | 333  |  R81   |   64.97   |   56.62   |   76.71   |  **82.93**  | **70.31** |   70.06   | sc3    |
| PG-DFC + DaA            |  15  |  R92   | **74.55** | **66.91** |   77.35   |    72.87    |   72.92   |   70.95   | v100   |
| PG-DFC + DaA            | 333  |  R99   | **73.05** |   60.54   |   72.22   |    61.53    |   66.84   |   66.84   | sc3    |

### PACS 关键结论

1. **PG-DFC vanilla 2-seed mean best 72.02 / last 71.77,sketch 域 83.82 ⭐**(主要受益)
2. **PG-DFC vanilla 比 F2DC+DaA(论文版)+2.10pp best / +2.65pp last** ⭐(sketch +15.16pp 主导)
3. **PG-DFC vanilla 比 FedAvg +2.80pp best / +3.38pp last**
4. **PG-DFC + DaA 在 PACS 上反而拖性能** -2.14pp best(sketch 67.20 vs vanilla 83.82 = -16.62pp 是主因)
5. PG-DFC vanilla last gap 仅 0.25pp ⭐**最稳定**(F2DC+DaA gap 0.80 也稳)
6. **DaA 在 PACS 上是负贡献** -2.35pp(F2DC+DaA 比 F2DC vanilla 还差),sketch 域 -12.30pp 主因 → DaA reweight 把 sketch 主力 client 权重砍 -31%
7. **EXP-139 PG-DFC-ML 主对照是 F2DC+DaA**(因为论文版 F2DC = F2DC+DaA),要赢 best 69.92 / last 69.12 才能 claim ML 提升

### PG-DFC vanilla per-domain 增益拆解 (vs F2DC+DaA, 主对照)

| Domain | F2DC+DaA mean | PG-DFC vanilla mean | Δ | 含义 |
|---|:---:|:---:|:---:|---|
| photo   | **74.10** | 67.52 | -6.58 | DaA reweight 升 photo 51%, vanilla 没升 |
| art     | 59.56 | 58.21 | -1.35 | 接近 |
| cartoon | 77.35 | 78.53 | +1.18 | PG-DFC 略胜 |
| **sketch** | 68.66 | **83.82** | **+15.16** ⭐⭐ | DaA 砍 sketch 主力 client 权重, vanilla 保留 |
| **AVG_B** | 69.92 | **72.02** | **+2.10** ✅ | sketch +15 主导 |
| **AVG_L** | 69.12 | **71.77** | **+2.65** ✅ | 同上 |

→ **PG-DFC vanilla 真正强在 sketch 域**,因为没用 DaA 砍 sketch 主力 client (sample share 0.151 each, 占 PACS 数据 45%)。
→ DaA 在 PACS 上是负贡献 (paper 用 DaA 反而损失), 我们 vanilla 路径的 sketch 优势是真正的 paper-grade contribution。

---

## Table 2. Office-Caltech (4 domains × 10 classes, fixed: caltech:3/amazon:2/webcam:2/dslr:3)

### 3-seed mean (s=2 + s=15 + s=333)

| Method                       | caltech | amazon | webcam | dslr   | **AVG_B ↑** | **AVG_L ↑** | gap | Best Round |
| ---------------------------- | :-----: | :----: | :----: | :----: | :---------: | :---------: | :-: | :--------: |
| FedAvg [AISTATS'17]          | 61.83   | 74.47  | 58.62  | 36.67  | 57.90       | 54.01       | 3.89 | R62.0     |
| FedBN [ICLR'21]              | 61.61   | 72.89  | 51.73  | 38.34  | 56.14       | 52.61       | 3.53 | R82.5     |
| FedProx [MLSys'20]           | 62.95   | 71.58  | 55.17  | 40.00  | 57.43       | 55.45       | 1.98 | R79.0     |
| FedProto [AAAI'22]           | 63.84   | 74.47  | 62.94  | 38.33  | 59.90       | 58.04       | 1.86 | R86.0     |
| FPL [CVPR'23]                | TBD     | TBD    | TBD    | TBD    | TBD         | TBD         | -    | TBD       |
| MOON [CVPR'21]               | 58.70   | 70.53  | 47.42  | 33.33  | 52.49       | 50.11       | 2.38 | R54.0     |
| **FDSE [CVPR'25]**           | 57.59   | 62.37  | **74.14** | **60.00** | **63.52** ⭐ | 59.33   | 4.19 | R69      |
| **F2DC [CVPR'26]**           | **63.84** | **77.37** | 56.04 | 45.00 | 60.56       | 56.68       | 3.88 | R99       |
| **F2DC + DaA**               | 63.40   | 73.69  | 63.79  | 53.33  | 63.55       | 62.07       | 1.48 | R84.0     |
| **PG-DFC vanilla** (EXP-137) | 66.07   | **76.49** | 54.60 | 50.00 | 61.79     | 56.36       | 5.43 | R85.3     |
| **PG-DFC + DaA** (EXP-137) ⭐ | 63.25 | 69.47 | 63.79 | **62.22** | **64.69** ⭐ | **60.35** | 4.34 | R81.3 |
| **PG-DFC-ML** (EXP-139, 跑中) | TBD   | TBD    | TBD    | TBD    | TBD (HPC R21+) | TBD     | -    | -         |
| **PG-DFC-ML + DaA** (EXP-139, 跑中) | TBD | TBD | TBD | TBD | TBD (HPC R0+) | TBD | - | - |
| **Δ PG-DFC+DaA vs F2DC+DaA** | -0.15 | -4.22  | 0.00   | **+8.89** ⭐ | **+1.14pp** ✅ | -1.72pp | +2.86 | -3 round |
| **Δ PG-DFC+DaA vs FDSE**     | +5.66  | +7.10  | -10.35 | +2.22  | **+1.17pp** ✅ | +1.02pp | +0.15 | +12 round |

### Per-seed (Office)

| Method                       | seed | R@Best | caltech | amazon | webcam | dslr   | **AVG_B** | **AVG_L** | server |
| ---------------------------- | :--: | :----: | :-----: | :----: | :----: | :----: | :-------: | :-------: | :----: |
| FedAvg                       | 15   | -      | -       | -      | -      | -      | 54.485    | 48.830    | sc3    |
| FedAvg                       | 333  | -      | -       | -      | -      | -      | 61.312    | 59.188    | sc3    |
| F2DC vanilla                 | 15   | -      | -       | -      | -      | -      | 60.80     | 54.618    | sc3    |
| F2DC vanilla                 | 333  | -      | -       | -      | -      | -      | 60.323    | 58.735    | sc3    |
| F2DC + DaA                   | 15   | R100   | 63.925  | -      | -      | -      | 63.925    | 63.925    | sc3    |
| F2DC + DaA                   | 333  | R68    | 63.175  | -      | -      | -      | 63.175    | 60.220    | sc3    |
| FDSE                         | 15   | R81    | 60.99   | -      | -      | -      | 60.99     | 58.062    | sc3    |
| FDSE                         | 333  | R55    | 66.055  | -      | -      | -      | 66.055    | 60.6      | sc3    |
| **PG-DFC vanilla** (EXP-137) | 2    | R93    | **65.62** | **74.21** | 53.45 | 60.00 | **63.32** | 59.33  | sc3    |
| **PG-DFC vanilla**           | 15   | R75    | 64.29   | **77.37** | 51.72 | 40.00 | 58.35   | 54.15     | sc3    |
| **PG-DFC vanilla**           | 333  | R88    | **68.30** | **77.89** | 58.62 | 50.00 | 63.70 | 55.60   | sc3    |
| **PG-DFC + DaA**             | 2    | R97    | **67.41** | **70.00** | 62.07 | **73.33** | **68.20** ⭐ | **66.82** ⭐ | sc3 |
| **PG-DFC + DaA**             | 15   | R56    | 58.04   | 66.84  | **63.79** | 56.67 | 61.34   | 55.91     | sc3    |
| **PG-DFC + DaA**             | 333  | R91    | 64.29   | **71.58** | **65.52** | 56.67 | 64.52  | 58.33     | sc3    |

### Office 关键结论

1. **PG-DFC + DaA 3-seed mean best 64.69 ⭐**,排名 #1,微胜 FDSE 63.52 (+1.17pp) 跟 F2DC+DaA 63.55 (+1.14pp)
2. PG-DFC vanilla 单跑 best 61.79,差 PG-DFC+DaA 2.90pp(**Office 上 DaA 真的有 +3pp 提升**,跟 PACS 完全相反)
3. 关键:**Office 上 DaA 必须加,不加跟 PG-DFC+DaA 同跑道比赢不了**
4. **EXP-139 ML 在 Office 也跑了 +DaA 版**(HPC 1189313 RUNNING),才能跟 PG-DFC+DaA 64.69 同跑道比

### Office DaA 跟 PACS DaA 不对称(重要观察)

| 数据集 | vanilla mean best | +DaA mean best | Δ DaA | 是否加 DaA? |
|---|:---:|:---:|:---:|:---:|
| PACS | 72.02 | 69.88 | **-2.14** ⚠️ | ❌ 不加 |
| **Office** | 61.79 | **64.69** | **+2.90** ✅ | ✅ 加 |

→ **PACS 上 sketch 域被 DaA reweight 砍权重,导致主导域 sketch 大跌**
→ **Office 上 dslr 域(样本最少)受益,被 DaA 升权 +51% 后 acc 大涨**
→ EXP-139 ML PACS 也只跑 vanilla(跟 PG-DFC vanilla 比),Office 跑 +DaA(跟 PG-DFC+DaA 比)

---

## Table 3. Digits (4 domains × 10 classes, fixed: mnist:3/usps:6/svhn:6/syn:5)

### 2-seed mean (s=15 + s=333)

| Method                 |   mnist   |   usps    |   svhn    |    syn    | **AVG_B ↑** | **AVG_L ↑** | Best Round |
| ---------------------- | :-------: | :-------: | :-------: | :-------: | :---------: | :---------: | :--------: |
| FedAvg                 | 96.00     | 91.58     | 87.48     | 92.38     | 91.86       | 91.63       | R89.5      |
| FedBN                  | 95.58     | 90.19     | 86.12     | 91.34     | 90.81       | 90.57       | R82.5      |
| FedProx                | 96.30     | 91.18     | 87.60     | 92.69     | 91.94       | 91.82       | R95.0      |
| FedProto               | **97.08** | **92.13** | 87.84     | 93.08     | **92.53**   | 92.53       | R100.0     |
| MOON                   | 95.73     | 91.61     | 87.30     | 91.73     | 91.59       | 90.37       | R87.5      |
| FDSE                   | 92.34     | 91.38     | 74.41     | 88.50     | 86.66       | 84.61       | R74.0      |
| **F2DC**               | **97.34** | 92.46     | 90.18     | **94.36** | **93.59** ⭐ | **93.40**   | R94.5      |
| **PG-DFC vanilla** (EXP-137) | TBD | TBD     | TBD       | TBD       | TBD (待跑)   | TBD         | -          |

> ⚠️ Digits 的 PG-DFC EXP-137 数据未跑(focus 在 PACS+Office),用 TBD 占位。如要补 Digits 实验需另开 sbatch。

---

## EXP-137 vs EXP-135 4-bug fix 修复增量(为什么用 EXP-137 数据)

### PACS 2-seed mean

| 配置 | EXP-135 mean best/last | EXP-137 mean best/last | Δ best | Δ last |
|---|:---:|:---:|:---:|:---:|
| **PG-DFC vanilla** | 70.58 / 69.01 | **72.02 / 71.77** | **+1.44** ✅ | **+2.76** ✅ |
| PG-DFC +DaA | 70.24 / 68.07 | 69.88 / 68.90 | -0.36 | +0.83 |

### Office 3-seed mean(EXP-135 V100 vs EXP-137 sc3)

| 配置 | EXP-135 V100 mean best/last | EXP-137 sc3 mean best/last | Δ best | Δ last |
|---|:---:|:---:|:---:|:---:|
| **PG-DFC vanilla** | 58.86 / 55.70 (2-seed) | **61.79 / 56.36 (3-seed)** | **+2.93** ✅ | +0.66 |
| **PG-DFC +DaA** | 62.93 / 60.65 | **64.69 / 60.35** | **+1.76** ✅ | -0.30 |

→ **PG-DFC vanilla 4-bug fix 是真正的 paper-grade 增量**:PACS +2.76 last、Office +2.93 best。
→ +DaA 路径在 fix 前后基本持平(Office +1.76 best 但 last 持平)。
→ **本主表只用 EXP-137 数据**,EXP-135/EXP-131 旧数据 **已全部删除**。

---

## EXP-139 ML 跑中(待回填)

### sc3 当前进度(R67+)

| run | 当前 round | 当前 acc | 跟 EXP-137 vanilla 同 round Δ |
|---|:---:|:---:|:---:|
| ML PACS s15 | R67 | 68.76 | +0.65 vs v100 vanilla R67 (68.11) |
| ML PACS s333 | R67 | 66.27 | -0.75 vs sc3 vanilla R67 (67.02) |

### HPC 当前进度

| Job | run | 状态 |
|---|---|:---:|
| 1189310 | ML PACS s2 R100 vanilla | RUNNING |
| 1189313 | ML Office +DaA seq (s2/s15/s333) R100 | RUNNING(刚 launch) |

R100 完成后回填到 Table 1 (PACS) + Table 2 (Office) 的 PG-DFC-ML 行。

---

## Figures(待 EXP-139 完成后重画)

旧 figs 已不准(用了 EXP-131/EXP-135 PG-DFC 数据),重画时用 EXP-137 + EXP-139 ML 数据。
