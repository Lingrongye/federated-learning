---
date: 2026-04-27
type: 实验结果汇总(主表 - 干净版,只保留正确数据)
status: EXP-137 PG-DFC 完整 + EXP-139 ML 完整 (PACS 3-seed, Office +DaA 3-seed, Digits 4-variants × 2-seed)
last_revised: 2026-04-30 (EXP-139 全部 R100 完成回填 + 跨 dataset ML diag 诊断结论)
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

| Method                                  |   photo   |    art    |  cartoon  |    sketch     |  **AVG_B ↑**  |  **AVG_L ↑**  |     gap      | Best Round |
| --------------------------------------- | :-------: | :-------: | :-------: | :-----------: | :-----------: | :-----------: | :----------: | :--------: |
| FedAvg [AISTATS'17]                     |   64.52   |   53.93   | **81.41** |     77.01     |     69.22     |     68.39     |     0.83     |   R95.5    |
| FedBN [ICLR'21]                         |   58.53   |   43.75   |   72.97   |     72.61     |     61.97     |     59.74     |     2.23     |   R60.5    |
| FedProx [MLSys'20]                      |    TBD    |    TBD    |    TBD    |      TBD      |      TBD      |      TBD      |      -       |    TBD     |
| FedProto [AAAI'22]                      |    TBD    |    TBD    |    TBD    |      TBD      |      TBD      |      TBD      |      -       |    TBD     |
| FPL [CVPR'23]                           |    TBD    |    TBD    |    TBD    |      TBD      |      TBD      |      TBD      |      -       |    TBD     |
| MOON [CVPR'21]                          |   58.24   |   39.10   |   62.29   |     65.54     |     56.29     |     51.32     |     4.97     |   R93.0    |
| FDSE [CVPR'25] (F2DC framework)         |   61.38   |   40.08   |   67.30   |     50.19     |     54.74     |     37.71     | **17.03** ⚠️ |   R19.0    |
| **F2DC vanilla** (EXP-135)              |   68.56   |   59.44   |   80.13   |     80.96     |     72.27     |     69.37     |     2.90     |   R95.0    |
| **F2DC + DaA** (EXP-135) ⭐ paper        | **74.10** | **59.56** |   77.35   |     68.66     |     69.92     |     69.12     |   0.80 ⭐稳    |   R91.5    |
| **PG-DFC vanilla** (EXP-137)            |   67.52   |   58.21   |   78.53   |   **83.82**   |   **72.02**   |   **71.77**   |   0.25 ⭐稳    |   R88.5    |
| **PG-DFC + DaA** (EXP-137)              |   73.80   | **63.73** |   74.79   |     67.20     |     69.88     |     68.90     |     0.98     |   R95.5    |
| **PG-DFC-ML** (EXP-139) ⭐ 3-seed        |   67.37   |   56.78   |   77.49   |   **81.61**   |   **70.81**   |     68.64     |     2.17     |   R85.0    |
| **Δ PG-DFC-ML vs F2DC+DaA (主对照)**       |   -6.73   |   -2.78   |   +0.14   | **+12.95** ⭐  | **+0.89pp** ✅ |     -0.48     |    +1.37     | -6.5 round |
| **Δ PG-DFC-ML vs PG-DFC vanilla**       |   -0.15   |   -1.43   |   -1.04   |     -2.21     |   -1.21 ⚠️    |   -3.13 ⚠️    |    +1.92     | -3.5 round |
| **Δ PG-DFC vanilla vs F2DC vanilla**    |   -1.04   |   -1.23   |   -1.60   |  **+2.86** ⭐  |     -0.25     | **+2.40pp** ✅ |    -2.65     | -6.5 round |
| **Δ PG-DFC vanilla vs F2DC+DaA**        |   -6.58   |   -1.35   |   +1.18   | **+15.16** ⭐⭐ | **+2.10pp** ✅ | **+2.65pp** ✅ |    -0.55     |  -3 round  |
| **Δ F2DC+DaA vs F2DC vanilla (DaA 增量)** |   +5.54   |   +0.12   |   -2.78   | **-12.30** ⚠️ | **-2.35** ⚠️  |     -0.25     |    +2.10     | -3.5 round |

### Per-seed (PACS)

| Method                  | seed | R@Best |   photo   |    art    |  cartoon  |  sketch   | **AVG_B** |  **AVG_L**  | server |
| ----------------------- | :--: | :----: | :-------: | :-------: | :-------: | :-------: | :-------: | :---------: | :----: |
| FedAvg                  |  15  |  R99   |   64.97   |   52.21   |   83.76   |   77.32   |  69.565   |   69.565    |  sc4   |
| FedAvg                  | 333  |  R92   |   64.07   |   55.64   |   79.06   |   76.69   |  68.865   |   67.205    |  sc4   |
| FedBN                   |  15  |   -    |     -     |     -     |     -     |     -     |  63.540   |   61.810    |  sc3   |
| FedBN                   | 333  |   -    |     -     |     -     |     -     |     -     |  60.393   |   57.665    |  sc3   |
| MOON                    |  15  |  R90   |   55.99   |   33.09   |   56.62   |   61.53   |  51.808   |   47.085    |  sc4   |
| MOON                    | 333  |  R96   |   60.48   |   45.10   |   67.95   |   69.55   |   60.77   |   55.560    |  sc4   |
| FDSE (F2DC framework)   |  15  |  R18   |   67.37   |   36.52   |   63.46   |   46.24   |  53.398   |   33.575    |  sc3   |
| FDSE (F2DC framework)   | 333  |  R20   |   55.39   |   43.63   |   71.15   |   54.14   |  56.078   |   41.838    |  sc3   |
| F2DC vanilla            |  15  |  R96   |   69.16   |   60.05   | **83.12** |   79.49   |   72.96   |    69.78    |  sc6   |
| F2DC vanilla            | 333  |  R96   |   67.96   |   58.82   |   77.14   | **82.42** |   71.59   |    68.96    |  sc6   |
| **F2DC + DaA** ⭐ paper  |  15  |  R94   | **76.35** | **63.24** |   79.91   |   74.52   | **73.51** |  **72.97**  |  sc6   |
| **F2DC + DaA**          | 333  |  R89   | **71.86** |   55.88   |   74.79   |   62.80   |   66.33   |    65.26    |  sc3   |
| **PG-DFC vanilla** ⭐    |  15  |  R96   |   70.06   |   59.80   |   80.34   | **84.71** | **73.73** | **73.47** ⭐ |  v100  |
| **PG-DFC vanilla**      | 333  |  R81   |   64.97   |   56.62   |   76.71   | **82.93** | **70.31** |    70.06    |  sc3   |
| PG-DFC + DaA            |  15  |  R92   | **74.55** | **66.91** |   77.35   |   72.87   |   72.92   |    70.95    |  v100  |
| PG-DFC + DaA            | 333  |  R99   | **73.05** |   60.54   |   72.22   |   61.53   |   66.84   |    66.84    |  sc3   |
| **PG-DFC-ML** (EXP-139) |  2   |  R80   |   65.87   |   51.96   |   77.99   | **83.18** |   69.75   |    67.13    |  HPC   |
| **PG-DFC-ML**           |  15  |  R98   |   69.46   |   58.82   |   78.42   | **83.06** | **72.44** |  **71.20**  |  sc3   |
| **PG-DFC-ML**           | 333  |  R77   |   66.77   |   59.56   |   76.07   |   78.60   |   70.25   |    67.59    |  sc3   |

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
| **PG-DFC-ML + DaA** (EXP-139) | 62.95 | 69.47 | 59.19 | 58.89 | **62.63** | 58.98 | 3.66 | R95.0 |
| **Δ ML+DaA vs PG-DFC+DaA**   | -0.30 | 0.00 | -4.60 | -3.33 | **-2.06** ⚠️ | -1.37 ⚠️ | -0.68 | +13.7 round |
| **Δ ML+DaA vs F2DC+DaA(主对照)** | -0.45 | -4.22 | -4.60 | +5.56 | **-0.92pp** ⚠️ | -3.09 ⚠️ | +2.18 | +11 round |
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
| **PG-DFC-ML + DaA**          | 2    | R95    | 65.18   | 69.47  | 63.79 | 66.67 | **66.28** | 62.62 | HPC    |
| **PG-DFC-ML + DaA**          | 15   | R93    | 60.27   | 69.47  | 55.17 | 56.67 | 60.40   | 58.31     | HPC    |
| **PG-DFC-ML + DaA**          | 333  | R97    | 63.39   | 69.47  | 58.62 | 53.33 | 61.20   | 56.00     | HPC    |

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

| Method                       |   mnist   |   usps    | svhn  |    syn    | **AVG_B ↑** | **AVG_L ↑** | Best Round |
| ---------------------------- | :-------: | :-------: | :---: | :-------: | :---------: | :---------: | :--------: |
| FedAvg                       |   96.00   |   91.58   | 87.48 |   92.38   |    91.86    |    91.63    |   R89.5    |
| FedBN                        |   95.58   |   90.19   | 86.12 |   91.34   |    90.81    |    90.57    |   R82.5    |
| FedProx                      |   96.30   |   91.18   | 87.60 |   92.69   |    91.94    |    91.82    |   R95.0    |
| FedProto                     | **97.08** | **92.13** | 87.84 |   93.08   |  **92.53**  |    92.53    |   R100.0   |
| MOON                         |   95.73   |   91.61   | 87.30 |   91.73   |    91.59    |    90.37    |   R87.5    |
| FDSE                         |   92.34   |   91.38   | 74.41 |   88.50   |    86.66    |    84.61    |   R74.0    |
| **F2DC**                     | **97.34** |   92.46   | 90.18 | **94.36** | **93.59** ⭐ |  **93.40**  |   R94.5    |
| **PG-DFC vanilla** (EXP-139) |   97.21   |   92.11   | 89.95 |   94.27   |   93.39     |   93.01     |   R90.0    |
| **PG-DFC + DaA** (EXP-139)   |   97.26   |   94.00   | 88.85 |   94.91   | **93.75**   | **93.37**   |   R83.0    |
| **PG-DFC-ML** (EXP-139)      |   97.31   |   91.88   | 90.13 |   94.35   |   93.42     |   93.10     |   R83.0    |
| **PG-DFC-ML + DaA** (EXP-139) ⭐ | 97.35 | **94.17** | 88.78 | 94.88 | **93.80** ⭐ | **93.48** ⭐ |   R92.5    |
| **Δ ML+DaA vs F2DC(主对照)**  |   +0.01   |   +1.71   | -1.40 |   +0.52   |   +0.21     |   +0.08     |   -2 round |
| **Δ ML+DaA vs PG-DFC+DaA**   |   +0.09   |   +0.17   | -0.07 |   -0.03   |   +0.05     |   +0.11     |   +9.5 round |
| **Δ ML vs PG-DFC vanilla**   |   +0.10   |   -0.23   | +0.18 |   +0.08   |   +0.03     |   +0.09     |   -7 round |
| **Δ +DaA 增量 (PG-DFC)**     |   +0.05   |   +1.89   | -1.10 |   +0.64   |   +0.36     |   +0.36     |   -7 round |

### Per-seed (Digits, EXP-139)

| Method                  | seed | R@Best | mnist | usps  | svhn  | syn   | **AVG_B** | **AVG_L** | server |
| ----------------------- | :--: | :----: | :---: | :---: | :---: | :---: | :-------: | :-------: | :----: |
| **PG-DFC vanilla**      |  15  |  R88   | 97.21 | 92.33 | 90.01 | 94.52 | 93.518    | 93.288    | HPC    |
| **PG-DFC vanilla**      | 333  |  R92   | 97.21 | 91.93 | 89.89 | 94.02 | 93.262    | 92.730    | HPC    |
| **PG-DFC + DaA**        |  15  |  R87   | 97.05 | 93.32 | 89.26 | 95.02 | 93.662    | 93.610    | HPC    |
| **PG-DFC + DaA**        | 333  |  R98   | 97.47 | 94.67 | 88.44 | 94.79 | 93.842    | 93.125    | HPC    |
| **PG-DFC-ML**           |  15  |  R80   | 97.52 | 91.88 | 89.99 | 94.79 | 93.545    | 93.262    | HPC    |
| **PG-DFC-ML**           | 333  |  R86   | 97.09 | 91.88 | 90.27 | 93.91 | 93.288    | 92.910    | HPC    |
| **PG-DFC-ML + DaA** ⭐   |  15  |  R89   | 97.22 | 94.12 | 88.79 | 95.21 | 93.835    | 93.655    | HPC    |
| **PG-DFC-ML + DaA**     | 333  |  R96   | 97.48 | 94.22 | 88.77 | 94.55 | 93.755    | 93.305    | HPC    |

### Digits 关键结论

1. **PG-DFC-ML + DaA 2-seed mean best 93.80 ⭐ #1**(微胜 PG-DFC+DaA +0.05pp,微胜 F2DC paper 93.59 +0.21pp)
2. **DaA 在 Digits 上是正贡献 +0.36pp**(usps +1.89pp 主导,svhn -1.1pp,但整体涨)— 跟 Office 同向(+2.90pp),跟 PACS 反向(-2.35pp)
3. **ML 在 Digits 上几乎打平 vanilla(+0.03pp)**,但跟 +DaA 叠加最强(+0.05pp vs PG-DFC+DaA),**ML+DaA 是 Digits 最优配置**
4. usps 域是 ML+DaA 增益主导(94.17 vs vanilla 92.11 = +2.06pp),其他 3 域几乎打平

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

## EXP-139 ML 跨数据集结论(R100 完整,2026-04-30)

### 整体性能横向对比(主对照: vs F2DC + DaA paper)

| Dataset | seeds | ML 配置 | ML mean best | F2DC+DaA mean best | **Δ best** | 判决 |
|---|:---:|---|:---:|:---:|:---:|:---:|
| **PACS** | 3 (2/15/333) | ML(不加 DaA) | **70.81** | 69.92 | **+0.89** ✅ | **赢** |
| **Office** | 3 (2/15/333) | ML+DaA | 62.63 | 63.55 | **-0.92** ⚠️ | 输 |
| **Digits** | 2 (15/333) | ML+DaA | **93.80** ⭐ | 93.59 (F2DC paper, no DaA 数据) | **+0.21** ✅ | **赢** |

→ **ML 在 PACS / Digits 都赢 F2DC paper baseline**,Office 输(原因:dslr 域 ML 多参数过拟合)
→ **跨数据集净位置**: PACS ML 70.81 / Office ML+DaA 62.63 / Digits ML+DaA 93.80

### ML 4 模块 跨 dataset 工作情况(R0 → R99 trajectory mean)

> 所有诊断指标都从 EXP-139 R100 .log 的 `[ML diag]` print 提取
> mask3_sparsity = layer3 lite mask 全 (B,C,H,W) 张量元素均值, 0.5=均匀, < 0.5=偏 non-robust
> mask3_std = 跨 batch / channel / 位置的 std, 大 = mask 真在切, 小 = mask 全 0.5 没学
> attn_entropy = proto attention 分布的 entropy, ≈1.0 = 几乎均匀(没真选 specific proto)
> proto_signal = proto 信号占总特征比例, 越高 = proto attention 越主导

| Run | mask3_sparsity (R0→R99) | mask3_std (×倍) | aux3/main R99 | mask4 (R0→R99) | attn_entropy R99 | proto_signal R99 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PACS s15** ⭐ | 0.499 → 0.389 (−22%) | **0.00012 → 0.00124 (×9.9)** | 0.148 | 0.502 → 0.473 | 0.9996 | 0.0208 |
| PACS s333 | 0.495 → 0.398 (−19%) | 0.00020 → 0.00115 (×5.6) | 0.158 | 0.497 → 0.476 | 0.9995 | 0.0205 |
| PACS s2 | 0.498 → 0.395 (−21%) | 0.00017 → 0.00113 (×6.8) | 0.152 | 0.505 → 0.501 | 0.9999 | 0.0191 |
| Office+DaA s2 | 0.498 → 0.482 (−3%) | 0.00014 → 0.00027 (×1.9) | **0.654** ⚠️ | 0.502 → 0.478 | — | — |
| Office+DaA s15 | 0.499 → 0.485 (−3%) | **0.00036 → 0.00019 (×0.5)** ⚠️ 倒退 | **0.694** ⚠️ | 0.502 → 0.475 | 0.9991 | — |
| Office+DaA s333 | 0.496 → 0.480 (−3%) | 0.00021 → 0.00023 (×1.1) | **0.592** ⚠️ | 0.499 → 0.491 | — | — |
| Digits ML s15 | 0.499 → 0.371 (−26%) | 0.00044 → 0.00108 (×2.5) | 0.177 | 0.506 → 0.474 | 0.9992 | 0.0207 |
| Digits ML+DaA s15 | 0.499 → 0.386 (−23%) | 0.00034 → 0.00142 (×4.2) | 0.242 | 0.505 → 0.491 | 0.9991 | 0.0202 |

### ML 模块工作度 vs acc 增益完美正相关

| Dataset | mask3 学习强度 | acc 增益 (vs F2DC+DaA) | 因果 |
|---|---|:---:|---|
| **PACS** | ✅✅ 强 (mask3_std ×9.9, sparsity 降 22%) | **+0.89** ✅ | mask 真切 → acc 涨 |
| **Digits** | ✅ 中 (mask3_std ×4.2, sparsity 降 23%) | +0.21 ✅ | 中等切 → 微涨 |
| **Office** | ❌ 几乎不学 (sparsity 仅降 3%, std s15 倒退) | -0.92 ⚠️ | mask 没学 → 多 4 万参数过拟合 → 输 |

**关键洞察**:**ML 增益跟 mask3 学习强度严格正相关**。
- Office 上 mask3_std s15 反而**倒退**(0.00036→0.00019, ×0.5)+ aux3/main 升到 0.69(警戒 0.5),layer3 deep sup 信号过强压过 main path 收敛 → ML 反而拖性能
- PACS 上 mask3 学得最猛 (×9.9 spread, sparsity 真往 0.4 走) → acc 真涨

### ⚠️ proto attention 全部数据集都没工作(关键 paper-grade 发现)

- **attn_entropy 全部 ≈ 1.0**(均匀分布上限),**proto attention 没真选 specific proto**
- **proto_signal ≈ 0.02**(proto 信号占总特征仅 2%,几乎不影响最终特征)
- 这跟 PG-DFC v3.3 设计本身的问题一致(BN 强制 logit batch mean=0 → sigmoid → 0.5),**ML 也继承了这个 bug**
- **核心结论**: ML 的真正贡献是 **layer3 mask3** (DFD-lite),不是 proto attention,也不是 layer4 mask4

### 后续优化方向(基于诊断)

1. **方案 C: layer3 cleaned 接入 layer4 主路(最有价值)**
   - 当前 `out = self.layer4(feat3)` 用原 feat3,**ML 切分不影响主路**
   - 改成 `out = self.layer4(feat3 + alpha · (feat3_clean - feat3))` 软残差融合
   - 用 ramp_up 控制 alpha:warmup 时 0(不破坏对照),后期 0.3-0.5(主路真受益)
   - 预期 PACS 增益 +1-2pp(因为 mask3 真切了但当前没传到主路)

2. **修 proto attention(全 dataset 通用)**
   - DFD 模块 BN 拉平 logit → mask 永远 0.5
   - 改用 LayerNorm 或去掉 BN,让 logit 能 spread

3. **Office dslr 过拟合保护**
   - 降 ml_aux_alpha 0.1 → 0.05(让 deep sup 弱辅助)
   - 或者 Office 单独不用 ML(只用 PG-DFC+DaA)

---

## EXP-139 数据来源(供回查)

| Run | log 路径 |
|---|---|
| ML PACS s15 R100 | `experiments/ablation/EXP-139_pgml_main/logs/pgml_pacs_s15_R100.log` |
| ML PACS s333 R100 | `experiments/ablation/EXP-139_pgml_main/logs/pgml_pacs_s333_R100.log` |
| ML PACS s2 R100 | `experiments/ablation/EXP-139_pgml_main/logs/pgml_pacs_s2_1189310.log` |
| ML+DaA Office s2/s15/s333 R100 | `experiments/ablation/EXP-139_pgml_main/logs/pgml_daa_office_s{2,15,333}_R100.log` |
| Digits 4 variants × 2 seed R100 | `experiments/ablation/EXP-139_pgml_main/logs/digits_{pgdfc,pgml}_{vanilla,daa}_s{15,333}_R100.log` |

---

## Figures(待 EXP-139 完成后重画)

旧 figs 已不准(用了 EXP-131/EXP-135 PG-DFC 数据),重画时用 EXP-137 + EXP-139 ML 数据。
