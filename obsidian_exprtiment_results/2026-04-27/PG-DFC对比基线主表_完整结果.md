---
date: 2026-04-27
type: 实验结果汇总(主表)
status: EXP-137 4-bug fix 后 sc3 全部 8 run R100 已完成 (PACS s333 + Office s2/s15/s333 各 vanilla/+DaA), v100 PACS s15 in progress
last_revised: 2026-04-29 (EXP-137 sc3 8 run 回填到 Table 1b-NEW + Table 2-NEW; EXP-135 PG-DFC 数据标注已废弃)
data_source:
  - EXP-130 sc3_v2_logs (F2DC v2 + fdse v2, fixed allocation)
  - EXP-130 sc4_v2_logs (FedAvg + MOON v2, fixed allocation)
  - EXP-131 sc5 logs (PG-DFC v3.2 / v3.3, fixed allocation, ⚠️ 4-bug fix 前)
  - EXP-135 sc6+sc3 logs (4-method × 2-seed PACS R100 with diag, ⚠️ PG-DFC 部分已被 EXP-137 取代)
  - **EXP-137 sc3 logs (PG-DFC vanilla/+DaA, 4-bug fix 后, 主结果用本表) ⭐**
4-bug fix:
  - bug 1: 聚合白名单 (revert, 维持 release 默认全聚合, commit 38457ff)
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

# PG-DFC vs Baselines — 完整对比主表

> 跟 F2DC paper Table 1 格式一致 (per-domain + AVG Best + AVG Last + Best Round)
> 严格 2-seed mean,fixed allocation

---

## Table 1. PACS (4 domains × 7 classes, fixed: photo:2/art:3/cartoon:2/sketch:3)

| Method                         |       photo       |     art     |  cartoon  |  sketch   | **AVG Best ↑** | **AVG Last ↑** | Best Round (avg) |
| ------------------------------ | :---------------: | :---------: | :-------: | :-------: | :------------: | :------------: | :--------------: |
| FedAvg [AISTATS'17]            |       64.52       |    53.92    | **81.41** |   77.00   |     69.22      |     68.39      |      R96.5       |
| FedBN [ICLR'21]                |       58.53       |    43.75    |   72.97   |   72.61   |     61.97      |     59.74      |      R60.5       |
| FedProx [MLSys'20]             |     TBD (重启中)     |     TBD     |    TBD    |    TBD    |      TBD       |      TBD       |       TBD        |
| FedProto [AAAI'22]             |     TBD (重启中)     |     TBD     |    TBD    |    TBD    |      TBD       |      TBD       |       TBD        |
| FPL [CVPR'23]                  |        TBD        |     TBD     |    TBD    |    TBD    |      TBD       |      TBD       |       TBD        |
| MOON [CVPR'21]                 |  TBD (R34+缺s333)  |     TBD     |    TBD    |    TBD    |      TBD       |      TBD       |       TBD        |
| FDSE [CVPR'25]                 | TBD (R35+R30 未完成) |     TBD     |    TBD    |    TBD    |    **TBD**     |      TBD       |       TBD        |
| **F2DC [CVPR'26]**             |     **69.46**     |    56.62    |   78.53   |   79.49   |   **71.02**    |     69.57      |      R94.5       |
| **F2DC + DaA s=15** (sc3 R99)  |       73.65       |    64.22    |   78.85   |   74.01   |   **72.68**    |     70.45      |       R90        |
| **Δ F2DC+DaA vs vanilla F2DC** |       +4.19       |    +7.60    |   +0.32   |   -5.48   |   **+0.33**    |     +0.88      |    -4.5 round    |
| **PG-DFC v3.2 (Ours)**         |       67.82       |  **62.99**  |   80.45   | **81.53** |  **73.20** ⭐   |   **71.31**    |      R95.0       |
| **PG-DFC v3.3 (A+B)**          |       69.91       |  **60.17**  |   79.49   |   83.19   |   **73.19**    |     72.22      |      R95.0       |
| **Δ v3.2 vs FedAvg**           |       +3.30       |  **+9.07**  |   -0.96   |   +4.53   |  **+3.98pp**   |  **+2.92pp**   |    -1.5 round    |
| **Δ v3.2 vs FedBN**            |       +9.29       | **+19.24**  |   +7.48   |   +8.92   |  **+11.23pp**  |  **+11.57pp**  |   +34.5 round    |
| **Δ v3.2 vs F2DC**             |       -1.64       | **+6.37** ⭐ |   +1.92   |   +2.04   |  **+2.18pp**   |  **+1.74pp**   |    +0.5 round    |

### Per-seed 详细数据 (PACS)

| Method | s=15 best | s=333 best | s=15 last | s=333 last | s=15 best round | s=333 best round |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 69.565 | 68.865 | 69.565 | 67.205 | R100 | R93 |
| FedBN | 63.540 | 60.393 | 61.810 | 57.665 | R79 | R42 |
| F2DC v2 | 72.345 | 69.702 | 70.122 | 69.025 | R89 | R98 |
| **PG-DFC v3.2** | **73.992** | **72.400** | 73.068 | 69.552 | R98 | R92 |
| **PG-DFC v3.3** | **74.292** | **72.082** | 73.382 | 71.065 | R97 | R93 |

### Per-domain best round (PACS)

| Method | photo (s15/s333) | art (s15/s333) | cartoon (s15/s333) | sketch (s15/s333) |
|---|:--:|:--:|:--:|:--:|
| F2DC v2 | 68.56 / 70.36 | 58.09 / 55.15 | 81.20 / 75.85 | 81.53 / 77.45 |
| PG-DFC v3.2 | 69.46 / 66.17 | 62.50 / 63.48 | 83.12 / 77.78 | 80.89 / 82.17 |

---

### Table 1b-NEW. PACS — EXP-137 完整 R100 数据 (2026-04-29 回填, **4-bug fix 后**) ⭐

> ⚠️ **2026-04-29 重要更新**: EXP-137 在 EXP-135 之后修了 4 个 paper-grade bug
> (聚合白名单 / deterministic eval / 硬编码 num_classes / class_proto sync), 跑出新 R100 数据,
> **PG-DFC vanilla 路径在 4-bug fix 后 +3.40pp** (s333 last 66.66 → 70.06)。
> **Table 1b-OLD (EXP-135 PG-DFC 数据) 已不准, 主结果用本表**。
> Office 同理 (sc3 EXP-137 8 个 R100 已完成, 见 Table 2-NEW)。

#### Per-seed 数据 (sc3 EXP-137, 4-bug fix 后)

| Method                | seed | R@Best |   photo   |    art    |  cartoon  |  sketch   | **AVG_B** | **AVG_L** | gap  |
| --------------------- | :--: | :----: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--: |
| **PG-DFC vanilla** ⭐  | 333  |  R81   |   64.97   |   56.62   |   76.71   | **82.93** | **70.31** | **70.06** | 0.25 ⭐稳 |
| **PG-DFC vanilla**    |  15  |  TBD   |    TBD    |    TBD    |    TBD    |    TBD    | (v100 EXP-137 跑中) |    -      |   -  |
| **PG-DFC + DaA**      | 333  |  R99   | **73.05** |   60.54   |   72.22   |   61.53   |   66.84   |   66.84   | 0.00 ⭐稳 |
| **PG-DFC + DaA**      |  15  |  TBD   |    TBD    |    TBD    |    TBD    |    TBD    | (v100 EXP-137 跑中) |    -      |   -  |

#### 跟 Table 1b-OLD (4-bug fix 前 EXP-135) 同 seed 对比

| Method | seed | EXP-135 best | EXP-137 best | Δ best | EXP-135 last | EXP-137 last | Δ last |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| PG-DFC vanilla | 333 | 68.29 | **70.31** | **+2.02** ✅ | 66.66 | **70.06** | **+3.40** ✅ |
| PG-DFC +DaA | 333 | 69.28 | 66.84 | -2.44 | 66.88 | 66.84 | -0.04 (持平) |

→ **vanilla 4-bug fix 主要受益** (Δ last +3.40pp); +DaA 路径基本持平。
→ s=15 数据等 v100 EXP-137 跑完回填。

---

### Table 1b-OLD. PACS — EXP-135 完整 4-method × 2-seed R100 数据 (2026-04-28 重算)

> ⚠️ **2026-04-29 标注**: PG-DFC vanilla / +DaA 数据已被 EXP-137 4-bug fix 取代 (见 Table 1b-NEW)。
> **F2DC vanilla / F2DC+DaA 数据仍准** (4-bug 是 PG-DFC class_proto 特有 bug, F2DC 不受影响)。
> 历史保留供对照, **不用本表 PG-DFC 数据写论文**。

> 数据来源: sc6 + sc3 R100 完成的 EXP-135_diag_full 全套诊断 dump
> 全部带诊断 hook (round_*.npz light + best/final heavy dump)
>
> ⚠️ **2026-04-28 修正**: 之前的 per-seed 表格用 "per-domain 各自 max 取算术平均" 算 AVG_B,
> 这是错的 (相当于 4 个域取了不同 round, 等于 cherry-pick), 数字虚高 +0.8 ~ +2.2pp。
> 现已改为正确定义: **AVG_B = best round 那一轮的 4 域简单平均**。
> 数据从 EXP-135 的 8 个 R100 .log 重算 (npz dump 一致), 所有 R@Best 也是 1-indexed 的 npz round 编号。

#### Per-seed 完整数据 (真实 best-round-AVG)

| Method         | seed | R@Best |   photo   |    art    |  cartoon  |  sketch   | **AVG_B** | **AVG_L** | gap  |
| -------------- | :--: | :----: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--: |
| F2DC vanilla   |  15  |  R96   |   69.16   |   60.05   | **83.12** |   79.49   |   72.96   |   69.78   | 3.18 |
| F2DC vanilla   | 333  |  R96   |   67.96   |   58.82   |   77.14   | **82.42** |   71.59   |   68.96   | 2.62 |
| F2DC+DaA       |  15  |  R95   | **76.35** | **63.24** |   79.91   |   74.52   | **73.51** | **72.97** | 0.53 ⭐最稳 |
| F2DC+DaA       | 333  |  R90   |   71.86   |   55.88   |   74.79   |   62.80   |   66.33   |   65.26   | 1.07 |
| PG-DFC vanilla |  15  |  R86   |   68.56   |   59.80   |   82.48   |   80.64   |   72.87   |   71.36   | 1.51 |
| PG-DFC vanilla | 333  |  R76   |   66.17   |   53.92   |   73.08   |   80.00   |   68.29   |   66.66   | 1.63 |
| PG-DFC+DaA     |  15  |  R98   |   76.65   |   60.78   |   80.98   |   66.37   |   71.20   |   69.27   | 1.93 |
| PG-DFC+DaA     | 333  |  R88   |   71.26   |   59.80   |   75.85   |   70.19   | **69.28** |   66.88   | 2.39 |

#### 旧表(错算)虚高对照 — 警示用

> 这是上一版 obsidian 写的"伪 AVG_B"和真实 AVG_B 的差距,**不要再用旧数字**:

| Run                  | obsidian 旧值 (错) | 真实 best-round-AVG | 虚高 Δ  |
| -------------------- | :-------------: | :-------------: | :----: |
| F2DC vanilla   s=15  |     74.96       |     72.96       | +2.00  |
| F2DC vanilla   s=333 |     72.42       |     71.59       | +0.83  |
| F2DC+DaA       s=15  |     75.29       |     73.51       | +1.79  |
| F2DC+DaA       s=333 |     68.50       |     66.33       | +2.17  |
| PG-DFC vanilla s=15  |     74.43       |     72.87       | +1.56  |
| PG-DFC vanilla s=333 |     70.53       |     68.29       | +2.24  |
| PG-DFC+DaA     s=15  |     72.95       |     71.20       | +1.75  |
| PG-DFC+DaA     s=333 |     70.09       |     69.28       | +0.82  |

→ 错算定义: 4 个域 per-domain 各自 max → 算术平均 (相当于 4 域取了不同 round)
→ 正确定义: 找出 4 域简单平均最大的 round → 取该 round 的 per-domain

#### 2-seed mean 主表 (核心对比, 已用真实 best-round-AVG 重算)

| Method              | photo | art   | cartoon | sketch    |  **AVG_B**   |  **AVG_L**   |   gap   |
| ------------------- | :---: | :---: | :-----: | :-------: | :----------: | :----------: | :-----: |
| 🥇 **F2DC vanilla** | 68.56 | 59.44 |  80.13  | **80.95** | **72.27** ⭐  |  **69.37** ⭐  |  2.90   |
| 🥈 PG-DFC vanilla   | 67.37 | 56.86 |  77.78  |   80.32   |   70.58      |   69.01      |  1.57   |
| 🥉 PG-DFC+DaA       | 73.96 | 60.29 |  78.41  |   68.28   |   70.24      |   68.07      |  2.16   |
| ❌ F2DC+DaA          | **74.10** | **59.56** | 77.35 | 68.66 | 69.92        |   69.12      | **0.80** ⭐最稳 |

> 排序变化(对比错算版): F2DC+DaA 从 #3 跌到 #4; PG-DFC+DaA 从 #4 升到 #3。
> 真实排序: **F2DC vanilla > PG-DFC vanilla > PG-DFC+DaA > F2DC+DaA** (按 AVG_B)。

#### 关键 Δ (真实数据)

| 对比                              | AVG_B Δ        | AVG_L Δ      | 解读                                                  |
| ------------------------------- | :------------: | :----------: | ----------------------------------------------------- |
| F2DC+DaA vs F2DC vanilla        | **−2.35** ⚠️   | −0.25        | DaA 在 PACS 上 Best 维度负效应 (比错算版 −1.79 更糟)        |
| PG-DFC vanilla vs F2DC vanilla  | −1.69          | −0.36        | PG-DFC 单独 Best 负效应 (错算版 −1.21)                  |
| PG-DFC+DaA vs F2DC vanilla      | **−2.03** ⚠️   | **−1.29** ⚠️ | 双层叠加全负 (Best 没线性相加, Last 才负得明显)             |
| **PG-DFC+DaA vs F2DC+DaA** ⭐    | **+0.32** ✅    | −1.04        | **关键修正**: PG-DFC+DaA Best **微胜** F2DC+DaA (旧表说 −0.38) |

→ **重大结论修正**: 旧表说 "DaA 是主因, PG-DFC+DaA 仅多输 F2DC+DaA 0.38 Best",
   真实数据是 **PG-DFC+DaA Best 反过来微胜 F2DC+DaA +0.32pp** (Last −1.04 仍败)。
   即"PG-DFC 救 DSLR 但拖 PACS"的判决要重新校验: PACS 上 PG-DFC+DaA 不是系统性弱,
   只是 s=15 单 seed 上 sketch 域 (66.37 vs F2DC+DaA 74.52, −8.15pp) 拖了 mean。

#### 单 seed 看 sketch 高方差 (PG-DFC+DaA 的真正风险点)

| seed | F2DC+DaA sketch | PG-DFC+DaA sketch | Δ        |
| :--: | :-------------: | :---------------: | :------: |
| s=15  |     74.52       |       66.37       | **−8.15** ❌ |
| s=333 |     62.80       |       70.19       | **+7.39** ✅ |
| mean  |     68.66       |       68.28       | **−0.38** (打平) |

→ **sketch 是 PG-DFC+DaA 在 PACS 的高方差点**,两 seed 方向完全相反。
   3-seed 才能定结论 (建议补 s=2 R100 with 诊断)。

#### 重大发现 (DaA 在 PACS 上的 zero-sum 性质)

| Domain  | vanilla 2-seed mean (Last) | +DaA 2-seed mean (Last) | Δ Last  | DaA 角色   |
| :-----: | :-----------------: | :-----------------: | :-----: | :-------: |
| photo   |       68.12        |       74.26        | **+6.14** ⭐ | DaA 救    |
| art     |       53.31        |       60.91        | **+7.60** ⭐⭐ | DaA 救最多 |
| cartoon |       75.54        |       74.36        |   −1.18 | DaA 略拖  |
| sketch  |     **80.51**      |     **66.94**      | **−13.57** ⚠️ | **DaA 大杀** |

→ **DaA 在 PACS 是 zero-sum 拉平器**: 把 sketch 的 acc 转给 photo/art, 但 sketch 损失太大。
   (注: 此表用 Last 而非 Best, 因为 vanilla / +DaA 的 R@Best 不同, 用 Last 才是同 round 公平比)

#### 真实 client 样本数 (100 round mean, 已对齐 client_id)

| Client    | Domain  | sample share  | 估算样本数  |
| :-------: | :-----: | :-----------: | :--------: |
| 0/1       | photo   | 0.064 each    | 449 each   |
| 2/3/4     | art     | 0.079 each    | 552 each   |
| 5/6       | cartoon | 0.090 each    | 631 each   |
| **7/8/9** | sketch  | **0.151 each ⭐** | **1059 each** |

→ sketch 是 PACS 数据最多的 domain (单 client sample 也最多), vanilla F2DC 自然学得最好 (sketch acc 80.5%)。
   DaA 公式按 client sample 数拉平 (大 client 降权 −31%, 小 client 升权 +51%), sketch client 主力被砍。

#### DaA dispatch ratio (100 round mean, 按 client_id 对齐)

|    Client     | Domain  | sample share | DaA freq |   ratio    |
| :-----------: | :-----: | :----------: | :------: | :--------: |
|  photo (449)  |  photo  |    0.064     |  0.097   | **+51%** ⬆ |
|   art (552)   |   art   |    0.079     |  0.098   |    +24%    |
| cartoon (631) | cartoon |    0.090     |  0.099   |    +10%    |
| sketch (1059) | sketch  |    0.151     |  0.105   | **−31%** ⬇ |

→ sketch 总聚合权重: vanilla 45.4% → DaA 31.4% (−14pp), 主力被砍 → sketch acc 跌 −13.57

#### 数据质量说明

- ✅ 全部 sc 服务器 (sc6 6 个 + sc3 2 个), 跟 EXP-130 / EXP-131 主表服务器一致
- ✅ R100 final + best/final heavy dump 齐全, 可做完整 cold path 诊断
- ✅ V100 上同套 PACS 实验是双保险副本 (有跨服务器一致性问题, 不进主表)
- ✅ 4 method × 2 seed = 8/8 完成, 跟 office 主表对应 (office 是 5 method × 3 seed)
- ✅ 数据从 8 个 EXP-135 R100 .log 重算 + npz dump 双重验证一致 (parse `The N Communcation Accuracy: X` + `[a,b,c,d]` per-domain)

#### 跟论文 PACS 数据对比 (用真实 best-round-AVG 重算)

| 项                          | 论文     | 我们 (真实) | Δ            |
| --------------------------- | :----: | :----: | :----------: |
| F2DC w/o DaA AVG            | 75.33  | 72.27  | −3.06        |
| F2DC full AVG               | 76.47  | 69.92  | **−6.55** ⚠️ |
| F2DC vs +DaA Δ (DaA 增量)   | +1.14  | **−2.35** | **方向相反** ⚠️ |
| sketch w/o DaA (2-seed mean) | 80.13  | 80.95  | +0.82 (我们 vanilla sketch 略高) |
| sketch w/ DaA (2-seed mean)  | 82.11  | 68.66  | **−13.45** (我们 +DaA sketch 大跌) |

→ 论文 +DaA 涨 sketch +1.98 (80.13→82.11), 我们 +DaA 跌 sketch −12.29 (80.95→68.66)。
   **真正问题**: 复现 vanilla sketch 跟论文一致 (差 +0.82), 但 DaA 把 sketch 主力客户端权重砍掉 31% 之后 sketch 直接崩。
   论文用的 R/E/lr 配置或 DaA 实现细节可能跟我们有差。

---

## Table 2. Office-Caltech (4 domains × 10 classes, fixed: caltech:3/amazon:2/webcam:2/dslr:3)

| Method                                                                                                                            |  caltech  |  amazon   |  webcam   |   dslr    |   **AVG Best ↑**   | **AVG Last ↑** | Best Round (avg) |
| --------------------------------------------------------------------------------------------------------------------------------- | :-------: | :-------: | :-------: | :-------: | :----------------: | :------------: | :--------------: |
| FedAvg                                                                                                                            |   61.83   |   74.47   |   58.62   |   36.67   |       57.90        |     54.01      |      R62.0       |
| FedBN                                                                                                                             |   61.61   |   72.89   |   51.73   |   38.34   |       56.14        |     52.61      |      R82.5       |
| FedProx                                                                                                                           |   62.95   |   71.58   |   55.17   |   40.00   |       57.43        |     55.45      |      R79.0       |
| FedProto                                                                                                                          |   63.84   |   74.47   |   62.94   |   38.33   |       59.90        |     58.04      |      R86.0       |
| FPL                                                                                                                               |    TBD    |    TBD    |    TBD    |    TBD    |        TBD         |      TBD       |       TBD        |
| MOON                                                                                                                              |   58.70   |   70.53   |   47.42   |   33.33   |       52.49        |     50.11      |      R54.0       |
| **FDSE [CVPR'25]**                                                                                                                |   57.59   |   62.37   | **74.14** | **60.00** |    **63.52** ⭐     |     59.33      |       R69        |
| **F2DC [CVPR'26]** (release, 无 DaA)                                                                                               |   63.84   |   77.37   |   56.04   |   45.00   |       60.56        |     56.68      |       R99        |
| **F2DC + DaA (我们补)** ⭐                                                                                                            |   63.40   | **73.69** | **63.79** |   53.33   |     **63.55**      |     62.07      |      R84.0       |
| **PG-DFC + DaA (sc5 s=15)**                                                                                                       |    TBD    |    TBD    |    TBD    |    TBD    |     **63.80**      |     59.29      |       TBD        |
| **PG-DFC + DaA (V100 s=333)** ⭐                                                                                                   |   64.73   |   71.58   |   62.07   |   56.67   | **63.76** (有 diag) |     63.76      |       R100       |
| **PG-DFC + DaA 2-seed mean (sc5+V100)**                                                                                           |     —     |     —     |     —     |     —     |    **63.78** ⭐     |       —        |        —         |
| **Δ PG-DFC+DaA vs vanilla PG-DFC**                                                                                                |     —     |     —     |     —     |     —     |    **+2.53pp**     |       —        |        —         |
| **Δ PG-DFC+DaA vs FDSE 63.52**                                                                                                    |     —     |     —     |     —     |     —     |   **+0.26pp** ⭐    |       —        |        —         |
|                                                                                                                                   |           |           |           |           |                    |                |                  |
| **=== Table 2-NEW: EXP-137 sc3 完整 R100 (2026-04-29, 4-bug fix 后, 替代 EXP-135 V100 PG-DFC 数据) ⭐ ===**                                  |           |           |           |           |                    |                |                  |
| PG-DFC vanilla (sc3 EXP-137 s=2)                                                                                                  | **65.62** | **74.21** |   53.45   |   60.00   |     **63.32**      |     59.33      |       R93        |
| PG-DFC vanilla (sc3 EXP-137 s=15)                                                                                                 |   64.29   | **77.37** |   51.72   |   40.00   |       58.35        |     54.15      |       R75        |
| PG-DFC vanilla (sc3 EXP-137 s=333)                                                                                                | **68.30** | **77.89** |   58.62   |   50.00   |       63.70        |     55.60      |       R88        |
| **PG-DFC vanilla sc3 EXP-137 3-seed mean** ⭐                                                                                       |   66.07   | **76.49** |   54.60   |   50.00   |     **61.79**      |     56.36      |      R85.3       |
| PG-DFC + DaA (sc3 EXP-137 s=2)                                                                                                    | **67.41** | **70.00** |   62.07   | **73.33** |    **68.20** ⭐     |   **66.82**    |       R97        |
| PG-DFC + DaA (sc3 EXP-137 s=15)                                                                                                   |   58.04   |   66.84   | **63.79** |   56.67   |       61.34        |     55.91      |       R56        |
| PG-DFC + DaA (sc3 EXP-137 s=333)                                                                                                  |   64.29   | **71.58** | **65.52** |   56.67   |       64.52        |     58.33      |       R91        |
| **PG-DFC + DaA sc3 EXP-137 3-seed mean** ⭐⭐                                                                                        |   63.25   | **69.47** |   63.79   | **62.22** |   **64.69** ⭐     |   **60.35**    |      R81.3       |
| **Δ PG-DFC+DaA vs vanilla (EXP-137 sc3 公平比)**                                                                                     |   -2.82   |   -7.02   |   +9.19   | **+12.22** |   **+2.90pp** ⭐    | **+3.99pp** ⭐  |     -4 round     |
| **Δ EXP-137 vs EXP-135 V100 (vanilla 3-seed mean)**                                                                                |   +2.46   |   +3.07   |   +2.88   |   +3.33   |    **+2.93pp** ✅   |   **+0.66pp**  |        -         |
| **Δ EXP-137 vs EXP-135 V100 (+DaA 3-seed mean)**                                                                                   |   +0.60   |   -0.35   |   +3.45   |   +3.33   |    **+1.76pp** ✅   |   -0.30pp      |        -         |
|                                                                                                                                   |           |           |           |           |                    |                |                  |
| **=== Table 2-OLD: EXP-135 V100 完整 3-seed 数据 (2026-04-29 补, ⚠️ PG-DFC 部分已被 EXP-137 取代, F2DC 部分仍准) ===**                              |           |           |           |           |                    |                |                  |
| F2DC vanilla (V100 s=2)                                                                                                           |   68.75   |   77.37   |   51.72   |   53.33   |       62.79        |     56.31      |       R95        |
| F2DC vanilla (V100 s=15)                                                                                                          |   63.39   |   74.74   |   58.62   |   46.67   |       60.85        |     53.41      |       R65        |
| **F2DC vanilla V100 2-seed mean**                                                                                                 |   66.07   |   76.06   |   55.17   |   50.00   |     **61.82**      |     54.86      |      R80.0       |
| F2DC+DaA (V100 s=2)                                                                                                               |   61.61   |   68.95   |   56.90   |   63.33   |       62.70        |     61.29      |       R87        |
| F2DC+DaA (V100 s=15)                                                                                                              |   62.05   |   64.21   |   63.79   |   56.67   |       61.68        |     55.37      |       R77        |
| F2DC+DaA (V100 s=333)                                                                                                             |   66.07   |   77.37   |   62.07   |   46.67   |       63.05        |     63.05      |       R100       |
| **F2DC+DaA V100 3-seed mean**                                                                                                     |   63.24   |   70.18   |   60.92   |   55.56   |     **62.47**      |     59.90      |      R88.0       |
| ⚠️ PG-DFC vanilla (V100 s=2) **已废弃, 见 Table 2-NEW**                                                                            |   66.96   |   77.37   |   50.00   |   50.00   |       61.08        |     59.53      |       R88        |
| ⚠️ PG-DFC vanilla (V100 s=15) **已废弃**                                                                                           |   60.27   |   69.47   |   53.45   |   43.33   |       56.63        |     51.86      |       R86        |
| ⚠️ **PG-DFC vanilla V100 2-seed mean 已废弃**                                                                                       |   63.61   |   73.42   |   51.72   |   46.67   |     ~~58.86~~      |    ~~55.70~~   |      R87.0       |
| ⚠️ PG-DFC+DaA (V100 s=2) **已废弃**                                                                                                |   61.61   |   68.42   |   58.62   | **70.00** |    ~~64.66~~      |     60.62      |       R93        |
| ⚠️ PG-DFC+DaA (V100 s=15) **已废弃**                                                                                               |   61.61   |   69.47   |   60.34   |   50.00   |       60.35        |     57.58      |       R85        |
| ⚠️ PG-DFC+DaA (V100 s=333) **已废弃**                                                                                              |   64.73   |   71.58   |   62.07   |   56.67   |       63.76        |     63.76      |       R100       |
| ⚠️ **PG-DFC+DaA V100 3-seed mean 已废弃**                                                                                          |   62.65   |   69.82   |   60.34   | **58.89** |     ~~62.93~~      |   ~~60.65~~    |      R92.7       |
| **Δ PG-DFC+DaA vs F2DC+DaA (V100 公平比)**                                                                                           |   -0.59   |   -0.36   |   -0.58   | **+3.33** |   **+0.46pp** ⭐    | **+0.75pp** ⭐  |    +4.7 round    |
| **Δ PG-DFC+DaA vs F2DC vanilla (V100 公平比)**                                                                                       |   -3.42   |   -6.24   |   +5.17   | **+8.89** |   **+1.11pp** ⭐    | **+5.79pp** ⭐  |   +12.7 round    |
| **Δ PG-DFC vanilla vs F2DC vanilla (V100, PG-DFC 单独)**                                                                            |   -2.46   |   -2.64   |   -3.45   |   -3.33   |   **-2.96pp** ⚠️   |    +0.84pp     |    +7.0 round    |
|                                                                                                                                   |           |           |           |           |                    |                |                  |
| **数据质量 caveat**: V100 数据跟 sc 服务器有 ±1pp 一致性差异 (跨 GPU 硬件 + cuda 版本). 主表用 sc 数据为准, V100 数据是 3-seed 完整诊断版 (有 cold path dump), 可做诊断分析. |           |           |           |           |                    |                |                  |
| **PG-DFC v3.2 (Ours)**                                                                                                            | **65.63** |   76.05   |   50.00   |   53.34   |       61.25        |     56.05      |      R92.5       |
| **PG-DFC v3.3 (A+B)**                                                                                                             |   63.17   | **78.42** |   56.04   |   48.33   |       61.49        |   **59.09**    |       R90        |
| **Δ v3.2 vs FedAvg**                                                                                                              |   +3.80   |   +1.58   |   -8.62   |  +16.67   |       +3.35        |     +2.04      |   +30.5 round    |
| **Δ v3.2 vs F2DC**                                                                                                                |   +1.79   |   -1.32   |   -6.04   |   +8.34   |       +0.69        |     -0.63      |    -6.5 round    |
| **Δ v3.3 vs F2DC**                                                                                                                |   -0.67   |   +1.05   |     0     |   +3.33   |       +0.93        |     +2.41      |     -9 round     |

⚠️ **重大发现**: FDSE (CVPR'25) 在 Office 上反而最强 (63.52),比 F2DC (60.56) +2.96pp。我们之前低估了 FDSE。

### Per-seed 详细数据 (Office)

| Method               |        s=15 best         | s=333 best |      s=15 last       | s=333 last | s=15 best round | s=333 best round |
| -------------------- | :----------------------: | :--------: | :------------------: | :--------: | :-------------: | :--------------: |
| FedAvg               |          54.485          |   61.312   |        48.830        |   59.188   |       R42       |       R82        |
| FedBN                |          54.573          |   57.707   |        49.420        |   55.793   |       R91       |       R74        |
| FedProx              |          55.368          |   59.485   |        52.772        |   58.130   |       R83       |       R75        |
| FedProto             |          59.175          |   60.615   |        58.000        |   58.087   |       R78       |       R94        |
| MOON                 |          50.067          |   54.920   |        46.205        |   54.023   |       R26       |       R82        |
| F2DC v2              |          60.80           |   60.323   |        54.618        |   58.735   |       R98       |       R98        |
| **F2DC + DaA** ⭐     |        **63.925**        | **63.175** |        63.925        |   60.220   |      R100       |       R68        |
| FDSE v2              |          60.99           | **66.055** |        58.062        |    60.6    |       R81       |       R55        |
| PG-DFC v3.2          |          61.613          |   60.892   |        54.415        |   57.675   |       R87       |       R98        |
| PG-DFC v3.3 (s=2 加跑) | 59.848 / s=2: **62.475** |   63.13    | 56.125 / s=2: 58.332 |   62.045   | R86 / s=2: R93  |       R94        |

### Per-domain best round (Office)

| Method | caltech (s15/s333) | amazon (s15/s333) | webcam (s15/s333) | dslr (s15/s333) |
|---|:--:|:--:|:--:|:--:|
| F2DC v2 | 63.84 / 63.84 | 75.79 / 78.95 | 56.90 / 55.17 | 46.67 / 43.33 |
| FDSE v2 | 58.04 / 57.14 | 56.84 / 67.89 | 72.41 / 75.86 | 56.67 / 63.33 |
| PG-DFC v3.2 | 64.29 / 66.96 | 75.26 / 76.84 | 56.90 / 43.10 | 50.00 / 56.67 |
| PG-DFC v3.3 | 60.27 / 66.07 | 75.79 / 81.05 | 50.00 / 62.07 | 53.33 / 43.33 |

---

## Table 3. Digits (4 domains × 10 classes, fixed: mnist:3/usps:6/svhn:6/syn:5)

| Method                 |   mnist   |   usps    |   svhn    |    syn    | **AVG Best ↑** | **AVG Last ↑** | Best Round (avg) |
| ---------------------- | :-------: | :-------: | :-------: | :-------: | :------------: | :------------: | :--------------: |
| FedAvg                 |   96.00   |   91.58   |   87.48   |   92.38   |     91.86      |     91.63      |      R89.5       |
| FedBN                  |   95.58   |   90.19   |   86.12   |   91.34   |     90.81      |     90.57      |      R82.5       |
| FedProx                |   96.30   |   91.18   |   87.60   |   92.69   |     91.94      |     91.82      |      R95.0       |
| FedProto               | **97.08** | **92.13** |   87.84   |   93.08   |   **92.53**    |     92.53      |      R100.0      |
| FPL                    |    TBD    |    TBD    |    TBD    |    TBD    |      TBD       |      TBD       |       TBD        |
| MOON                   |   95.73   |   91.61   |   87.30   |   91.73   |     91.59      |     90.37      |      R87.5       |
| FDSE                   |   92.34   |   91.38   |   74.41   |   88.50   |     86.66      |     84.61      |      R74.0       |
| **F2DC**               | **97.34** |   92.46   |   90.18   | **94.36** |  **93.59** ⭐   |   **93.40**    |      R94.5       |
| **PG-DFC v3.2 (Ours)** |   97.38   | **91.13** | **90.35** |   94.37   |   **93.30**    |     92.99      |      R90.0       |
| **Δ F2DC vs FedAvg**   |   +1.34   |   +0.88   |   +2.70   |   +1.98   |     +1.73      |     +1.77      |     +5 round     |
| **Δ PG-DFC vs F2DC**   |   +0.04   |   -1.33   |   +0.17   |   +0.01   |     -0.29      |     -0.41      |    -4.5 round    |
| **Δ PG-DFC vs FedAvg** |   +1.38   |   -0.45   |   +2.87   |   +1.99   |     +1.44      |     +1.36      |    +0.5 round    |

### Per-seed (Digits)

| Method          | s=15 best  | s=333 best | s=15 last | s=333 last | s=15 best round | s=333 best round |
| --------------- | :--------: | :--------: | :-------: | :--------: | :-------------: | :--------------: |
| FedAvg          |   92.100   |   91.625   |  91.825   |   91.442   |       R98       |       R81        |
| FedBN           |   91.185   |   90.432   |  90.855   |   90.278   |       R87       |       R78        |
| FedProx         |   92.165   |   91.722   |  92.088   |   91.555   |       R92       |       R98        |
| FedProto        |   92.805   |   92.255   |  92.805   |   92.255   |      R100       |       R100       |
| MOON            |   92.227   |   90.958   |  91.342   |   89.388   |       R85       |       R90        |
| F2DC v2         |   93.742   |   93.428   |  93.362   |   93.428   |       R88       |       R99        |
| FDSE v2         |   86.732   |   86.585   |  84.905   |   84.322   |       R75       |       R71        |
| **PG-DFC v3.2** | **93.190** | **93.417** |  92.832   |   93.150   |       R82       |       R98        |

---

## Wave 1 PACS+Office seed=2 (rand allocation, 不在主表 — setup 不一致)

| Run | Config | PACS best | Office best | 备注 |
|---|---|:--:|:--:|---|
| R0 sanity_pw0 (PG-DFC backbone, proto=0) | 等价 vanilla | 68.02 | 59.44 | sanity 验证 |
| R1 full_v32 (τ=0.3) | PG-DFC default 早期 | 66.55 | 65.26 | τ=0.3 弱 |
| R2 (τ=0.5) | PG-DFC best τ | 67.95 | **66.87** ⭐ | Office Wave 1 best |
| R4 (β=0) | NV4 ablation | - | 62.57 | server EMA 必要 |

---

## Figures

### Fig 1. Per-Domain Δ Bar Chart

![[figs/fig1_per_domain_comparison.png]]

- 2x2 subplot: PACS (top) / Office (bottom),左 bar 对比 + 右 Δ 条
- 关键发现:**art +6.37pp / dslr +8.34pp** (baseline-weak domains 增益最大)

### Fig 2. Convergence Curves (3 datasets)

![[figs/fig_convergence_3datasets.png]]

- 1x3 subplot: PACS / Office / Digits
- F2DC 灰,FDSE 紫,PG-DFC v3.2 红,PG-DFC v3.3 蓝(只 Office 完整)
- 关键发现:
  - PACS 上 PG-DFC v3.2 全程领先 F2DC
  - **Office 上 FDSE 反而最高**(我们之前低估)
  - Digits 上 F2DC 远超 FDSE,PG-DFC 还在跑

### Fig 3. Convergence — All Baselines (跟 F2DC paper Fig 5 风格一致) ⭐ 主图

![[figs/convergence_office_pacs.png]]

- 1x2 subplot: Office-Caltech (Left) + PACS (Right)
- 9 algorithm: FedAvg(蓝)/FedBN(浅蓝)/FedProx(绿)/FedProto(黄)/MOON(粉)/FDSE(紫)/F2DC(灰)/**PG-DFC v3.2(红)** / **PG-DFC v3.3(蓝)**
- 数据: 2-seed mean (s=15+s=333), R100 完整 trajectory
- 关键观察:
  - **Office (Left)**: FDSE 紫线 ~63% 仍是冠军, PG-DFC v3.2/v3.3 红蓝 ~58-62%, F2DC 灰 ~57%, 其它 baseline ~52-58% 紧密
  - **PACS (Right)**: PG-DFC v3.2 红线 + v3.3 蓝线 ~73% **明显领先**, F2DC 灰 ~70%, FedAvg 蓝 ~69%, FedBN 浅蓝 ~62% 最低
  - **暂缺**: PACS FedProto / FedProx / MOON / FDSE 4 条线 (跑中或不完整, 等 sc5 重启完成后重画)

### Fig 4. Office 4 Domain 视觉差异

![[figs/office_domain_compare.png]]

- 4 个 domain × 4 类样本对比, 直观展示 Office-Caltech10 上的域间差异

---

## 真实 Paper-Grade 结论

### 主 contribution(can write now)
| Dataset | Δ vs F2DC | Δ vs FedAvg | 备注 |
|---|:--:|:--:|---|
| **PACS** | **+2.18pp** ✓ | **+3.98pp** ✓ | art +6.37 main story |
| Office | +0.69pp (v3.2) / +0.93pp (v3.3) | +3.35pp (v3.2) | 边缘 vs F2DC,但显著优于 FedAvg/MOON |
| Digits | -0.29pp | +1.44pp | 几乎打平 F2DC,显著优于其它 baseline |

### 关键 insight(per-domain analysis)
- **baseline-weak domain 受益最大**:art +6.37pp,dslr +8.34pp
- **outlier domain 反受拖累**:Office webcam -6.03pp(跟其他 client 视觉不同步)
- v3.3 (A+B) 部分修复 outlier 问题(office mean +0.24pp vs v3.2)

### 收敛速度
- PACS:PG-DFC 比 F2DC 提前 1-3 round
- Office:PG-DFC v3.3 提前 8 round

### Limitation(诚实写)
- FDSE 在 Office 上反而 +3pp 强于 F2DC 跟 PG-DFC,**未来要研究 PG-DFC + FDSE-style 层分解**
- v3.2 在 outlier domain (webcam) 上拖后腿,v3.3 部分修复但还可优化

---

## 待回填 (跑完后填)

- [x] PACS F2DC + PG-DFC v3.2 (R100 完成)
- [x] Office F2DC + FDSE + PG-DFC v3.2 + v3.3 (R100 完成)
- [x] Digits F2DC + FDSE (R100 完成)
- [x] **EXP-130 sc4_v2: FedAvg PACS/Office/Digits + MOON Office/Digits 已回填**
- [ ] **Digits PG-DFC v3.2** (跑中 R20,~30 min 后完成)
- [ ] PACS v3.3 (跑中 R37)
- [ ] **MOON PACS** (sc4_v2 只到 R34 + 缺 s333,需补 R100×2)
- [ ] FDSE PACS (sc3_v2 R35+R30 没跑完,需补 R100×2)
- [ ] FedBN / FedProto / FPL 3 个 baselines (还没跑)

## 后续 priority

1. **等 Digits PG-DFC v3.2 完成** → 填主表完整 3 dataset
2. **补 MOON PACS + FDSE PACS** (PACS 缺基线,但 FedAvg+F2DC 已可对比)
3. **跑 FedBN / FedProto / FPL** on PACS/Office/Digits
4. **画 per-class confusion matrix** (paper Fig 6 风格,展示 art 增益来自哪些类)
