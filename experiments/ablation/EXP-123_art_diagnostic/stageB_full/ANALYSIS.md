# EXP-123 Stage B 完整分析 (8/9 runs + orth_s333 mid-run)

**生成时间**: 2026-04-24 ~10:45

## 数据状态

| Method | Seeds | 状态 |
|---|---|---|
| **FedBN** | 2, 15, 333 | ✅ 全 3 seeds R=200 完成 |
| **orth_only** | 2, 15 | ✅ 2/3 完成 |
| **orth_only** | **333** | 🟡 Round 129/200 (65%), 预计 13:30 完成 |
| **FDSE** | 2, 15, 333 | ✅ 全 3 seeds R=200 完成 |

**以下所有数据: orth_only 仅 2 seeds mean**, 完整 3-seed 待 orth_s333 完成补齐。

---

## 1. 3-seed mean — 整体 accuracy

| Method | AVG Best | AVG Last | Art | Cartoon | Photo | Sketch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FDSE** (3 seeds) | **81.54** | — | 64.71 | 85.18 | **86.83** | 89.46 |
| **orth_only** (🟡 2 seeds) | 79.95 | — | 62.50 | 86.96 | 82.34 | 88.01 |
| **FedBN** (3 seeds) | 79.23 | — | 62.25 | 85.90 | 80.24 | 88.52 |

### 排序 + Δ

- **FDSE > orth > FedBN** (差 2.31 pp 和 0.72 pp)
- FDSE 主要靠 Photo (+6.6pp vs FedBN), Art 仅 +2.5pp
- orth 比 FedBN 好 0.72pp, 分布在 Art +0.25, Cartoon +1.06, Photo +2.10, Sketch -0.51

---

## 2. Per-class 7×4 Matrix (3-seed mean, at best round)

### FedBN

| Domain | dog | elephant | giraffe | guitar | horse | house | person | 行均 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Art | 60.98 | 66.58 | 75.62 | **37.25** ⚠️ | **45.93** ⚠️ | 78.28 | 60.31 | 60.71 |
| Cartoon | 76.45 | 92.81 | 89.19 | 93.52 | 79.86 | 88.76 | 84.02 | 86.37 |
| Photo | 64.08 | 74.69 | 78.95 | 87.62 | **48.07** ⚠️ | 93.68 | 99.17 | 78.04 |
| Sketch | 77.44 | 91.56 | 88.85 | 98.78 | 87.83 | 97.78 | 97.22 | 91.35 |
| 列均 | 69.74 | 81.41 | 83.15 | 79.29 | **65.42** | 89.63 | 85.18 | — |

### orth_only (2 seeds)

| Domain | dog | elephant | giraffe | guitar | horse | house | person | 行均 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Art | 57.87 | 68.01 | 78.33 | **43.94** | **50.85** | 89.63 | 53.49 | 63.16 |
| Cartoon | 77.42 | 90.74 | 87.63 | 97.06 | 81.70 | 91.00 | 89.91 | 87.92 |
| Photo | 62.53 | 77.78 | 81.58 | 90.99 | **55.86** | 92.36 | 98.75 | 79.98 |
| Sketch | 76.03 | 89.12 | 91.62 | 97.37 | 87.66 | 100.00 | 95.83 | 91.09 |
| 列均 | 68.46 | 81.41 | 84.79 | 82.34 | 69.02 | 93.25 | 84.50 | — |

### FDSE

| Domain | dog | elephant | giraffe | guitar | horse | house | person | 行均 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Art | 47.62 | 59.02 | 66.73 | **61.57** | **61.45** | 83.03 | 69.42 | 64.12 |
| Cartoon | 73.16 | 94.15 | 91.29 | 88.56 | 76.27 | 94.07 | 82.05 | 85.65 |
| Photo | 78.70 | 86.42 | 80.70 | 83.28 | **65.57** | 98.61 | 100.00 | 84.75 |
| Sketch | 80.80 | 89.35 | 92.19 | 98.39 | 88.29 | 100.00 | 95.69 | 92.10 |
| 列均 | 70.07 | 82.24 | 82.73 | 82.95 | 72.90 | 93.93 | 86.79 | — |

---

## 3. 最 Hard Cells (FedBN 视角)

### Top 10 最难 cells (FedBN accuracy 排序)

| Rank | Cell | FedBN | orth | **FDSE** | FDSE Δ | orth Δ |
|:-:|---|:-:|:-:|:-:|:-:|:-:|
| 1 | (Art, **guitar**) | 37.25 | 43.94 | **61.57** | **+24.33** ⭐ | +6.69 |
| 2 | (Art, **horse**) | 45.93 | 50.85 | **61.45** | **+15.52** ⭐ | +4.92 |
| 3 | (Photo, **horse**) | 48.07 | 55.86 | **65.57** | **+17.50** ⭐ | +7.79 |
| 4 | (Art, person) | 60.31 | 53.49 | 69.42 | +9.11 | -6.82 |
| 5 | (Art, dog) | 60.98 | 57.87 | 47.62 | **-13.36** | -3.11 |
| 6 | (Photo, dog) | 64.08 | 62.53 | 78.70 | +14.62 | -1.55 |
| 7 | (Art, elephant) | 66.58 | 68.01 | 59.02 | **-7.56** | +1.43 |
| 8 | (Photo, elephant) | 74.69 | 77.78 | 86.42 | +11.73 | +3.09 |
| 9 | (Art, giraffe) | 75.62 | 78.33 | 66.73 | **-8.89** | +2.71 |
| 10 | (Cartoon, dog) | 76.45 | 77.42 | 73.16 | -3.29 | +0.97 |

### ⭐ 关键 3 个 cells 对 AVG Best 差距解释

| Cell | FDSE > FedBN (pp) | 贡献到 AVG Best |
|---|:-:|:-:|
| (Art, guitar) | +24.33 | 1/28 × 24.33 = +0.87 |
| (Art, horse) | +15.52 | +0.55 |
| (Photo, horse) | +17.50 | +0.63 |
| **合计** | — | **+2.05 pp** |

FDSE 整体赢 FedBN **+2.31 pp**, 其中 **+2.05 pp 来自这 3 个 cell**! 即**FDSE 的优势 ~90% 集中在这 3 个 (domain, class) cells**.

---

## 4. FDSE 的弱点 cells (反超点)

| Cell | FedBN | FDSE | FedBN 赢 |
|---|:-:|:-:|:-:|
| (Art, dog) | 60.98 | 47.62 | **+13.36** |
| (Art, giraffe) | 75.62 | 66.73 | +8.89 |
| (Art, elephant) | 66.58 | 59.02 | +7.56 |

**Art 里 dog/elephant/giraffe 3 个 animal class FDSE 反而差**, 但 guitar/horse/person 涨 > 9pp.

### Art 域整体为什么只 +3.4pp (64.12 vs 60.71)?
- 涨的 (guitar +24, horse +15, person +9, house +5, giraffe/elephant/dog 输 ~7-13)
- 每 class 相抵, 净涨 3.4pp

**结论: FDSE 在 Art 域并非 uniform 好**, 而是"换了 trade-off".

---

## 5. Confidence / Calibration 诊断

### Per-domain ECE (3-seed mean)

| Method | Art ECE | Cartoon | Photo | Sketch |
|---|:-:|:-:|:-:|:-:|
| FedBN | **0.190** | 0.051 | 0.110 | 0.045 |
| orth_only | **0.180** | 0.079 | 0.095 | 0.065 |
| FDSE | **0.177** | 0.062 | **0.062** | 0.053 |

### Per-domain over_confident_wrong_ratio (错但 conf>0.8)

| Method | Art | Cartoon | Photo | Sketch |
|---|:-:|:-:|:-:|:-:|
| FedBN | **14.7%** | 4.7% | 8.0% | 4.4% |
| orth_only | **13.0%** | 6.6% | 8.1% | 5.7% |
| FDSE | **12.6%** | 5.8% | 5.0% | 5.6% |

### 观察

1. **Art ECE ~0.18 跨所有方法**: Art calibration 是 PACS 固有难点, FDSE 也未解决
2. **Photo ECE FDSE 0.062 vs FedBN 0.110**: FDSE 校准 Photo 降了近一半, 解释 Photo +6.6pp 的一部分
3. **Art 13-15% sample 过度自信错**: 表示 orth/FedBN 对 Art 常"很肯定地错了"

---

## 6. R_best (best round) 分布

| Method | s=2 R_best | s=15 R_best | s=333 R_best |
|---|:-:|:-:|:-:|
| FedBN | 133 | 87 | **37** ⚠️ |
| orth_only | 168 | 130 | 🟡 (R=129 mid-run) |
| FDSE | **188** | 119 | **182** |

**特征**:
- **FDSE 普遍 R=180+ 才达峰** (慢热型), 小 R 不够
- **FedBN s=333 R_best=37**: 训练 10h 居然 R=37 时就达峰然后下滑了 100+ rounds → 严重 overfit 或 seed unlucky
- **orth_only 中等**: R=130-168

---

## 7. 方向判断 (综合)

| 方向 | 证据强度 | 判断 |
|---|:-:|:-:|
| **I stage-aware dynamic** | 🟡 中 | FedBN s=333 R=37 早熟 + FDSE 晚熟. 动态 schedule 可能解决 Unlucky seed 不好.但非核心 |
| **II calibration-aware** | 🟢 Photo 部分有效 | FDSE Photo ECE 0.062 vs FedBN 0.110 成立; 但 Art ECE 跨方法 ~0.18 没解决 |
| **III per-cell hardness** | 🟢🟢 **最强证据** | 3 hard cells (Art-guitar, Art-horse, Photo-horse) 贡献 FDSE +2.05/+2.31 的 90% |
| **IV local protocol reform** | ⚪ 未验证 | 无 gradient alignment 数据 |

---

## 8. 推荐方法方向

### 方向 III 细化 (最具体, 证据最强):

**"Style-confused animal/instrument hardness hook"**:

观察: FDSE 优势 cells 全是 `(非 Photo, animal/instrument)` — painting / real-photo 里的 horse/guitar. 这些 object 在 photo 里 "具体", 在 Art painting 里 "风格化/抽象", FedBN/orth 混淆了。

**方法构想**:
1. **Per-cell hardness tracker**: 每 round 算 `(client_domain, class)` 准确率, 识别 hardness < 60% 的 cell 为 "hard target"
2. **Cross-domain class bank**: 让 Art client 的 guitar/horse 样本 anchor 到 Cartoon/Sketch/Photo 的同类原型 (不需要 Art 自己有好的 guitar 表示, 借用其他 domain)
3. **Per-cell Loss re-weight**: 对 hard cells 的 class 样本上 α 倍 CE loss weight

### 预期效果:

| Cell | orth 现在 | 若 SAR 到 FDSE 水平 | 贡献 |
|---|:-:|:-:|:-:|
| (Art, guitar) | 43.94 | 55-60 | +0.4-0.6pp |
| (Art, horse) | 50.85 | 55-60 | +0.2-0.3pp |
| (Photo, horse) | 55.86 | 60-65 | +0.2-0.3pp |
| **合计 AVG Best** | — | — | **+0.8-1.2pp** |

若成立, orth 80 + 1.0 = 81, 与 FDSE 81.54 持平或略输。

### 若还要涨:
- 同时加 方向 II (per-domain temperature calibration) for Photo: 预期 +0.3-0.5pp
- Total 最高 81.5-82, 或接近/胜 FDSE

---

## 9. 待完成 (当 orth_s333 跑完, ~13:30)

- [ ] 更新 orth 为 3 seeds mean (预期小幅变化)
- [ ] 生成完整 3-seed std bar (现在只有 mean)
- [ ] 可选: 置信度 histogram 画图 (confidence_hist_dist 有数据, 每 50 round 一次)
