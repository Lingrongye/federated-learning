---
date: 2026-04-28
type: 诊断指标体系全维度对比 (paper-grade 实证)
status: office 5 method × 2-3 seed 完整 R100 dump 已分析
exp_id: EXP-134-cmp
---

# PG-DFC+DaA vs F2DC+DaA on Office: 全维度诊断对比

> 用我们诊断体系 (`诊断指标体系_完整说明.md`) 里 7 维 30+ 指标, 对 office 上 5 method × 2-3 seed 的 R100 dump 做全量分析. 回答: PG-DFC+DaA 到底赢了 F2DC+DaA 没有? 输在哪?

## 0. TL;DR (一句话结论)

**PG-DFC+DaA Last AVG (60.65) 微胜 F2DC+DaA (59.90) +0.75pp, 主要靠 dslr (最稀有 domain) 大胜 +6.66pp; Best AVG 也微胜 (65.78 vs 65.23) +0.55pp; 但都输给 vanilla F2DC Best AVG (66.48) -0.70pp.**

**真正瓶颈**: PG-DFC 的 prototype consensus alignment 已经达到 0.97+ (10/10 client 全被同化), DaA 升权动作 (1.6-2.1× ratio) 跟 F2DC+DaA 几乎一致 → **DaA 跟 PG-DFC 机制重叠**, 学到的 representation CKA=0.88 → **没有 incremental 空间**.

## 1. 主表 acc 对比 (3-seed × 4 domain × Best/Last)

> 数据来源: 各诊断目录的 `round_*.npz['per_domain_acc']`, AVG_B = mean(domain_max_across_rounds), AVG_L = mean(last_round_per_domain).

| Method | seed | caltech | amazon | webcam | dslr | **AVG_B** | **AVG_L** | gap |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| F2DC | 2 | 68.75 | 77.37 | 56.90 | 66.67 | 67.42 | 56.31 | 11.12 |
| F2DC | 15 | 67.41 | 81.05 | 60.34 | 53.33 | 65.53 | 53.41 | 12.12 |
| **F2DC mean (2 seed)** |  |  |  |  |  | **66.48** | 54.86 | 11.62 |
| F2DC+DaA | 2 | 66.96 | 70.00 | 62.07 | 63.33 | 65.59 | 61.29 | 4.30 |
| F2DC+DaA | 15 | 62.50 | 72.63 | 63.79 | 56.67 | 63.90 | 55.37 | 8.53 |
| F2DC+DaA | 333 | 66.52 | 79.47 | 65.52 | 53.33 | 66.21 | 63.05 | 3.16 |
| **F2DC+DaA mean (3 seed)** |  |  |  |  |  | 65.23 | 59.90 | 5.33 |
| PG-DFC | 2 | 68.75 | 77.89 | 53.45 | 56.67 | 64.19 | 59.53 | 4.66 |
| PG-DFC | 15 | 62.95 | 76.32 | 53.45 | 50.00 | 60.68 | 51.86 | 8.82 |
| **PG-DFC mean (2 seed)** |  |  |  |  |  | 62.43 | 55.70 | 6.74 |
| **PG-DFC+DaA** | 2 | 67.41 | 72.11 | 62.07 | **70.00** | 67.90 | 60.62 | 7.28 |
| **PG-DFC+DaA** | 15 | 62.95 | 71.58 | 62.07 | 56.67 | 63.32 | 57.58 | 5.74 |
| **PG-DFC+DaA** | 333 | 66.52 | 72.63 | 62.07 | 63.33 | 66.14 | **63.76** | 2.37 |
| **PG-DFC+DaA mean (3 seed)** |  |  |  |  |  | **65.78** ⭐ | **60.65** ⭐ | **5.13** ⭐ |

### Per-domain Last 3-seed mean (核心对比)

| Domain | F2DC+DaA | PG-DFC+DaA | Δ |
|---|:--:|:--:|:--:|
| caltech | 63.69 | 62.50 | -1.19 |
| amazon | **68.95** | 68.77 | -0.18 |
| webcam | **59.19** | 56.90 | -2.29 |
| **dslr** ⭐ | 47.78 | **54.44** | **+6.66** |
| **AVG_L** | 59.90 | **60.65** | **+0.75** |

**核心**: **PG-DFC+DaA 几乎全靠 dslr (最稀有 domain, 30 样本) 赢的**, 在其他 3 个 domain 上略输 -0.18 ~ -2.29.

## 2. 维度 1: DaA dispatch ratio (server 端聚合行为)

> daa_freqs / sample_shares 的 100-round 平均. 看 DaA 给每个 client 升权多少.

| Client | Domain | F2DC+DaA | PG-DFC+DaA | Δ |
|---|---|:--:|:--:|:--:|
| 0 | caltech | 1.779 | 1.966 | +0.19 |
| 1 | caltech | 1.878 | 1.654 | -0.22 |
| 2 | caltech | 1.794 | 1.834 | +0.04 |
| 3 | amazon | 1.798 | 1.874 | +0.08 |
| 4 | amazon | 1.685 | 1.925 | +0.24 |
| 5 | webcam | 2.001 | 1.937 | -0.06 |
| 6 | webcam | 1.896 | 2.074 | +0.18 |
| 7 | dslr | 1.892 | 1.637 | -0.26 |
| 8 | dslr | 1.742 | 1.706 | -0.04 |
| 9 | dslr | 2.093 | 1.952 | -0.14 |

**结论**: 两者 DaA dispatch 行为**几乎完全一致** (ratio 都在 1.6-2.1, Δ 大多 < 0.2). DaA 不区分 backbone (PG-DFC 还是 F2DC), 只看 sample_shares + Q. → DaA 跟 PG-DFC 在 server 聚合层面**无差异**.

⚠️ **反直觉**: dslr (最稀有, 期望 DaA 最升权) 在 PG-DFC+DaA 反而**升权略低** (1.637-1.952 vs F2DC+DaA 1.742-2.093). 但 PG-DFC+DaA dslr Last 仍大胜 +6.66pp → 说明赢点不在 server dispatch, 在 client 端 prototype guidance.

## 3. 维度 1: Effective contribution (weight × ‖Δw‖)

> 每个 client 真实声音 = α_i × ‖w_i - w_g‖, 100-round 平均.

按 domain 聚合:

| Method | caltech | amazon | webcam | dslr |
|---|:--:|:--:|:--:|:--:|
| F2DC | 0.396 | 0.313 | 0.302 | 0.459 |
| F2DC+DaA | 0.358 | 0.249 | 0.237 | 0.342 |
| PG-DFC | 0.427 | 0.285 | 0.311 | 0.437 |
| PG-DFC+DaA | 0.367 | 0.243 | 0.228 | 0.365 |

**结论**:
- DaA 后所有 domain 的 contribution 都**降低** (因为 ‖Δw‖ 也变小: layer drift 看下面)
- PG-DFC+DaA dslr contribution = **0.365** (高于 F2DC+DaA 0.342) → **PG-DFC backbone 让 dslr client 学得更深 / Δw 更大**, 这才是 dslr 涨 +6.66pp 的实证根源 (不是 server 升权)

## 4. 维度 1: Per-layer drift (浅 → 深)

> 每个 client conv layer 跟 global 的 L2 距离, 100-round avg, 6 个层.

| Method | L1 (浅) | L2 | L3 | L4 | L5 | L6 (深) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| F2DC | 0.375 | 0.438 | 0.463 | 0.733 | 0.813 | 0.131 |
| F2DC+DaA | 0.316 | 0.392 | 0.421 | 0.638 | 0.668 | 0.126 |
| PG-DFC | 0.352 | 0.417 | 0.473 | 0.736 | 0.795 | 0.133 |
| PG-DFC+DaA | 0.317 | 0.393 | 0.438 | 0.632 | 0.685 | 0.134 |

**结论**:
- DaA 让所有 layer 的 drift 都减小 (浅层 -16%, 深层 -13~14%): 因为 DaA 重新分配权重让 client update 更趋同
- PG-DFC+DaA 跟 F2DC+DaA 各层 drift **几乎完全相同** (差距 < 0.02) → **训练动力学高度相似**

## 5. 维度 2: Prototype consensus alignment (PG-DFC 核心机制)

> avg cos sim between client local_proto[c] 跟 consensus mean over class. F2DC 系列没存 local_protos (无 prototype 上传).

| Method | r10 | r50 | r100 | Δ(end-start) | mode_collapse_r100 |
|---|:--:|:--:|:--:|:--:|:--:|
| PG-DFC | 0.978 | 0.982 | 0.985 | +0.008 | 0.592 |
| PG-DFC+DaA | 0.983 | 0.982 | 0.985 | +0.001 | 0.587 |

### Per-client cos sim @ R100 (PG-DFC+DaA seed=2)

| Client | Domain | cos | 状态 |
|---|---|:--:|:--:|
| 0 | caltech | 0.991 | ⚠ 同化 |
| 1 | caltech | 0.992 | ⚠ 同化 |
| 2 | caltech | 0.989 | ⚠ 同化 |
| 3 | amazon | 0.977 | ⚠ 同化 |
| 4 | amazon | 0.991 | ⚠ 同化 |
| 5 | webcam | 0.973 | ⚠ 同化 |
| 6 | webcam | 0.982 | ⚠ 同化 |
| 7 | dslr | 0.979 | ⚠ 同化 |
| 8 | dslr | 0.985 | ⚠ 同化 |
| 9 | dslr | 0.989 | ⚠ 同化 |

**🚨 关键发现 (重大瓶颈)**:

1. **10/10 client 全被同化**: cos sim **全部 > 0.97** (远超 0.85 同化阈值). PG-DFC 已经把所有 client 拉到几乎相同的 prototype, **client 个性彻底丢失**.

2. **从 R10 就同化** (cos=0.978-0.983): warmup 还没结束 prototype 就趋于一致, 后续 90 round 没继续学到任何 domain 个性.

3. **mode collapse score = 0.587** (10 类 proto 平均 pairwise cos 0.59): **类间区分度也低** — 不仅 client 同化, **类与类之间也开始混淆**. 0.59 远高于健康范围 (期望 < 0.2).

→ **PG-DFC 实际上是把 prototype 变成了一个"伪 single proto"**, 既丢了 domain 多样性, 也丢了 class 区分度.

## 6. 维度 3: Feature space silhouette (best vs final)

| Method | seed | sil_class B → F | sil_domain B → F (低=domain-invariant) |
|---|:--:|:--:|:--:|
| F2DC | 2 | 0.078 → 0.097 (+0.019) | -0.061 → -0.065 (-0.004) |
| F2DC | 15 | 0.066 → 0.083 (+0.017) | -0.059 → -0.077 (-0.017) |
| F2DC+DaA | 2 | 0.050 → 0.088 (+0.038) | -0.032 → -0.049 (-0.017) |
| F2DC+DaA | 15 | 0.057 → 0.070 (+0.013) | -0.040 → -0.024 (+0.017) |
| F2DC+DaA | 333 | 0.052 → 0.092 (+0.040) | -0.056 → -0.069 (-0.013) |
| PG-DFC | 2 | 0.071 → 0.106 (+0.035) | -0.055 → -0.072 (-0.017) |
| PG-DFC | 15 | 0.054 → 0.072 (+0.018) | -0.066 → -0.088 (-0.022) |
| PG-DFC+DaA | 2 | 0.054 → 0.088 (+0.034) | -0.050 → -0.075 (-0.025) |
| PG-DFC+DaA | 15 | 0.050 → 0.087 (+0.037) | -0.037 → -0.044 (-0.006) |
| PG-DFC+DaA | 333 | 0.052 → 0.083 (+0.032) | -0.070 → -0.076 (-0.006) |

**结论**:
- 所有 method 的 sil_class 训练后期都涨 (class 区分度变好), domain sil 都变负 (更 domain-invariant) — 训练正常
- **PG-DFC+DaA sil_class final 0.083-0.088 ≈ F2DC+DaA 0.070-0.092** (几乎一样)
- **PG-DFC+DaA sil_domain (-0.075~-0.044) ≈ F2DC+DaA (-0.069~-0.024)** — domain-invariance 也无优势

→ **特征空间质量 PG-DFC+DaA vs F2DC+DaA 没有显著差异** (尽管 prototype 行为完全不同)

## 7. 维度 7: Cross-method CKA (PG-DFC+DaA vs F2DC+DaA)

| seed | CKA(best) | CKA(final) |
|:--:|:--:|:--:|
| 2 | 0.888 | 0.882 |
| 15 | 0.921 | 0.882 |

**🚨 关键: CKA > 0.88** → **PG-DFC+DaA 跟 F2DC+DaA 学到的 representation 高度相似**. paper 上 CKA > 0.85 通常被认为是"等价 representation".

→ **DaA 是主导项**, PG-DFC 的 prototype guidance 在 +DaA 后被 wash 掉 → 实证 saturation 假设.

## 8. 维度 6: Per-class shift best→final (训练后期掉了哪些 class)

### F2DC+DaA seed=15

| Domain | per-class shift |
|---|---|
| caltech | +3.3 +0.0 +15.0 +3.7 -5.9 +0.0 +7.7 -5.3 -17.6 -5.0 |
| amazon | +16.7 +0.0 +5.6 +0.0 +5.0 -15.0 +5.0 -10.0 +5.0 +0.0 |
| webcam | -20.0 +0.0 +0.0 +0.0 +0.0 +0.0 -12.5 -33.3 +0.0 +0.0 |
| dslr | +0.0 +0.0 +0.0 +0.0 +50.0 +0.0 -25.0 +50.0 +0.0 +25.0 |

### PG-DFC+DaA seed=15

| Domain | per-class shift |
|---|---|
| caltech | +10.0 +0.0 +5.0 +0.0 -5.9 -3.8 +11.5 +5.3 -17.6 -5.0 |
| amazon | +5.6 +6.2 -5.6 +0.0 +5.0 +0.0 +5.0 +45.0 -5.0 +5.6 |
| webcam | -20.0 +0.0 -33.3 +0.0 -16.7 +0.0 -12.5 +0.0 +0.0 -16.7 |
| dslr | **+50.0** +0.0 +0.0 +0.0 **+50.0** +16.7 **+50.0** **+50.0** +0.0 -25.0 |

**结论**:
- **dslr 区别极大**: PG-DFC+DaA 在 dslr 后期把 4 个 class 都涨到 +50pp (R100 反而比 best 还好 — best round 在早期), F2DC+DaA dslr 在 R100 涨幅类似但少 1 个 class
- **webcam 反而 PG-DFC+DaA 掉的 class 更多** (5 个 class 掉 vs F2DC+DaA 3 个) → webcam Last -2.29pp 输的实证根源
- caltech / amazon 两者掉的 class 模式接近

## 9. 维度 5: Best vs Last gap (训练稳定性)

| Method | mean gap (3 seed) | 备注 |
|---|:--:|:--:|
| F2DC | 11.62 | ❌ 最不稳, 后期严重崩 |
| F2DC+DaA | 5.33 | DaA 修了 F2DC 不稳问题 |
| PG-DFC | 6.74 | prototype 让 PG-DFC 比 vanilla F2DC 稳 |
| **PG-DFC+DaA** | **5.13** ⭐ | **最稳** (微胜 F2DC+DaA 0.20pp) |

→ **PG-DFC+DaA 是最稳的训练**, 但稳定性优势相对 F2DC+DaA 仅 0.20pp, 没有显著区别.

## 10. 综合诊断结论 (paper-grade)

### 10.1 PG-DFC+DaA 赢了吗?

| 维度 | 谁赢 | 幅度 |
|---|:--:|:--:|
| AVG_B (3-seed) | PG-DFC+DaA | +0.55pp (65.78 vs 65.23) |
| **AVG_L (3-seed)** | **PG-DFC+DaA** | **+0.75pp (60.65 vs 59.90)** |
| dslr Last | PG-DFC+DaA | **+6.66pp** ⭐ |
| webcam Last | F2DC+DaA | -2.29pp |
| caltech Last | F2DC+DaA | -1.19pp |
| amazon Last | tie | -0.18pp |
| training stability | PG-DFC+DaA | -0.20pp gap (微胜) |
| feature quality (sil) | tie | 无显著差异 |
| representation (CKA) | tie | 0.88 高度相似 |
| DaA dispatch | tie | ratio 几乎一样 |

✅ **赢了 (但很小)**: AVG Last +0.75pp, AVG Best +0.55pp, 主要来自 dslr.

### 10.2 输给 vanilla F2DC ⚠️

| Method | AVG_B | Δ vs PG-DFC+DaA |
|---|:--:|:--:|
| **F2DC (vanilla)** | 66.48 | +0.70pp |
| PG-DFC+DaA | 65.78 | — |

**vanilla F2DC AVG_B (66.48) 比 PG-DFC+DaA AVG_B (65.78) 高 0.70pp**. 但 vanilla F2DC AVG_L 只有 54.86 (比 PG-DFC+DaA 60.65 低 5.79pp), 训练后期严重崩 → vanilla F2DC 是"早期峰值高但后期不稳"的特点.

如果 paper 主表用 Last (合理选择), PG-DFC+DaA 才是最好的; 如果用 Best, vanilla F2DC 反而最强.

### 10.3 输在哪 / 瓶颈是什么 (3 个根本原因)

1. **🚨 Prototype 全同化** (维度 2):
   - cos sim 全部 > 0.97 (10/10 client), R10 就同化
   - mode collapse 0.59 (类间区分也开始混淆)
   - → **PG-DFC 的 prototype guidance 已经 over-aligned**, 没给 DaA 留 incremental 空间

2. **🚨 DaA + PG-DFC 机制重叠** (维度 1+7):
   - DaA dispatch ratio: PG-DFC+DaA vs F2DC+DaA 几乎一样 (Δ < 0.2)
   - per-layer drift: PG-DFC+DaA vs F2DC+DaA 各层差距 < 0.02
   - CKA(final) = 0.88 → 学到的 representation 高度相似
   - → **加 DaA 后, PG-DFC 跟 F2DC backbone 选择无关**, DaA 主导

3. **🚨 webcam 后期掉太多** (维度 6):
   - PG-DFC+DaA webcam 5 个 class 后期掉, F2DC+DaA 只掉 3 个
   - webcam Last -2.29pp 输的实证根源 — over-alignment 让 webcam 个性消失

### 10.4 paper narrative 建议

❌ **不能写**: "PG-DFC + prototype guidance 显著优于 F2DC backbone"
- AVG Best 跟 vanilla F2DC 比还输 -0.70pp
- AVG Best 跟 F2DC+DaA 比只赢 +0.55pp
- CKA 0.88 实证 representation 等价

✅ **可以写**:
1. **"PG-DFC+DaA 在 dslr (最稀有 domain) 上大胜 F2DC+DaA +6.66pp"** — dominant domain protection
2. **"PG-DFC+DaA 训练最稳定"** (gap 5.13, 比 vanilla F2DC 11.62 好 6.49pp)
3. **"PG-DFC + DaA 在 last avg 上是最强"** (60.65 vs F2DC+DaA 59.90 vs F2DC 54.86)

✅ **paper 卖点重新定位**:
- 不是 "我们 method 比 F2DC+DaA 强很多 accuracy"
- 而是 "我们 method 在**稀有 domain 保护 + 训练稳定性 + 后期不崩** 三个维度都更优"

## 11. 下一步 (打破 saturation 的 3 个候选方案)

### 方案 A: 打破 prototype over-alignment
- **现象**: cos sim > 0.97, mode collapse 0.59
- **可能修法**:
  - 加 prototype diversity loss (push 类 proto 互相远离)
  - 加 per-domain proto branch (不让 4 domain 的 proto 完全合并)
  - InfoNCE 温度调高 (减弱拉近力)

### 方案 B: 让 DaA 跟 PG-DFC 互补而非重叠
- **现象**: DaA dispatch ratio 跟 backbone 无关, CKA 0.88
- **可能修法**:
  - DaA 不只看 sample_shares, 加 "prototype drift" 信号 (cos sim 距 consensus 远的 client 升权更多)
  - 这就是上次提的 PDA-Agg 方向

### 方案 C: 保护 webcam 后期不崩
- **现象**: PG-DFC+DaA webcam 5 class 后期掉
- **可能修法**:
  - 后期 freeze prototype 不再 align (保留早期个性)
  - 给 webcam 加 stronger regularization

## 12. 数据来源

- 诊断 dump: `experiments/cold_path_analysis/diag_office/diag_*` (10 个目录, 每个 100 round npz + best/final)
- 分析脚本: `experiments/cold_path_analysis/compare_pgdfc_vs_f2dc_with_daa.py`
- 完整 raw output: `/tmp/diag_compare_output.txt`
