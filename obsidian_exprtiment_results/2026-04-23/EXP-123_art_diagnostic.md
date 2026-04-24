# EXP-123 | PACS Art Domain 失败根因诊断 — 数据驱动选方向

## 基本信息
- **日期**: 2026-04-23 启动, Stage B 2026-04-24 完成
- **目标**: 通过诊断数据, 回答 "PACS 的 Art domain 为什么只有 64-65%" 的根因, 指导下一步方法设计
- **状态**: 🟢 **Stage B 完成 8/9 runs (2026-04-24 early morning)** — orth_s333 待补 (Round 169/200, 还需 ~1h)

---

## 🎯 Stage B 核心结论 (2026-04-24, 8/9 runs 完整)

**最关键发现**: FDSE 的优势 **90% 来自 3 个 hard cells**, 不是 uniform!

| Cell | FedBN | orth_only | **FDSE** | FDSE Δ vs FedBN |
|---|:-:|:-:|:-:|:-:|
| (Art, **guitar**) | 37.25 | 43.94 | **61.57** | **+24.33** ⭐ |
| (Art, **horse**) | 45.93 | 50.85 | **61.45** | **+15.52** |
| (Photo, **horse**) | 48.07 | 55.86 | **65.57** | **+17.50** |

这 3 个 cells 贡献 FDSE AVG +2.05pp (总 FDSE-FedBN gap 2.31pp). 其他 25 cells 几乎相抵。

**反噬 cells** (FDSE 反输):
- (Art, dog): FedBN 60.98 > FDSE 47.62 (-13.36)
- (Art, giraffe): FedBN 75.62 > FDSE 66.73 (-8.89)
- (Art, elephant): FedBN 66.58 > FDSE 59.02 (-7.56)

### 3-seed mean 对比 (FDSE 3 seeds, FedBN 3 seeds, orth 仅 2 seeds — s=333 待补)

| Method | AVG Best | Art | Cartoon | Photo | Sketch |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **FDSE** (3) | **81.54** | 64.71 | 85.18 | **86.83** | 89.46 |
| **orth_only** (🟡 2) | 79.95 | 62.50 | 87.92 | 79.98 | 91.09 |
| **FedBN** (3) | 79.23 | 62.25 | 85.90 | 80.24 | 88.52 |

### Confidence / Calibration

Art ECE **所有方法 ~0.18** — 是 PACS 固有难点, 不是方法差异.
Photo ECE: FDSE 0.062 vs FedBN 0.110 → FDSE 校准 Photo 好, 部分解释 +6.6pp.

### 方向判决

| 方向 | 证据强度 |
|---|:-:|
| I stage-aware | 🟡 |
| II calibration | 🟢 (Photo 有效, Art 所有方法都差) |
| **III per-cell hardness** | **🟢🟢 最强** (3 cells 贡献 90%) |
| IV local protocol | ⚪ |

**决定方向**: III (per-cell hardness) → EXP-124 PCH (CE re-weight for hard cells, hw=2.0).

详细分析见 `stageB_full/ANALYSIS.md`.

---

---

## 🔴 Stage A 结果 (2026-04-23, paper-standard metric v2)

**修正**: 早期 v1 脚本用 per-(seed, client) 各自峰值 (每 client 不同 round), 高估; v2 改用 paper-standard (所有 client 在同一 best global round 取值)。

### 3-seed mean ± std (paper-standard)

| Method | AVG Best | AVG Last | Art | Cartoon | Photo | Sketch |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **orth_only** | **80.41 ± 1.08** | **79.42 ± 0.89** | 63.40 ± 4.78 | **86.89** | **81.64** | **89.71** |
| fdse | 79.91 ± 0.74 | 77.55 ± 0.65 | **66.50 ± 5.00** | 84.47 | 80.84 | 87.84 |
| **Δ (orth − FDSE)** | **+0.50 ✅** | **+1.87 ✅** | **-3.10 ❌** | +2.42 ✅ | +0.80 ✅ | +1.87 ✅ |

### 核心发现

1. **整体仍赢** (+0.50 AVG Best) — 满足 CLAUDE.md 硬指标 (>79.91) ✅
2. **Art 域输 -3.10** — 3/3 seeds 一致 FDSE 赢 (非波动, 系统性劣势)
3. **Art 方差极大** (both ≈5pp) — Art 是 PACS 固有难点 (painting 风格抽象)
4. **Last 指标优势更大** (+1.87) — orth_only 训练稳定, FDSE 后期下滑 (80.81→78.09 for s=2)
5. **只要攻下 Art** 就能整体 3pp 以上领先

### 已有数据回答的问题 (不需 Stage B hook)

- **Q: FDSE 在 Art 上做对了什么?**
  - FDSE peak rounds 普遍靠后 (R=185/120/181 vs orth_only R=150/98/135)
  - FDSE 是"慢热"型 — DSEConv 迭代擦除可能给 Art 这种风格差异大的域更多学习时间
  - FDSE Art std 稍大 (5.00 vs 4.78) — 也不稳定, 不是完美方案

- **Q: Art 是 class 问题还是 domain 问题?**
  - v2 没提取 per-class (需 Stage B) — 但 std 4.78 across seeds 提示 seed 间 Art 差 10+pp, 可能是不同 seed 学到不同 class 组合

- **Q: orth_only 比 FDSE 早收敛但在 Art 更弱, 意味什么?**
  - 可能是 orth_only 在 R100 就 early-stop 了 Art 学习 → **方向 I (stage-aware) 有戏**
  - orth_only s=15 在 R98 就达峰, Art 只有 60.78 — 如果继续练 Art 会不会爬?

### 下一步决策树

- **选项 A**: 跑 Stage B (9 runs × diagnostic hooks, seetacloud2 ~5h) — 彻底回答 4 个 Q
- **选项 B**: Stage A 数据够用, 直接设计 Art-targeted method (方向 I+III 优先)
- **选项 C**: 轻量 Stage B — 只跑 3 seeds × orth_only with hook, 对准 orth 的 Art 失败模式

**倾向 B/C**: Stage A 已给出方向 (FDSE "慢热" + orth_only "早熟"), 直接做 Art-aware 方法实验。Stage B 的 ECE/gradient alignment 是 nice-to-have, 不是 blocker。

---

---

## 为什么做这个实验

### 背景 (前情回顾)

- 我们当前 PACS 最强方法: **orth_only** (feddsa_scheduled mode=0, EXP-109), 3-seed mean AVG Best = **80.64** (胜 FDSE 本地复现 79.91 **+0.73**, 胜 FedBN 80.41 +0.23)
- 注意: DomainNet 上最强是 orth_uc1 (feddsa_sgpa uw=1 uc=1, EXP-115, 72.49), 但 PACS 上 orth_uc1 **没跑过 3-seed R200** — 本 NOTE 主要诊断 PACS, 所以选 orth_only
- 要达到 paper 要求必须**超过 FDSE 82.17** → 差 **+1.53pp**
- **per-domain 分解** (EXP-109 数据):
  - Art: 64.87 ± 3.40 (std 大, 最弱)
  - Cartoon: 87.75 ± 0.53
  - Photo: 84.63 ± 1.13
  - Sketch: 90.31 ± 0.00
- **Art 提 5pp (65→70) = Avg +1.25 → 逼近 FDSE**

### 读完 5 篇 paper 的共同盲点

5 篇 (FDSE / F2DC / I2PFL / FedCCRL / FedOMG) 都**直接给方法, 没先诊断**. 结果:
- FDSE 擦除派: 假设 domain-related = 噪声
- F2DC 校准派: 反实验证明 57% class signal 在里面
- I2PFL 反稀释派: 假设 outlier prototype 应上权
- **都没回答: Art 到底为什么低? 是 domain gap? class 不均衡? gradient 冲突? 还是 calibration 差?**

**我们不能重复这个错误**. 先诊断, 再设计方法.

---

## 4 个候选方向 (等诊断数据选)

| 方向 | 假设 | 需要看到的诊断结果 | 若确认 → 方法 |
|:-:|---|---|---|
| **I: Stage-aware dynamic** | 最优策略随 round 变化 | feature space 早期晚期差异大 | 动态 λ_erase(t), λ_refine(t) |
| **II: Calibration-aware** | Art 过度自信错误, 不是不会分 | Art ECE >> 其他 domain ECE | per-domain temperature scaling |
| **III: Per-cell hardness** | 瓶颈在 (Art, 某 2-3 class), 不是 Art 整体 | per-(domain, class) 矩阵有明显 hard cell | cross-domain class bank targeted |
| **IV: Local protocol reform** | Art client local training 和 global 冲突 | Art gradient 和其他 domain inner product 负 | client-side adversarial / online mixstyle |

---

## 变体通俗解释

本实验**不引入新方法**. 只在现有 3 个 baseline 上加**诊断 hook**:

| 变体 | 本质 | 为什么要跑 |
|:-:|---|---|
| **FedBN** | 什么都不做, BN 本地化 + FedAvg | **lower bound baseline** — Art 在最朴素方法下是啥样 |
| **orth_only** | 我们当前 PACS 最强 (feddsa_scheduled mode=0, 只正交头 + CE, 无 SGPA 架构无 whitening 无 centers) | **我们现在的 working point** — Art 是什么卡点 |
| **FDSE** | CVPR'25 SOTA 擦除派 | **直接竞品** — FDSE 在 Art 上是不是也卡? 他们隐瞒了什么 |

**FedBN vs orth_only 的对比关键**:
- 如果 FedBN Art = 60, orth_only Art = 65 → orth_only 的 +5 来自 Art, 方法方向对
- 如果 FedBN Art = 65, orth_only Art = 65 → orth_only 的进步来自其他 domain, Art **架构卡死**
- 如果 FedBN Art < orth_only < FDSE → FDSE 有处理 Art 的东西我们没学

### 为什么不只跑 orth_only?

- orth_only 是**复杂 baseline**, 它好可能是因为**某些组件压根没动 Art**, 诊断信号会混淆
- FedBN 是**最干净 baseline**, 它的 Art 数据告诉我们 **Art 的 intrinsic 难度**
- FDSE 是**对手数据**, 他们 Art 的 per-class 分布告诉我们 **FDSE 擦除对 Art 的影响**

**三者并看才能辨认模式**.

---

## 实验设计

### 阶段 A — 从已有 record 提取 (0 成本, 今天就能做)

flgo 的 record JSON 默认记录了:
- `local_test_accuracy_dist`: per-client per-round acc (已有)
- `local_val_accuracy_dist`: per-client per-round val
- `mean_local_train_loss`: 全局 train loss 曲线

可**立即**从以下 record 提取 4 种分析:

1. **Per-domain accuracy curve** (已有, 4 domain × 200 round)
2. **Domain Gap**: (best domain acc - worst domain acc) per round
3. **Convergence timing**: 每 domain 什么时候到 peak? 是否 Art 晚收敛?
4. **Last-round stability**: 最后 20 round Art 是否震荡? (std)

**Record 来源**:
- FedBN PACS: `task/PACS_c4/record/fedbn_*.json` (3 seed)
- orth_only PACS: `task/PACS_c4/record/feddsa_scheduled_lo1.0*sm0*.json` (3 seed) — 来自 EXP-109 (mode=0 纯正交头)
- FDSE PACS: `task/PACS_c4/record/fdse_*.json` (3 seed) — 项目已跑过

### 阶段 B — 新跑 diagnostic runs (1 晚, 需要加 hook)

**现有 record 无法提供**以下信息 (需代码 hook):

1. **Per-(domain, class) accuracy matrix** (4×7=28 cells) — 需 hook server eval
2. **Per-domain ECE** (Expected Calibration Error) — 需 hook logits 不只 top-1
3. **Per-domain gradient alignment on server** — 需 log client gradient / 模型 delta
4. **Feature space t-SNE @ R={50, 100, 200}** — 需 hook feature save

**实施**:
- 写 1 个 `diagnostic_hooks.py` 挂到 flgo 现有 callback
- 每 50 round 触发一次诊断 (共 4 次, R=50/100/150/200)
- 输出 4 份 JSON 到 `diagnostic/round_{R}/` 下

**Runs**:
- FedBN + orth_only + FDSE, 各 3 seed {2, 15, 333} = **9 runs**
- 单 seed 诊断容易被 seed-noise 误导, 3 seed mean ± std 才能判断 pattern 是 systematic 还是偶然
- R=200, E=5, B=50 对齐 EXP-109

**资源**: 9 runs × R=200 greedy 并行 on **seetacloud2 RTX 4090 24GB** (空闲):
- 单 run ~2-2.5GB 显存, 9 并行 ~22GB ✅
- 单 run 4090 wall ~4h, 9 并行 wall ~5h (早期不饱和)
- 今晚启动, 明早出结果

---

## 判决规则 (诊断 → 方向映射)

诊断数据看回答下面 4 个问题:

### Q1: Art 是 class-level 不均衡 还是 domain-level 整体难?

- **证据**: Per-(domain, class) matrix
- **若**: Art 里有 2-3 class 明显 < 50%, 其他 class ≥ 80% → **方向 III (per-cell) 有戏**
- **若**: Art 所有 class 都在 55-70% range → 不是 class 问题, Art 整体弱

### Q2: Art 是 calibration 差还是 representation 差?

- **证据**: Per-domain ECE + 错误预测的 confidence 分布
- **若**: Art ECE > 0.15 + 错误预测 confidence 平均 > 0.7 → **方向 II (calibration) 有戏**
- **若**: Art ECE < 0.08 (和其他 domain 差不多) → calibration 不是瓶颈

### Q3: Art local training 和其他 domain 冲突吗?

- **证据**: Server-side 每 round Art 的 gradient 和其他 3 domain 的 cosine similarity
- **若**: Art 和其他 domain 的 gradient cos_sim < 0.3 (或负) → **方向 IV (local reform) 有戏**
- **若**: Art gradient 和其他一致 → 不是冲突问题

### Q4: Art 的 feature space 早期晚期差异大吗?

- **证据**: t-SNE @ R=50 vs R=200
- **若**: R=50 时 Art 所有 class 挤成一团, R=200 时还没散开 → **方向 I (dynamic stage) 有戏**
- **若**: R=100 Art 就已经稳定, 后面没变化 → stage-aware 没意义

### Optional Q5: orth_only vs FDSE 的 Art 差异来自哪里?

- 如果 FDSE Art > orth_only Art → FDSE 有处理 Art 的秘密 (层分解 DSEConv?)
- 如果 orth_only Art > FDSE Art → 我们方向对, 继续深挖
- 如果两者差不多 → Art 是**普遍难题**, 没有现成 silver bullet

---

## Stage B 诊断 hook 实施 (2026-04-23)

### Hook 架构

**文件**:
- `FDSE_CVPR25/diagnostics/per_class_eval.py` — 算法无关的纯函数 `run_diagnostic(model, dataset, device, num_classes, ...)`. 返回 per-class acc/conf/support, confidence stats (mean/std/p10/p50/p90/ECE/over_conf_err_ratio/wrong_conf_mean), 可选 histogram (correct vs wrong × 20 bins)
- `FDSE_CVPR25/logger/__init__.py` — 新增 `PerRunDiagLogger(PerRunLogger)`, 每 round 调 diag; 每 50 rounds dump histogram

### 记录字段 (添加到现有 record JSON, 不破坏结构)

| Key | 类型 | 说明 |
|---|---|---|
| `per_class_test_acc_dist` | `list[round][client][class]` float | per-(domain, class) 准确率 |
| `per_class_test_conf_dist` | `list[round][client][class]` float | per-class 平均 max softmax |
| `per_class_test_support_dist` | `list[round][client][class]` int | 每 class 样本数 |
| `confidence_stats_dist` | `list[round][client]` dict | {mean, std, p10, p50, p90, ece, over_conf_err_ratio, wrong_conf_mean} |
| `confidence_hist_dist` | `list` (每 50 rounds 一个) | {round, per_client:[{hist_correct, hist_wrong, bins}]} |

### 测试覆盖

**单元测试** `tests/test_per_class_eval.py` (7 tests, 全过):
1. 完美预测 ECE ≈ 0
2. per-class acc 正确性 (class 不均衡)
3. over-confident wrong ratio
4. histogram 包含正确
5. 空 class 返回 NaN
6. ECE 函数直接测试
7. model.train()/eval() 状态恢复

**集成测试** `tests/test_per_run_diag_logger.py` (6 tests, 全过):
1. PerRunDiagLogger 继承 PerRunLogger
2. mock model diag forward
3. log_once mock 结构正确
4. round 0 含 histogram, round 1-49 不含
5. num_classes 推断 (PACS=7/Office=10/fallback to last Linear)
6. 输出 JSON-serializable

### Smoke test 流程 (CLAUDE.md 17.4 完整验证清单)

本地 smoke 受限 (PACS raw 数据不在本机), 在 seetacloud2 跑 R=3 smoke:
- `fedbn + feddsa_scheduled + fdse × seed=2 × R=3`
- 用 `--logger PerRunDiagLogger`
- 验证:
  - 不崩溃
  - record JSON 出现新字段
  - 数值合理 (per-class acc ∈ [0,1], ECE ∈ [0, 0.5])

通过后 → 启动 9 runs R=200.

### 资源 + 时间 (更新)

| 阶段 | 计算 | 时间 |
|---|---|---|
| 本地单测 (13 tests) | 本地 Python CPU | 已完成 ✅ |
| 服务器 R=3 smoke × 3 algo | seetacloud2 ~10min | 启动中 |
| 9 runs R=200 × diagnostic hook | seetacloud2 greedy 并行 | ~5h wall |
| 分析 + 选方向 | 本地 | 1h |

---

## 资源 + 时间

| 阶段 | 计算 | 时间 |
|:-:|---|---|
| A: record 提取 + 可视化 | 本地 Python, 0 GPU | 30-60 min |
| B: 3 runs diagnostic | lab-lry GPU 1 or seetacloud2, 3-4h wall | 1 晚 |
| 分析 + 写 diagnostic report | 本地 | 1-2h |
| 选方向 + 设计 EXP-124 方法 | — | 1 天 |

**总**: 今天 1 晚跑 + 明天 1 天分析 = **明天晚上**决定下一步方向.

---

## 成功标准

诊断完毕 4 个问题**至少 2 个得到明确回答** (Y 或 N), 指向下一步具体方法方向.

**不是** "pointless 跑一遍看曲线" — 每个 diagnostic hook 都对应**一个候选方向**的 decisive 证据.

---

## 下一步 (本 NOTE 审核后)

1. ✅ 写 `scripts/analyze_art_from_records.py` — 从已有 record 提取阶段 A 数据
2. ✅ 写 `FDSE_CVPR25/diagnostic_hooks.py` — 诊断 hook
3. ✅ 写启动脚本 `run_exp123.sh` — 3 runs with hook
4. ⏸ 等审核通过, 启动阶段 A 看初步结果再决定是否启动阶段 B

---

## 📎 相关文件

- 诊断 hook 代码: `FDSE_CVPR25/diagnostic_hooks.py` (待写)
- record 提取脚本: `scripts/analyze_art_from_records.py` (待写)
- 历史 record: `task/PACS_c4/record/` (fedbn / fdse / feddsa_sgpa)
- 报告输出: `experiments/ablation/EXP-123_art_diagnostic/report/`
- 上游: EXP-109 (orth_only PACS), EXP-080 (orth_only PACS), 5 篇 paper 精读笔记 (`obsidian_exprtiment_results/知识笔记/论文精读_*.md`)

---

## 🧠 设计哲学 (写给未来的自己)

**不要** 直接看 paper 抄方法 → 这是 "缝合" 陷阱. 5 篇 paper 实验 setup 差异巨大 (ResNet-18/10, MobileNetV3, ViT; FedDG vs FedDA; 4/6/10/100 clients), 数字都不可直比. 方法也各有盲点.

**要** 从**自己数据的诊断**出发, 识别 **FDSE/F2DC/I2PFL 都没看到的问题**, 设计 novel 切入点. 这是 CVPR 级别 paper 的真正做法, 不是跟风.

本 EXP-123 的价值不在 accuracy 数字, 而在**数据驱动地决定下一步方向**. 决策质量 > 实验数量.
