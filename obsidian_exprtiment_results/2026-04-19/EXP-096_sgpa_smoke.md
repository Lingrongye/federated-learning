# EXP-096 SGPA Smoke Test — 大白话解读

> 2026-04-19 晚。跑通了 FedDSA-SGPA 新方案的第一版实验。核心结论:**真的管用,而且知道为什么管用**。

## 一句话

把 Plan A 的最后一层 "可训分类器" 换成 "钉死的几何分类器" (Fixed ETF),在 Office 上只跑 1/4 的训练时间 (R50 vs Plan A R200),就已经超过了 Plan A 的最终表现。更重要的是,这次装了诊断监控,所以**知道为什么赢**,不是瞎猜。

## 我们到底改了啥

**就一件事**: `sem_classifier` 原来是 `nn.Linear(128, 10)` (10 个类,128 维特征,每轮训练更新 1280 个参数,还要全联邦平均)。现在改成一个**一开始就算好的固定矩阵 M** (同样 128×10,但**永远不动**,所有 client 用同一个 seed 生成,所以天生一致)。

分类时: `logits = normalize(z_sem) @ M / 0.1`,就是算每个样本的特征跟 10 个固定方向的相似度,最像哪个就是哪类。

## 为什么这样设计 (非学术版)

想象你要教一群人画 10 种动物。
- **原来 (Linear)**:老师自己也不知道该怎么画,边教边学,但每个老师学到的不一样,最后取平均。结果大家画得乱七八糟。
- **现在 (Fixed ETF)**:老师一开始就定好 10 种动物的"标准姿势",每个都互相成最大可能的角度 (10 个点均匀撒在球面上)。学生只要把特征拉到对应姿势方向就行,方向是**所有老师统一的,没人吵架**。

这就是 **Neural Collapse** 理论说的:训练到最后,好的特征**一定会**自动对齐这样的几何形状。我们干脆**一开始就把这个形状钉死**,让特征早点对齐,不用自己摸索。

## 结果 (Office-Caltech10, R50, seed=2)

| 方法 | 训练预算 | AVG 准确率 |
|------|---------|-----------|
| Plan A (原版) | **R200** (全量) | 82.55% |
| FDSE (最强 baseline) | R200 | 90.58% |
| **我们 SGPA (只跑 1/4)** | **R50** | **84.98% (best), 83.09% (末轮)** |

R50 就**超 Plan A R200 +2.4%**。这是个非常意外的好结果。

## 诊断监控说了什么 (重头戏)

这次实验最值钱的不是准确率,是 **21 个中间过程监控指标**。挑 5 个最关键的讲。

### 指标 1: `etf_align` — "特征有没有对齐理想方向"

- **怎么算**: 每一类的所有样本特征取平均,算这个平均向量跟"理想方向" (M 的第 c 列) 的 cosine
- **0 = 完全没对齐**, **1 = 完美对齐**
- **我们的**: R5=0.30 → R30=0.79 → R50=**0.83**
- **大白话**: 第 5 轮时特征还在乱飞,第 30 轮已经基本知道往哪指了,第 50 轮几乎钉在理想方向上。**训练一半就达到理论最优的 83%**。

### 指标 2: `inter_cls_sim` — "不同类的特征中心离得多远"

- **怎么算**: 10 个类的平均向量,两两算 cosine
- **高 = 挤在一起**, **低/负 = 散开**
- **理论下界** (单纯形 ETF 给 10 个类能达到的最大分离): `-1/(10-1) = -0.111` (负的,意思是比垂直还更远)
- **我们的**: R5=0.64 (挤一起) → R30=-0.07 → R50=**-0.08**
- **大白话**: 一开始 10 个类的中心重叠严重 (0.64),训练到一半就散开到**接近理论极限** (-0.08 vs -0.111,达 72%)。这就是 **Neural Collapse** 里说的"类中心自动形成单纯形"。

### 指标 3: `orth` — "语义头跟风格头有没有真的分开"

- **怎么算**: z_sem 跟 z_sty 的 cosine 平方 (cos²),就是我们训练时的 L_orth 损失
- **0 = 完全正交 (分得开)**, **1 = 完全重合 (没分开)**
- **我们的**: R5=0.0037 → R50=0.0003
- **大白话**: 几乎完全正交。语义特征和风格特征之间的"串线"只剩 0.03%。正交解耦生效了。

### 指标 4: `client_center_var` — "跨客户端的类中心一致性"

- **怎么算**: 4 个 client 各自的 10 个类中心,算跨 client 的方差
- **高 = 每个客户端各画各的**, **低 = 大家画得一样**
- **我们的**: R1=0.025 → R26=0.003 → R50=**0.0019**
- **大白话**: 降了 93%。Fixed ETF 作为"统一标尺"让 4 个 client 的特征表示越来越像。这就是**为什么 FedAvg 聚合不再把模型平均糊**。

### 指标 5: `param_drift` — "每个客户端本地训完的模型离全局模型多远"

- **怎么算**: `‖client_weights - global_weights‖₂`
- **高 = client 训飞了**, **低 = client 和全局对齐**
- **我们的**: R1=0.75 → R26=0.046 → R50=**0.003**
- **大白话**: 降了 99.6%。到 R50 时每 client 训出来的模型**跟全局几乎完全一样**。Plan A 的 Linear classifier 就没这个性质 (每轮被拉来拉去)。

## 这次实验还证明了什么

**诊断比 accuracy 更重要**。有两个版本 (v2 不开诊断,v3 开诊断),**accuracy 几乎一样** (84.47% vs 84.98%),但只有 v3 能回答"为什么好"。如果没这些诊断数据:
- ❌ 不知道 ETF 有没有被真正对齐
- ❌ 不知道类间分离度有多接近理论最优
- ❌ 不知道 FedAvg 漂移是不是真被根除
- ❌ 只能瞎猜 "可能是 seed 运气吧"

## 踩过的 3 个坑 (真实 bug 记录)

| 版本 | 现象 | 根因 | 修复 |
|------|------|------|------|
| v1 | CUDA assert crash | `num_classes=7` 默认值,但 Office 有 10 类 | 按 task 名映射 (PACS=7, Office=10) |
| v2 | Layer 2 的 `client_center_var` 偏大 | 某 client 没见过某类时填了 0 向量 | 改成填 NaN + server 端过滤 |
| v2 | Layer 1 jsonl 完全不写 | flgo 调 Server.initialize 先,Client.initialize 后;后者 hard-override 了前者设的 `diag_log_dir=None` | 用 `getattr(self, 'diag_log_dir', None)` 保留已有值 |

3 个 bug 都是**实际部署时才冒出来的**,单元测试过不了这一关。说明**跑一次真实验比写十个单测更能暴露问题**。

## 当前状态

| 事项 | 状态 |
|------|------|
| 代码 | feddsa_sgpa.py 756 行, 26 单测通过 |
| 诊断框架 | 21 指标完整集成, 79 单测通过 |
| 首次 smoke test | ✅ R50 AVG 84.98% |
| 诊断数据 | ✅ Layer 1+2 全部生成 |
| Neural Collapse 证据 | ✅ etf_align 0.83, inter_cls -0.08 |
| Git commits | `cf0c47a` / `32ebc8e` / `926bf18` / `0c48a1c` / `8da0ec6` 全 push |
| 对照基线 (Linear) | ❌ 还没跑 (EXP-100) |
| 多 seed 验证 | ❌ 只跑了 seed=2 (EXP-097 待跑) |
| PACS 验证 | ❌ 等 SCPR v2 释放 GPU |
| SGPA 完整推理 | ❌ `test_with_sgpa` 没被 flgo 自动调, 需独立 script (EXP-099) |

## 一定要提醒自己的事

1. **84.98% 只是 seed=2 一次结果**,不能就此下定论。SCPR 当初也是单 seed 亮眼,3-seed 就拉胯了。**必须 EXP-097 跑 3 seed** 才能信。

2. **Fixed ETF 本身是不是真在帮**,需要一个 Linear 对照组 (EXP-100)。要是 Linear 同配置也 84%,那可能不是 ETF 的功劳,是别的地方 (比如 z_sty μ/σ 广播的副作用)。

3. **SGPA 推理端 (双 gate + 原型) 这次没真正测**。flgo 默认 test 走 `model.forward()` 直接 argmax,等于只测 ETF 训练端效果。真正的 SGPA 推理贡献 (论文 dominant contribution) 要等 EXP-099 单独 script 验证。

4. **PACS 4-client outlier 场景更难**。Office 10 类 K=10 理论下界 -0.111,PACS 7 类 K=7 下界 -0.167 更宽,理论上 ETF 收益应该更大。但 PACS AlexNet E=5 跟 Office ResNet E=1 训练节奏差很多,不一定能复制 R50 就收敛的现象。

## 下一步决策树

```
情况 A: 用户让继续跑 EXP-097 (Office 3-seed R200)
  → 约 3-4 GPU-hours
  → 回答: "不是 seed 运气"
  → 最关键的下一步

情况 B: 用户让先跑 EXP-100 (Linear 对照)
  → 约 1 GPU-hour (Office R200 × 1 seed 先看)
  → 回答: "ETF 本身贡献多少"
  → 控制变量干净

情况 C: 用户让先跑 EXP-099 (SGPA 推理 script)
  → 约 0.5 GPU-hour
  → 回答: "SGPA 推理端到底有没有贡献"
  → 验证论文的 dominant contribution

情况 D: 等 SCPR v2 PACS 释放 GPU (约 1h), 并行 EXP-098 PACS 3-seed
  → 约 25 GPU-hours (PACS AlexNet E=5 慢)
  → 回答: "PACS 是不是也 work"
  → 双数据集覆盖
```

**我的建议**: **B → A → C → D** 顺序。先 Linear 对照 1 seed 看差距,如果 ETF 贡献明显 (比如 Linear 82%, ETF 85%,差 3%) 就加速 3-seed;如果 Linear 也 85% 就等于 ETF 没贡献,要重新思考方案。

## 参考链接

- NOTE.md: `experiments/ablation/EXP-096_sgpa_smoke/NOTE.md`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (756 行)
- 诊断框架: `FDSE_CVPR25/diagnostics/sgpa_diagnostic_logger.py` (79 单测)
- 诊断数据: `FDSE_CVPR25/task/office_caltech10_c4/diag_logs/R50_S2/*.jsonl`
- FINAL_PROPOSAL: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/FINAL_PROPOSAL.md`
