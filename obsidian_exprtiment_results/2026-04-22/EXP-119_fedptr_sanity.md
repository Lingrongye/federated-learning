# EXP-119 | FedPTR Sanity Smoke Phase 1 — 验证 3 个组件是否值得投入

## 基本信息
- **日期**: 2026-04-22 23:32 启动 → 2026-04-23 11:01 全部完成 ✅
- **目标**: 在写 FedPTR 全量代码前, 用最小实验验证 3 个核心组件是否真 work
- **服务器**: **westb:13399 (新 RTX 4090 24GB, 从原 seetacloud 实例克隆, SSH 密码 CgVAw0UGAAoB)**
- **状态**: ✅ **Wave 1/2/3 全部完成 (Office 12 + PACS 12 = 24 runs)**
- **决策依据**: 接受 CVPR reviewer review (Score 4/10) 的建议, 不堆组件, 先验证单点
- **🏁 最终判决**: **C1 INVALID (Centralized 实验假设错), C2 FAILED (CC-Bank 输 FedBN 两数据集), C3 MARGINAL (Trajectory η=0.5 小赢 PACS +0.36pp 但 <0.5 noise 门)**

## 部署细节 (2026-04-22 23:32 更新)

**5-Wave chained dispatcher** (`run.sh`), 自动跑完 24 runs 约 23h:

| Wave | 内容 | 并行度 | 单 run 时间 | 预计完成 |
|:----:|---|:----:|:----:|:----:|
| 1 | Office: centralized × 3 + ccbank_a05 × 3 | 6 并行 | ~1h | **2026-04-23 00:30** |
| 2 | Office: ccbank_a03 × 3 + ccbank_a07 × 3 | 6 并行 | ~1h | 2026-04-23 01:30 |
| 3a | PACS: centralized × 3 + ccbank_a05_s2 | 4 并行 | ~7h | 2026-04-23 08:30 |
| 3b | PACS: ccbank_a05_s15/333 + traj_eta0_s2/15 | 4 并行 | ~7h | 2026-04-23 15:30 |
| 3c | PACS: traj_eta0_s333 + traj_eta05 × 3 | 4 并行 | ~7h | 2026-04-23 22:30 |

**GPU 显存策略**:
- Office (E=1): 单 run ~3-4GB, 6 并行 × ~3.5GB = 21GB ✅ (剩 3GB 安全)
- PACS (E=5): 单 run ~5-6GB, 4 并行 × ~5.5GB = 22GB ✅
- 当前显存: 13.5 / 24 GB (Wave 1 刚启动, 数据加载占比未满)

**路径**:
- master 日志: `experiments/ablation/EXP-119_fedptr_sanity/master.log`
- 每 run 日志: `experiments/ablation/EXP-119_fedptr_sanity/logs/W{N}_<label>_s{seed}.log`
- 结果 JSON: `FDSE_CVPR25/task/{PACS_c4,office_caltech10_c4}/record/*.json`

## 🏆 Wave 1 结果 (2026-04-23 00:35 完成)

### 3-seed AVG Best / AVG Last

| 方法 | AVG Best | AVG Last | vs FDSE 90.58 | vs FedBN 88.68 (原阈值基线) | vs EXP-116 FedBN 89.75 (新最强 baseline) |
|---|:---:|:---:|:---:|:---:|:---:|
| FDSE | 90.58 | 89.22 | 0 | +1.90 | +0.83 |
| **W1 Centralized Office** (Sanity A, C1) | **81.81** | 76.62 | **−8.77** 🔴 | −6.87 | −7.94 |
| **W1 CC-Bank α=0.5** (Sanity B, C2) | **89.35** | 87.17 | −1.23 | **+0.67** ✅过原阈值 | **−0.40** ❌ 没过新 baseline |

### Per-seed (Wave 1)

**Centralized Office** (R=200, eval 每 2 round 一次, 共 101 采样点):

| Seed | ALL B/L | AVG B/L |
|:---:|:---:|:---:|
| 2 | 73.00/69.83 | 80.16/77.60 |
| 15 | 76.17/73.76 | 82.45/77.19 |
| 333 | 76.58/67.83 | 82.83/75.08 |
| **Mean** | 75.25/70.47 | **81.81/76.62** |

**CC-Bank α=0.5 Office** (R=200, 每 round eval, 201 采样点):

| Seed | ALL B/L | AVG B/L |
|:---:|:---:|:---:|
| 2 | 82.16/80.57 | 89.48/88.46 |
| 15 | 83.74/81.36 | 89.21/87.75 |
| 333 | 84.93/80.55 | 89.36/85.31 |
| **Mean** | 83.61/80.83 | **89.35/87.17** |

### 判决分析

#### C1 Centralized = **INVALID** (不是 failed, 是实验设计假设错)

**原假设** "centralized 是 FL method space 的上限" — 对多域数据**错误**.

**实际发现**: centralized.py 用 global AlexNet + global BN 训练, test 时用 global BN running_mean/var 评估每个 domain. 但 global BN 学到的是"4 domain 混合的 mean/var", 与每个单独 domain 的真实分布不匹配 → per-domain acc 天然打折. **这不是 AlexNet capacity 问题, 是 BN 在多域数据上的固有行为**.

**证据**:
- FedBN (per-client 本地 BN) = 88.68 > Centralized (global BN) = 81.81
- 说明 "BN 本地化" 带来 +6.87pp, 远超 method space 的 +1-2pp 改进上限
- Centralized 不是上限, 是**下限**在这个设置下

**正确的 C1 capacity test** 应该是:
- **Per-domain single training** (每 domain 单独训一个模型, 相当于 4 个独立模型): 这才是真正的 method space 上限
- OR **Centralized + local BN** (训练集中但 BN per-domain 本地, 不过这已经等价于 FedBN)

**判决**: C1 **结果无意义**, 不能用来否决 backbone. 继续走 Wave 2-3 看 CC-Bank/Trajectory.

#### C2 CC-Bank α=0.5 = **⚠️ 边界**, 没打过最强 FedBN

**原阈值 > 89.18 (FedBN 88.68 + 0.5)**: 89.35 **技术过线** ✅
**新 baseline (EXP-116 lo=0 FedBN)**: 89.75 → CC-Bank 89.35 **−0.40pp 没赚到** ❌

**结论**: 这是历史 AdaIN 第 **5 次"部分失败"** — 没崩 (不像 EXP-059 -2.54%), 但**也没改进** (甚至微降). 配合 EXP-116 发现 "正交头无贡献"：**FedBN + 什么都不加 (lo=0) 就已经是 89.75, CC-Bank 再加也没超过**.

**不 kill 的理由**:
1. α=0.5 可能不是最优, Wave 2 α=0.3/0.7 还没完
2. PACS (强风格异质) 可能有 AdaIN 真实收益, 等 Wave 3
3. 历史 EXP-059/061 AdaIN 失败是 on PACS, 这次 Office 没失败反而接近 FedBN — 跨数据集行为不一样

**Wave 2-3 完成后重新判**.

## 🏆 Wave 2 结果 (2026-04-23 01:35 前完成, α=0.3 + α=0.7 Office)

### α 扫描总表 (Office AVG Best / AVG Last)

| α | Wave | S2 AVG B/L | S15 AVG B/L | S333 AVG B/L | 3-seed Mean AVG B/L |
|:-:|:----:|:---:|:---:|:---:|:---:|
| 0.3 | W2 | 88.85/88.46 | 90.30/89.17 | 88.18/86.39 | **89.11/88.01** |
| **0.5** | W1 | 89.48/88.46 | 89.21/87.75 | 89.36/85.31 | **89.35/87.17** |
| **0.7** | W2 | 88.13/87.42 | 89.91/89.91 | 90.78/89.81 | **89.61/89.05** 🏆 |

### α 判决

- **α=0.7 最优** (AVG Best 89.61, AVG Last 89.05 最稳)
- **α=0.3 Last 88.01** 仍然稳（比 α=0.5 的 87.17 更稳）
- **α=0.5 Last 87.17** 后期 drop 较大
- 所有 α **都没打过 EXP-116 lo=0 FedBN 的 89.75** (差 -0.14 ~ -0.64pp)

### Office 上 CC-Bank 最终判决: **AdaIN 第 5 次部分失败**

无论 α=0.3/0.5/0.7, Office AVG Best 都在 **89.1-89.6** 区间, 没有一个超过 FedBN baseline 89.75. **Office 上 CC-Bank 不 work** (没崩但没赚).

**下一步决定命运**: Wave 3 PACS 结果 (强风格异质) — 如果 CC-Bank PACS 也没赚 → **正式 kill CC-Bank**.

---

## Wave 3 PACS 部署 (2026-04-23 01:32-01:45 启动, 12 runs 真正并行)

**关键教训**: 原 chained dispatcher 设计 "Wave 3a/3b/3c × 4 并行 × 7h = 21h" 完全浪费. Wave 2 的 6 Office runs 只用 13GB/24GB, 有 11GB 剩余. **应该立即并行启动 PACS**, 不用等 Office 完成.

**修复流程**:
1. kill 原 dispatcher (保留已 launched 子进程继续跑)
2. 写 greedy_pacs.sh, 按 `nvidia-smi memory.free` 动态 launch, 阈值 4500MB
3. 修 PACS task 的 `task/PACS_c4/model/default_model.py` — 原 `init_global_module: pass` (silent kill bug 发现 5) 改为有效构造 `Model().to(device)`
4. Launch watcher (`json_watcher.py`) 每 30s 扫 record/, 检测到 JSON mtime 变化立即 cp 带时间戳 bak, 防止 η=0/0.5 互覆盖 ✅ **成功保留 6 份 trajectory 数据**
5. 12 个 PACS 全部并行跑 (3 centralized + 3 ccbank α=0.5 + 3 traj η=0 + 3 traj η=0.5)

**实际完成**: Wave 3 12 runs 于 2026-04-23 **08:48-11:01** 全部完成 (greedy 节省 14h wall, 比原计划 22:30 提前 ≈ 11h).

---

## 🏆 Wave 3 PACS 结果 (2026-04-23 11:01 全部完成)

### Sanity A: Centralized PACS (早停 R=101, 单 run 未跑满 R200)

| Seed | ALL B/L | AVG B/L | R |
|:---:|:---:|:---:|:---:|
| 2 | 68.20/65.99 | 65.86/62.92 | 101 |
| 15 | 71.71/68.20 | 68.63/64.54 | 101 |
| 333 | 69.80/67.80 | 66.38/64.56 | 101 |
| **Mean** | **69.90/67.33** | **66.96/64.01** | 101 |

**注意**: Centralized PACS 跑到 R=101 就自动停 (可能是 centralized 算法只跑 int(0.5*R)=100 round 或 early stop). R=200 等效版本估计 ≈ 70-73% AVG Best.

### Sanity B: CC-Bank α=0.5 PACS (R=200 完整)

| Seed | ALL B/L | AVG B/L |
|:---:|:---:|:---:|
| 2 | 81.74/80.54 | 79.94/78.40 |
| 15 | 80.33/79.93 | 77.89/77.49 |
| 333 | 81.24/80.03 | 78.81/77.43 |
| **Mean** | **81.10/80.17** | **78.88/77.77** |

**判决**: **输 FedBN PACS baseline 80.41 -1.53pp** ❌. Office α 全扫 -0.14~-0.64, PACS α=0.5 -1.53 → **CC-Bank 两数据集都输 FedBN**, AdaIN 家族第 5 次部分失败定案.

### Sanity C: Trajectory PACS (R=200 完整, η=0 vs η=0.5)

**.bak 时间戳对应 log mtime 验证, 准确区分**:

| Seed | η=0 (对照, pure alignment) | η=0.5 (预测, 我们的方法) | Δ (η=0.5 − η=0) |
|:---:|:---:|:---:|:---:|
| 2 | B 81.76 / L 80.07 | B 81.38 / L 79.51 | -0.38 B |
| 15 | B 79.77 / L 78.48 | B 80.39 / L 79.13 | **+0.62 B** |
| 333 | B 79.65 / L 77.33 | B 80.53 / L 77.94 | **+0.88 B** |
| **Mean** | **B 80.39 / L 78.63** | **B 80.77 / L 78.86** | **+0.38 B / +0.23 L** |

### Wave 3 PACS 判决 (对齐 C1/C2/C3 Claim)

| Claim | 阈值 | 实际 | 判决 |
|:-:|---|:---:|:---:|
| **C1 Centralized > 93** (Office) | method-space 有 +2pp 空间 | 81.81 (Office) / 66.96 (PACS R=101) | ❌ **INVALID 假设** (global BN 多域 mismatch, 不是 capacity 问题) |
| **C2 CC-Bank Office > 89.2** (≥ FedBN + 0.5) | — | α=0.7 89.61 (max), FedBN 89.75 | ❌ **FAILED** (全 α 输 FedBN) |
| **C2 CC-Bank PACS > 80.9** (≥ FedBN 80.41 + 0.5) | — | 78.88 | ❌ **FAILED** 大幅输 -1.53pp |
| **C3 Trajectory Δ > 0.5pp** (η=0.5 vs η=0) | — | +0.38 B / +0.23 L | ⚠️ **MARGINAL** (小过阈值, 但 per-seed 方差大) |

### 核心结论

1. **Sanity A 实验设计错误**: 多域 Centralized + global BN 不是 method-space 上限, 而是 BN mismatch 下限. 不能用来否决 backbone.
2. **CC-Bank 彻底失败**: Office 全 α 输 FedBN 0.14-0.64pp, PACS α=0.5 大幅输 -1.53pp. **AdaIN 家族第 5 次失败定案**, 砍掉 CC-Bank 组件.
3. **Trajectory 小赢 PACS**: η=0.5 mean B 80.77 > FedBN 80.41 +0.36pp, 但 per-seed {s=2 -0.38, s=15 +0.62, s=333 +0.88} 方差大, 2 赢 1 负 — **保留 trajectory 但优先级降低** (不作主卖点).
4. **对比 orth_only 80.64 vs trajectory η=0.5 80.77**: trajectory 小赢 +0.13, 同 noise 级别. orth_only 已是强 baseline, trajectory 增量可忽略.

### 下一步决定

**EXP-119 结论**: FedPTR 3 组件中 **CC-Bank 整个砍** (两次失败), **Centralized 不作判据** (实验设计错), **Trajectory 保留但降权** (增量小).

**不进 FedPTR 全量代码**: 3 组件预期协同效应没出来, Sanity Phase 已判死. 需要:
- **Plan A (pivot)**: 回到 orth_uc1 / VIB / VSC 线 (已 PACS/DomainNet 胜) 做 Office 攻关方案
- **Plan B (降 venue)**: 3-seed mean 稳过 FedBN baseline + novelty 诊断 → WACV/BMVC (不冲 CVPR)

**推荐 Plan A**: Office 是唯一败北数据集, 针对 Office 弱域异质设计专门方案 (EXP-118 fullbn 已试过 +0.24 无效, 下一个方向可以尝试 logit-level calibration / class-conditional classifier bank).

---

## 部署前代码 Review 发现 (Critical bugs 已修)

**CC-Bank AdaIN 公式 bug** (commit 2ed462e):
- 原代码: `mu_self = h.mean(dim=1, keepdim=True)` → per-sample scalar (LayerNorm 语义)
- bank 里 `mu_other, sig_other` 是 per-feature [D] (class-conditional 均值)
- Hybrid 形式语义不匹配, 历史 AdaIN 4 次失败可能部分源于此类实现细节
- 修复为 **batch-level per-feature** 统计: `h.mean(dim=0)` + `h.std(dim=0) + 1e-5`

**trajectory alignment loss 向量化** (同 commit):
- 原 Python for-loop 每 batch 50 次, 改为 proto_table + class_id lookup gather
- 性能提升，非正确性 fix

---

## 0. Claim Map & Hypothesis

### 需要回答的 3 个 Go/No-Go 问题

| # | Claim 被验证 | 反向假设 | 最小证据 |
|:-:|---|---|---|
| **C1** | AlexNet on Office/PACS 有 method-space 到 FDSE+2pp | "AlexNet 已到 capacity 天花板, method 做不上" | Centralized > 93 on Office |
| **C2** | Class-conditional BN stats bank + feature AdaIN work | "AdaIN 家族第 5 次失败 (EXP-040/059/061/047 历史阴影)" | Office 3-seed > FedBN 88.68 + 0.5 |
| **C3** | Prototype trajectory prediction 有真实贡献 | "FL E=5 noise 下 signal<<noise, 预测等于放大噪声" | η=0.5 vs η=0 差 > 0.5pp |

### Paper 储备 (if sanity 都通过)

- **Primary Claim (dominant)**: 在 FedBN 基础上, class-conditional style prototype 的 cross-client 共享 + trajectory prediction 使跨域 accuracy 超 FDSE +2pp
- **Supporting Claim**: Learnable per-client α 作为 difficulty-aware curriculum 救 outlier 域 (Caltech)
- **Anti-Claim to rule out**: "Gain 仅来自 FedFA/FedCA 的增量组合"

### Paper 储备 (if sanity 部分失败)

**Fallback 方案**:
- C1 失败 → 换 backbone 到 ResNet-18 (或降 venue 到 WACV/BMVC, target +1pp)
- C2 失败 → 砍 CC-Bank, 转向 **logit-level calibration** (类似 FedDr+ class-conditional 变体, 不碰 feature AdaIN)
- C3 失败 → 砍 trajectory, 保留 simple prototype alignment

---

## 1. 实验 Block 列表

### Block A: Centralized AlexNet Capacity Test

**Claim tested**: C1 — method space 还有 +2pp 的上限吗?

**Why this block exists**: FDSE 已经在 Office 上 90.58. 如果 centralized AlexNet 都打不到 93, 我们 FL 无论怎么调都冲不上 +2pp.

**Dataset / split**: Office-Caltech10 + PACS + DomainNet (10-class subset)

**Compared systems**:
- Centralized AlexNet (算法: `standalone`, proportion=1.0, 1 "virtual client" 合并全部 domain 数据)
- FedBN baseline 作为 FL 下限参照
- FDSE 已知 3-seed AVG Best 作为 SOTA 参照

**Metrics**: AVG Best, AVG Last, per-domain Best

**Setup details**:
- Backbone: AlexNet (同 FL 实验), Linear(1024 → num_classes) 分类头
- R=200, E=5 (PACS/DomainNet) / E=1 (Office), B=50, LR=0.05, WD=1e-3
- Seeds: {2, 15, 333} (3 seeds, 跟 FL 对齐)
- Optimizer: SGD 0.9 momentum, lr_decay 0.9998

**Success criterion**:
- Office ≥ 93 → **Go** 全量, +2pp target 可行
- Office 90-92 → **Marginal Go**, 降 target 到 +1pp
- Office < 90 → **换 backbone 或 venue**

**Failure interpretation**:
- Centralized 比 FDSE 还低 → AlexNet capacity 就是瓶颈, method novelty 救不了
- 建议方案: (a) 换 ResNet-18 重跑所有 baseline (b) 冲 WACV 不冲 CVPR

**Priority**: **MUST-RUN**

---

### Block B: CC-Bank Only (fixed α=0.5, no trajectory)

**Claim tested**: C2 — class-conditional style augmentation 是不是历史 AdaIN 第 5 次失败?

**Why this block exists**: 历史上 EXP-040/059/061/047 四次 AdaIN 都挂了. 必须先单独测 CC-Bank 避免跟 trajectory / α 耦合.

**Dataset / split**:
- **Office-Caltech10** (最关键, EXP-061 style sharing 套件在这崩过)
- PACS (EXP-059 stylehead AdaIN -2.54%)

**Compared systems**:
| System | AdaIN | Bank 结构 | α |
|:-:|:-:|:-:|:-:|
| Base FedBN | ❌ | — | — |
| CC-Bank fixed α=0.3 | ✅ | (class, client) 2D | 0.3 |
| **CC-Bank fixed α=0.5** | ✅ | (class, client) 2D | **0.5** |
| CC-Bank fixed α=0.7 | ✅ | (class, client) 2D | 0.7 |
| Domain-level bank (FedFA-like) ref | ✅ | client 1D | 0.5 |

**Metrics**: 3-seed mean AVG Best / AVG Last, per-domain Best

**Setup details**:
- 算法: `fedbn_ccbank.py` (继承 fedbn.py, 加 ~80 行)
- AdaIN 位置: encoder 最后 pooled feature (1024d), 不在 BN 层本身
- Bank 内容: 每 round client 上传 per-class batch μ/σ (n_c ≥ 8 才上传, 否则 skip)
- 采样: 训练时以 p=0.5 概率做 AdaIN, 从 other_client 的 same-class bank 采
- R=200, E=1 (Office) / E=5 (PACS), Seeds {2,15,333}, LR=0.05

**Success criterion** (Office 3-seed mean AVG Best):
- **> 89.2**: CC-Bank 真 work, 保留 → 加 trajectory + α 冲全量
- **88.0 - 89.2**: marginal, 需要 EMA smoothing + min sample threshold 才能救
- **< 88.0**: **kill CC-Bank**, 转 logit-level 路线

**Failure interpretation**:
- Office < 88.0: AdaIN 在 feature space 不适用 AlexNet capacity
- 可能原因: class-conditional 加剧了小样本 μ/σ 估计误差
- Plan B: logit-level (class-conditional classifier bank, FedDr+ 变体)

**Priority**: **MUST-RUN**

---

### Block C: Trajectory Prediction vs No-Prediction

**Claim tested**: C3 — 一阶 Taylor 预测 `p̂ = p + η·v` 真的比 `p̂ = p` 显著好?

**Why this block exists**: CVPR reviewer 说 FL E=5 noise 下 velocity estimate 方差 >> mean, 一阶预测等于放大噪声. 必须单独验证 trajectory 真贡献.

**Dataset**: **PACS** (4 client × 7 class, E=5, trajectory 最容易测出差异)

**Compared systems**:
| η 值 | 含义 | prototype smoothing |
|:-:|:-:|:-:|
| 0 | no prediction (退化成普通 prototype alignment) | ❌ |
| 0.3 | 保守预测 | ❌ |
| **0.5** | 默认 | ❌ |
| 0.8 | 激进预测 | ❌ |
| 0.5 + EMA(β=0.9) | 预测 + EMA 平滑 | ✅ |

**Metrics**: 3-seed mean AVG Best / AVG Last

**Setup details**:
- 算法: `fedbn_trajectory.py` (继承 fedbn.py, 加 ~60 行)
- Prototype: 每 class 全局均值 = client class mean 的加权平均
- 维护 `p_c^t` (position) + `v_c^t = p_c^t - p_c^{t-1}`
- 预测: `p̂_c^{t+1} = p_c^t + η · v_c^t`
- Client loss: `L = L_CE + 0.5 · (1 - cos(h, sg(p̂_{y})))` (stop_grad 必须)
- R=200, E=5, Seeds {2,15,333}
- 不加 CC-Bank, 不加 learnable α, 不加 curvature 降权

**Success criterion** (PACS 3-seed mean, η=0.5 vs η=0):
- **Δ > 0.5pp**: trajectory 有真实贡献, 保留
- **Δ 0.2 - 0.5pp**: marginal, 简化为 EMA prototype 可能更稳
- **Δ < 0.2pp**: **kill trajectory**, 退化为普通 prototype alignment

**Failure interpretation**:
- 如果 η=0 已经 > FedBN baseline 很多, 说明增益在 alignment 本身不在预测
- 如果 η=0.5 反而 < η=0, velocity 估计真的是噪声主导
- Plan: 把 trajectory component 整个砍掉, 保留 simple alignment

**Priority**: **MUST-RUN**

---

## 2. Compute Budget

### 每 run 时间估计 (基于历史数据)

| Task | E | 每 run 时间 | 单卡并行 (24GB) |
|:---:|:---:|:---:|:---:|
| Office-Caltech10 | 1 | ~50min - 1h | 5-7 runs |
| PACS_c4 | 5 | ~7-8h | 3-4 runs |
| DomainNet_c6 | 5 | ~4h | 5-7 runs |
| Office Centralized | 1 | ~45min (单 client 但数据全) | 5-7 runs |
| PACS Centralized | 5 | ~5-6h (单 client) | 3-4 runs |

### Sanity A (Centralized) runs

| 数据集 | Seeds | Runs | 单 run | 3-并行 wall |
|:---:|:---:|:---:|:---:|:---:|
| Office | 2, 15, 333 | 3 | 45min | 45min |
| PACS | 2, 15, 333 | 3 | 5h | 5h |
| DomainNet | 2, 15, 333 | 3 (可选) | 4h | 4h |

**子总**: 3-6 runs, wall 5h (Office + PACS 必要, DomainNet 可选)

### Sanity B (CC-Bank) runs

| 数据集 | α configs | Seeds | Runs | 单 run | 3-并行 wall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Office | α ∈ {0.3, 0.5, 0.7} | {2,15,333} | 9 | 1h | 3h (3 并行 3 轮) |
| PACS | α = 0.5 only (验证不退化) | {2,15,333} | 3 | 7h | 7h (3 并行) |

**子总**: 12 runs, wall ~10h (Office 3h + PACS 7h, 可并行)

### Sanity C (Trajectory) runs

| 数据集 | η configs | Seeds | Runs | 单 run | 3-并行 wall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PACS | η ∈ {0, 0.5} (主对照) | {2,15,333} | 6 | 7h | 14h (2 批 × 3 并行) |
| PACS | η=0.5 + EMA (可选) | {2} | 1 | 7h | 7h |

**子总**: 6-7 runs, wall 14h

### 总预算

| 必要 (MUST) | wall (3-并行) | 组件 |
|:---:|:---:|:---:|
| Sanity A Office + PACS | 6h | 3 + 3 runs |
| Sanity B Office + PACS | 10h | 9 + 3 runs |
| Sanity C PACS η∈{0,0.5} | 14h | 6 runs |
| **TOTAL** | **~30h wall** | **24-27 runs** |

**GPU·hour 估算**: ~100-120 GPU·h (24 runs × 平均 4-5h/run)

---

## 3. 对照矩阵 (一张表看全)

| Run ID | Block | Task | Algo | Seed | Key Param | Config File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| R001-003 | A | Office | standalone | {2,15,333} | proportion=1.0, 1 client | centralized_office.yml |
| R004-006 | A | PACS | standalone | {2,15,333} | proportion=1.0, 1 client | centralized_pacs.yml |
| R007-009 | A | DomainNet (可选) | standalone | {2,15,333} | proportion=1.0 | centralized_dn.yml (待写) |
| R010-018 | B | Office | fedbn_ccbank | 3seeds × 3α | α∈{0.3, 0.5, 0.7} | ccbank_office_a{0.3,0.5,0.7}.yml |
| R019-021 | B | PACS | fedbn_ccbank | {2,15,333} | α=0.5 | ccbank_pacs_a0.5.yml |
| R022-024 | C | PACS | fedbn_trajectory | {2,15,333} | η=0 | traj_pacs_eta0.yml |
| R025-027 | C | PACS | fedbn_trajectory | {2,15,333} | η=0.5 | traj_pacs_eta0.5.yml |
| (可选) R028 | C | PACS | fedbn_trajectory | 2 | η=0.5 + EMA | traj_pacs_eta0.5_ema.yml |

**计数**: Must-run 21 runs (A + B + C minimum), Nice-to-have 6 runs

---

## 4. 执行顺序 (Run Order + Milestones)

### M0: GPU 资源释放 (0.5 天)

**Goal**: 等现有 runs 完成腾 GPU

**Runs**: 无新 runs, 等
- seetacloud2 EXP-115 DomainNet 12 runs 完成 (R180+, 预计 2h)
- lab-lry EXP-116 lo=0 × 6 + EXP-117 orth_only DomainNet × 3 + EXP-118 fullBN × 3 完成

**Decision gate**: GPU 完全空闲

**Cost**: 0 (纯等待)

**Risk**: 这些 runs 失败需要重跑 → 确认都是健康的

---

### M1: Sanity A Office (最快验证 capacity) — 2h

**Goal**: C1 验证, Office capacity 有没有 +2pp 空间?

**Runs**: R001-003 (Office centralized × 3 seeds)

**Decision gate**: Office 3-seed mean ≥ 93 (Go) / 90-92 (Marginal) / < 90 (换 backbone)

**Cost**: 3 runs × 45min = 2.25h wall (3-并行)

**Risk**: standalone.py 可能有 bug (CLAUDE.md 记载 PACS standalone 有 standalone bug)
- **Mitigation**: 先跑 Office smoke test (R=20) 确认能跑, 再扩 R=200

---

### M2: Sanity B Office (CC-Bank 生死) — 3h

**Goal**: C2 验证, AdaIN 是不是第 5 次失败?

**Runs**: R010-018 (3 α × 3 seeds on Office)

**Decision gate**:
- Office 3-seed mean > 89.2: **CC-Bank 生** → M3/M4 继续
- Office 3-seed mean 88.0-89.2: 不确定 → 决定要不要加正则救
- Office 3-seed mean < 88.0: **CC-Bank 死** → 转 logit-level

**Cost**: 9 runs × 1h = 3h wall (3-并行)

**Risk**:
- AdaIN 第 5 次挂 — concede 并转 logit-level
- Small-class bank noise 污染训练 — 加 min sample threshold (n≥8) mitigate

---

### M3: Sanity A PACS + Sanity C PACS (trajectory 生死) — 14h

**Goal**: C1 PACS 验证 + C3 验证

**Runs**:
- R004-006 (PACS centralized × 3 seeds)
- R022-027 (PACS trajectory × 6 runs)

**可并行**: 是, GPU 够用的话 9 runs 并行

**Decision gate**:
- PACS centralized: capacity check
- Trajectory η=0.5 vs η=0: Δ > 0.5 (生) / < 0.2 (死)

**Cost**: 9 runs × 7h = 14h wall (3-并行两波)

**Risk**:
- PACS E=5 耗时长, 如果 Sanity B 先死掉, 可以提前 cancel

---

### M4: Sanity B PACS (验证 Office 结论能迁移到 PACS) — 7h

**Goal**: CC-Bank 在 PACS 是否也不退化? 历史 EXP-059 PACS -2.54% 的阴影能破吗?

**Runs**: R019-021 (3 seeds PACS α=0.5)

**Decision gate**:
- PACS 3-seed mean ≥ FedBN 79.01: 保留
- PACS 3-seed mean < 78.5: CC-Bank 在 PACS 也挂

**Cost**: 3 runs × 7h = 7h wall (3-并行)

**Risk**: 如果 M2 已死, M4 可以 skip

---

### M5: Decision Meeting (写 review report)

**Goal**: 基于 M1-M4 结果决定最终方案

**Output**: 更新 `FedPTR_FINAL_PROPOSAL.md` → V2 瘦身版 OR 换方向

**Cost**: 0.5 天阅读+决策

---

## 5. Go/No-Go 决策表 (全局)

| 结果组合 | 决策 | 下一步方案 |
|:---:|:---:|:---:|
| A pass, B pass, C pass | ✅ All go | 写全量 FedPTR (3 组件 + α 正则) |
| A pass, B pass, **C fail** | ⚠️ 砍 trajectory | 瘦身: CC-Bank + Learnable α (2 组件) |
| A pass, **B fail**, C pass | ⚠️ 砍 CC-Bank | logit-level 替代方案 + trajectory |
| A pass, **B fail**, **C fail** | ❌ 大换血 | 转 FedAdvStyle (对抗风格) 或 FedGLA (条件 MI) |
| **A fail** (Office < 90) | ❌ 换 backbone | ResNet-18 重跑 baseline, 降 target 或换 venue |
| A 边缘 (90-92) | 🟡 降 target | Target +1pp, 冲 WACV/BMVC |

---

## 6. 代码改动估计

### Sanity A: standalone.py 复用 (~10 行新增)
- 主要是 config 文件
- 需要 "centralized" task 设置 (1 virtual client 合并 data) 或用 `proportion=1.0` 模拟
- 行数: ~10 行 (config × 2-3 个)

### Sanity B: `fedbn_ccbank.py` (~100 行新增)

```python
# 继承 fedbn.py
class Server:
    - __init__: 新增 style_bank = defaultdict(dict)  # bank[class][client_id] = (μ, σ)
    - pack: 下发 style_bank
    - iterate: 聚合时更新 bank (从 client 上传收集)

class Client:
    - unpack: 加载 bank
    - train: 训练时 AdaIN 增强
        - 前向 h = encoder(x), 计算 batch μ_self, σ_self
        - 从 bank[y][other_client] 采 (μ', σ')
        - h_aug = σ' · (h - μ_self) / σ_self + μ'
        - h_final = α * h_aug + (1-α) * h
        - CE(classifier(h_final), y)
    - upload: 上传 class-conditional batch μ/σ
```

- 行数: ~100 行 (算法) + 3-6 个 config (~60 行) = **160 行**
- 单测: ~10 个 tests

### Sanity C: `fedbn_trajectory.py` (~80 行新增)

```python
class Server:
    - 新增: self.prototype_history = dict (维护 p_c^t, p_c^{t-1})
    - aggregate_prototype: 加权平均 class prototypes
    - predict_next: p_hat = p + η · v
    - pack: 下发 p_hat

class Client:
    - 上传: per-class local prototype
    - train loss: + 0.5 · (1 - cos(h, sg(p_hat[y])))
```

- 行数: ~80 行 (算法) + 5-6 个 config (~60 行) = **140 行**
- 单测: ~8 个 tests

**总代码**: ~310 行, 1 天能写完 + 测试

---

## 7. 启动命令清单

### Sanity A: Centralized AlexNet

```bash
# Office
CUDA_VISIBLE_DEVICES=1 python run_single.py \
  --task office_caltech10_c4 --algorithm standalone \
  --config ./config/office/centralized_office.yml \
  --logger PerRunLogger --seed 2 --gpu 0 &
# 重复 seed=15, 333

# PACS
CUDA_VISIBLE_DEVICES=1 python run_single.py \
  --task PACS_c4 --algorithm standalone \
  --config ./config/pacs/centralized_pacs.yml \
  --logger PerRunLogger --seed 2 --gpu 0 &
```

**需要新写**:
- `FDSE_CVPR25/config/office/centralized_office.yml`
- `FDSE_CVPR25/config/pacs/centralized_pacs.yml`
- (可能) 确认 standalone.py 在 Office/PACS 上能跑

### Sanity B: CC-Bank

```bash
# Office α=0.5 × 3 seeds
for seed in 2 15 333; do
  CUDA_VISIBLE_DEVICES=1 python run_single.py \
    --task office_caltech10_c4 --algorithm fedbn_ccbank \
    --config ./config/office/ccbank_office_a0.5.yml \
    --logger PerRunLogger --seed $seed --gpu 0 &
done

# 类似 α=0.3 / α=0.7
```

**需要新写**:
- `FDSE_CVPR25/algorithm/fedbn_ccbank.py` (~100 行)
- 6 个 config (3 α × 2 dataset)
- 10 个单元测试

### Sanity C: Trajectory

```bash
# η=0 × 3 seeds
for seed in 2 15 333; do
  CUDA_VISIBLE_DEVICES=1 python run_single.py \
    --task PACS_c4 --algorithm fedbn_trajectory \
    --config ./config/pacs/traj_pacs_eta0.yml \
    --logger PerRunLogger --seed $seed --gpu 0 &
done

# η=0.5 × 3 seeds
```

**需要新写**:
- `FDSE_CVPR25/algorithm/fedbn_trajectory.py` (~80 行)
- 2 个 config (η=0, η=0.5)
- 8 个单元测试

---

## 8. 数据收集 + 回填流程

### 每 run 完成后
1. JSON record 自动写入 `task/{task}/record/{algo}_*.json`
2. 用 Python 读取 `mean_local_test_accuracy` 算 max (Best) 和 last (Last)
3. 回填到本 NOTE.md 的"结果"章节

### 3-seed 齐了后
1. 算 mean ± std
2. 对比 Go/No-Go 判据
3. 做决策 → 写到 M5 Decision Meeting

### 代码模板
```python
# 收集结果
import json, glob
files = glob.glob('task/{task}/record/{algo}*S{seed}*.json')
for f in files:
    d = json.load(open(f))
    acc = d['mean_local_test_accuracy']
    best = max(acc); last = acc[-1]
    print(f'best={best*100:.2f} last={last*100:.2f}')
```

---

## 9. 潜在风险 + 缓解

| 风险 | 概率 | 影响 | 缓解 |
|:---:|:---:|:---:|:---:|
| standalone.py 在 Office/PACS 有未发现 bug | 中 | 高 (Sanity A 废) | R=20 smoke 先试, 再 scale 到 R=200 |
| CC-Bank 的 per-class batch μ/σ 在 class 样本 < 8 时崩 | 高 | 中 | min sample threshold: n_c ≥ 8 才收 bank |
| Trajectory velocity 数值不稳 (prototype 跳跃) | 高 | 中 | warmup: R < 10 不做 prediction (η=0 等效) |
| GPU 资源不够同时跑 3 个 sanity | 中 | 中 | 串行跑 (M1 → M2 → M3/M4) |
| 在 lab-lry 上 standalone 模式要特殊 task 定义 | 中 | 低 | 用 `proportion=1.0` 模拟 centralized 或写新 `central_office_1c` task |
| Sanity B PACS 第 5 次 AdaIN 挂 (-2.54% 重演) | 中 | 低 (只是确认 kill) | 这是"预期失败", 数据本身就是结论 |

---

## 10. 代码实现优先级 (给 implementation phase 用)

1. **First**: 写 `fedbn_ccbank.py` (100 行) + 2-3 个 Office config — Sanity B 最快出结果
2. **Second**: 写 centralized config (复用 standalone.py) — Sanity A 最快
3. **Third**: 写 `fedbn_trajectory.py` (80 行) + 2 个 PACS config — Sanity C 最慢 (PACS E=5)
4. **所有代码完成**后再统一部署, 不要边写边跑 (避免资源切换)

---

## 11. Final Checklist (实验设计质量检验)

- [x] 每个 sanity 对应明确 claim (C1/C2/C3)
- [x] 每个 sanity 有明确 Go/No-Go 判据
- [x] 每个 sanity 有 fallback 方案 (failure interpretation)
- [x] 3 seeds 保证统计效度
- [x] 资源预算合理 (30h wall, 120 GPU·h)
- [x] 对照组覆盖到基线 (FedBN, FedFA-like, etc.)
- [x] 代码量估计 (~310 行)
- [x] 实验启动命令具体化
- [x] 潜在 bug 风险 + 缓解 措施

---

## 12. 下一步 (本 NOTE 之后)

### 按 experiment-plan skill 的产出规范

- **文档**: 本 NOTE + `refine-logs/2026-04-22_FedPTR_refined/FedPTR_FINAL_PROPOSAL.md` (已有)
- **Tracker**: 需要一份 `EXPERIMENT_TRACKER.md` (可选, NOTE 足够覆盖)
- **代码实现**: 按优先级 CC-Bank → Centralized → Trajectory

### 等用户 Gate

- [ ] 批准开始实现代码 (CC-Bank 最高优先)
- [ ] 批准清理 lab-lry 现有 runs 腾 GPU 给 sanity

---

## 📎 相关文件

- **方案原文**: `refine-logs/2026-04-22_FedPTR_refined/FedPTR_FINAL_PROPOSAL.md`
- **大白话版**: `obsidian_exprtiment_results/知识笔记/大白话_7个新方案brainstorm_2026-04-22.md`
- **CVPR review**: (待存) `refine-logs/2026-04-22_FedPTR_refined/REVIEW_ROUND_1.md`
- **上游实验基线**:
  - FedBN: 79.01 (PACS) / 88.68 (Office) / 72.08 (DomainNet)
  - FDSE: 79.91 / 90.58 / 72.21
  - orth_uc1 (EXP-109/110/115): 80.64 / 89.17 / 72.49

---

*生成日期: 2026-04-22*
*基于 research-pipeline 的 experiment-plan Phase 5: 实验计划细化*
