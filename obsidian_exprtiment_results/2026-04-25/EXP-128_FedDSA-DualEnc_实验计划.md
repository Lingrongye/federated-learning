# FedDSA-DualEnc 实验计划

**Problem**: 联邦跨域学习 (Feature Skew/Domain Shift) 下,3-seed mean Best accuracy 必须在 PACS + Office-Caltech10 同时**实测胜过 FDSE 本地复现 baseline** (PACS > 79.91, Office > 90.58),而非论文报告数。
**Method Thesis**: 在 AlexNet from scratch + 联邦 personalization 设定下,通过 (1) 双 encoder 物理解耦 (E_sem 512d + E_sty 16d VAE) + (2) AdaIN-style modulation decoder + (3) 跨 client 风格仓库 (μ, σ) 共享 + (4) cycle anatomy consistency (L_saac),实现"风格资产化跨域增强"。
**Date**: 2026-04-25
**Framework**: FDSE_CVPR25 (flgo-based,**非 PFLlib**)
**EXP ID**: EXP-128 ~ EXP-135 (multi-stage)

---

## 一、Claim Map

### Primary Claim (C1) — 主创新
> **在 AlexNet from scratch + FL personalization 设定下,首次将"跨 client 风格仓库 + cycle anatomy consistency"组合用于跨域分类,3-seed mean Best 同时打过 FDSE baseline (PACS > 79.91, Office > 90.58)。**

| 关键证据 | 防御 |
|---|---|
| 3-seed mean PACS AVG Best > 79.91 (硬指标) | Block 1 主表 |
| 3-seed mean Office AVG Best > 90.58 (硬指标) | Block 1 主表 |
| Per-domain 至少 2/4 域超过 FDSE 同 seed 复现 | Block 1 主表 |
| 跨 client 风格 swap 可视化(4×4 grid)显示风格真换了 | Block 6 |
| Cycle anatomy 一致性可视化:swap 后 z_sem 几乎不变 | Block 6 |

### Supporting Claim (C2) — 方法论贡献
> **首次给"双 encoder 解耦 + 风格 swap + cycle 一致性" design family 提供 probe-based ablation methodology:每个 loss 移除后报告 probe 数字 (z_sty 类 probe / z_sem 域 probe / SVD effective rank / 重建 PSNR),把"loss 互锁防坍缩"从口述论证升级为可测假设。**

| 关键证据 | 防御 |
|---|---|
| 去 L_saac 后 z_sty 类 probe 飙升 > 50% (类别泄漏) | Block 2 |
| 去 L_dsct 后 z_sty SVD ER 暴跌 < 5 (mode collapse) | Block 2 |
| 去 L_rec 后 decoder 风格多样性 → 0 (decoder 偷懒) | Block 2 |
| Full method 同时维持 z_sty class probe < 25%, z_sem domain probe < 50% | Block 7 |

### Anti-Claim 必须排除 (A1-A4)

| Anti-claim | 排除证据 |
|---|---|
| **A1: 收益只是来自 decoder 多出来的 ~3M 参数容量** | Block 2 ablation: 去 L_saac 留 decoder + L_rec → acc 应明显下降 |
| **A2: 收益只是来自类似 MixStyle 的特征扰动增强** | Block 4: 跟 PARDON/FISC (混合空间风格 swap 无 cycle) 对比,我们 cycle 必须显著领先 |
| **A3: 收益只是来自 VAE 正则** | Block 3: 单加 VAE bottleneck 不加 swap/cycle → 无显著提升 |
| **A4: 跟 BiProto 一样最终 mode collapse,只是 R200 没看出来** | Block 5: 全程监控 z_sty SVD ER,必须始终 > 10;BiProto 实测 ER 2.73 作为反例 |

---

## 二、Paper Storyline

### 主论文必证(MAIN PAPER)
- **C1 主指标**: PACS + Office 双数据集打过 FDSE
- **C2 ablation**: 4-loss 各自必要性 + 对应 probe 飙升
- **跨 client 必要性**: 没有跨 client 风格仓库就退化

### 附录补充(APPENDIX)
- 4×4 风格 swap 可视化 + cycle 验证图
- 完整 probe 曲线(每 10 round 数据点)
- AdaIN 注入位置消融(浅/中/深 vs 全部)
- λ 系数 grid sweep(λ_saac ∈ {0.5, 1.0, 2.0})

### 故意不做(CUT)
- DomainNet:CLAUDE.md §0 硬规定不换数据集
- ResNet-18/ViT 替代 backbone:超出本论文 scope
- Image-level MAE:小数据 + AlexNet 必崩,不浪费 GPU
- Differential Privacy 噪声分析:超出 scope

---

## 三、Experiment Blocks (5 个 must-run + 2 个 nice-to-have)

### Block 1: Main Anchor Result (主表) — MUST-RUN ★

- **Claim tested**: C1 (3-seed mean Best 双数据集打过 FDSE)
- **Why this block exists**: CLAUDE.md §0 硬指标,这是论文存在的前提
- **Dataset / split / task**:
  - PACS_c4 (Photo, Art, Cartoon, Sketch),personalization 4 client × 4 域
  - Office_c4 (Amazon, Caltech, DSLR, Webcam),personalization 4 client × 4 域
- **Compared systems** (3 seeds {2, 15, 333} × 2 datasets):
  - FedAvg(基线下界)
  - FedBN(域适应轻量基线)
  - FedProto(原型派代表)
  - FPL(原型 + InfoNCE)
  - **FDSE**(必比 baseline,擦除派 SOTA)
  - orth_only(我们 R200 现状,sanity 比对)
  - **FedDSA-DualEnc full**(我们方案)
  - BiProto(失败基线,作为反面参考)
- **Metrics** (按重要度):
  1. **3-seed mean AVG Best** (主指标,胜负判决)
  2. 3-seed mean AVG Last
  3. 3-seed mean ALL Best/Last (paper Table 时一起报)
  4. Per-domain Best/Last(看是否有 outlier 域)
  5. 3-seed std (稳定性)
- **Setup details**:
  - Backbone: AlexNet from scratch(沿用 FDSE 同款)
  - Z dim: z_sem 512d, z_sty 16d (VAE μ + log_var)
  - Optimizer: SGD lr=0.005 momentum=0.9 wd=5e-4 (沿用 orth_only 已调通超参)
  - Rounds: 200, local_epoch: 5 (PACS) / 1 (Office,沿用 EXP-119 设定)
  - Loss 系数: λ_rec=0.001, λ_saac=1.0, λ_dsct=0.01, λ_kl=0.01 (KL warmup R0-R10)
  - 聚合: encoder.* + E_sem.* + decoder.* + classifier.* → FedAvg, E_sty.* + bn.* → 本地
  - Style bank: server 维护全局 (μ, σ) buffer,每轮广播别 client 的统计量
- **Success criterion**:
  - **必须**: PACS 3-seed mean AVG Best > 79.91 AND Office 3-seed mean AVG Best > 90.58
  - 期望:per-domain 至少 2/4 域超过 FDSE 同 seed 复现,std < 1.0
- **Failure interpretation**:
  - 仅 PACS 过:走"PACS 强、Office 待补"路线,先 ablation 找 Office 失败原因(大概率 Caltech outlier)
  - 仅 Office 过:违背 PACS 现状(orth_only 已 80.64),debug DualEnc 是否破坏了 orth_only 已有优势
  - 都没过:**直接 kill,不浪费后续 GPU**,重新评估方向
- **Table / figure target**: Paper Table 1 (主表,所有 baseline 对比)

---

### Block 2: Novelty Isolation — 4-Loss Ablation — MUST-RUN ★

- **Claim tested**: C2 + A1/A3 排除 (各 loss 必要性 + decoder 不是单独贡献者)
- **Why this block exists**: reviewer 第一句话会问"哪个 loss 最重要?去掉哪个就崩?"。这是 paper 核心 Method Section 后半段
- **Dataset / split / task**: Office_c4 (更难数据集,差异更显著),seed=2, R200
- **Compared systems** (8 个变体):
  | # | 变体 | L_CE | L_rec | L_saac | L_dsct | L_kl | 假设 |
  |---|---|---|---|---|---|---|---|
  | 1 | Full method | ✅ | ✅ | ✅ | ✅ | ✅ | 主对照,基准 |
  | 2 | -L_saac | ✅ | ✅ | ❌ | ✅ | ✅ | acc ↓, **z_sty 类 probe ↑** |
  | 3 | -L_rec | ✅ | ❌ | ✅ | ✅ | ✅ | acc 微 ↓, decoder 训不动 |
  | 4 | -L_dsct | ✅ | ✅ | ✅ | ❌ | ✅ | **z_sty SVD ER ↓**,可能 collapse |
  | 5 | -L_saac -L_rec | ✅ | ❌ | ❌ | ✅ | ✅ | 等价 orth_only + VAE,看是否退化 |
  | 6 | Decoder-only (CE + rec only) | ✅ | ✅ | ❌ | ❌ | ✅ | 排除 A1: 单 decoder 容量不够 |
  | 7 | orth_only baseline | ✅ | ❌ | ❌ | ❌ | ❌ | 我们当前现状,sanity |
  | 8 | + L_orth + HSIC (BiProto-style 显式正交) | ✅ | ✅ | ✅ | ✅ | ✅ | + ❌ A4: 是否互补 |
- **Metrics**:
  - acc (per-domain Best/Last + AVG)
  - **probe 数字**(★ 关键):z_sty class probe / z_sty domain probe / z_sem class probe / z_sem domain probe (每 10 round)
  - z_sty SVD effective rank (每 round)
  - decoder 风格多样性 (4 张同图 4 个不同风格 swap 的 pairwise L2)
  - 重建 PSNR (每 round)
  - 各 loss 梯度 norm (训练时实时记)
- **Setup details**: 同 Block 1,只改 algo_para 关闭对应 loss
- **Success criterion**:
  - **必须**:Full > -L_saac > -L_rec ≈ -L_dsct > orth_only (acc 单调)
  - **必须**:去 L_saac 后 z_sty 类 probe **必须从 < 25% 飙升到 > 50%**(否则 L_saac 没起作用)
  - **必须**:去 L_dsct 后 z_sty SVD ER **必须从 > 10 暴跌到 < 5** (否则 L_dsct 没起作用)
  - 期望:Full vs Decoder-only 差距 ≥ 1.0pp(排除 A1)
- **Failure interpretation**:
  - probe 数字不变化 → loss 没起作用,要 debug 实现
  - acc 单调性反转 → 设计有问题,可能某些 loss 互相破坏
- **Table / figure target**: Paper Table 2 (loss ablation) + Figure 2 (probe 曲线)

---

### Block 3: Simplicity Check — Extras Don't Help — MUST-RUN ★

- **Claim tested**: A1/A3 排除 + 论证当前方法已足够
- **Why this block exists**: reviewer 会问"为什么不加 MAE / R-Drop / explicit orth?"这块直接回应
- **Dataset / split / task**: Office_c4, seed=2, R200
- **Compared systems** (4 个):
  - Full method (基准)
  - **+ Feature-level mask + R-Drop** (在 z_sem 上做 50% dropout,两次前向 KL 一致性)
  - **+ Image-level CutMix** (训练时 50% 概率 CutMix)
  - **+ L_orth (cos²) + HSIC** (BiProto-style 显式解耦正则)
- **Metrics**: AVG Best/Last, probe 数字, 训练稳定性 (loss curve smoothness)
- **Setup details**: 同 Block 1 + 加上对应增强项
- **Success criterion**:
  - **期望**: 加任何 extras 后 acc 不显著提升 (ΔAVG Best < +0.3pp) 或反而下降
  - **如果 extras 反而涨 ≥ 0.5pp**:必须重新评估主方法,可能漏了关键机制
- **Failure interpretation**:
  - 加 R-Drop 涨明显 → z_sem 表征容量是真瓶颈 → 论文要承认
  - 加 L_orth + HSIC 涨明显 → cycle 不够,需要显式正交补充 → 改主方法
- **Table / figure target**: Paper Table 3 (Simplicity ablation)

---

### Block 4: Cross-Client Necessity Check — MUST-RUN ★

- **Claim tested**: C1 cross-client 部分 + A2 排除 (不是 MixStyle-style 特征扰动)
- **Why this block exists**: 防 reviewer 说"intra-client 风格交换就够了,跨 client 没意义"
- **Dataset / split / task**: Office_c4, seed=2, R200
- **Compared systems** (4 个):
  - **A. No swap** (z_sty_swap = z_sty,L_saac 退化为自一致)
  - **B. Intra-client swap** (从同 client 内 batch 别样本采 z_sty,CDDSA 等价)
  - **C. Cross-client mean** (用别 client (μ, σ) 均值,FISC 等价)
  - **D. Cross-client random linear (我们)** (server bank 任意线性组合 α ~ U(-1,1))
- **Metrics**: AVG Best/Last + 风格仓库通信代价 (KB/round)
- **Setup details**: 同 Block 1,只改 style bank 采样函数
- **Success criterion**:
  - **必须**: D > C > B > A (cross-client random > cross-client mean > intra-client > no swap)
  - **期望**: D vs B 差距 ≥ 1.0pp (跨 client 必要性证据)
- **Failure interpretation**:
  - D ≈ B → 跨 client 没意义,论文卖点崩,改投"intra-client cycle"路线
  - D < B → 风格仓库设计有 bug
- **Table / figure target**: Paper Table 4 (Style swap source ablation)

---

### Block 5: BiProto Mode Collapse Regression Test — MUST-RUN ★

- **Claim tested**: A4 排除 (我们没在 R200 之后 mode collapse)
- **Why this block exists**: BiProto 实测 z_sty SVD ER = 2.73 (mode collapse),我们必须证明 cycle 替代 EMA 自循环后,collapse 不复发
- **Dataset / split / task**: Office_c4, seed=2, **R400 加长** (R200 不够看长期稳定性)
- **Compared systems**:
  - **BiProto (反例)**:已有结果 ER = 2.73 @ R200
  - **Full method**:每 10 round 算 z_sty SVD ER
  - **orth_only (sanity)**:已知 z_sty 没有 EMA 循环,ER 应该健康
- **Metrics**:
  - z_sty SVD effective rank vs round (折线图)
  - z_sty class probe vs round (类别泄漏监控)
  - z_sem domain probe vs round (风格泄漏监控)
  - 训练 loss 曲线(看是否后期发散)
- **Setup details**: 同 Block 1,但 R=400
- **Success criterion**:
  - **必须**: Full method z_sty SVD ER 全程 > 10
  - **必须**: z_sty class probe 全程 < 30%
  - **期望**: ER 在 [15, 25] 区间稳定
- **Failure interpretation**:
  - ER 在 R200 后开始下降 → cycle 强度不够,可能要加 KL/L_dsct 权重
  - class probe 飙升 → cycle 没生效,bug 检查
- **Table / figure target**: Paper Figure 3 (Stability over rounds)

---

### Block 6: Style Swap Visualization (Qualitative) — MUST-RUN ★

- **Claim tested**: C1 可视化证据 + A2 排除 (decoder 真在用 z_sty,不是 trivial)
- **Why this block exists**: 论文必须能让 reviewer "看到风格换了" — 否则任何数字都缺直观信任
- **Dataset / split / task**: PACS + Office,best ckpt (Block 1 选最佳 seed)
- **What to dump**:
  - **4×4 风格 swap grid**:行 = 原图所在域,列 = 注入风格所在域,对角线 = 自重建,非对角 = 跨域 swap
  - **Cycle 验证图**:原图 → swap 图 → cycle 还原图,3 列对比
  - **Decoder 偷懒检测图**:同一 z_sem + 4 个不同 z_sty (来自 4 域) → 4 张重建图,pairwise L2 数值标注
- **Setup details**: 离线脚本读 ckpt,在 val set 上采样
- **Success criterion**:
  - **必须**: 4×4 grid 中非对角图风格明显与列方向风格相似 (photo 列偏自然色,sketch 列偏黑白线条)
  - **必须**: 4×4 grid 中行方向语义保持 (狗还是狗,不变成猫)
  - **必须**: cycle 还原图 ≈ 原图 (PSNR > 18 dB)
  - **必须**: 4 张不同风格重建图 pairwise L2 > 0.05 (decoder 不偷懒)
- **Failure interpretation**:
  - 风格不变 → decoder 没真用 z_sty,需要加大 λ_rec 或调整 AdaIN 注入位置
  - 语义变了 → cycle 弱,需要加大 λ_saac
- **Table / figure target**: Paper Figure 1 (Method 总览图) + Figure 4 (qualitative results)

---

### Block 7: Probe-Based Decoupling Verification — MUST-RUN ★

- **Claim tested**: C2 (probe methodology) + 解耦真实性
- **Why this block exists**: 这是我们 second novelty,把"loss 互锁防坍缩"从 claim 升级为 measurable evidence
- **Dataset / split / task**: PACS + Office,seed=2,R200
- **What to measure** (4 个 probe,每 10 round):
  - **P1: z_sty class probe** (期望 < 25% / 风险 > 50%) — 风格码是否偷类别
  - **P2: z_sty domain probe** (期望 > 80%) — 风格码是否真捕捉到域
  - **P3: z_sem domain probe** (期望 < 50%) — 语义码是否含域信息
  - **P4: z_sem class probe** (期望 > 60%) — 语义码是否真分类有效
- **Probe MLP 设置**:
  - 2 层 MLP, hidden 256, ReLU
  - 训 5 epoch (Quick) / 10 epoch (Decisive at R200)
  - 冻住 backbone, 只训 probe 头
  - Sub-sample 200 examples per domain (~30s)
- **Compared systems**:
  - Full method (期望:P1<25%, P2>80%, P3<50%, P4>60%)
  - orth_only (期望:解耦不充分,P1, P3 居中)
  - BiProto (反例:大概率 P1 > 50% 类别泄漏)
- **Success criterion**: 至少 3/4 probe 满足期望,P1 < 25% 不可妥协
- **Failure interpretation**:
  - P1 高 → cycle 没把类别压出 z_sty,改大 λ_saac
  - P3 高 → z_sem 偷风格,需要在 z_sem 上加额外 regularizer
- **Table / figure target**: Paper Figure 5 (Probe 曲线 + Heatmap) + Table 5 (probe 终值汇总)

---

### Nice-to-Have Block 8: AdaIN 注入位置消融 — APPENDIX

- 浅层注入 / 中层注入 / 深层注入 / 全注入
- 风险:小动作,不影响主 claim,Office 数据小可能噪声大于信号
- 优先级低,只在 Block 1-7 完成后跑

### Nice-to-Have Block 9: λ 系数 grid sweep — APPENDIX

- λ_saac ∈ {0.5, 1.0, 2.0}, λ_rec ∈ {0.0005, 0.001, 0.005}
- 12 个组合 × 1 seed = 12 runs
- 优先级低,只为附录 robustness 表

---

## 四、Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | GPU-hours | Risk |
|---|---|---|---|---|---|
| **M0 Sanity** | 代码正确,所有 loss 跑得通,可视化输出正常 | 1 client × 5 round × Office | 4 loss 都有非零梯度,4×4 grid 输出非全黑 | 0.5 | 低 |
| **M1 Baseline 复用** | 沿用已有 orth_only / FDSE / BiProto / FedAvg / FedBN / FedProto / FPL 的 3-seed × 2 dataset 结果 | 0 (复用) | — | 0 | — |
| **M2 Pilot Office seed=2** | DualEnc 在 Office (更难) 跑通,看是否有破 90.58 苗头 | 1 run × Office × R200 | AVG Best > 88 (能涨过 orth_only 89.09 的 80%) | 5 | 中 |
| **M3 Pilot PACS seed=2** | DualEnc 在 PACS 跑通,看是否破坏 orth_only 现状 | 1 run × PACS × R200 | AVG Best > 78 (不掉太多) | 8 | 中 |
| **★ Decision Gate 1** | M2+M3 都通过 → 上 3-seed 主表;任一不通过 → kill 或 debug | — | — | — | — |
| **M4 主表 3-seed** | Block 1 完整 3-seed × 2 dataset | 6 runs (DualEnc only,baseline 已有) | PACS > 79.91 AND Office > 90.58 | 50 | 高 |
| **★ Decision Gate 2** | M4 双过 → 进入 ablation 阶段;不过 → 评估论文方向 | — | — | — | — |
| **M5 Block 2 (4-loss ablation)** | 8 个变体 × 1 seed × Office | 8 runs | 单调性 + probe 飙升 | 40 | 中 |
| **M6 Block 3+4 (simplicity + cross-client)** | 7 个变体 × 1 seed × Office | 7 runs | extras 不显著涨 + cross-client 必要 | 35 | 中 |
| **M7 Block 5 (R400 stability)** | DualEnc R400 长跑 + BiProto R400 反例 | 2 runs × Office | ER 全程 > 10 | 30 | 中 |
| **M8 Block 6+7 (可视化 + probe)** | 离线脚本 dump 4×4 grid + 4-probe 曲线 | 0 训练 (读 ckpt) | 风格真换 + P1<25% | 5 | 低 |
| **M9 Polish (Block 8+9 if time)** | AdaIN 位置 + λ grid | 12 runs | — | 60 | 低 |

**总计**: 必须 33 runs, ~170 GPU-hours; 含 Polish 45 runs, ~230 GPU-hours

**预估时间**(按 CLAUDE.md §17.8 greedy parallel 策略,2 GPU 同时):
- M0-M3 sanity + pilot: **2 天**
- M4 主表: **3-4 天**
- M5-M7 ablation: **2-3 天**
- M8 离线分析: **0.5 天**
- M9 polish (可选): **3 天**
- **总计 8-13 天**(含调试)

---

## 五、Compute and Data Budget

| 项 | 估值 |
|---|---|
| 总 GPU-hours (must-run) | ~170 |
| 总 GPU-hours (with appendix) | ~230 |
| 主力服务器 | seetacloud (RTX 4090 24GB) |
| 辅助服务器 | seetacloud2 (RTX 4090 24GB),lab-lry (双 3090) |
| 数据准备 | 已完成(PACS_c4 / Office_c4 在 FDSE_CVPR25/task/ 下) |
| 风格仓库存储 | 内存中维护,~250KB,不落盘 |
| 模型 ckpt 存储 | ~30MB/run × 33 runs ≈ 1GB |
| 可视化输出 | ~2MB/round × 8 round dump × 33 run ≈ 530MB |
| 人工评估 | 看 4×4 grid 判断风格真换否(~10 分钟/dataset) |
| **最大瓶颈** | **M4 主表 3-seed × 2 dataset × R200**,单 run 7-8h × 6 = 42-48h wall (greedy 2-GPU 后 ~24h) |

---

## 六、Risks and Mitigations

| Risk | 触发条件 | Mitigation |
|---|---|---|
| **R1: cycle 走 trivial 解** (z_sem 学成 image hash,L_saac 永远 0 但 acc 不涨) | L_saac < 0.001 全程 + acc 下降 | 强制 λ_CE = 1.0 + 监控 z_sem 类内/类间方差比;触发后改用 z_sem.detach()_after_E_sem 减弱 cycle |
| **R2: VAE posterior collapse** (z_sty → N(0,I),decoder 完全忽略) | KL loss → 0 + decoder 风格多样性 → 0 | KL warmup R0-R10 关闭,然后线性涨到 0.01;触发后 free-bits 技术 |
| **R3: decoder 训不动** (Office 2500 样本 + 256×256 重建) | recon PSNR < 12 全程 | 简化 decoder 至 2 conv block + 重建分辨率降至 128×128 |
| **R4: Caltech outlier 拖累 Office AVG** | Office AVG Best 卡在 89-90 上不去 | Caltech 客户端单独加 weight × 1.5,或 client-wise lr 调整 |
| **R5: Cycle 二次前向计算开销** | 单 epoch 时间翻倍,M4 6 runs 跑不完 | cycle frequency 改为每 2 batch 一次;或用 torch.compile 加速 |
| **R6: BiProto-style mode collapse 复现** | z_sty SVD ER 在 R100+ 暴跌 | 实时监控 ER,触发警报;调大 λ_dsct 至 0.05 |
| **R7: AdaIN 注入位置敏感** | full method acc 比 orth_only 反而低 | 先固定中层注入(标准位置),后续 ablation 再 sweep |
| **R8: Style bank 通信延迟** | flgo pack/unpack 序列化大 tensor 慢 | 先存 (μ,σ) 概要 (per-domain mean/std),不存 per-sample |
| **R9: 三个 seed mean 方差大无法 declare 胜过** | 3-seed std > 1.5 | 加 seed 到 5 个;或用 paired t-test |
| **R10: 8-13 天预算超支** | 实验 deadline | 砍 Block 8+9 nice-to-have;Block 4 简化为 3 变体 |

---

## 七、Implementation Spec(代码改动清单)

### 7.1 文件结构 (FDSE_CVPR25 框架,**非 PFLlib**)

```
FDSE_CVPR25/
├── algorithm/
│   ├── fdse.py                          [已存在] FDSE baseline
│   ├── feddsa.py                        [已存在] orth_only 现状
│   └── feddsa_dualenc.py                [★ NEW] Server + Client + 4 loss + style bank
├── model/
│   ├── alex_dse.py                      [已存在] FDSE 用 AlexNet
│   └── dualenc_alexnet.py               [★ NEW] AlexNet + E_sem(FC) + E_sty(VAE) + Decoder(AdaIN)
├── utils/
│   ├── probes.py                        [★ NEW] 4 probe + diagnostic metrics
│   └── visualize.py                     [★ NEW] 4×4 swap grid + cycle dump
├── config/
│   ├── pacs/feddsa_dualenc.yml          [★ NEW]
│   └── office/feddsa_dualenc.yml        [★ NEW]
└── scripts/
    ├── dump_style_swap_grid.py          [★ NEW] 离线读 ckpt 做可视化
    └── run_decouple_probes.py           [★ NEW] 离线 probe
```

### 7.2 核心算法实现 (feddsa_dualenc.py 伪码)

```python
class Server(BasicServer):
    def __init__(self, ...):
        self.style_bank = {}  # {client_id: {'mu': T, 'sigma': T}}

    def iterate(self):
        # 1. 选 active clients
        # 2. 广播 model + 别 client 的 style_bank
        for cid in active_clients:
            other_bank = {k: v for k, v in self.style_bank.items() if k != cid}
            client_pkg = self.pack(cid, other_bank)
            client.set_parameters(client_pkg)
            new_pkg = client.train_with_swap()
            # 3. 收 client 上传的 z_sty stats
            self.style_bank[cid] = new_pkg['style_stats']
        # 4. 差异化聚合
        self.aggregate(packages, exclude_keys=['E_sty.*', 'bn.*'])

class Client(BasicClient):
    def train_with_swap(self):
        for batch in self.data:
            pooled = self.backbone(batch.x)
            z_sem = self.E_sem(pooled)
            mu, log_var = self.E_sty(pooled)
            z_sty = reparam(mu, log_var)

            # L_CE
            logits = self.classifier(z_sem)
            L_CE = F.cross_entropy(logits, batch.y)

            # L_rec
            x_hat = self.decoder(z_sem, z_sty)
            L_rec = F.l1_loss(x_hat, batch.x)

            # L_saac (cycle anatomy consistency)
            z_sty_swap = sample_from_other_clients(self.received_bank, batch_size=B, K=4)
            x_swap = self.decoder(z_sem, z_sty_swap)
            pooled_swap = self.backbone(x_swap)
            z_sem_swap = self.E_sem(pooled_swap)
            L_saac = F.l1_loss(z_sem.detach(), z_sem_swap)  # ★ detach GT 防自循环

            # L_dsct (InfoNCE on z_sty)
            L_dsct = info_nce(z_sty, positives_same_client, negatives_other_clients, tau=0.1)

            # L_kl
            L_kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1).mean()
            kl_weight = min(1.0, current_round / 10) * lambda_kl  # warmup

            # Total
            L = L_CE + lambda_rec*L_rec + lambda_saac*L_saac + lambda_dsct*L_dsct + kl_weight*L_kl
            L.backward()
            self.optimizer.step()

        # 上传本 epoch z_sty 统计
        return {'params': state_dict, 'style_stats': self.collect_local_stats()}
```

### 7.3 网络定义 (dualenc_alexnet.py 关键块)

```python
class SRM(nn.Module):
    """16d z_sty → (γ, β) for AdaIN"""
    def __init__(self, style_dim=16, hidden=256, out_channels):
        self.fc1 = nn.Linear(style_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 * out_channels)

class DualEncAlexNet(nn.Module):
    def __init__(self, num_classes=7):
        # AlexNet backbone (沿用 FDSE alex_dse.py)
        self.features = AlexNetFeatures()  # → (B, 256, 6, 6)
        self.fc1 = nn.Linear(256*6*6, 1024)

        # 双 encoder
        self.E_sem = nn.Linear(1024, 512)
        self.E_sty_mu = nn.Linear(1024, 16)
        self.E_sty_logvar = nn.Linear(1024, 16)

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

        # Decoder (3 个 AdaIN-modulated upsample blocks)
        self.dec_fc = nn.Linear(512, 256*8*8)
        self.dec_block1 = AdaINBlock(256, 128, style_dim=16)  # 8→16
        self.dec_block2 = AdaINBlock(128, 64, style_dim=16)   # 16→32
        self.dec_block3 = AdaINBlock(64, 32, style_dim=16)    # 32→64
        self.dec_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),   # 128→256
            nn.Tanh(),
        )

class AdaINBlock(nn.Module):
    def __init__(self, in_c, out_c, style_dim):
        self.conv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        self.srm = SRM(style_dim, 256, out_c)
        self.relu = nn.ReLU()
    def forward(self, F, z_sty):
        F = self.conv(F)
        gamma, beta = self.srm(z_sty)
        F = adain(F, gamma, beta)
        return self.relu(F)

def adain(F, gamma, beta):
    mean = F.mean(dim=(2,3), keepdim=True)
    std = F.std(dim=(2,3), keepdim=True) + 1e-5
    F_norm = (F - mean) / std
    return gamma.view(*gamma.shape, 1, 1) * F_norm + beta.view(*beta.shape, 1, 1)
```

### 7.4 Probe 实现 (probes.py)

```python
@torch.no_grad()
def collect_features(client, val_loader):
    """Forward 一遍,收集 z_sem/z_sty/labels/domain_ids"""
    z_sems, z_stys, labels, domains = [], [], [], []
    for batch in val_loader:
        pooled = client.backbone(batch.x)
        z_sems.append(client.E_sem(pooled).cpu())
        mu, _ = client.E_sty(pooled)
        z_stys.append(mu.cpu())
        labels.append(batch.y.cpu())
        domains.append(batch.domain_id.cpu())
    return [torch.cat(t) for t in [z_sems, z_stys, labels, domains]]

def train_probe_mlp(features, targets, num_classes, epochs=5, hidden=256):
    probe = nn.Sequential(nn.Linear(features.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in range(epochs):
        for batch in DataLoader(zip(features, targets), batch_size=64, shuffle=True):
            x, y = batch
            loss = F.cross_entropy(probe(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return probe

def run_4_probes(client, val_loader, save_path):
    z_sems, z_stys, labels, domains = collect_features(client, val_loader)
    results = {}
    # P1: z_sty → class (期望 < 25%)
    probe = train_probe_mlp(z_stys, labels, num_classes=7)
    results['P1_z_sty_class'] = eval_acc(probe, z_stys, labels)
    # P2: z_sty → domain (期望 > 80%)
    probe = train_probe_mlp(z_stys, domains, num_classes=4)
    results['P2_z_sty_domain'] = eval_acc(probe, z_stys, domains)
    # P3: z_sem → domain (期望 < 50%)
    probe = train_probe_mlp(z_sems, domains, num_classes=4)
    results['P3_z_sem_domain'] = eval_acc(probe, z_sems, domains)
    # P4: z_sem → class (期望 > 60%)
    probe = train_probe_mlp(z_sems, labels, num_classes=7)
    results['P4_z_sem_class'] = eval_acc(probe, z_sems, labels)
    json.dump(results, open(save_path, 'w'))
```

### 7.5 可视化 (visualize.py)

```python
def dump_4x4_style_swap_grid(client, server, round_idx, save_dir):
    sample_imgs = sample_4_images_one_per_domain(client.val_data)  # (4, 3, 256, 256)
    grid = torch.zeros(4, 4, 3, 256, 256)
    for i in range(4):
        x = sample_imgs[i:i+1]
        z_sem_i = client.E_sem(client.backbone(x))
        for j in range(4):
            if i == j:
                z_sty_j = client.E_sty(client.backbone(x))[0]  # μ
            else:
                z_sty_j = sample_from_bank(server.style_bank, target_domain=j)
            x_recon = client.decoder(z_sem_i, z_sty_j)
            grid[i, j] = x_recon[0]
    save_image_grid(grid, f'{save_dir}/round_{round_idx:03d}_swap_grid.png',
                    row_labels=['photo', 'art', 'cartoon', 'sketch'],
                    col_labels=['→photo', '→art', '→cartoon', '→sketch'])

def dump_cycle_verification(client, round_idx, save_dir):
    x = sample_one_image(client.val_data)
    z_sem = client.E_sem(client.backbone(x))
    z_sty_orig = client.E_sty(client.backbone(x))[0]
    z_sty_swap = sample_from_other_domain(client.style_bank)
    x_swap = client.decoder(z_sem, z_sty_swap)
    z_sem_swap = client.E_sem(client.backbone(x_swap))
    x_cycle = client.decoder(z_sem_swap, z_sty_orig)  # 用回原 z_sty
    grid = torch.stack([x[0], x_swap[0], x_cycle[0]], dim=0)
    save_image_grid(grid, f'{save_dir}/round_{round_idx:03d}_cycle.png',
                    col_labels=['原图', 'swap 图', 'cycle 还原'])
```

### 7.6 配置文件 (config/office/feddsa_dualenc.yml)

```yaml
algorithm: feddsa_dualenc
algo_para:
  # Loss 权重
  - lambda_rec: 0.001
  - lambda_saac: 1.0
  - lambda_dsct: 0.01
  - lambda_kl: 0.01
  - kl_warmup_rounds: 10
  # Architecture
  - z_sem_dim: 512
  - z_sty_dim: 16
  - srm_hidden: 256
  # Style bank
  - bank_K: 4              # 每次采样别 client 的 K 个 (μ,σ)
  - alpha_range: [-1, 1]   # CDDSA 风格
  # Probe
  - probe_freq: 10         # 每 10 round 跑 probe
  - viz_freq: 25           # 每 25 round dump 可视化
  - probe_subsample: 200   # 每 client 200 个样本算 probe
# Optimizer (沿用 orth_only 已调通超参)
lr: 0.005
momentum: 0.9
weight_decay: 5e-4
local_epochs: 1            # Office 用 1
batch_size: 64
num_rounds: 200
```

---

## 八、Final Checklist

- [x] 主 paper Table 1 (Block 1) 覆盖:6 baseline + DualEnc + BiProto on PACS+Office
- [x] Novelty 隔离 (Block 2):8 个 4-loss 变体
- [x] Simplicity 防御 (Block 3):4 个 extras 不该助益
- [x] Frontier necessity:**N/A,本论文不依赖 LLM/VLM/Diffusion/RL 现代组件,故跳过 Block "frontier necessity check"**
- [x] Cross-client 必要性 (Block 4):4 种风格采样源对比
- [x] Stability regression (Block 5):BiProto vs Full ER 长跑
- [x] 可视化 (Block 6):4×4 grid + cycle dump
- [x] Probe methodology (Block 7):4-probe 曲线 + 终值表
- [x] Nice-to-have 与 must-run 分离:Block 8+9 是附录,不阻塞主流程
- [x] Anti-claim A1-A4 全部有对应 ablation 排除
- [x] Decision gate 明确:M3 后判 Pilot,M4 后判主表

---

## 九、Sync 状态

- [x] 写入 `refine-logs/2026-04-25_FedDSA-DualEnc/EXPERIMENT_PLAN.md`
- [x] 写入 `refine-logs/2026-04-25_FedDSA-DualEnc/EXPERIMENT_TRACKER.md`
- [x] 同步到 `obsidian_exprtiment_results/2026-04-25/EXP-128_FedDSA-DualEnc_实验计划.md`
- [ ] 待用户确认后,基于本计划写正式 algorithm code (1.5-2 周开发周期)
