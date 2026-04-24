# 联邦学习跨域项目现状报告 (2026-04-24)

> 用于**重新设计算法**的完整上下文. 整合项目目标 / 实验历史 / 最新诊断发现 / FDSE 胜因 / 现有差距 / 设计约束.

---

## 一. 项目目标 (强制, 来自 CLAUDE.md)

**唯一胜利标准**: 3-seed {2, 15, 333} mean AVG Best accuracy **必须超过 FDSE 本地复现 baseline** (不是论文数字)。

| Dataset | FDSE paper | FDSE 本地复现 | **必须达到** |
|---|:-:|:-:|:-:|
| **PACS** AVG Best | 82.17 | **79.91** | >79.91 |
| **Office-Caltech10** AVG Best | 91.58 | **90.58** | >90.58 |

### 硬性禁止
- ❌ 禁止换数据集 (不准因打不过 FDSE 就 pivot 到 FEMNIST 等)
- ❌ 禁止改 paper 叙事为"诊断论文"
- ❌ 禁止用 non-inferior/持平叙事
- ❌ 禁止"有创新就算赢", accuracy 数字超 FDSE 才是胜利

---

## 二. 数据集

| Dataset | 客户端 (domain) | 类别数 | Image size | 特点 |
|---|:-:|:-:|:-:|---|
| **PACS** | 4 (Art, Cartoon, Photo, Sketch) | 7 | 256×256 | 风格差异大, Art 最难 |
| **Office-Caltech10** | 4 (Amazon, Caltech, DSLR, Webcam) | 10 | 256×256 | 风格差异小, DSLR 样本少 |
| **DomainNet** | 6 (Clip, Info, Paint, Quick, Real, Sketch) | 10 (子集) | 256×256 | 多 domain, 中等难度 |

Backbone: **AlexNet** (60M params, fc 输出 1024 pooled). 统一 bs=50, num_epochs=5, num_rounds=200.

---

## 三. 当前最强方法 + 数据集对比 (2026-04-24)

### PACS (3-seed mean AVG Best)

| 方法 | AVG Best | Art | Cartoon | Photo | Sketch | 状态 |
|:---:|:-:|:-:|:-:|:-:|:-:|:-:|
| **FDSE 本地** | 79.91 (EXP-081 旧) / **81.54** (EXP-123 新) | 64.71 | 85.18 | **86.83** | 89.46 | 基线 |
| **orth_only** (feddsa_scheduled sm=0) | **80.41** (EXP-123 v2) / 79.95 (EXP-123 Stage B 2 seeds mean) | 62.50 | 87.92 | 79.98 | 91.09 | **我们最强** |
| FedBN | 79.23 | 62.25 | 85.90 | 80.24 | 88.52 | 基线 |
| FedAvg | ~74 (估) | — | — | — | — | 弱基线 |

**观察**: orth_only 旧估 80.64, Stage B R200 3-seed 重跑得 79.95 (2 seeds, s=333 未完成). FDSE 3 seeds R200 **重跑得 81.54** (之前误记 79.91, 是我们 EXP-081 旧配置)。**实际 FDSE 比 orth_only 领先 ~1.6pp**, 目标 hardest 比之前想象更严峻.

### Office-Caltech10 (3-seed mean AVG Best)

| 方法 | AVG Best | 状态 |
|---|:-:|:-:|
| **FDSE 本地** | 90.58 | 基线 |
| **FedBN lo=0 (EXP-116)** | **89.75** | **本地最强** |
| orth_only | 89.09 (EXP-113) | 我们 |
| sgpa_only (uw=0 uc=0, EXP-120) | 89.14 | — |
| sgpa_w (uw=1) | 88.68 | 有害 |
| sgpa_c (uc=1) | 88.74 | 有害 |

**观察**: Office 所有方法都**输 FDSE 1-2pp**. 甚至 FedBN 也输. 这是我们的**最大短板**.

### DomainNet (3-seed mean AVG Best)

| 方法 | AVG Best | 状态 |
|---|:-:|:-:|
| **orth_uc1** (feddsa_sgpa uw=1 uc=1, EXP-115) | **72.49** | **我们最强** |
| orth_only (sm=0, EXP-117) | 72.23 | — |
| FDSE 本地 | 72.21 | 基线 |
| sgpa_only (uw=0 uc=0, EXP-120 R=105 mid) | 72.62 | 接近 |

**观察**: DN 上 orth_uc1 **赢 FDSE +0.28**, 但差距极小 (差在 seed noise 内). EXP-120 诊断 (Office + DN mid-run) 说明 whitening/centers **没有独立贡献**, SGPA 架构本身贡献 +0.39 (只在 DN, 不 generalize 到 Office).

### 跨数据集总结

- **PACS**: 赢 FDSE 本地 0.5, 但 **FDSE 新 3-seed 重跑 81.54** 让差距倒转为 **-1.13pp**
- **Office**: 输 1.49pp
- **DomainNet**: 赢 0.28pp

**现状**: 3 个数据集上, 只有 DomainNet 稳稳胜 FDSE, PACS 胜负在翻转, Office **全线败北**. **不存在跨数据集一致领先的方法**.

---

## 四. 方法演化史 (FedDSA 变体族)

### 4.1 核心架构 (unchanged)

**`FedDSAModel`** (feddsa_scheduled.py):
```
Input → AlexNetEncoder → h (1024d) 
                      ↓
           ┌──────────┴──────────┐
   semantic_head (MLP)      style_head (MLP)
        ↓                        ↓
      z_sem (128d)          z_sty (128d)
        ↓
     classifier (head)
        ↓
     logits (num_classes)
```

Loss 组成 (sm=0 orth_only mode):
- **L_CE**: task cross-entropy on `head(z_sem)`
- **L_orth**: `cos²(z_sem, z_sty)` — 正交解耦, 我们的核心 novelty

### 4.2 变体演化

| 变体 | 关键机制 | 最强 dataset | Notes |
|:---:|---|---|---|
| **feddsa** (EXP-010~036) | 基础双头 + InfoNCE | 早期探索 | 淘汰 |
| **feddsa_plus** (EXP-015, 018) | 3-stage schedule | 早期 | 复杂度过高 |
| **feddsa_scheduled sm=0 "orth_only"** (EXP-076, 109) | 只 CE + L_orth, 无 infoNCE/aug | **PACS 最强 80.64** | **当前 baseline** |
| **feddsa_scheduled sm=1-7** (EXP-076) | Bell-curve / cutoff / 各种 schedule | — | 都不如 sm=0 |
| **feddsa_sgpa** (EXP-096~099, 115) | + SGPA 双头, + whitening (uw), + class centers (uc) | **DN 最强 72.49 (uc1)** | Office 失败 |
| **feddsa_vib** (EXP-113) | + VIB info bottleneck | — | 未突破 |
| **feddsa_pch** (EXP-124, 进行中) | orth_only + hard-cell CE re-weight | pilot | **太粗糙** (hardcoded) |

### 4.3 一些已消融的方向

- **HSIC 核独立性** (lh>0): 多次扫, 没显著帮助
- **pcgrad / CKA / Triplet** (EXP-029, 031, 030): 都弱于 orth
- **Asymmetric heads** (EXP-042): 不显著
- **VAE-based style head** (EXP-041, 045): 复杂度高, 无显著增益
- **Multi-layer style** (EXP-040, 044): 早期尝试 multi-level, 效果一般

**重要**: **只在最终 feature 做一次正交解耦** (sm=0 orth_only) 目前是我们最稳的 setup.

---

## 五. EXP-123 Stage B 诊断 (2026-04-24 关键发现)

### 5.1 诊断 hook 设计

写了 `diagnostics/per_class_eval.py` + `PerRunDiagLogger`, 每 round 记录:
- **per_class_test_acc_dist**: 4 client × 7 class matrix
- **confidence_stats_dist**: mean/std/p10/p50/p90/ECE/over_conf_err_ratio per client
- **confidence_hist_dist**: 每 50 rounds dump

### 5.2 核心发现 (基于 8/9 runs 完整, orth_s333 Round 189 待补)

**FDSE 的 +2.31pp 优势 ~90% 来自 3 个 hard cells**:

| Cell | FedBN | orth_only | **FDSE** | FDSE Δ vs FedBN |
|---|:-:|:-:|:-:|:-:|
| (Art, **guitar**) | 37.25 | 43.94 | **61.57** | **+24.33** ⭐ |
| (Art, **horse**) | 45.93 | 50.85 | **61.45** | **+15.52** |
| (Photo, **horse**) | 48.07 | 55.86 | **65.57** | **+17.50** |
| **总贡献到 AVG** | — | — | — | **+2.05 pp** |

**反噬 cells** (FDSE 反输 FedBN):
- (Art, dog): FedBN 60.98 > FDSE 47.62 (**-13.36**)
- (Art, giraffe): FedBN 75.62 > FDSE 66.73 (-8.89)
- (Art, elephant): FedBN 66.58 > FDSE 59.02 (-7.56)

### 5.3 Confidence / Calibration

| Method | Art ECE | Cartoon | Photo | Sketch |
|---|:-:|:-:|:-:|:-:|
| FedBN | **0.190** | 0.051 | 0.110 | 0.045 |
| orth_only | **0.180** | 0.079 | 0.095 | 0.065 |
| FDSE | **0.177** | 0.062 | **0.062** | 0.053 |

**观察**:
- **Art ECE 跨方法 ~0.18**: 是 PACS 固有难点, 方法改进无效
- **Photo ECE**: FDSE 0.062 vs FedBN 0.110 → FDSE 把 Photo calibration 降了 50%, 这部分解释 +6.6pp
- **Art over_confident_wrong**: 所有方法 13-15% (严重过自信错)

### 5.4 R_best (best round) 分布

| Method | s=2 R_best | s=15 R_best | s=333 R_best |
|---|:-:|:-:|:-:|
| FedBN | 133 | 87 | **37 ⚠️** |
| orth_only | 168 | 130 | 🟡 (R=189 mid) |
| FDSE | **188** | 119 | **182** |

**观察**: FDSE 普遍 R=180+ 才达峰 (**慢热**), orth 中等 (R=130-168), FedBN s=333 R=37 早熟后下滑 100+ rounds (过拟合).

---

## 六. FDSE 胜因根本分析

结合 5 篇精读论文 + 我们诊断数据:

### 6.1 FDSE 怎么实现 "+Art guitar/horse"?

**FDSE 架构**:
- 每层 Conv 分解为 `DFE (domain-free extractor)` + `DSE (domain-specific eraser)`
- DSE 用 depthwise conv, 每层擦除 style 残留
- 迭代擦除 → Art 的 guitar/horse 特征**每层都向"无 style"靠近** → classifier 看到的是 style-free 的 guitar/horse

**我们的 orth_only 做了什么**:
- 只在**最后 pooled feature** 做一次 `cos²(z_sem, z_sty)` 正交
- style 信息可能在**前面几层已经泄漏**进 `z_sem`
- Art 的 guitar 风格极端 (卡通化 / 抽象化), 单次正交不够

### 6.2 为什么 FDSE 的 Art dog/giraffe/elephant 反输?

**假设**: FDSE 擦除太狠, Art 上 "常见动物" 的 style 信息其实**部分有用** (比如 Art dog 的 photoshop 效果颜色提示 class). FDSE 把这种"有益 style" 也擦了 → 反伤.

这暗示 FDSE 不是最优解, 只是**对 extreme style 场景适用**.

### 6.3 我们的机会 — FDSE 有 trade-off, 不是通吃

- FDSE 在 PACS 靠 3 cells 赢 2pp, 其他 cells 其实与 FedBN 相当甚至输
- Office 上 FDSE 领先 1.5pp, 但具体来源不明 (没做 per-cell 诊断)
- DomainNet 上 FDSE 其实**落后我们** 0.28pp

**结论**: FDSE 不是 "universal winner", 它的优势**集中在 PACS 的 style-confused cells**. 如果我们能做**更 balanced 的 style-invariance** (不过度擦除), 理论上可以通吃。

---

## 七. 5 篇精读论文的方法学启示

| 论文 | 核心方法 | 对我们启示 | 状态 |
|:--:|---|---|:--:|
| **FDSE (CVPR'25)** | 层级 DFE+DSE 擦除 | 多层处理比单层好, 但擦除过头反伤 | 🟢 已深度理解 |
| **FPL (CVPR'23)** | FINCH 聚类多原型 + InfoNCE | 多原型能表达多模态, 但 cluster 噪声敏感 | 🟡 部分借鉴过 |
| **FedPLVM (NeurIPS'24)** | 双层聚类 + α-sparsity | 困难域自适应, α 超参敏感 | 🟡 EXP-077 借鉴 safety valve |
| **F2DC (AAAI'25)** | "Calibrate rather than eliminate" + corrector | 反对纯擦除, 保留 57% class signal | 🟢 最近精读 |
| **FedCCRL (ICDCS'25)** | MixStyle + AugMix | 数据增强层, 简单但非解耦 | 🟢 最近精读 |
| **FedAlign (CVPR'W25)** | 双阶段 embedding+prediction 对齐 | 轻量隐私保护 | 🟢 最近精读 |
| **FedOMG (ICML'25)** | Server-side gradient matching | 与 feature-space 方法正交, 可叠加 | 🟢 最近精读 |
| **I2PFL** | APA + GPCL intra/inter-domain | APA (feature MixUp + MSE) +4.80pp on Art | 🟢 最近精读 |

### 关键启示 (from I2PFL + F2DC)

- **I2PFL APA (feature MixUp + MSE)**: 在 Art 涨 4.8pp, **远超 GPCL 对比学习** 的 +0.42 → feature-level mixup 比 contrastive 更直接
- **F2DC "calibrate not eliminate"**: 我们 orth + FDSE 都是 "eliminate style", 但 F2DC 证明 **57% class info 在 "style" 里**, 擦了就失了

---

## 八. 我们距离 FDSE 的差距来源分解

**核心假设**: 我们 orth_only 的 z_sem 里仍有**大量 style leak**, 导致 classifier 学到的决策边界是 domain-dependent 的.

证据链:
1. Stage B per-class matrix 显示 orth_only 在 (Art, guitar) 43.94, FDSE 61.57 (差 17pp)
2. orth_only 只做**最后一层**正交, FDSE 逐层擦除
3. 我们的 L_orth 只是 cos² → 只约束**线性不相关**, 不是信息独立 (HSIC 过但弱)
4. Art 的 guitar 风格变异巨大 (painting 里吉他可能是**抽象形状**), 前面层 feature 已经 style-tangled

**三个可能修复路径**:

### 路径 A: 更深的解耦 (架构级)
- Multi-level orthogonal: 不是 1 次, 是每层/每几层做一次
- 类似 FDSE 但保留 orth 而非 erase
- **风险**: 架构复杂, 调参多

### 路径 B: 更强的 loss (loss 级)
- 从 cos² 升级到 HSIC (RBF kernel)
- 加 **condition on class**: "same class across domains should have close z_sem"
- **风险**: HSIC 超参敏感, 我们扫过没突破

### 路径 C: 不解耦, 而是 alignment (representation 级)
- 让 Art 的 horse z_sem **直接对齐到** Photo 的 horse z_sem
- Cross-domain class prototype anchor
- Classifier 看到跨 domain 一致的 z_sem 分布
- **风险**: 需要 server 维护 prototype bank, 通信开销

---

## 九. EXP-124 PCH (在跑) 的局限 — 用户正确指出

**EXP-124 FedDSA-PCH** 硬编码 hard cells (Art {guitar, horse, person}, Photo {horse}), CE loss w=2x.

**问题**:
- Hard cells **per-dataset 不同**. PACS 的 hard 在 DomainNet 不是 hard.
- 需要每个 dataset 先跑诊断 + hardcode → **不是通用算法**
- Paper 不接受这种 "per-dataset hack"

**PCH 的真正价值**: 验证 "**per-cell hardness 思路是否可行**" (pilot). 结果数据会说明:
- 若 PCH 涨 ≥1pp: per-cell 方向成立, 但要做 **adaptive 版**
- 若 PCH 涨 <0.3pp: 说明 loss-level 调节不够, 必须到**表示层面**

**PCH 不是我们要发的方法**, 只是诊断的 follow-up 验证.

---

## 十. 下一步算法设计要求

### 必要条件 (hard constraint)

1. **跨数据集通用**: PACS + Office + DomainNet 都要胜 FDSE
2. **无 per-dataset hack**: 不许 hardcoded per-domain/per-class priors
3. **自适应**: 根据数据自动识别"难"样本/"难"特征方向
4. **保留 novelty**: 不能照抄 FDSE 多层擦除 (不新), 不能用 per-cell re-weight (粗糙)
5. **理论 story**: 有一个简洁的数学叙事 (例如 "更好的 style-invariance")

### 实施约束

1. Backbone 固定 AlexNet (不能换更大模型)
2. 通信量 ≤ FDSE (最好不增加)
3. 超参 ≤ 3 个新增 (现在 feddsa_scheduled 已 16 个)
4. **必须胜过数据**: FDSE PACS 81.54, Office 90.58, DN 72.21

### 可行的方向候选 (brainstorm 起点, 未验证)

**候选 1 — Adaptive multi-level orthogonal**
- 在 AlexNet 的 pool2, pool5, fc1 三个位置都做 orth(z_sem_l, z_sty_l)
- 每层有独立 projection head
- Loss weight 自动学习 (类似 uncertainty weighting)
- **风险**: 新超参 3 个, 调参复杂

**候选 2 — Cross-domain class prototype anchor**
- 每 client 上传 per-class mean(z_sem)
- Server 维护 global class prototype
- Client loss += `||z_sem - global_proto[y]||²`
- 让同 class 跨 domain 的 z_sem 自然对齐
- **风险**: 通信量 +num_classes × dim, 通信增加少

**候选 3 — Conditional HSIC**
- 升级 L_orth 从 cos² 到 RBF HSIC
- Condition on class y: HSIC(z_sem | y, z_sty | y)
- 每 class 独立约束, 捕获非线性 style leak
- **风险**: RBF kernel 计算开销, 小 batch 噪声大

**候选 4 — Style swap + consistency (I2PFL APA 变体)**
- Feature-level MixUp: `z_sem_aug = (1-α)·z_sem + α·z_sem_other_domain_same_class`
- CE on mixed feature 保证跨 domain 特征一致性
- **风险**: 本质是 data aug, 不够 novel. 但 I2PFL 实测 Art +4.8pp

**候选 5 — Calibrated style erase (F2DC 启示)**
- 不纯擦除, 而是**保留 class-correlated part of style**
- 用 class-conditional erase: z_sem = f(z - style·(1-class_bias))
- 复杂但可能 work

---

## 十一. 开放问题 (需要你决策)

### Q1: 方法哲学选哪个?

- (A) **更深解耦** (路径 A): 多层正交. 接近 FDSE 但不一样.
- (B) **更强 loss** (路径 B): HSIC + class-conditional. 轻改但效果存疑.
- (C) **Alignment** (路径 C): cross-domain class prototype. 新颖但通信变化.
- (D) **Hybrid**: 路径 B + C combine.

### Q2: novelty 点打哪?

- (a) 数学: 新的 disentanglement objective
- (b) 架构: 新的 multi-level 结构
- (c) 分布: 新的 representation alignment 方式
- (d) 优化: 新的 federated aggregation 策略

### Q3: 是否换 backbone?

- CLAUDE.md 要求 AlexNet, 但可考虑 ResNet-18 with FedBN (论文新范式, 多篇 2025 SOTA 都用)
- 或加 lightweight head 在 AlexNet 上

---

## 十二. 当前实验状态 (2026-04-24 夜)

**跑中的实验**:

| EXP | 算法 | 状态 | 预计完成 |
|---|---|:-:|:-:|
| EXP-123 orth_s333 | orth_only Round 189/200 | 🟡 | ~20min (sc2) |
| EXP-124 PCH × 3 seeds | feddsa_pch hw=2 | 🟡 ramp up | ~4-5h (lab-lry) |

**等待数据回来后可以**:
1. 完整 orth_only 3-seed mean (填完 s=333)
2. PCH 结果判决 per-cell 方向
3. 基于上面的分析重新设计算法 (本报告 十. 候选 1-5)

---

## 十三. 资源

- **代码仓库**: `D:/桌面文件/联邦学习/`
- **CLAUDE.md**: 项目指令 + 硬指标
- **本地 Python**: `D:/anaconda/python.exe`
- **服务器**:
  - seetacloud2: RTX 4090 24GB (当前 orth_s333 跑)
  - lab-lry GPU 1: RTX 3090 24GB (当前 PCH 3 seeds 跑)
  - seetacloud: AutoDL 按量 (待用)
- **EXP 目录**: `experiments/ablation/EXP-XXX_name/`
- **Obsidian 笔记**: `obsidian_exprtiment_results/`
- **Key analyses**:
  - `experiments/ablation/EXP-123_art_diagnostic/stageB_full/ANALYSIS.md`
  - `obsidian_exprtiment_results/2026-04-24/EXP-123_Stage_B_Analysis.md`

---

## 摘要 (TL;DR)

我们 3 个数据集的局势:

- **PACS**: orth_only 80 vs FDSE 81.5 (新重跑), **差 1.5pp**
- **Office**: orth_only 89 vs FDSE 90.6, **差 1.5pp**
- **DomainNet**: 我们 72.5 vs FDSE 72.2, **赢 0.3pp** (勉强)

**关键诊断**: FDSE 的优势不是 "全面好", 而是 "在 style-confused cells 压倒性好". 这给我们机会 — 做**更 balanced 的 style-invariance** 可能通吃.

**不可走的路**:
- ❌ 加 layer 多做 erase (照抄 FDSE)
- ❌ Hardcode hard cells per dataset (EXP-124 PCH 的问题)
- ❌ 过度 engineering (超参 > 3 个新)

**可走的路** (待 brainstorm):
- 🟢 Adaptive multi-level orth (with learnable weights)
- 🟢 Cross-domain class prototype anchor (简单 + 通信增加小)
- 🟢 Class-conditional HSIC
- 🟢 Feature-level class-preserving style swap

等 EXP-124 结果 + orth_s333 完成后, 正式 brainstorm 下一个算法.
