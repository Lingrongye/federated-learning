# 研究方案 Round 0:Hierarchical IBN-Augmented Decoupling (HIBD)

> 面向跨域联邦学习的层级结构化解耦方法 — Plan A 升级版
> 基础框架:FedDSA(Decouple-Align,Share 已证伪)
> 目标会议:CVPR / ICCV / NeurIPS

---

## Problem Anchor(不可变,每轮复制)

- **Bottom-line problem**:在 K=4 FedBN 跨域 FL 设置下,当前 FedDSA Plan A 的解耦**只在特征输出层做一次**(sem_head + style_head + cos²/HSIC)。Backbone 中间层特征早已被域偏移污染,下游所有基于 prototype 的对齐/共享方案(85+ 实验全部证伪,包括刚完成的 SCPR v1/v2)都在"已污染特征"上动手,徒劳。

- **Must-solve bottleneck**:
  1. **解耦太 shallow**:只在最后一层 projection head 前做,backbone 中间层未触及
  2. **Plan A 已是 ceiling**:所有"下游"改动(InfoNCE / M3 / SCPR)都 ≤ Plan A,意味着问题在上游
  3. **FDSE(我们主 baseline)用层级 DSE 结构分解**:每卷积层分为 DFE(共享)+ DSE(本地擦除偏移),新参数较多 → 我们需要**更轻量、更 sharp 的层级 decouple**
  4. **Style/Semantic 分离的理论 + FL 工程结合点缺失**:Plan A 的 `cos²(z_sem, z_sty) → 0` 只对 fc2 后 feat,中间层 BN / conv 特征里大量残余域偏移未被显式解耦

- **Non-goals**(严格排除):
  1. ❌ Prototype-level routing / alignment(EXP-095 v1/v2 证伪)
  2. ❌ 大量新 trainable 参数(soft cap < backbone 10%)
  3. ❌ 改骨干结构/深度(不 scale up AlexNet/ResNet-18)
  4. ❌ 引入 VLM / CLIP 预训练(破坏公平对比)
  5. ❌ 增加 K(保持 PACS/Office = 4 客户端)

- **Constraints**:
  - Backbone:AlexNet-from-scratch (PACS) / ResNet-18 (Office),参数量不变
  - FedBN 原则保留(BN 参数本地)
  - cos² + HSIC Plan A 解耦**保留**,作为最外层约束
  - 3-seed {2, 15, 333} 对齐 EXP-084 / EXP-095 v1/v2 基线
  - 必须 beat PACS AVG Best 3-seed mean **82.31%**(Plan A)**至少 +0.5%**
  - **快速验证**:R=20 sanity(30min)看 trend,R=50 mid(~2h)决定,R=200 full 定稿

- **Success condition**:
  1. PACS 3-seed mean AVG Best ≥ **82.81%**(+0.5% 门槛)
  2. Office 3-seed mean AVG Best ≥ **83.05%**(+0.5% 对比 Plan A 82.55%)
  3. R=20 sanity 即可看到中间层 cos²(z_sem^{(l)}, z_sty^{(l)}) 持续下降(解耦信号 active)
  4. 代码增量 **< 150 行**,**0 新 trainable 参数**
  5. 跟 FDSE 对比 sharp:参数量远少于 FDSE 的 DSE(DSE ≈ DFE/94,我们 **0 新参数**)

---

## Technical Gap

### Plan A 的解耦是"末梢级"

当前 Plan A orth_only 的 loss 只在最后 feat→head 处:
```
h = backbone(x)                       # 1024d,已含域偏移
z_sem = sem_head(h)                   # 128d,sem 投影
z_sty = style_head(h)                 # 128d,sty 投影
L_orth = cos²(z_sem, z_sty)           # 只在末梢层强制正交
```

**问题**:backbone 的 conv 层 / BN 层没有任何解耦信号,**中间层的 feature 已被域分布污染**,即使最后加 orth loss,信号回传到 backbone 的梯度**太稀**。解耦能力受限。

### 为什么"下游"所有方案失败

- EXP-095 v1/v2 SCPR:在**已污染**的 z_sem 上做 prototype attention,attention 看到的 prototypes 本身就带有域偏移,style 加权无从发力
- EXP-059 / EXP-078d:在 z_sem 或 h 上做 AdaIN 增强,污染特征,CE 回退
- 所有 InfoNCE 变体:对齐目标 = mean-of-污染-prototypes

**根本原因:backbone 中间层无解耦约束,污染早已形成。**

### FDSE 的做法与我们的差异化机会

FDSE (CVPR 2025) 在每个 conv 层**结构性分解**:
```
x → DFE_conv(通道减半 oup/2) → dse_bn → DSE_conv(分组深度卷积) → concat → dfe_bn → out
```
DFE 共享(FedAvg),DSE 本地(擦除域偏移),DSE ≈ DFE/94 参数。

**我们需要**:
- 比 FDSE **更轻**(DSE 再小也是新参数)
- 比 FDSE **差异化更 sharp**:我们不改结构,只改**normalization scheme**

### 切入点:IBN(Instance-Batch Normalization)

**关键观察**:Pan et al. (ECCV 2018) IBN-Net 证明 **Instance Normalization(IN)通道自然分离 style,Batch Normalization(BN)通道自然保留 batch-level content**,IN+BN 混合显著提升 domain generalization。

但 IBN-Net 的原始设计**面向单机 DG**,没有 FL 设置,也没有**显式正交约束**。

**我们的机会**:把 IBN 原语**移植到 FedBN 框架**,并在 IBN 层上**显式加正交损失**,形成**层级解耦约束**。

### 最小充分干预

- **不改 backbone 结构**(不加新 conv/新 linear)
- **不加可训练参数**(IN 是 parameter-free,跟 BN 共用 gamma/beta)
- **只改 normalization**:选 backbone 的 **early 1-2 个 BN 层**替换为 IBN(parallel IN + BN,拼接或选择)
- **层级正交损失**:取 IBN 内 IN 通道输出 vs BN 通道输出的 flat representation,加 cos²

---

## Method Thesis

> **一句话 thesis**:把 FedDSA 的"末梢层 cos² 解耦"升级为"**层级 IBN-Orthogonal 解耦**"—— 在 backbone 的前若干个 BN 层替换为 **IBN(Instance+Batch Norm parallel)** 并**显式施加正交约束**,让 style(IN 通道)与 semantic(BN 通道)在每一层就显式分离,不再把解耦压力全压到末梢层。

- **最小充分干预**:
  - 替换 2-3 个 early BN 层为 IBN(结构不变,normalization scheme 改变)
  - 在 IBN 位置加 per-layer cos² 正交 loss
  - **0 新 trainable 参数**,~100 行代码
- **当下可行性**:IBN 是 2018 ECCV 的成熟原语,有 reference impl;FedBN 框架本就本地化 BN stats,IBN 无缝兼容
- **差异化**:
  - vs FDSE:我们不改 conv/linear 结构,只改 normalization,参数增量 **0**(FDSE 有 DSE 新参数)
  - vs IBN-Net:我们加**显式正交 loss + FL 本地 BN 聚合规则**(IBN-Net 是单机 DG,无 FL)
  - vs Plan A:我们在 backbone 前端加层级解耦约束(Plan A 只在末梢)

---

## Contribution Focus

- **唯一主贡献**:**Hierarchical IBN-Orthogonal Decoupling (HIBD)**
  > 在 FedBN 框架下,将 backbone 前 1-2 个 BN 层替换为 IBN(IN 和 BN 并行),并在每个 IBN 层的 IN/BN 通道输出上施加 cos² 正交损失。IN 通道自然承载 style,BN 通道保留 content;层级正交约束让 backbone 前端就开始解耦,不再把压力全压到末梢层。**0 新 trainable 参数,~100 行代码**。

- **可选辅贡献**:~~无~~(保持 single contribution,避免 sprawl)

- **非贡献**(明确排除):
  - 不做 prototype routing
  - 不做 alignment / InfoNCE(Plan A 本来就没,HIBD 也不加)
  - 不做风格共享

---

## Proposed Method

### Complexity Budget

| 组件 | 状态 |
|------|------|
| 骨干卷积/linear(AlexNet/ResNet-18) | **冻结结构** |
| 前 1-2 个 BN 层 | **替换为 IBN**(结构改 normalization,不改上下 conv 连接) |
| sem_head / style_head / classifier | **复用 Plan A** |
| 末梢层 cos² + HSIC(Plan A) | **保留** |
| **IBN 模块**(IN + BN 并行) | **新增,但 IN 无 trainable 参数**(gamma/beta 共享或独立,见下) |
| **Per-layer cos² orth loss**(层级) | **新增 loss**,无参数 |

- **新 trainable 组件:0**
- **新 loss 项:1**(层级正交 `L_orth_layer`)
- **新超参:2**(`num_ibn_layers`:替换几个 BN;`λ_orth_layer`:层级 loss 权重)

### System Overview

```
                  Plan A (baseline)                     HIBD (OURS, +4 BN→IBN)
                        │                                         │
               ┌────────┴────────┐                       ┌────────┴────────┐
               │                 │                       │                 │
   Conv1 → BN1 ─── relu → maxpool                 Conv1 → IBN1 ─── relu → maxpool
   Conv2 → BN2 ─── relu → maxpool                 Conv2 → IBN2 ─── relu → maxpool     ← 早期层用 IBN
   Conv3 → BN3 ─── relu ← ...                     Conv3 → BN3 ─── relu ← ...           ← 后期 BN 不动
                  ...                                       ...
   fc2 → BN7                                       fc2 → BN7
   ↓                                               ↓
   h (1024d)                                       h (1024d)
   ↓                                               ↓
   sem_head(h) → z_sem                             sem_head(h) → z_sem
   style_head(h) → z_sty                           style_head(h) → z_sty
   L_orth_end = cos²(z_sem, z_sty)                 L_orth_end = cos²(z_sem, z_sty)
                                                   + L_orth_layer = Σ_l cos²(IN_l, BN_l)  ← 新
```

### IBN 模块定义(详细)

**IBN-b**(原 IBN-Net Type-b):
```python
class IBN(nn.Module):
    def __init__(self, channels, in_ratio=0.5):
        super().__init__()
        self.in_channels = int(channels * in_ratio)    # 前 50% 通道走 IN
        self.bn_channels = channels - self.in_channels # 后 50% 通道走 BN
        self.instance_norm = nn.InstanceNorm2d(self.in_channels, affine=False)  # 0 参数
        self.batch_norm = nn.BatchNorm2d(self.bn_channels)                      # gamma/beta
    def forward(self, x):
        split = torch.split(x, [self.in_channels, self.bn_channels], dim=1)
        out_in = self.instance_norm(split[0])
        out_bn = self.batch_norm(split[1])
        return torch.cat([out_in, out_bn], dim=1)
```

- **InstanceNorm2d (affine=False)**:0 可训练参数
- **BatchNorm2d**:参数跟原 BN 一样(gamma/beta,但只一半通道)
- **总参数变化**:从 `2C`(原 BN) → `C`(IBN 的 BN 通道)→ **减少一半**!(但可忽略,因为 BN 参数量 vs 主干 conv 本来就很小)

### Core Mechanism — Per-layer Orthogonal Loss

对每个 IBN 层 l,取其 IN 输出和 BN 输出的 spatial mean pooling:
```python
# In the forward pass of each IBN layer, hook both outputs
in_feat_l = global_avg_pool(out_in)    # [B, C_in]
bn_feat_l = global_avg_pool(out_bn)    # [B, C_bn]

# Normalize, align dim via proj (optional, if C_in != C_bn; here C_in = C_bn)
in_flat = F.normalize(in_feat_l, dim=1)
bn_flat = F.normalize(bn_feat_l, dim=1)

# Per-sample cos similarity (IN vs BN should orthogonal)
cos_sim = (in_flat * bn_flat).sum(dim=1)   # [B]
L_orth_layer_l = (cos_sim ** 2).mean()
```

**总层级正交损失**:
```
L_orth_layer = (1 / N_IBN) × Σ_l L_orth_layer_l       # 平均各层
```

**最终训练目标**:
```
L = L_CE + λ_orth_end × L_orth_end(Plan A)
       + λ_hsic × L_HSIC(Plan A)
       + λ_orth_layer × L_orth_layer(NEW)
```

默认 `λ_orth_layer = 0.5`(比末梢 λ_orth=1.0 弱,因为层级信号本来就多),`num_ibn_layers = 2`(AlexNet 的 BN1/BN2,ResNet-18 的 layer1/layer2 首 BN)。

### FedBN / FL 规则

- **IN 通道**:per-sample normalize,**无 running stats,无 FL 同步问题**
- **BN 通道**:跟原 BN 一样,按 FedBN 本地化(running_mean/var 不聚合,gamma/beta 聚合 **或** 按需决定,保持与 Plan A 一致)
- **IBN 层本身不引入 FL 聚合复杂度**

### 梯度回传路径

```
L_orth_layer (early)
   ↓
IN output, BN output (IBN layer)
   ↓
backbone conv (更深的梯度!)
```

这让 backbone **前端 conv 层**直接收到"style/content 分离"的梯度信号,比 Plan A 的"只在末梢回传"更有效。

### Training Plan

- **超参**(继承 Plan A + 新增):
  - `lo = 1.0`(末梢 cos² 权重,Plan A)
  - `lh = 0.0`(HSIC 权重,Plan A 默认 0 从 EXP-017)
  - `λ_orth_layer = 0.5`(新)
  - `num_ibn = 2`(新,替换前 2 个 BN)
  - LR = 0.05, R = 200, E = 5(PACS)/ E = 1(Office), B = 50
  - seeds = {2, 15, 333}

- **无新 trainable 参数**(IN affine=False)
- **训练开销**:IN 每层 forward 略贵(每 sample 算 mean/var),但 backbone 主干 conv 开销 >> IN,总开销增加 <5%

### Failure Modes and Diagnostics

| Failure | Signal | Fallback |
|---------|--------|----------|
| `L_orth_layer` 从始至终不下降(IN ≈ BN feature)| cos²(IN_l, BN_l) 稳在 0.5+ | 检查 IN channels 是否真在 spatial-normalize |
| PACS AVG Best < Plan A 82.31% | 3-seed 均值 | 减少 `num_ibn` 到 1,避免过度 IN 破坏 semantic |
| 训练不稳定(IN 依赖 batch=1 行为差)| batch size 小时 IN 学习不稳 | 切换到 `GroupNorm`(GN affine=False)作 style 分支 |
| cos²(IN, BN) 下降但 accuracy 不升 | 解耦 ≠ 有效 | 启动 `λ_orth_layer` sweep {0.1, 0.5, 1.0} |

### Novelty and Elegance Argument

- **最小机制**:~100 行,0 新 trainable,1 新 loss,1-2 新超参
- **理论根基**:IBN-Net (ECCV 2018) 的 style/content 分离 + Plan A 的正交约束 framework
- **差异化 2×2 矩阵**:

|   | 单机 DG(无 FL) | FL 设置 |
|---|----------------|---------|
| 单一 BN + 末梢正交 | SWA-Gaussian 等 | **Plan A(我们,baseline)** |
| 层级 IN+BN 分离 | IBN-Net (ECCV 2018) | **HIBD(我们,novel)** |

- **vs FDSE**:FDSE 用 DSE 参数分解(新参数 O(C²/94)),HIBD 用 IN 通道分离(新参数 = 0)
- **vs FedBN**:FedBN 只本地化 BN stats,HIBD 将 IN 融入 normalization 结构并显式正交
- **vs Plan A**:层级 cos² 扩展到前 2 个 BN 位置,解决"shallow decouple"问题

---

## Claim-Driven Validation Sketch

### Claim A(主):PACS 上 HIBD > Plan A + 0.5%

- **对比**:
  - A.1 Plan A orth_only(82.31%)
  - A.2 **HIBD**(num_ibn=2, λ_layer=0.5)
- **指标**:3-seed {2, 15, 333} mean AVG Best / AVG Last / per-domain
- **预期**:A.2 ≥ 82.81%,且 per-domain(Art/Sketch outlier)至少不崩
- **R=20 sanity 指标**:per-layer `cos²(IN_l, BN_l)` 曲线从 0.5+ 下降到 < 0.3(解耦 active)
- **决定性**:A.2 < A.1 → 方法证伪

### Claim B(辅,机制诊断):解耦信号从末梢扩散到前端

- **诊断指标**(零 GPU 成本):
  - 末梢 `cos²(z_sem, z_sty)` vs 中间 `cos²(IN_l, BN_l)` 跨训练轮次的演化
  - 预期:HIBD 下末梢 cos² 比 Plan A **更低**(前端已分离,末梢更容易)+ 中间层 cos² 也 < 0.3
- **支持理论**:验证"shallow decouple 是 Plan A 瓶颈"的假设

### 实验预算

| Milestone | Runs | 预算 |
|-----------|------|------|
| **M0 Sanity** R=20 s=2 | 1 PACS | 10 min ✅ 极快 |
| **M1 Mid** R=50 × 3 seeds(PACS)| 3 | 2h |
| **M2 Full** R=200 × 3 seeds(PACS)| 3 | 8h |
| **M3 Full Office** R=200 × 3 | 3 | 4h(并行)|
| **M4 Ablation**(num_ibn={1,2,3} × R=50)| 3 | 3h |
| **总** | ~13 runs | ~17 GPU·h |

---

## Experiment Handoff Inputs

- **Must-prove claims**:A(PACS HIBD > Plan A + 0.5%)+ B(层级解耦信号 active)
- **Must-run ablations**:num_ibn ∈ {1, 2, 3},λ_orth_layer ∈ {0.1, 0.5, 1.0}
- **Critical metrics**:PACS AVG Best/Last + per-domain(Art/Sketch),per-layer cos² 曲线
- **Highest-risk assumptions**:
  1. IN 通道真的承载 style(IBN-Net 在 DG 有 +2-3% reportable gain,FL 设置未测)
  2. 层级 orth loss 不会破坏 CE(需 sanity 验证 R=20)
  3. num_ibn=2 是 sweet spot(太多 IN 破坏 semantic,太少不够)

---

## Compute & Timeline Estimate

- **GPU·h**:~17(PACS + Office 主 + ablation)
- **数据**:复用现有 PACS / Office 任务
- **人力 timeline**:
  - 0.5 天:实现 HIBD 模块 + 单测
  - 0.5 天:codex review + 修
  - 0.5 天:R=20 sanity + R=50 mid
  - 1 天:R=200 full + Office
  - 0.5 天:回填 NOTE + Obsidian 同步
  - **共 3 天 end-to-end**
