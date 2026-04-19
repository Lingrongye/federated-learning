# Round 1 Refinement — 修 L_orth_layer 数学退化 + 收窄 Claim + 加 IBN-only 消融

---

## Problem Anchor(复制自 Round 0,不变)

(略,见 round-0)

---

## Anchor Check

- 原 bottleneck 保持:Plan A 解耦太 shallow,backbone 中间层未被直接约束
- Reviewer 的 3 条 CRITICAL 都是方法细化,**均不改变 anchored problem**
- Reviewer 担心过度 claim 会 drift 到"generic IBN for DG",但我们坚持 FedBN 框架 + early layer 约束 = **对准原 bottleneck**
- **结论:无 drift**

---

## Simplicity Check

- Round 0 的主 contribution:"Hierarchical IBN-Orthogonal Decoupling" + "层级 style/semantic 分离"
- Reviewer 指出:IN/BN concat 后被下一层 conv 混合,**不存在持久化分离路径**,claim 过强
- Round 1 收窄:**"Parameter-free early-split regularization for FedBN that supports the existing end-layer decouple"**
- 不增加 contributions,只修细节

---

## Changes Made(针对 R1 review)

### 1. [CRITICAL] 修正 `L_orth_layer` 数学退化 bug

- **Reviewer 说**:`InstanceNorm2d(affine=False)` 后 `global_avg_pool` = 0,原公式 `cos²(GAP(IN), GAP(BN))` 恒为 0,loss 无法回传
- **我确认**:这是真 bug!IN 的 per-channel spatial mean 就是 0(normalization 定义)
- **Action**:替换为 **spatial cross-correlation decorrelation**(保留 spatial 结构):

```python
# 在 IBN.forward() 内保存:
self._cache_in = out_in        # [B, C_in, H, W]
self._cache_bn = out_bn        # [B, C_bn, H, W]

# 在 L_orth_layer 计算时:
xin = F.normalize(self._cache_in.flatten(2), dim=2)   # [B, C_in, H*W]
xbn = F.normalize(self._cache_bn.flatten(2), dim=2)   # [B, C_bn, H*W]
# 跨通道 channel-wise correlation matrix
corr = torch.bmm(xin, xbn.transpose(1, 2)) / xin.size(-1)   # [B, C_in, C_bn]
L_orth_layer_l = corr.square().mean()
```

- **物理意义**:强制 IN 通道的 spatial activation pattern 与 BN 通道的 spatial activation pattern **不相关**(类似 CKA/HSIC 在 spatial 维度的 mini-batch 版本)
- **数学上**:此 loss 在 IN affine=False 下**不退化**(normalize 之后的 spatial sequence 不为 0)
- **Impact**:method specificity 从"形式上能实现,实际数学失效"→"数学正确,可训练"

### 2. [CRITICAL] 收窄 Contribution claim

- **Reviewer 说**:IN/BN 被下一层 conv 混合,无持久分离路径,不能 claim "hierarchical style/semantic 分解"
- **Action**:改主 thesis 为:
  > **"在 FedBN 框架下,用 IN+BN parallel + spatial 去相关作为 parameter-free early-split 正则,辅助 Plan A 末梢解耦,不新增可训练参数"**
- **删除**:所有"层级 style/semantic persistent separation"言辞
- **保留**:"per-layer decorrelation 让 backbone 前端 conv 梯度直接接收 style-content 分离信号"(这是**可证伪**的 mechanism claim,不是过度承诺)
- **Impact**:contribution quality 从"过度承诺的新方法"→"诚实的轻量正则"

### 3. [CRITICAL] 加 IBN-only ablation 作为必跑实验

- **Reviewer 说**:无 IBN-only 对照,无法区分"IBN 结构"和"新 orth loss"哪个在起作用
- **Action**:主实验从 2 个 config 扩展到 **3 个 config**:
  - **A.1 Plan A** orth_only(end-layer cos²)
  - **A.2 IBN-only**(IBN 替换前 2 BN,**不加** L_orth_layer)
  - **A.3 HIBD**(IBN + L_orth_layer)
- **决定性**:
  - A.3 > A.2 → decorrelation loss 的贡献
  - A.2 > A.1 → IBN 结构本身的贡献(DG 固有优势)
  - A.3 > A.1 但 A.2 ≈ A.1 → 主功劳是 loss,IBN 只是 loss 的 enabler

### 4. [IMPORTANT] Venue-ready claim 一句话

- **新 headline claim**:"Replacing early FedBN layers with split IN/BN plus local decorrelation improves cross-domain FL without adding trainable parameters."
- **对 FDSE 定位**:FDSE 是 heavier alternative(DSE 新参数),我们是 lighter & orthogonal direction
- **避免声称**:不是"novel 2×2 cell",而是"cheap focused early-layer fix for end-layer decouple"

### 5. [MINOR] 澄清 HSIC 权重

- **Reviewer 以为**:我们在 Plan A 里"偷偷改"HSIC 权重从 0.1 到 0.0
- **澄清**:EXP-017(85 实验中的关键发现)已证明 HSIC=0 是 Plan A 最优配置,后续所有实验沿用此默认。**本方案不改 Plan A 权重配置**
- **Action**:Round 1 proposal 里明确标注 `lh = 0.0 (inherited from EXP-017 optimal Plan A)`

### 6. [简化] 用 module 自带 cache 代替 forward hook

- **Reviewer 说**:"Avoid hooks if possible; let each IBN module expose cached `out_in` / `out_bn` directly"
- **Action**:在 IBN.forward() 里直接 `self._cache_in/_cache_bn`,client.train() 遍历 IBN layers 拿 cache
- 代码更简洁,无 hook 管理复杂度

---

## Revised Proposal(Round 1 完整版)

# Research Proposal Round 1: Hierarchical IBN-Decorrelated FedBN (HIBD-r1)

## Problem Anchor(同上)

## Technical Gap

Plan A's `L_orth = cos²(z_sem, z_sty)` operates only at the final 128d projection. Backbone conv/BN receive only indirect gradient → 85+ downstream experiments on polluted mid-features all stuck at Plan A ceiling.

FDSE (CVPR 2025) per-layer DFE(shared)+DSE(local-erase) has ~DSE=DFE/94 new params → heavy. **Our differentiation: 0 new params, only normalization scheme + decorrelation loss.**

## Method Thesis(收窄)

> **Headline**: Replace early FedBN layers with split IN/BN plus parameter-free spatial decorrelation loss. This is an **early-layer regularizer that supports the existing end-layer Plan A decouple**, not a claimed structural style/semantic decomposition.

## Contribution Focus

- **唯一主贡献**:**Parameter-free early-split decorrelation for FedBN**
  > 将 backbone 前 2 个 BN 替换为 IBN(InstanceNorm + BN parallel on split channels),并在 IBN 位置施加 spatial cross-correlation 去相关损失。**0 新可训练参数**,~100 行代码,作为 Plan A 末梢正交约束的前端补充。
- **非 claim**:不声称"层级 style/semantic persistent decomposition"(因 IN/BN concat 后被下一层 conv 混合,无持久路径)。

## Proposed Method

### Complexity Budget

| 组件 | 状态 |
|------|------|
| 主干 conv / linear | **冻结结构** |
| 前 2 个 BN(AlexNet bn1/bn2 或 ResNet-18 layer1/layer2 首 BN)| **替换为 IBN** |
| Plan A 末梢 `L_orth_end = cos²(z_sem, z_sty)` | **保留**(λ=1.0)|
| Plan A `L_HSIC`(λ=0.0 继承 EXP-017 最优) | **保留配置,不改** |
| sem_head / style_head / classifier | **复用** |
| **IBN 模块**(IN+BN parallel,affine=False for IN)| **新增,0 可训练参数** |
| **L_orth_layer**(spatial cross-correlation)| **新增 loss,0 参数** |

- 新 trainable:**0**
- 新 loss:**1**(spatial decorrelation)
- 新超参:**2**(`num_ibn=2`,`λ_orth_layer=0.5`)

### IBN 模块(带 cache,无 forward hook)

```python
class IBN(nn.Module):
    def __init__(self, channels, in_ratio=0.5):
        super().__init__()
        self.in_ch = int(channels * in_ratio)
        self.bn_ch = channels - self.in_ch
        self.instance_norm = nn.InstanceNorm2d(self.in_ch, affine=False)  # 0 params
        self.batch_norm = nn.BatchNorm2d(self.bn_ch)                      # FedBN-local
        self._cache_in = None
        self._cache_bn = None

    def forward(self, x):
        x_in, x_bn = torch.split(x, [self.in_ch, self.bn_ch], dim=1)
        out_in = self.instance_norm(x_in)
        out_bn = self.batch_norm(x_bn)
        self._cache_in = out_in     # [B, C_in, H, W]
        self._cache_bn = out_bn     # [B, C_bn, H, W]
        return torch.cat([out_in, out_bn], dim=1)
```

### Core Mechanism — Spatial Cross-Correlation Decorrelation(修正版)

**关键修正**:原 GAP-based 公式在 `IN(affine=False)` 下数学退化(per-channel spatial mean = 0)。改用 **spatial pattern 去相关**:

```python
def scpr_layer_loss(ibn_module):
    xin = F.normalize(ibn_module._cache_in.flatten(2), dim=2)   # [B, C_in, H*W]
    xbn = F.normalize(ibn_module._cache_bn.flatten(2), dim=2)   # [B, C_bn, H*W]
    # 跨通道 cross-correlation
    corr = torch.bmm(xin, xbn.transpose(1, 2)) / xin.size(-1)   # [B, C_in, C_bn]
    return corr.square().mean()

L_orth_layer = (1 / N_IBN) * Σ_l scpr_layer_loss(ibn_l)
```

**物理意义**:在 IBN 层,强制 IN 通道的空间激活模式与 BN 通道的空间激活模式**不相关**。
**数学正确性**:`normalize(flatten)` 后每条通道向量单位长度 ≠ 0,cross-correlation 矩阵**不退化**。

### 总 Loss

```
L = L_CE + λ_orth_end × L_orth_end(Plan A, λ=1.0)
       + λ_hsic × L_HSIC(λ=0.0 继承 EXP-017 最优)
       + λ_orth_layer × L_orth_layer(NEW, λ=0.5 默认)
```

### Integration

- 修改点:`FDSE_CVPR25/algorithm/feddsa_scheduled.py`:
  - `AlexNetEncoder.__init__` 里 `self.bn1 = IBN(64)`, `self.bn2 = IBN(192)`(替换原 `nn.BatchNorm2d`)
  - `Client.train()` 里 forward 后遍历 `self.ibn_layers` 拿 cache,算 `L_orth_layer`,加到 total loss
- 不用 forward hook,IBN 模块自带 cache
- 代码增量:~80 行(IBN 类 30 + 替换 BN 10 + loss hook 20 + config 10 + test 10)

### FedBN 规则

- IN 通道:parameter-free,per-sample normalize,**无 FL 同步问题**
- IBN 内 BatchNorm:跟 Plan A FedBN 一致(本地 running_mean/var;gamma/beta 按现有 feddsa_scheduled 规则)

### Failure Modes

| Failure | Signal | Fallback |
|---------|--------|----------|
| L_orth_layer 不下降 | corr.square().mean() 稳 > 0.1 20+ 轮 | 检查 normalize dim;λ 增大到 1.0 |
| PACS < Plan A | 3-seed AVG Best < 82.31% | 减 num_ibn 2→1 |
| 小 batch IN 不稳 | loss spike | 换 GroupNorm(affine=False) |
| decouple active but no acc gain | corr 下降但 acc 不升 | **Plan A 已近 ceiling**,接受 negative result |

## Claim-Driven Validation(加 IBN-only ablation)

### Claim A(主):HIBD > Plan A PACS AVG Best by ≥ +0.5% 且 decorrelation loss 有独立贡献

- Configs(**3 个,新增 IBN-only**):
  - **A.1 Plan A** orth_only
  - **A.2 IBN-only**(替换 bn1/bn2 为 IBN,**不加** L_orth_layer)— **ISOLATION CONTROL**
  - **A.3 HIBD**(IBN + L_orth_layer,λ=0.5)
- 3 seeds {2, 15, 333},R=200,PACS
- **预期决断**:
  - A.3 ≥ A.1 + 0.5%(主 claim)
  - A.3 > A.2(证明 decorrelation loss 有独立贡献,不是 IBN 结构本身)
  - A.2 ≈ A.1 或微胜(IBN 结构在 K=4 FL 下的独立效应)
- **诊断**:corr.square() 曲线(诊断,不作主证据)

### Claim B(机制,诊断):末梢 cos²(z_sem, z_sty) 在 HIBD 下比 Plan A 更低

- 零 GPU 成本,在 A.1 vs A.3 训练过程中都记录末梢 cos²
- 预期:A.3 的末梢 cos² < A.1 的末梢 cos²(前端 decorrelation 减轻末梢压力)
- **重要**:R1 reviewer 提醒"loss 下降是 tautological",这个末梢 cos² 下降**不是** tautological(我们不直接优化末梢 cos² 和 L_orth_layer 的联动),是**可证伪**的间接证据

### 实验预算

| Milestone | Runs | Time |
|-----------|------|------|
| **M0 Sanity** R=20 s=2(HIBD)| 1 | 10min |
| **M1 Mid** R=50 × 3 configs × 3 seeds(PACS) | 9 | 6h |
| **M2 Full** R=200 × 3 configs × 3 seeds(PACS)| 9 | 24h(并行 3)|
| **M3 Office** R=200 × 3 configs × 3 seeds | 9 | 12h |
| **M4 Ablation**(num_ibn ∈ {1, 3},λ ∈ {0.1, 1.0})× R=50 × 3 seeds | 12 | 8h |
| **总** | ~40 runs | ~50 GPU·h(并行 4 → 12h wall) |

---

*Round 1 refinement 完成,修复 math bug + 收窄 claim + 加 IBN-only ablation,准备 Round 2 codex 审核。*
