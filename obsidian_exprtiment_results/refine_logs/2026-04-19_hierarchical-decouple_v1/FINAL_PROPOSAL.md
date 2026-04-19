# Research Proposal: Early-Split IN-BN Decorrelation for FedBN (ES-IBND)

> 面向跨域联邦学习的前端轻量解耦方法 — 支撑 Plan A 末梢正交的前端正则
> 基础框架:FedDSA(Decouple-only,Share 已证伪)
> 目标会议:CVPR / ICCV / NeurIPS
> **Refine 最终评分:9.0/10 READY ✅**(4 轮 GPT-5.4 xhigh 审核)

---

## Problem Anchor

- **Bottom-line**:K=4 FedBN 跨域 FL 下,当前 FedDSA Plan A 的解耦**只在末梢 projection 层做一次** cos²(z_sem, z_sty)。85+ 实验(含刚失败的 SCPR v1/v2)全部在"已被污染的中间特征"上做下游对齐 → 都卡在 Plan A ceiling。
- **Must-solve**:
  1. shallow decouple 造成 backbone 前端无直接解耦梯度信号
  2. Plan A 已是 ceiling(82.31% AVG Best,81.17% AVG Last)
  3. 需要比 FDSE 更轻量的层级方案(FDSE 新增 DSE ≈ DFE/94 参数)
- **Non-goals**:prototype routing(证伪),新 trainable > 10% backbone,放大 backbone,VLM/CLIP,K > 4
- **Constraints**:AlexNet(主)/ ResNet-18(附录),FedBN,Plan A 目标**不变**(HSIC=0.0 继承 EXP-017),3-seed {2,15,333},代码 < 150 行,**无新增可训练参数**
- **Success**:PACS AVG Last ≥ **81.67** (3-seed mean, std ≤ 1.5);A.3 > A.2 on AVG Last(证明 decorrelation loss 独立贡献)

---

## Technical Gap

Plan A 的 `cos²(z_sem, z_sty)` **只在最后 128d projection**。Backbone conv/BN 层只收到间接弱梯度 → 中间特征早已污染 → 所有下游 prototype/alignment 都是徒劳。

FDSE(CVPR 2025)用**每层 DFE+DSE 结构分解**(DSE 是 grouped depth-wise conv,新参数 ≈ DFE/94)。我们的差异化:**不改结构,只改 normalization scheme + 加 decorrelation loss,无新增 trainable**。

**切入点**:IBN-Net(Pan et al. ECCV 2018)证明 **IN 通道自然分离 style,BN 通道保留 content**,IN+BN 混合在单机 DG 有 +2-3% reportable gain。**我们将 IBN 移植到 FedBN 框架,并在 IBN 层加 pairwise spatial cosine decorrelation 损失**。

---

## Method Thesis

> **Replacing early FedBN layers with split IN/BN plus pairwise spatial cosine decorrelation improves cross-domain FL without adding trainable parameters.**

Early-split(前 2 BN → IBN)+ pairwise spatial cosine decorrelation,作为 Plan A 末梢正交的前端正则。**不声称**持久化 style/semantic 分离(IN/BN concat 后被下一层 conv 混合,无持久路径)。

---

## Contribution Focus

- **唯一主贡献**:**Early-Split IN-BN Decorrelation (ES-IBND)** for FedBN — 0 新可训练参数,~80 行代码,pairwise spatial cosine decorrelation 作为 Plan A 末梢 cos² 的**前端补充正则**
- **非贡献**:不做 prototype routing / alignment / style sharing(均已证伪)

---

## Proposed Method

### Complexity Budget

| 组件 | 状态 |
|------|------|
| 主干 conv/linear(AlexNet/ResNet-18)| **冻结结构** |
| **Plan A 目标不变;HSIC 系数继续为 0.0(继承 EXP-017 最优)** | **原封保留** |
| sem_head / style_head / classifier | **复用** |
| 前 2 个 BN(AlexNet bn1/bn2) | **替换为 IBN**(IN affine=False + BN 各占 50% 通道) |
| **IBN 模块**(含 cache) | **新增,无新可训练**(IN 无参数,BN 参数跟原 BN 一样) |
| **L_orth_layer**(spatial cosine decorrelation) | **新增 loss,无参数** |

**Summary**:
- 新可训练参数:**0**
- 新 loss:1
- 新超参:2(`num_ibn=2` locked,`λ_orth_layer=0.5` locked)
- 新代码:~80 LOC

### IBN Module(cache-based,无 forward hook)

```python
class IBN(nn.Module):
    def __init__(self, channels, in_ratio=0.5):
        super().__init__()
        self.in_ch = int(channels * in_ratio)
        self.bn_ch = channels - self.in_ch
        self.instance_norm = nn.InstanceNorm2d(self.in_ch, affine=False)
        self.batch_norm = nn.BatchNorm2d(self.bn_ch)
        self._cache_in = None
        self._cache_bn = None
    def forward(self, x):
        xin, xbn = torch.split(x, [self.in_ch, self.bn_ch], dim=1)
        out_in = self.instance_norm(xin)
        out_bn = self.batch_norm(xbn)
        self._cache_in = out_in   # [B, C_in, H, W]
        self._cache_bn = out_bn   # [B, C_bn, H, W]
        return torch.cat([out_in, out_bn], dim=1)
```

### Core Mechanism — Pairwise Spatial Cosine Decorrelation(Locked 公式)

```python
def layer_decorr_loss(ibn):
    xin = F.normalize(ibn._cache_in.flatten(2), dim=2)   # [B, Cin, HW] unit
    xbn = F.normalize(ibn._cache_bn.flatten(2), dim=2)   # [B, Cbn, HW] unit
    corr = torch.bmm(xin, xbn.transpose(1, 2))           # [B, Cin, Cbn]
    return corr.square().mean()

L_orth_layer = (1 / N_IBN) * sum_l layer_decorr_loss(ibn_l)
```

**语义**:对每 (IN 通道, BN 通道) 对,计算两者 spatial activation 序列的 cos 相似度平方 → 推向 0(正交)。
**数学正确**:`F.normalize` 单位化后 bmm 产生 [-1, 1] cosine,无 `/HW` 重复;loss 在 [0, 1] 可直接阈值解读。

### 总损失

```
L = L_CE + 1.0 × L_orth_end(Plan A cos²)
       + 0.0 × L_HSIC(EXP-017 最优)
       + 0.5 × L_orth_layer(NEW, locked)
```

### Integration

- **主 AlexNet**:`AlexNetEncoder.bn1 = IBN(64)`,`self.bn2 = IBN(192)`
- **附录 ResNet-18**:`layer1[0].bn1 = IBN`,`layer2[0].bn1 = IBN`
- `Client.train()` 后 forward,遍历 `self.ibn_layers` 收集 cache 算 `L_orth_layer`
- 代码量 ~80 行,**无 forward hook**

### FedBN 规则

- IN 通道:parameter-free,per-sample,**无 FL 同步**
- IBN 内 BN(半通道):同 Plan A FedBN(本地 running stats)

### Failure Modes

| Failure | Signal | Fallback |
|---------|--------|----------|
| L_orth_layer 不下降 | `corr.square().mean()` 稳 > 0.1 > 20 轮 | 查 `normalize` dim;λ 调到 1.0 |
| PACS AVG Last < Plan A | 3-seed mean < 81.17 | num_ibn 2→1;λ 降到 0.1 |
| IN batch-1 不稳 | loss NaN | 换 `GroupNorm(affine=False)` 作 style 分支 |
| Loss 下降但 acc 不升 | corr ↓ 但 AVG 停 | **证据表明 Plan A 已 ceiling**,接受 negative result,**不加新模块** |

### Novelty 与 Elegance

- **~80 LOC,0 新 trainable,1 新 loss,2 新超参 locked**
- **vs FDSE**:FDSE 用 DSE 新参数分解,我们**无新参数**
- **vs FedBN**:FedBN 只本地化 BN stats,我们加 IN 分支 + 显式正交 loss
- **vs IBN-Net**:IBN-Net 是单机 DG,我们加**显式正交 loss + FedBN 共存规则**
- **vs Plan A**:前端 2 层直接施加解耦梯度,减轻末梢层压力

---

## Claim-Driven Validation

### Claim A(主):ES-IBND > Plan A AVG Last ≥ +0.5,且 decorrelation loss 有独立贡献

**Configs(3)**:
- A.1 Plan A orth_only
- **A.2 IBN-only**(IBN 替换,**不加** L_orth_layer)← isolation control
- **A.3 ES-IBND**(IBN + L_orth_layer λ=0.5)

**数据**:PACS,AlexNet,3 seeds {2, 15, 333},R=200

**主表报告标准(mean ± std over 3 seeds)**:

| Config | AVG Best | AVG Last | ALL Best | ALL Last | Art | Cartoon | Photo | Sketch |
|--------|----------|----------|----------|----------|-----|---------|-------|--------|
| A.1 Plan A | 82.31 ± ? | 81.17 ± ? | 80.41 | 79.42 | ... | ... | ... | ... |
| A.2 IBN-only | ? | ? | ? | ? | ... | ... | ... | ... |
| A.3 ES-IBND | ? | ? | ? | ? | ... | ... | ... | ... |

**主决策指标:AVG Last**(Best 仅 supportive)

**Decisive 条件**:
- A.3 AVG Last ≥ **81.67**(+0.5 over Plan A 81.17)★ 主 claim
- A.3 AVG Last std ≤ 1.5(稳定性)
- A.3 > A.2 on AVG Last(decorrelation loss 独立贡献)
- A.3 per-domain ≥ A.1 per-domain − 1% on worst domain(无 hidden crash)

### Claim B(机制诊断,非 tautological):ES-IBND 末梢 cos²(z_sem, z_sty) 比 Plan A 低

- 零 GPU 成本,训练过程 track 末梢 cos² 演化
- 不 tautological(我们不直接优化末梢 cos² 和 L_orth_layer 的联动)
- 证据:前端 decorrelation **传播**到末梢,证明机制不止局部

### 附录实验(Office-Caltech10 ResNet-18)

1 个 compact table,3-seed mean ± std 对 A.1/A.2/A.3。插入点:`layer1[0].bn1`, `layer2[0].bn1`。

### 实验预算

| Milestone | Runs | Time |
|-----------|------|------|
| **M0 Sanity** R=20 s=2 (A.3) | 1 | 10 min |
| **M1 Full** R=200 × 3 configs × 3 seeds PACS | 9 | 24h(并行 3)|
| **M2 Appendix** R=200 × 3 configs × 3 seeds Office | 9 | 12h(ResNet-18 E=1 快)|
| **总** | ~19 runs | ~36 GPU·h(并行 3)|

---

## Experiment Handoff Inputs

- **Must-prove claims**:A(AVG Last +0.5 + decorrelation 独立贡献)+ B(机制诊断)
- **Must-run ablations**:Plan A / IBN-only / ES-IBND 三配置;λ_orth_layer ∈ {0.1, 0.5, 1.0} 扫作 appendix
- **Critical metrics**:PACS AVG Last(主决策)、AVG Best、ALL B/L、per-domain(worst-domain 检查)
- **Highest-risk assumptions**:
  1. IN 通道真承载 style(IBN-Net 报告 +2-3% DG,FL 设置未测)
  2. 在 K=4 下,layer-level orth loss 不干扰 CE
  3. 前 2 层 IBN 足够(不用深层)

---

## Compute & Timeline

- **GPU·h**:~36(主 + 附录),单卡并行 3 ≈ 12-16h wall-clock
- **人力 timeline**:
  - Day 1:实现 IBN 模块 + 单测 + codex review
  - Day 2:M0 sanity(10 min)→ M1 Full PACS(24h 后台)
  - Day 3:M2 Office 附录 + 回填 NOTE + Obsidian 同步
  - **总 3 天 end-to-end**

---

*文档版本:R4 Final(2026-04-19 refine session v1)*
*基础:GPT-5.4 xhigh 4 轮 refine 收敛,最终评分 9.0/10 READY*
