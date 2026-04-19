# Round 2 Refinement — 锁定 loss 公式 + 改名去 hierarchical + 报 final+best

---

## Problem Anchor(不变)
(略)

## Anchor Check
- R2 reviewer 全部是措辞 / 公式精确度问题,不改变 anchor
- **无 drift**

## Simplicity Check
- 主贡献已收窄(R1 完成),本轮只做**文档级**调整
- 不增加 contribution,不改 ablation 结构

---

## Changes Made(针对 R2 critique)

### 1. [CRITICAL] 锁定 loss 公式 — 选 cosine decorrelation(去掉 /HW)

**原写法(不一致)**:
```python
xin = F.normalize(out_in.flatten(2), dim=2)   # 单位化 [B, Cin, HW]
xbn = F.normalize(out_bn.flatten(2), dim=2)
corr = torch.bmm(xin, xbn.transpose(1, 2)) / xin.size(-1)   # ← /HW 重复
```

**问题**:normalize 后已经单位向量,bmm 产生 [-1, 1] cos 相似度矩阵,再除 HW 让数值 ≈ 0,诊断阈值 `> 0.1` 不可解读。

**Reviewer 建议两选一,我选 cosine decorrelation**:
```python
def layer_decorr_loss(ibn):
    xin = F.normalize(ibn._cache_in.flatten(2), dim=2)   # [B, Cin, HW] unit
    xbn = F.normalize(ibn._cache_bn.flatten(2), dim=2)   # [B, Cbn, HW] unit
    corr = torch.bmm(xin, xbn.transpose(1, 2))           # [B, Cin, Cbn], no /HW
    return corr.square().mean()
```

**语义**:对每个 sample / 每对 (IN_ch, BN_ch),计算 spatial 激活序列的 cos 相似度;平方 → 强制接近 0(正交);mean 跨 batch、跨 channel pair。

**阈值意义**:corr in [-1,1],square ∈ [0,1]。预期训练末 `corr.square().mean()` 从 0.1-0.3 降到 < 0.05。

### 2. [CRITICAL] 改名去 "Hierarchical"(只用前 2 层,不是真 hierarchy)

- **R2 reviewer 说**:"'hierarchical' is stronger than what is actually implemented(只有 2 个 IBN,不算 depth stages)"
- **新名字**:**Early-Split IN-BN Decorrelation for FedBN (ES-IBN-D)** 或简写 **ES-IBND**
- **论文标题参考**:"Early-Split Instance-Batch Decorrelation for Cross-Domain Federated Learning"

### 3. [CRITICAL] 措辞:no added trainable parameters

- **原**:"0 new trainable parameters"(reviewer 担心被 BN 参数 count 反驳)
- **新**:"**no added trainable parameters**"(更精确措辞,因为 IBN 的 BN 分支仍有 BN 参数,只是总数不增加)

### 4. [CRITICAL] 主表报 **AVG Best + AVG Last + ALL Best + ALL Last**

- **原**:只 AVG Best 作主指标(被 reviewer 说 "soft")
- **新**:主结果表必须 4 个指标齐报(Best/Last × ALL/AVG)
- **Final(Last)指标更严格**,表征收敛质量

### 5. [简化] 锁定默认超参,ablation 最小化

- `num_ibn = 2` 锁定默认(bn1/bn2),**不扫 {1, 3}**(reviewer 说 "too broad search space")
- `in_ratio = 0.5` 锁定默认,不扫
- `λ_orth_layer = 0.5` 锁定默认,但保留一个 3-value sweep(仅作 appendix)

---

## Revised Proposal(Round 2)

# Research Proposal Round 2: Early-Split IN-BN Decorrelation for FedBN (ES-IBND)

## Problem Anchor(preserved)

## Technical Gap
(同 R1)

## Method Thesis(锁定版)

> **Headline**:"Replacing early FedBN layers with split IN/BN plus pairwise spatial cosine decorrelation improves cross-domain FL **without adding trainable parameters**."

Early-split(前 2 BN → IBN)+ 空间 cosine decorrelation 作为 Plan A 末梢解耦的前端补充。**不声称**持久化 style/semantic 分离。

## Contribution Focus

- **唯一主贡献**:**Early-Split IN-BN Decorrelation(ES-IBND)** — 在 FedBN 框架下用 IN+BN parallel + spatial cosine decorrelation 作为 Plan A 末梢正交约束的前端正则,无新增可训练参数

## Proposed Method

### Complexity Budget
- Frozen: backbone conv/linear, Plan A L_orth_end(λ=1.0), L_HSIC(λ=0.0), sem_head, style_head, classifier
- Replaced: `bn1`, `bn2` → IBN instances (IN affine=False, BN on half channels)
- New: L_orth_layer(spatial cosine decorrelation)
- **No added trainable parameters**
- 2 new hyperparameters: `num_ibn=2` (locked default), `λ_orth_layer=0.5` (locked default)

### IBN Module(同 R1,cache-based)
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
        self._cache_in, self._cache_bn = out_in, out_bn
        return torch.cat([out_in, out_bn], dim=1)
```

### Core Mechanism — Spatial Cosine Decorrelation(锁定 R2 final 公式)

```python
def layer_decorr_loss(ibn):
    xin = F.normalize(ibn._cache_in.flatten(2), dim=2)   # [B, Cin, HW] unit vectors along HW
    xbn = F.normalize(ibn._cache_bn.flatten(2), dim=2)   # [B, Cbn, HW]
    corr = torch.bmm(xin, xbn.transpose(1, 2))           # [B, Cin, Cbn] cos similarities
    return corr.square().mean()                          # push toward 0

L_orth_layer = (1 / N_IBN) * sum_l layer_decorr_loss(ibn_l)
```

**无 /HW**(normalize 已单位化);loss 值在 [0,1],直观解读。

### Total Loss
```
L = L_CE + 1.0 * L_orth_end(Plan A) + 0.0 * L_HSIC(EXP-017 optimal) + 0.5 * L_orth_layer(NEW)
```

### Integration
- Edit `feddsa_scheduled.py`:
  - `AlexNetEncoder`:`self.bn1 = IBN(64)`,`self.bn2 = IBN(192)`
  - `Client.train()` 后 forward,遍历 `self.ibn_layers` sum decorrelation
- **~80 行**代码变更,无 forward hook

## Claim-Driven Validation

### Claim A(主):ES-IBND > Plan A AVG Best ≥ +0.5% 且 decorrelation loss 有独立贡献

- Configs(3):
  - A.1 Plan A orth_only
  - **A.2 IBN-only**(IBN 替换但无 L_orth_layer)← isolation control
  - **A.3 ES-IBND**(IBN + L_orth_layer)
- 3 seeds,R=200,PACS
- **报 4 指标**:AVG Best / AVG Last / ALL Best / ALL Last
- 预期:
  - A.3 AVG Best ≥ 82.81%
  - A.3 > A.2(decorrelation loss 独立贡献)
  - A.3 AVG Last 也 ≥ Plan A + 0.3%(训练稳定,不只是 peak)

### Claim B(机制诊断):末梢 cos² 在 ES-IBND 下比 Plan A 更低

零 GPU 成本;证据 early decorrelation 传播。

### Budget
- PACS 主实验:3 configs × 3 seeds = 9 runs × 8h = 72h 顺序,并行 3 → 24h
- Ablation(λ_orth_layer ∈ {0.1, 0.5, 1.0} × 3 seeds × R=50):9 runs × 2h = 18h 顺序,并行 3 → 6h
- Office(可选,复用 3 configs × 3 seeds):9 runs
- **总 ~30 runs,~15 GPU·h(并行 3)**

---

*Round 2 refinement 完成:loss 公式锁定(去 /HW),改名 ES-IBND,4 指标报告,锁定默认超参。准备 Round 3 codex。*
