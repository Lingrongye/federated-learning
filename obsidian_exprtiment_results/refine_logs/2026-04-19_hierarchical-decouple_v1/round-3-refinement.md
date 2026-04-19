# Round 3 Refinement — 吃下 4 个 exact single-line fix

## Anchor Check + Simplicity Check
- R3 reviewer 明确说 "core mechanism is proposal-ready, stop revising and run it"
- 4 个 exact fix 都是 paper hygiene,不改 mechanism
- 无 drift,不增 contribution

## Changes (R3 exact single-line fixes)

### 1. 措辞修正 Plan A HSIC 说明
**从**:`Plan A cos²+HSIC stays`
**改**:`Plan A objective stays unchanged; HSIC coefficient remains 0.0 per EXP-017 finding.`

### 2. Claim A 主决策指标改 AVG Last
**从**:AVG Best 为主决定
**改**:**AVG Last** 为主决定指标(Best 仅 supportive)
- 原因:Best 只反映训练过程中某一 peak,Last(R=200 最后一轮)反映收敛质量
- 新决定条件:A.3 AVG Last ≥ Plan A AVG Last + 0.5%(82.31 AVG Best → 81.17 AVG Last,目标 ≥ 81.67)

### 3. 报告标准 mean ± std
**新要求**:主表每个数字必须是 **3-seed mean ± std**(不只 mean)
- 捕获稳定性:若 std > 1.5,即使 mean 达标也要打折扣
- 示例表头:`AVG Best (mean ± std) | AVG Last (mean ± std)`

### 4. 主 claim 聚焦 AlexNet,ResNet-18 移 Appendix
- 主论文/主实验只跑 **PACS AlexNet**
- Office ResNet-18 移 Appendix,一句话说明 ResNet-18 的 2 个 BN 插入点(layer1.0.bn1 + layer2.0.bn1 首个 BN)
- 原因:AlexNet 是我们主战场,先证 mechanism 在 1 个 backbone 上 work

## Revised Proposal (Round 3 final)

# Early-Split IN-BN Decorrelation for FedBN (ES-IBND) — R3 Final

## Problem Anchor (preserved)
(略)

## Method Thesis(同 R2)
> "Replacing early FedBN layers with split IN/BN plus pairwise spatial cosine decorrelation improves cross-domain FL **without adding trainable parameters**."

## Proposed Method

### Complexity Budget
- **Plan A objective stays unchanged; HSIC coefficient remains 0.0 per EXP-017 finding.**
- Replaced: AlexNet bn1, bn2 → IBN(IN affine=False on half channels + BN on other half)
- New: L_orth_layer (spatial cosine decorrelation, no params)
- **No added trainable parameters**
- 2 new hyperparameters locked: num_ibn=2, λ_orth_layer=0.5

### IBN Module(同 R2)
(略)

### Core Mechanism (R2 locked formula,不变)
```python
def layer_decorr_loss(ibn):
    xin = F.normalize(ibn._cache_in.flatten(2), dim=2)   # [B, Cin, HW]
    xbn = F.normalize(ibn._cache_bn.flatten(2), dim=2)   # [B, Cbn, HW]
    corr = torch.bmm(xin, xbn.transpose(1, 2))           # [B, Cin, Cbn]
    return corr.square().mean()
```

### Total Loss
L = L_CE + 1.0 × L_orth_end + 0.0 × L_HSIC + 0.5 × L_orth_layer

### Integration — 主 backbone AlexNet
- `AlexNetEncoder.bn1 = IBN(64)`, `self.bn2 = IBN(192)`
- Client.train() 后 forward 收集 IBN caches 算 loss
- ~80 LOC

### Appendix Only — ResNet-18 extension
- 2 个 BN 插入点:`layer1[0].bn1`, `layer2[0].bn1`(stage 1/2 首 block 首 BN)
- 同一 IBN 类,同一 decorrelation loss
- Office 附录表呈现,不进主论文 headline

## Claim-Driven Validation (R3 final metrics)

### Claim A (main): ES-IBND > Plan A PACS AVG Last ≥ +0.5%

**Main decision metric: AVG Last**(不是 Best;Best 降级为 supportive)

Configs (3):
- A.1 Plan A orth_only
- A.2 IBN-only (IBN, no L_orth_layer)
- A.3 ES-IBND (IBN + L_orth_layer)

PACS 3 seeds {2, 15, 333}, R=200, AlexNet.

**Main Table 报告(mean ± std,3-seed)**:
| Config | AVG Best | AVG Last | ALL Best | ALL Last | Art | Cartoon | Photo | Sketch |
|--------|---------|----------|----------|----------|-----|---------|-------|--------|
| A.1 Plan A | 82.31 ± ? | 81.17 ± ? | ... | ... | ... | ... | ... | ... |
| A.2 IBN-only | — ± — | — ± — | ... | ... | ... | ... | ... | ... |
| A.3 ES-IBND | — ± — | — ± — | ... | ... | ... | ... | ... | ... |

Decisive on **AVG Last**:
- A.3 AVG Last ≥ 81.67(+0.5 over Plan A 81.17)← 主 claim
- A.3 AVG Last std ≤ 1.5(稳定性)
- A.3 > A.2 on AVG Last(decorrelation loss 独立贡献)
- A.2 vs A.1(IBN 结构 DG 效应)

### Claim B (mechanism diagnostic)
末梢 cos²(z_sem, z_sty) 在 A.3 下比 A.1 低。零 GPU 成本。

### Worst-domain check (reviewer 建议)
Main table 同时呈现 per-domain,避免"某个 domain 崩但平均赢"。若 A.3 on Sketch < Plan A on Sketch,即使 AVG 赢也要 flag 为 partial win。

## Budget: ~15 GPU-hours (PACS 3 configs × 3 seeds × 8h / 并行 3 + 附录 Office)
