# Round 1 Review — GPT-5.4 xhigh

**Verdict**: REVISE
**Overall**: **6.7/10**

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| Problem Fidelity | 8 | 直接针对 shallow-decouple 诊断,无 drift |
| Method Specificity | 6 ⚠️ | **L_orth_layer 数学退化**!IN+GAP 后恒为 0 |
| Contribution Quality | 6 ⚠️ | 过度声称"层级 style/semantic 分解",IN/BN concat 后 conv 混合 |
| Frontier Leverage | 8 | 克制合理,无需 VLM |
| Feasibility | 7 | 实现容易,科学风险大于工程风险 |
| Validation Focus | 6 ⚠️ | 缺 IBN-only ablation,AVG Best 太 soft |
| Venue Readiness | 6 ⚠️ | "0 new trainable"本身不够 strong |

## 3 个 CRITICAL

### 1. **致命 Math Bug**:`L_orth_layer` 在 `IN(affine=False) + GAP` 下数学退化

IN affine=False 的 per-channel spatial mean 定义上就是 0 → GAP(IN) = 0 → cos² = 0 / 0 → loss 无信号

**Fix**:改用 spatial cross-correlation(保留 H×W 空间结构)
```python
xin = F.normalize(out_in.flatten(2), dim=2)   # [B, Cin, HW]
xbn = F.normalize(out_bn.flatten(2), dim=2)   # [B, Cbn, HW]
corr = torch.bmm(xin, xbn.transpose(1, 2)) / xin.size(-1)
L_layer = corr.square().mean()
```

### 2. Claim 过度承诺

"hierarchical style/semantic separation" 不成立 — IN/BN concat 后下一层 conv 混合,无持久分离路径

**Fix**:narrow claim 为 **"parameter-free early-split regularization for FedBN that supports end-layer decouple"**

### 3. 缺 IBN-only 独立 ablation

无 IBN-only 配置时,不能区分"IBN 结构"和"L_orth_layer loss"的贡献

**Fix**:加 A.2 IBN-only config(IBN 替换但不加 orth loss),三组对比:Plan A / IBN-only / HIBD

## Simplification Suggestions

- Plan A 末梢 objective 不改(包括 HSIC=0 继承 EXP-017)
- 只用一个 backbone 验证 mechanism,先别 AlexNet + ResNet-18 同时证
- 避免 forward hook,用 IBN 自带 cache

## Modernization

NONE — VLM/diffusion 会 drift,不加

## Drift Warning

NONE(但警告:若声称 generic IBN for DG 会 drift;若加结构分支持久化路径会 drift 到 FDSE 复杂度)

## Reviewer Summary

> HIBD is the first pivot here that is actually anchored to the diagnosed failure mode. The proposal is admirably focused. BUT as written, the core layer loss is invalid; paper overclaims mechanism. Fix the local objective, freeze Plan A exactly, add the indispensable `IBN-only` ablation, narrow the claim.
>
> If corrected HIBD still cannot beat Plan A cleanly, I would treat that as evidence that Plan A is already near ceiling.
