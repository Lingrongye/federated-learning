# FedDSA-SGPA — Style-Gated Prototype Adjustment

> 2026-04-19 research-refine 3 轮打磨产物 (9.0/10 READY)。
> 完整 FINAL_PROPOSAL: `refine_logs/2026-04-19_feddsa-eta_v1/FINAL_PROPOSAL.md`

## 一句话方案

> 在 Plan A (正交双头解耦) 之上,**训练端**把 sem_classifier 换成**固定单纯形 ETF** (零可训参数消除 FedAvg 漂移);**推理端**用**白化空间 Mahalanobis z_sty 距离** + entropy 双 gate 做可靠性筛选,用可靠样本增量更新 **top-m 原型 bank**,用 cos(z_sem, proto) 分类,ETF 顶点作为冷启动 fallback。

完全 backprop-free inference + 训练成本 = Plan A。

## 核心创新点单一化 (Round 1 修)

- **Dominant**: **SGPA (Style-Gated Prototype Adjustment)** — 唯一 headline
- **Supporting**: Fixed Simplex ETF (仅作 classifier stabilizer + proto cold-start prior)
- **不是**: 新 loss / 新聚合 / 新 backbone / 跨 client raw 特征共享

## 三个关键决策 (为什么这么做)

### 决策 1: 为什么删 AdaIN(z_sem) 改 z_sty 距离 gate
- 原方案 AdaIN(z_sem, μ_sty, σ_sty): 空间不匹配 (μ,σ 来自 z_sty 却 apply 到 z_sem),B=1 时 std NaN,`exclude_self` 在 unseen target domain 无意义
- 新方案: `reliable = (H < τ_H) & (dist_min(z_sty, source_bank) < τ_S)` — 更 faithful 解耦叙事 (z_sty 只做 gate,z_sem 只做 classification),无 forward pass 开销,无 NaN 风险

### 决策 2: 为什么引入白化 (Round 2 修)
- FedBN 私有 BN → 客户端 A 的 z_sty 和客户端 B 的 μ_k 不在同一 affine 坐标系
- L2 normalize 去尺度但去不掉 client-specific affine 畸变
- Pooled whitening (Σ_global = Σ_within + Σ_between + εI,Σ^{-1/2} 广播给所有 client) → 在白化空间 Euclidean ≡ Mahalanobis → 提供 shared metric under pooled second-order approximation
- σ 终于被用上 (原方案 σ broadcast 但未用,Reviewer 抱怨过)

### 决策 3: 为什么 warm-up 5 batch + running quantile (Round 3 修)
- first-batch quantile: order-sensitive,第一 batch 运气差会永久影响 τ_H/τ_S
- running EMA + 5-batch warm-up: 稳定,但注意 warm-up 期间仍输出 **ETF fallback prediction** (避免 `continue` 跳过样本的 bug)

## 白化细节 (principled + honest)

```python
# Server
μ_global = (1/N) Σ_k μ_sty_k                      # [d]
Σ_within = (1/N) Σ_k Σ_sty_k                      # [d,d]
Σ_between = (1/N) Σ_k (μ_sty_k-μ_global)(...)^T   # [d,d]
Σ_global = Σ_within + Σ_between + ε·I             # ε=1e-3
Λ, Q = torch.linalg.eigh(Σ_global)                # symmetric eigendecomposition
Σ_inv_sqrt = Q @ diag(clamp(Λ, min=ε)^{-1/2}) @ Q^T
# Broadcast (source_μ_k, μ_global, Σ_inv_sqrt) ≈ 66KB/round
```

**诚实声明** (Round 3 修): 这给出的是 "**shared Mahalanobis metric under a pooled second-order approximation**",不是 "provably invariant to arbitrary private BN distortion"。后者 overreach。PACS N=4 时 Σ_between rank ≤3 可能退化 → fallback 到 Σ_within only。

## SGPA 推理全流程

```python
model.eval()
μ_k_white = (source_μ_k - μ_global) @ Σ_inv_sqrt   # [N, d] 预计算
with torch.no_grad():
  for batch_idx, (x, _) in enumerate(test_loader):
    z_sem, z_sty = model(x)
    logits_etf = F.normalize(z_sem, -1) @ M / τ_etf
    pred_etf = logits_etf.argmax(1)
    H = entropy(softmax(logits_etf))
    
    z_sty_white = (z_sty - μ_global) @ Σ_inv_sqrt
    dist_min = ((z_sty_white[:,None] - μ_k_white[None])**2).sum(-1).min(-1).values
    
    if batch_idx < 5:                               # Warm-up: 输出 ETF
      buf_H += H.tolist(); buf_D += dist_min.tolist()
      yield pred_etf; continue
    elif batch_idx == 5:
      τ_H = quantile(buf_H, 0.5); τ_S = quantile(buf_D, 0.3)
    else:
      τ_H = 0.95*τ_H + 0.05*quantile(H, 0.5)
      τ_S = 0.95*τ_S + 0.05*quantile(dist_min, 0.3)
    
    reliable = (H < τ_H) & (dist_min < τ_S)
    
    for c in range(K):                              # top-m proto update
      mask = reliable & (pred_etf == c)
      if mask.any():
        supports[c].extend([(H[i].item(), z_sem[i]) for i in mask.nonzero().flatten()])
        supports[c] = sorted(supports[c], key=lambda t: t[0])[:m_top]
        proto[c] = F.normalize(torch.stack([s[1] for s in supports[c]]).mean(0), -1)
    
    proto_logits = F.normalize(z_sem, -1) @ torch.stack(proto).T
    activated = torch.tensor([len(supports[c])>0 for c in proto_logits.argmax(1)])
    pred = torch.where(activated, proto_logits.argmax(1), pred_etf)
    yield pred
```

## 预期实验结果

| 数据集 | 指标 | Plan A | FedDSA-SGPA (target) | Δ |
|--------|------|--------|---------------------|---|
| PACS | AVG Best | 80.41 | ≥ 81.5 | +1.1 |
| PACS | ALL Best | 82.31 | ≥ 83 | +0.7 |
| Office (w/o SAS) | AVG Best | 82.55 | ≥ 84 | +1.5 |
| Office (w/ SAS) | AVG Best | 89.82 | ≥ 90.5 | +0.7 |
| 3-seed std | — | 0.99-1.37 | ≤ 1.5 | — |

## Novelty 护城河 (novelty check 过关)

- **FedDEAP (arXiv'25)**: CLIP prompt-tuning + ETF decouple → **我们正交**: scratch CNN + ETF 仅 supporting + SGPA 才是 dominant
- **FedETF (ICCV'23)**: label-skew label-balanced FL → 我们 feature-skew + 真 dominant SGPA
- **SATA (IVC'25)**: centralized 图像空间 style exchange → FL + Mahalanobis z_sty gate 替代 AdaIN
- **T3A (NeurIPS'21)**: centralized proto TTA → FL + 双 gate + ETF 冷启动 + top-m (vs EMA)
- **FedCTTA (arXiv'25)**: pure entropy TTA for FL → 我们双 gate,利用 decoupled style

**空白格子**: "Disentangled source-domain style statistics 作为 FL test-time reliability signal via pooled-whitening Mahalanobis" — 无占坑。

## 核心约束 (写入 CLAUDE.md 级别)

- **z_sty 只进 gate**,不进分类;**z_sem 只进分类**,不进 gate
- **ETF 永远 subordinate**: 不能在 abstract/intro/title 和 SGPA 并列
- **Non-goal**: no raw sample sharing, but lightweight (μ, Σ) broadcast OK
- **Wording honesty**: 不说 "provably invariant",说 "pooled second-order approximation"

## 下一步

**Task 57 已跳过** (user 决定)。后续真正要跑实验时:
1. 实现 Fixed ETF buffer + SGPA inference path in clientdsa.py + serverdsa.py
2. 单元测试: gate triggering, proto update, ETF cold-start, warm-up ETF fallback
3. Plan A checkpoint 复用 + SGPA 推理 → C1 消融
4. ETF retrain R200 × 3 seeds → C2 消融
5. 组合 C1+C2 → C3 主表
