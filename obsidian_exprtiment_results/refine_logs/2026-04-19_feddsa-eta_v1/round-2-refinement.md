# Round 2 Refinement — FedDSA-SGPA (Mahalanobis + Whitening + running quantile)

## Problem Anchor (Copy Verbatim, UPDATED non-goal)

- **Bottom-line problem**: (unchanged) 跨域 FL,在 Plan A 之上 +≥1% AVG,PACS+Office 通用,无持续崩溃机制
- **Must-solve bottleneck**: (unchanged) (a) sem_classifier 漂移;(b) z_sty + style_bank 推理未用
- **Non-goals (NARROWED per Round 2 feedback)**:
  - 不引入持续对齐损失
  - **不共享 raw feature/sample**,但允许 **lightweight source-domain summary statistics (μ_sty_k, Σ_sty_k)** 广播用于 test-time shift detection — 和 FedBN 广播 γ/β 同量级 (每轮 N × 2 × d_sty = 4×2×128 = 1024 floats)
  - 不用 CLIP/VLM
  - 不抢 "FL+ETF+decouple 第一"

## Anchor Check
- 原两个 bottleneck 依然被直接 attack ✅
- Non-goal 从 "不跨客户端风格共享" 放宽到 "不共享 raw feature,但允许 summary stats" — 这是必要的 narrowing,因为 Round 2 指出 source_style_bank 广播是 lightweight style sharing 的子集

## Simplicity Check
- **Dominant**: 仍是 SGPA 单点贡献,ETF supporting
- **新增**: Whitening 不是新贡献,只是 "让 cross-client z_sty 可比的 preprocessing step"
- **SGPA 实际唯一新机制**: double gate (entropy + Mahalanobis-in-whitened-space) + top-m proto bank

## Changes Made (from Round 1 refinement to Round 2)

### 1. CRITICAL: Mahalanobis gate in whitening space
- **Reviewer**: "FedBN 私有 BN → z_sty 跨 client affine distortion → L2 dist is heuristic"
- **New design**:
  ```python
  # Server aggregates global whitening stats (running, lightweight):
  #   μ_global = (1/N) Σ_k μ_sty_k        # [d_sty]
  #   Σ_global = (1/N) Σ_k Σ_sty_k + cov(μ_sty_k across clients)  # [d_sty, d_sty]
  # Per round broadcast: (μ_global, Σ_global^{-1/2}) — same size as source_style_bank was
  
  # Inference (per client, own test set):
  z_sty_white = (z_sty - μ_global) @ Σ_global_inv_sqrt     # [B, d_sty], whitened
  μ_k_white = (μ_k - μ_global) @ Σ_global_inv_sqrt          # [N, d_sty], pre-compute once
  
  # Mahalanobis in whitened = Euclidean
  dist = ((z_sty_white[:, None, :] - μ_k_white[None, :, :]) ** 2).sum(-1)  # [B, N]
  dist_min = dist.min(dim=-1).values                                        # [B]
  ```
- **为什么这有效**:
  - Whitening 把所有 z_sty 投影到 "cross-client common standard-normal space"
  - 在白化后空间做 Euclidean ≡ 原始空间 Mahalanobis
  - 理论上 principled: 假设 z_sty_k ~ N(μ_k, Σ_k),whitening 是最优跨客户端 comparability fix
  - Σ 终于被用上 (Round 2 抱怨 "σ 未用")
- **Overhead**: 每轮多广播 Σ^{-1/2} matrix [d×d = 128×128 = 16K floats],仍 lightweight
- **Numerical guards**: Σ + ε·I (ε=1e-3),`torch.linalg.cholesky()` 或 `symeig` 构造 Σ^{-1/2}

### 2. IMPORTANT: 删除 bn.track_running_stats=False
- **Reviewer**: "这个 flag 反而把 BN 推向 batch-stat,weakens 稳定"
- **Action**: 只保留 `model.eval() + torch.no_grad()`,BN 用 FedBN private running stats

### 3. IMPORTANT: Running quantile calibration
- **Reviewer**: "first-batch quantile is order-sensitive"
- **New design**:
  ```python
  # 推理时维护 running p50/p30 of H and dist_min
  # 前 W=5 batch 做 warm-up (只记录 stats,不做 gate),之后开始 gate
  if batch_idx < 5:
    warm_up_H.append(H); warm_up_dist.append(dist_min)
    reliable = torch.zeros_like(...).bool()   # warm-up 期间全 False
  else:
    if batch_idx == 5:
      τ_H = quantile(concat(warm_up_H), 0.5)
      τ_S = quantile(concat(warm_up_dist), 0.3)
    # 每 batch 小幅 EMA 更新 τ (稳健)
    τ_H = 0.95*τ_H + 0.05*quantile(H, 0.5)
    τ_S = 0.95*τ_S + 0.05*quantile(dist_min, 0.3)
    reliable = (H < τ_H) & (dist_min < τ_S)
  ```
- **Impact**: 跨 batch 稳定,不依赖单一 batch 样本;warm-up 期让 bank 收集冷启动

### 4. IMPORTANT: ETF 在 abstract/intro 保持 subordinate
- **Paper structure 约束**:
  - Abstract 主语: SGPA, ETF 出现在一句 "we use a fixed ETF classifier as stabilizer"
  - Intro 主线: "We enable test-time prototype correction in federated DG via disentangled style statistics as reliability signal"
  - Method section: SGPA 占 60%,ETF 占 20%,integration 占 20%
  - Title 候选: "Style-Gated Prototype Adjustment for Federated Domain Generalization"

### 5. MINOR: notation 统一
- Paper 只用一种:"**normalized L2**" (i.e., 1 - cosine on normalized vectors)
- 代码里 `1 - z_sty_n @ mu_bank_n.T` → 改名 `cos_dist`

---

## Revised Proposal: FedDSA-SGPA v2

### One-Sentence Thesis (unchanged)
> Disentangled source-domain style statistics serve as a **federated, backprop-free reliability signal** for test-time prototype correction.

### Key Math Update

```python
# ===== Training: each client per-round =====
for k in clients:
  μ_sty_k = z_sty.mean(0)                                # [d_sty]
  Σ_sty_k = (z_sty - μ_sty_k).T @ (z_sty - μ_sty_k) / (B-1)  # [d_sty, d_sty]
  upload (μ_sty_k, Σ_sty_k)

# ===== Server aggregation =====
μ_global = (1/N) Σ_k μ_sty_k                            # [d_sty]
Σ_within = (1/N) Σ_k Σ_sty_k                            # avg within-client cov
Σ_between = (1/N) Σ_k (μ_sty_k - μ_global)(μ_sty_k - μ_global)^T  # between-client cov
Σ_global = Σ_within + Σ_between + ε·I                   # [d_sty, d_sty], ε=1e-3
Σ_inv_sqrt = symeig(Σ_global).compute_inv_sqrt()        # Q Λ^{-1/2} Q^T

# ===== Broadcast to clients =====
payload = {
  'source_μ_k':  {k: μ_sty_k for k in N},               # [N, d_sty]
  'μ_global':    μ_global,                              # [d_sty]
  'Σ_inv_sqrt':  Σ_inv_sqrt                             # [d_sty, d_sty]
}

# ===== Inference (per client) =====
model.eval()
# whiten client's own source bank (one-time)
μ_k_white = (source_μ_k - μ_global) @ Σ_inv_sqrt        # [N, d_sty]

with torch.no_grad():
  for batch_idx, (x, y) in enumerate(test_loader):
    z_sem, z_sty = model(x)
    logits_etf = F.normalize(z_sem, dim=-1) @ M / τ_etf  # [B, K]
    H = -(softmax(logits_etf) * log_softmax(logits_etf)).sum(-1)
    
    # Whiten test z_sty and compute Mahalanobis-in-whitened-space
    z_sty_white = (z_sty - μ_global) @ Σ_inv_sqrt        # [B, d_sty]
    dist_mtx = ((z_sty_white[:, None] - μ_k_white[None]) ** 2).sum(-1)  # [B, N]
    dist_min = dist_mtx.min(-1).values                   # [B]
    
    # Running quantile (warm-up first 5 batches)
    if batch_idx < 5:
      reliable = torch.zeros(B).bool()
      warm_up_buffer_H.extend(H.tolist())
      warm_up_buffer_D.extend(dist_min.tolist())
    elif batch_idx == 5:
      τ_H = np.quantile(warm_up_buffer_H, 0.5)
      τ_S = np.quantile(warm_up_buffer_D, 0.3)
      reliable = (H < τ_H) & (dist_min < τ_S)
    else:
      τ_H = 0.95*τ_H + 0.05*np.quantile(H.cpu(), 0.5)
      τ_S = 0.95*τ_S + 0.05*np.quantile(dist_min.cpu(), 0.3)
      reliable = (H < τ_H) & (dist_min < τ_S)
    
    # Top-m per-class proto update
    pred_etf = logits_etf.argmax(1)
    for c in range(K):
      mask = reliable & (pred_etf == c)
      if mask.any():
        supports[c].extend([(H[i].item(), z_sem[i]) for i in mask.nonzero().flatten()])
        supports[c] = sorted(supports[c], key=lambda t: t[0])[:m_top]
        proto[c] = F.normalize(torch.stack([s[1] for s in supports[c]]).mean(0), dim=-1)
    
    # Classification
    proto_tensor = torch.stack(proto)
    proto_logits = F.normalize(z_sem, dim=-1) @ proto_tensor.T
    pred_proto = proto_logits.argmax(1)
    activated = torch.tensor([len(supports[c]) > 0 for c in pred_proto])
    pred = torch.where(activated, pred_proto, pred_etf)
```

### Communication Overhead (updated)
- 每轮 additional: 4×128 (μ_k) + 128×128 (Σ_inv_sqrt) + 128 (μ_global) = 16.6K floats = **66 KB**
- 与 FedBN 每轮传输 2×number_of_BN_channels γ/β 同量级,trivial

### Failure Modes (updated)
| 失败模式 | 检测 | Fallback |
|---------|------|---------|
| Σ_global 奇异 / 病态 | `torch.linalg.cond(Σ_global)` > 1e6 报警 | ε 增大到 1e-2 |
| PACS 4 client N=4 太少,Σ_between 秩 ≤ 3 | 监控 Σ_between 特征值分布 | 只用 Σ_within whitening |
| 前 5 batch warm-up 无 gate → 部分 test 样本没 TTA | 接受,影响 ≤ 5/测试 batch 数 | test_loader 只跑 1 遍 |
| Whitening 后 dist 分布异常 | 可视化 dist_min 直方图 | 回退纯 entropy gate |

### Validation (unchanged from Round 1 refinement)
C1 (SGPA +0.5~2%), C2 (ETF no harm + std ↓), C3 (组合达 success condition)

## Expected Score Improvements
- Feasibility 8.1 → 9+ (Mahalanobis whitening solves cross-client comparability)
- Validation Focus 8.3 → 9+ (ablation 新增: whitening on/off, Mahalanobis vs L2)
- Venue 8.5 → 9+ (FedBN 理论 justification 到位)
- Overall 8.5 → **9+** target
