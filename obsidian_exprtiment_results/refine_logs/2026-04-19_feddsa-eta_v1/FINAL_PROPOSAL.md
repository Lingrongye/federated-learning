# Research Proposal (FINAL): **FedDSA-SGPA**
## Style-Gated Prototype Adjustment for Federated Domain Generalization

---

## 1. Problem Anchor

- **Bottom-line problem**: 在跨域联邦学习 (feature-skew FL, PACS/Office-Caltech10) 场景下,客户端来自不同视觉域,如何在已稳定的"正交双头解耦"基线 Plan A (z_sem ⊥ z_sty + LR=0.05 + 可选 SAS) 之上再取得 ≥1% AVG 提升,PACS+Office 通用,不引入任何会长期崩溃的机制。
- **Must-solve bottleneck**:
  - (a) sem_classifier 参加 FedAvg 聚合是漂移源
  - (b) z_sty 和 style_bank 推理时完全未用,这是免费信号
- **Non-goals**:
  - 不引入持续对齐损失 (InfoNCE/Supcon/Triplet — 已证伪)
  - **No raw sample or per-instance feature sharing**; only client-level summary statistics (μ_sty, Σ_sty) are broadcast (~66KB/round, 同 FedBN γ/β 量级)
  - 不用 CLIP/VLM (留给 FedDEAP)
  - 不抢 "first FL+ETF+decouple" headline
- **Success condition**:
  - PACS AVG Best ≥ 81.5% (Plan A = 80.41); ALL Best ≥ 83% (Plan A = 82.31)
  - Office AVG Best ≥ 84% (w/o SAS) 或 ≥ 90.5% (w/ SAS)
  - 3-seed std ≤ 1.5%, drop ≤ 1%

---

## 2. One-Sentence Thesis

> 在 scratch-CNN 跨域联邦学习中,**正交解耦出的 source-domain style statistics** 作为 **federated, backprop-free reliability signal**,指导 test-time prototype correction;Fixed Simplex ETF 作为 classifier stabilizer + prototype cold-start prior。

---

## 3. Contribution Focus

| 类别 | 内容 |
|------|------|
| **Dominant (1)** | **Style-Gated Prototype Adjustment (SGPA)**: 双 gate (entropy + **Mahalanobis distance in whitened z_sty space**) 决定 test sample 是否更新 prototype bank;top-m per-class supports;cos(z_sem, proto) 分类;ETF fallback for cold-start classes |
| **Supporting (1)** | **Fixed Simplex ETF**: 替代 sem_classifier, 零可训参数, 消除 FedAvg 漂移 |
| **Non-contributions** | 不是新 loss / 不是新聚合 / 不是新 backbone / 不跨 client 共享 raw feature (只共享 lightweight (μ,Σ) summary stats) |

---

## 4. Method: FedDSA-SGPA

### 4.1 Complexity Budget
- **Frozen / reused**: Plan A 全部 (backbone, 双头解耦, L_orth, LR schedule, CE+CE_aug)
- **New trainable**: **None** (Fixed ETF = buffer, SGPA = backprop-free inference)
- **New communication**: μ_global [d] + Σ_inv_sqrt [d×d] + source_μ_k [N×d] per round ≈ **66 KB** (on par with FedBN γ/β)

### 4.2 System Overview

```
[TRAINING — = Plan A + 1 line of bookkeeping per client per round]
  x → backbone → double_head → z_sem, z_sty    [L_orth on (z_sem, z_sty)]
  
  logits = F.normalize(z_sem, dim=-1) @ M / τ_etf        # Fixed ETF buffer
  loss = CE(logits, y) + CE(logits_aug, y) + λ_orth·L_orth   # = Plan A loss

  # New: each client computes & uploads its local (μ_sty, Σ_sty)
  μ_sty_k = z_sty.mean(0); Σ_sty_k = cov(z_sty, unbiased=False)

[SERVER aggregation — per round]
  FedAvg: backbone + sem_head + sty_head                  # = Plan A
  FedBN: private BN (no aggregation)                      # = Plan A
  SAS (Office opt-in): sem_head style-weighted            # = Plan A
  Fixed ETF M: NOT aggregated (all clients share seeded buffer)
  NEW: build pooled second-order model:
    μ_global   = (1/N) Σ_k μ_sty_k                        # [d]
    Σ_within   = (1/N) Σ_k Σ_sty_k                        # [d,d]
    Σ_between  = (1/N) Σ_k (μ_sty_k-μ_global)(μ_sty_k-μ_global)^T   # [d,d]
    Σ_global   = Σ_within + Σ_between + ε·I               # ε=1e-3
    Λ, Q       = torch.linalg.eigh(Σ_global)              # symmetric eig
    Σ_inv_sqrt = Q @ diag(clamp(Λ, min=ε)^{-1/2}) @ Q^T   # [d,d]
  Broadcast: (source_μ_k, μ_global, Σ_inv_sqrt)

[INFERENCE — NEW, per client on own test set]
  model.eval()
  μ_k_white = (source_μ_k - μ_global) @ Σ_inv_sqrt        # [N, d], precompute
  
  τ_H, τ_S, supports, proto = calibrate()                 # see §4.4
  
  with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
      z_sem, z_sty = model(x)
      logits_etf = F.normalize(z_sem, -1) @ M / τ_etf
      pred_etf = logits_etf.argmax(1)
      H = entropy(softmax(logits_etf))
      
      z_sty_white = (z_sty - μ_global) @ Σ_inv_sqrt
      dist_min = ((z_sty_white[:, None] - μ_k_white[None])**2).sum(-1).min(-1).values
      
      # Warm-up: no gate, but still output ETF predictions
      if batch_idx < 5:
        buf_H += H.tolist(); buf_D += dist_min.tolist()
        yield pred_etf                                    # ← crucial: always output
        continue
      elif batch_idx == 5:
        τ_H = np.quantile(buf_H, 0.5)
        τ_S = np.quantile(buf_D, 0.3)
      else:
        τ_H = 0.95*τ_H + 0.05*np.quantile(H.cpu(), 0.5)
        τ_S = 0.95*τ_S + 0.05*np.quantile(dist_min.cpu(), 0.3)
      
      reliable = (H < τ_H) & (dist_min < τ_S)
      
      # Top-m per-class proto update
      for c in range(K):
        mask = reliable & (pred_etf == c)
        if mask.any():
          new_items = [(H[i].item(), z_sem[i]) for i in mask.nonzero().flatten()]
          supports[c].extend(new_items)
          supports[c] = sorted(supports[c], key=lambda t: t[0])[:m_top]
          proto[c] = F.normalize(torch.stack([s[1] for s in supports[c]]).mean(0), -1)
      
      proto_tensor = torch.stack(proto)
      proto_logits = F.normalize(z_sem, -1) @ proto_tensor.T
      pred_proto = proto_logits.argmax(1)
      activated = torch.tensor([len(supports[c]) > 0 for c in pred_proto])
      pred = torch.where(activated, pred_proto, pred_etf)
      yield pred
```

### 4.3 Fixed Simplex ETF (Supporting)

```python
# server init — all clients replicate with same seed
assert feat_dim >= num_classes                # PACS 128 ≥ 7 ✓, Office 128 ≥ 10 ✓
torch.manual_seed(0)
U, _ = torch.linalg.qr(torch.randn(feat_dim, num_classes))
I_K = torch.eye(num_classes); ones_K = torch.ones(num_classes, num_classes)
M = U @ (I_K - ones_K / num_classes)          # [d, K]
M = M * math.sqrt(num_classes / (num_classes - 1))
self.register_buffer('M', M)                   # NOT aggregated, NOT trainable

# forward (train & test)
logits = F.normalize(z_sem, dim=-1) @ self.M / self.tau_etf   # τ_etf = 0.1
```

**作用**: (a) classifier 无可训参数 → 不参加 FedAvg → 消除 Plan A 的 classifier 漂移源;(b) ETF 列向量作为 proto bank 冷启动 anchor: `proto_init[c] = M[:, c] / M[:, c].norm()`

### 4.4 Dominant Mechanism: SGPA

#### Gate Design
- **Entropy gate**: `H < τ_H` — 置信度筛选
- **Style distance gate**: `dist_min < τ_S` — Mahalanobis 距离 (在 pooled whitening 空间 = Euclidean)
- **Joint**: `reliable = (H < τ_H) & (dist_min < τ_S)`

#### Why Whitening Works (CLAIMS — deliberate wording)
**(formally partial, practically sufficient)**:
- Whitening provides a **shared Mahalanobis metric under a pooled second-order approximation** of source-domain style distribution
- 不声称 "provably invariant to arbitrary private BN affine" — 该声明 overreach
- 实际上 invariant 于 **shared affine transform**;但 private BN 可能引入 **client-specific affine distortion** 的残余分量,由 Σ_between 部分吸收
- Diagnostic 图表将展示 distance distribution before/after whitening 的直方图 (证明 whitening 使跨 client scatter 显著下降)

#### Prototype Bank
- `supports[c]`: list of up-to-m (H, z_sem) tuples,按 H 升序保留 top-m
- `m_top = max(K*5, 20)` → PACS K=7 → 35, Office K=10 → 50
- `proto[c] = F.normalize(mean({s[1] for s in supports[c]}))`
- 初始: `proto[c] = F.normalize(M[:, c])` (ETF vertex)
- `activated[c] = len(supports[c]) > 0`

#### Calibration
- **Warm-up W=5 batches**: 不激活 gate,输出 ETF prediction,累积 buf_H, buf_D
- **Init at batch 5**: `τ_H = quantile(buf_H, 0.5)`, `τ_S = quantile(buf_D, 0.3)`
- **Running EMA**: `τ_H ← 0.95·τ_H + 0.05·quantile(current_H, 0.5)`

---

## 5. Integration (PFLlib)

### `PFLlib/system/flcore/clients/clientdsa.py`
- `__init__`: 
  - `self.M = register_buffer(construct_etf(feat_dim, num_classes))`
  - 删除 `self.sem_classifier = nn.Linear(...)`
- `train()`: `logits = F.normalize(z_sem, -1) @ self.M / self.tau_etf`;其余 Plan A 不变
- `local_aggregate_stats()`: 每轮结束计算 `μ_sty_k, Σ_sty_k`,返回 server
- `test_with_sgpa()`: 新函数,见 §4.2 pseudocode;默认 test path 调用它

### `PFLlib/system/flcore/servers/serverdsa.py`
- `aggregate_parameters()`: 跳过 `self.sem_classifier` (Fixed ETF 不聚合)
- `aggregate_style_stats()`: 新函数,收集各 client (μ, Σ),构造 (μ_global, Σ_inv_sqrt, source_μ_k)
- `send_payload()`: 下发增量 payload

### 代码增量估计
- Client: ~80 行 (含 test_with_sgpa)
- Server: ~60 行 (含 aggregate_style_stats)
- **Total: ~140 行** (远小于 FDSE 的 DSE 层分解)

---

## 6. Novelty vs Closest Work (新 rephrase)

- **FedDEAP (arXiv'25 Oct)**: CLIP prompt-tuning 下训练时做 ETF-constrained transformation。**我们正交**: scratch CNN, 推理时用 disentangled style 做 reliability signal,**ETF 仅作 subordinate stabilizer**。
- **FedETF (ICCV'23)**: label-skew FL fixed ETF。**差异**: 我们 feature-skew + SGPA 是真 dominant。
- **SATA (IVC'25)**: centralized 图像空间 style exchange。**差异**: FL + Mahalanobis z_sty gate 替代 AdaIN,backprop-free,更 faithful 解耦。
- **T3A (NeurIPS'21 Spot)**: centralized proto TTA。**差异**: FL + 双 gate + ETF 冷启动 + top-m (vs all reliable EMA)。
- **FedCTTA (arXiv'25)**: pure entropy TTA for FL。**差异**: 双 gate (entropy + Mahalanobis),利用 decoupled style,非 entropy-only。

**Moat**: "Disentangled source-domain style statistics 作为 FL test-time reliability signal, via pooled-whitening Mahalanobis" — 2024-2026 文献无占坑。

---

## 7. Validation Plan (Claim-Driven)

### Claim 1 (Dominant): SGPA on Plan A checkpoint **免费 +0.5~2%**
- **Minimal experiment**: Plan A checkpoint × {no TTA, entropy-only gate, dist-only gate, both gates} × {PACS, Office} × 3 seeds {2,15,333}
- **Baselines**: Plan A + TENT, Plan A + FedCTTA
- **Metric**: AVG Best, AVG Last, ALL Best, ALL Last, drop, 3-seed std
- **Expected**: PACS +0.5~1.5%, Office +0.3~1.5%, std 不增加

### Claim 2 (Supporting): Fixed ETF is compatible prior
- **Minimal**: {Linear, Fixed ETF} × {PACS, Office} × 3 seeds (no SGPA)
- **Expected**: ETF mean Δ ≤ 0.3%, 3-seed std 下降 ≥ 0.2%

### Claim 3 (Integration): SGPA + ETF 达 success condition
- **Minimal**: Fixed ETF train + SGPA inference × {PACS, Office} × 3 seeds
- **Baselines**: FDSE R200, Plan A (既有)
- **Expected**: PACS AVG Best ≥ 81.5%, Office AVG Best ≥ 84%/90.5%

### Diagnostic Ablations (新增 per Round 3)
- **Whitening on/off**: SGPA w/ whitening vs w/o (raw L2) — 预期 whitening +0.5~1%
- **Covariance decomposition**: Σ_within only vs Σ_within + Σ_between — 预期 full covariance 略好
- **Distance distribution**: before/after whitening 直方图 (证明 cross-client scatter 下降)

---

## 8. Compute & Timeline

| 阶段 | 内容 | GPU-hours |
|------|------|-----------|
| Plan A retrain with ETF | 3-seed × {PACS, Office} × R200 | 27 |
| SGPA inference (reuse ckpt) | 3-seed × {PACS, Office} × 4 config (ablations) | 5 |
| Ablation: Linear vs ETF (C2) | 3-seed × {PACS, Office} × R200 | 27 (已有 Plan A = Linear baseline,只需 ETF) |
| Diagnostic: whitening on/off, Σ decomp | 推理期 | 2 |
| **Total** | | **~35 GPU-hours** (~2-3 days, 单卡 24GB) |

---

## 9. Failure Modes & Diagnostics

| 失败模式 | 检测 | Fallback |
|---------|------|---------|
| Σ_global 奇异 (PACS N=4 下 Σ_between rank ≤ 3) | `cond(Σ_global) > 1e6` | 只用 Σ_within + 较大 ε=1e-2 |
| dist_min 分布 bimodal / 无 gap | whitening histogram 报警 | 退化到纯 entropy gate |
| ETF 在 PACS 4-outlier 下 hurt mean > 0.5% | per-seed compare | 保留 Linear,只用 SGPA |
| Warm-up 后 τ_S 过紧 (reliable rate < 0.1) | 每 client 第 6 batch 监控 | τ_S 放宽 quantile 到 0.5 |
| Supports overflow / stale | len(supports[c]) > m_top 报警 | hard cap + LRU eviction |

---

## 10. Paper Title & Abstract Sketch

**Title**: *Style-Gated Prototype Adjustment for Federated Domain Generalization*

**Abstract** (draft):
> We address the test-time side of federated domain generalization (FedDG) under feature skew, where orthogonally-disentangled style features are available but unused at inference. We propose **Style-Gated Prototype Adjustment (SGPA)**, a backprop-free mechanism that (i) computes an entropy gate and a Mahalanobis style-distance gate over disentangled z_sty with pooled whitening across clients to form a reliability signal; (ii) uses reliable test samples to incrementally update a top-m prototype bank in the semantic space, and (iii) classifies via cosine similarity to the prototype bank with ETF-vertex fallback for cold-start classes. A Fixed Simplex ETF classifier replaces the trainable linear head as a parameter-free stabilizer and prototype cold-start prior. On PACS and Office-Caltech10 with ResNet-18 / AlexNet from scratch, SGPA provides a consistent +0.5–2% AVG-Best improvement over the strongest decoupled baseline, outperforming FDSE and Plan A with no additional training cost and communication overhead on the same order as FedBN parameter broadcasting.

---

## 11. Summary Table

| 维度 | Plan A (baseline) | FedDSA-SGPA (this proposal) |
|------|------------------|----------------------------|
| Backbone | ResNet-18 / AlexNet scratch | 同 |
| Decouple | cos² 双头 | 同 |
| LR | 0.05 | 同 |
| Aggregation | FedAvg + FedBN + SAS (opt) | 同 + lightweight (μ, Σ) 广播 |
| Classifier | Linear (trainable, FedAvg) | **Fixed Simplex ETF (zero param)** |
| Inference | argmax(Linear(z_sem)) | **SGPA: dual gate + top-m proto + ETF fallback** |
| New trainable | — | **None** |
| New communication/round | — | 66 KB (≈ FedBN γ/β) |
| Expected gain | — | PACS +1~1.5%, Office +1~2% |
