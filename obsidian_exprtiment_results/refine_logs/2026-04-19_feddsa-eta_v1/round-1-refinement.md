# Round 1 Refinement — FedDSA-SGPA (改名 + 三项 CRITICAL 修复)

## Problem Anchor (Copy Verbatim from Round 0)
- **Bottom-line problem**: 在跨域联邦学习 (feature-skew FL, PACS/Office-Caltech10) 场景下,客户端来自不同视觉域,如何在已稳定的"正交双头解耦"基线 (Plan A: z_sem ⊥ z_sty + LR=0.05 + 可选 SAS) 之上**再取得 ≥1% AVG 提升**,PACS+Office 通用,不引入任何会长期崩溃的机制。
- **Success condition** (澄清 Office 数字):
  - **PACS**: AVG Best ≥ 81.5% (Plan A = 80.41); ALL Best ≥ 83% (Plan A = 82.31)
  - **Office (w/o SAS comparator)**: AVG Best ≥ 84% (Plan A orth_only = 82.55); ALL Best ≥ 86%
  - **Office (w/ SAS comparator)**: AVG Best ≥ 90.5% (SAS = 89.82, FDSE = 90.58)
  - **稳定性**: drop ≤ 1%, 3-seed std ≤ 1.5%

## Anchor Check
- **原 bottleneck**:
  - (a) sem_classifier 参加 FedAvg 是漂移源
  - (b) z_sty + style_bank 推理时被浪费
- **修改版仍 attack 同一 bottleneck**:
  - (a) 由 Fixed ETF 消除 (降级为 supporting prior)
  - (b) 由 SGPA 显式利用 (dominant contribution)
- **Reviewer 拒收 drift**:
  - 若 ETF-only 实验失败,**不**退化为 "pure TTA on Plan A" — 保留训练端几何叙事最小化声明为 "prior initialization + variance stabilizer"

## Simplicity Check
- **Revised dominant contribution**: **SGPA = Style-Gated Prototype Adjustment**
  - 唯一新 named mechanism,novelty 单点
  - 用 z_sty distance to source-domain style_bank + entropy 双 gate 决定是否用此 test sample 更新 proto
  - proto 在 z_sem 空间做 top-m EMA,预测用 cos(z_sem, proto)
- **Demoted supporting component**: Fixed ETF (只作为 classifier stabilizer + proto cold-start prior)
- **Removed from core**: Learnable Prototype Classifier, FedDEAP-adapted-to-ResNet (moved to optional related work)
- **Remaining mechanism 还是最小吗?** 是。一个 gate (两个标量比较) + 一个 EMA + Fixed ETF buffer。没有新 learnable 参数。

## Changes Made

### 1. 改名 FedDSA-ETA → FedDSA-SGPA
- **Reviewer**: "contribution 散乱,2.5 个 headline 争抢 novelty"
- **Action**: 名字以 dominant mechanism SGPA (Style-Gated Prototype Adjustment) 为主
- **Impact**: 论文一句话叙事 = "disentangled source-domain style statistics 作为 federated backprop-free reliability signal 指导 TTA prototype correction"

### 2. SATA gate → z_sty distance gate (删 AdaIN)
- **Reviewer**: "AdaIN(z_sem, μ_sty, σ_sty) space mismatch, B=1 NaN, exclude_self 无意义"
- **Action**:
  ```python
  # 训练时每 client 维护自己的 source style bank 的 (μ,σ):
  #   source_style_bank = {client_k: (μ_k, σ_k) for k in 1..N}, shape [N, d_sty]
  # 推理时 (test 样本来自 unseen target domain):
  dist_to_source = min_k ||F.normalize(z_sty_test) - F.normalize(μ_k)||_2  # [B]
  reliable = (entropy(softmax(logits_ETF)) < τ_H) & (dist_to_source < τ_S)
  ```
- **Impact**:
  - 彻底删除 AdaIN(z_sem) swap forward pass
  - 无 space mismatch (z_sty 只用自己空间)
  - 无 std(B=1) NaN 风险
  - 更 faithful 解耦叙事 (z_sty 只做 gate,z_sem 只做 classification)
  - 两个超参: τ_H (entropy threshold) 和 τ_S (style distance threshold),per-dataset calibrate

### 3. T3A proto 改为 top-m supports per class
- **Reviewer**: "raw EMA on all reliable samples 可能被脏样本污染"
- **Action**:
  ```python
  # 每 class 保留 top-m (按 entropy 从小到大) reliable 样本作为 support
  # m=20 for PACS, m=50 for Office (可配置)
  # support_bank[c] = list of up to m latest reliable z_sem[i] for class c
  # proto[c] = normalize(mean(support_bank[c]))
  # 没有 support → proto[c] = ETF_vertex[c] (cold start)
  ```
- **Impact**: 比 EMA 更鲁棒 (丢弃最旧/最大 entropy 样本),top-m 足够做稳定均值

### 4. Mathematical Correctness 补充
- ETF 构造: `assert d >= K` (PACS K=7, Office K=10, d_sem=128 → 满足)
- β 固定为 1 (与 τ 冗余)
- 所有 std 用 `var(unbiased=False).sqrt()` + eps=1e-5
- 推理时: `model.eval()` + `torch.no_grad()` + `forward.bn.track_running_stats=False` 三重保险

### 5. 精简 Validation Claims
- C2 (原要求 "ETF beat Linear"): 放宽为 "ETF 不伤 mean,但 3-seed std 下降 ≥0.2%"
- 删除 Learnable Proto+CE baseline
- 删除 FedDEAP-adapted 主表对比 (放 related work 讨论)
- 保留 C1 (SGPA 免费 +0.5~2%) 为 dominant claim

---

## Revised Proposal: **FedDSA-SGPA (Style-Gated Prototype Adjustment)**

### One-Sentence Thesis
> 在 feature-skew 跨域联邦学习中,我们首次把**正交解耦出的 source-domain style statistics** 用作 **federated backprop-free reliability signal**,指导 test-time prototype 校正 — 所有训练时机制保持 Plan A 不变,Fixed ETF 仅作为 proto 冷启动 prior。

### Contribution Focus (collapsed)
- **Dominant (1)**: **SGPA** — 双 gate (entropy + z_sty-to-source-bank distance) 决定 test sample 是否更新 prototype bank;prototype bank 用 top-m per-class 支撑,分类用 cos(z_sem, proto)
- **Supporting (1)**: Fixed Simplex ETF 作为 sem_classifier (消除 FedAvg 漂移 + 提供 proto 冷启动顶点)
- **Explicit non-contributions**: 不是新损失,不是新聚合,不是新 backbone,不是跨客户端风格共享

### System Overview

```
[Training — 几乎=Plan A]
  x → backbone → double_head
       ↓
  z_sem ⊥ z_sty  [L_orth = cos²]
       ↓            ↓
   Fixed-ETF M   每 client 收集自己的 (μ_sty, σ_sty)
       ↓            ↓
  logits = F.normalize(z_sem) @ M / τ_etf
       ↓
  CE(logits) + CE(logits_aug) + λ_orth·L_orth   (= Plan A)

[Server aggregation]
  FedAvg: backbone + sem_head + sty_head   (Plan A 不变)
  FedBN: BN 本地
  SAS (Office opt-in): sem_head 风格加权
  Fixed ETF M: 不聚合 (seeded 初始化,所有 client 共享同一个 buffer)
  source_style_bank: 每轮聚合到 server 维护 N 个 (μ_k, σ_k)

[Inference — NEW (SGPA)]
  model.eval()
  with torch.no_grad():
    z_sem, z_sty = model(x)
    logits_etf = F.normalize(z_sem) @ M / τ_etf    # [B, K]
    H = entropy(softmax(logits_etf))                # [B]

    # Gate 1: entropy (置信度)
    # Gate 2: z_sty distance to nearest source style (是否在源域附近)
    z_sty_norm = F.normalize(z_sty, dim=-1)
    bank_norm = F.normalize(source_style_bank[:, 0, :], dim=-1)  # μ_k 部分 [N, d_sty]
    dist_min = (1 - z_sty_norm @ bank_norm.T).min(dim=-1).values  # [B]
    reliable = (H < τ_H) & (dist_min < τ_S)

    # Prototype 更新 (top-m per class)
    pred_etf = logits_etf.argmax(1)
    for c in range(K):
      mask_c = reliable & (pred_etf == c)
      if mask_c.any():
        support_bank[c].extend(z_sem[mask_c].detach().cpu())
        support_bank[c] = top_m_by_entropy(support_bank[c], H[mask_c], m=M_proto)
        proto[c] = F.normalize(torch.stack(support_bank[c]).mean(0), dim=-1)

    # 分类: 若该 class 的 proto 已激活 → cos 分类,否则 → ETF
    z_sem_norm = F.normalize(z_sem, dim=-1)
    proto_logits = z_sem_norm @ torch.stack(proto).T  # [B, K]
    pred_proto = proto_logits.argmax(1)
    activated = torch.tensor([len(support_bank[c])>0 for c in pred_proto])
    pred = torch.where(activated, pred_proto, pred_etf)
```

### Core Mechanism Details

#### Supporting: Fixed Simplex ETF
```python
# server init (all clients copy M)
assert feat_dim >= num_classes
torch.manual_seed(0)  # 所有 client 用同一个 seed
U = torch.linalg.qr(torch.randn(feat_dim, num_classes))[0]   # [d, K]
I_K = torch.eye(num_classes); ones = torch.ones(num_classes, num_classes)
M = U @ (I_K - ones / num_classes)  # [d, K], β=1 (省略,与 τ 冗余)
M = M * math.sqrt(num_classes / (num_classes - 1))
self.register_buffer('M', M)

# forward (训练/推理同)
logits = F.normalize(z_sem, dim=-1) @ self.M / self.tau_etf  # τ_etf = 0.1
```
**作用**: (a) classifier 无可训参数 → 不参加 FedAvg → 消除 Plan A 的 classifier 漂移;(b) ETF 列向量作为 prototype cold-start anchor。

#### Dominant: SGPA Gate + Prototype Bank
**训练阶段 (只加 1 件事)**: 每轮每 client 汇总本地 `(μ_sty^k, σ_sty^k) = (z_sty.mean(0), z_sty.var(0, unbiased=False).sqrt())`,上传 server。Server 维护 `source_style_bank ∈ ℝ^{N × 2 × d_sty}`。

**推理阶段**:
- `τ_H` calibration: 第一 batch 用 50%-quantile of H 作为 τ_H
- `τ_S` calibration: 第一 batch 用 30%-quantile of dist_min 作为 τ_S (要求 ≥30% samples 进入 reliable)
- Top-m support selection: `m = max(K*5, 20)` — PACS K=7 → m=35, Office K=10 → m=50

### Integration
- `PFLlib/system/flcore/clients/clientdsa.py`:
  - `__init__`: 替换 `self.sem_classifier = ETFBuffer(d=128, K=num_classes)`
  - `train()`: logits = F.normalize(z_sem) @ self.M / self.tau_etf
  - `test()`: 新增 `test_with_sgpa()` — 详见伪代码
  - 每轮本地汇总 `local_style_stats`,返回 server
- `PFLlib/system/flcore/servers/serverdsa.py`:
  - `aggregate_parameters()`: skip `self.sem_classifier` (Fixed ETF 不聚合)
  - 新增 `self.source_style_bank = {client_id: (μ, σ)}` 维护
  - 下发给 client 的 payload 增加 `source_style_bank`

### Failure Modes and Diagnostics

| 失败模式 | 检测 | Fallback |
|---------|------|---------|
| ETF mean 掉 >0.5% (PACS 4-outlier 假说) | per-seed ETF vs Linear | 保留 Linear,只用 SGPA |
| SGPA reliable rate 永远 >0.95 (gate 太松) | 每 client 第一 batch 的 reliable 比例 | τ_H calibration 用 50%-quantile |
| SGPA reliable rate <0.1 (gate 太紧) | 同上 | τ_S calibration 放宽到 50%-quantile |
| Support bank 溢出 | len(support_bank[c]) 报警 | top-m hard cap |
| 未见 target domain 的 z_sty 永远 far from source bank | 监控 dist_min 分布 | 动态 τ_S (test-time running percentile) |

### Novelty vs Closest Work (新 rephrase)

- **FedDEAP (arXiv 2510.18837, 2025-10)**: CLIP prompt-tuning 下训练时做 ETF-constrained semantic/domain transformation。**我们正交**: scratch CNN,parameter aggregation,**推理时**用 disentangled style 做 reliability signal 指导 prototype correction;ETF 仅作 stable prior。
- **FedETF (ICCV'23)**: label-skew FL 的 fixed ETF。**差异**: 我们 feature-skew,且 ETF 只是 supporting prior,dominant contribution 是 SGPA。
- **SATA (IVC'25, centralized)**: 图像空间 AdaIN style exchange 做 TTA reliability。**差异**: FL + 用 decoupled z_sty distance 替代图像空间 AdaIN,backprop-free,无需额外 forward pass。
- **T3A (NeurIPS'21 Spot)**: centralized proto TTA。**差异**: FL + z_sty-distance gate + ETF 顶点作为 cold start + top-m support (而非全部 reliable)。
- **FedCTTA (arXiv'25)**: pure entropy TTA for FL。**差异**: 我们双 gate (entropy + z_sty distance),利用 disentangled features,不仅 entropy。

**Moat**: "Disentangled style statistics 作为 FL test-time reliability signal" 是 2024-2026 文献完全空白的格子。

---

## Claim-Driven Validation (精简后)

### Claim 1 (Dominant): SGPA 在 Plan A 上免费 +0.5~2%
- **Minimal**: Plan A checkpoint × {w/o, SGPA full, entropy-only, z_sty-dist-only} × {PACS, Office} × 3 seeds
- **Baseline**: Plan A orth_only + TENT;Plan A + FedCTTA
- **Metric**: AVG Best, AVG Last, ALL Best, ALL Last, drop, 3-seed std
- **Expected**: PACS +0.5~1.5%, Office +0.3~1.5%, std ≤ Plan A

### Claim 2 (Supporting): Fixed ETF 是兼容 prior (not harm)
- **Minimal**: {Linear, Fixed ETF} × {PACS, Office} × 3 seeds (无 SGPA)
- **Metric**: AVG Best, 3-seed std
- **Expected**: ETF 不伤 (Δ ≤ 0.3%),std 降 ≥0.2%

### Claim 3 (Integration): SGPA + ETF 组合达 success condition
- **Minimal**: SGPA + ETF × {PACS, Office} × 3 seeds
- **Baseline**: FDSE R200, Plan A
- **Expected**: PACS AVG Best ≥ 81.5%, Office AVG Best ≥ 84% (no SAS) / ≥90.5% (w/ SAS)

## Experiment Handoff
- **Must-prove**: C1 (SGPA 免费 +), C2 (ETF no harm), C3 (组合达标)
- **Must-run ablations**: 双 gate / 单 gate / no gate (=T3A-like naive); τ_H / τ_S calibration; m (top-m)
- **Highest-risk**: PACS 4-outlier 下 z_sty 到 N=4 源域 distance gate 分布是否 bimodal

## Compute & Timeline
- ETF 训练: 27 GPU-hours (= Plan A 训练成本)
- SGPA 推理 (复用 checkpoint): 5 GPU-hours
- **Total**: ~32 GPU-hours (2 天)
