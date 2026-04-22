# EXP-113 FedDSA-VIB + VSC 完整流程 — 技术版

**对应大白话版**: [EXP-113_FedDSA-VIB-VSC_流程_大白话版](EXP-113_FedDSA-VIB-VSC_流程_大白话版.md)
**一句话 novelty**: First FL dual-head disentanglement with prototype-conditioned VIB, EMA-lagged stop-grad prior, σ-head localization, and Moyer-style probe evaluation.

---

## 1. Problem Statement

### 1.1 符号系统
- $x \in \mathcal{X}$: input image
- $h = \text{Encoder}(x) \in \mathbb{R}^{d_h}$ ($d_h = 1024$ for AlexNet)
- $z_{\text{sem}} \in \mathbb{R}^{d}$: semantic projection ($d = 128$)
- $z_{\text{sty}} \in \mathbb{R}^{d}$: style projection
- $y \in \{1,\ldots,K\}$: class label ($K=7$ PACS, $K=10$ Office)
- $c \in \{1,\ldots,N\}$: client/domain id

### 1.2 Empirical failure of prior attempts

| Method | linear probe $(z_\text{sty}\to y)$ | MLP-256 probe | PACS AVG Best |
|--------|:---------:|:-------:|:-------:|
| CDANN (EXP-108) | 0.962 | 0.962 | 80.08 ± 0.60 |
| orth_only (EXP-109) | 0.240 | **0.813** | 80.64 ± 1.46 |
| lo=3 strong_orth (EXP-111) | 0.339 | **0.714** | 81.33 (1 seed) |

**Core issue**: loss-level orthogonality/HSIC/CDANN **cannot eliminate nonlinear class leakage** in z_sty. Moyer (NeurIPS 2018) §5 predicted this 8 years ago.

### 1.3 Objective

针对 shared-trunk + loss-based decoupling 的非线性泄漏问题,设计:
1. **方法**: VIB compression on $z_\text{sem}$ forces minimum-sufficient semantic encoding, pushing style to $z_\text{sty}$
2. **评估**: Moyer 0/1/2/3-layer adversary sweep + Hewitt-Liang control task selectivity
3. **统计**: 10 seeds + paired design for sensitivity over orth_only std=1.46

---

## 2. Method

### 2.1 Architecture

```
Input x
   │
   ▼
AlexNet Encoder → h ∈ ℝ^1024
   │                │
   │                ├──→ StyleHead → z_sty ∈ ℝ^128  [FedBN private]
   │                │
   │                ├──→ μ_head (FedAvg aggregated)  ─┐
   │                │                                   ├──→ (μ, log_var)
   │                └──→ log_var_head (LOCAL) ─────────┘
   │                                                ▼
   │                                  q(z_sem | x) = N(μ, σ²)
   │                                                ▼
   │                                         z_sem ∼ N (train) / μ (eval)
   │                                                │
   │                                                ▼
   │                                  sem_classifier → ŷ
   │
   prototype_ema[y] ∈ ℝ^{K×128}  [EMA β=0.99, stop-grad, 不聚合]
   log_sigma_prior[y] ∈ ℝ^K     [LOCAL per-class learnable]
```

### 2.2 Loss Functions

#### 2.2.1 主任务 (A/B 共用)
$$L_\text{CE} = \text{CE}(\text{sem\_classifier}(z_\text{sem}), y)$$
$$L_\text{aug} = \text{CE}(\text{sem\_classifier}(z_\text{sem}^\text{aug}), y)$$
其中 $z_\text{sem}^\text{aug}$ 通过 h-space AdaIN (复用原 `_style_augment`)

#### 2.2.2 VIB Loss (A/B 共用) ★ 新
$$L_\text{VIB} = \frac{1}{B}\sum_{i=1}^{B} \text{KL}\Big(\mathcal{N}(\mu_i, \Sigma_i) \,\|\, \mathcal{N}(\tilde{\mu}_{y_i}, \Sigma_\text{prior}^{y_i})\Big)$$

**Closed-form KL**:
$$\text{KL} = \frac{1}{2}\sum_{d}\left[\log\frac{\sigma_{p,d}^2}{\sigma_{q,d}^2} + \frac{\sigma_{q,d}^2 + (\mu_{q,d} - \mu_{p,d})^2}{\sigma_{p,d}^2} - 1\right]$$

**Prior parameters**:
- $\tilde{\mu}_{y_i} = \text{prototype\_ema}[y_i]$ (stop-grad, EMA-lagged)
- $\Sigma_\text{prior}^{y_i} = \text{diag}(\exp(2 \cdot \log\_\sigma\_\text{prior}[y_i]))$ (per-class learnable, local)

**Warmup schedule**:
$$\lambda_\text{IB}(r) = \begin{cases} 0 & r < 20 \\ (r-20)/30 & 20 \le r < 50 \\ 1.0 & r \ge 50 \end{cases}$$

#### 2.2.3 Prototype alignment

- **方案 A (FedDSA-VIB)**: 保留原 InfoNCE
$$L_\text{InfoNCE} = -\frac{1}{B}\sum_i \log\frac{\exp(\cos(\mu_i, p_{y_i})/\tau)}{\sum_c \exp(\cos(\mu_i, p_c)/\tau)}$$

- **方案 B (FedDSA-VSC)**: 替换为 SupCon (Khosla NeurIPS 2020)
$$L_\text{SupCon} = -\frac{1}{B}\sum_i \frac{1}{|P(i)|}\sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \ne i} \exp(z_i \cdot z_a / \tau)}$$
其中 $P(i) = \{j : y_j = y_i, j \ne i\}$

#### 2.2.4 原 FedDSA 损失 (A/B 共用, 保留原权重)
- $L_\text{orth} = E[\cos^2(\mu_\text{sem}, z_\text{sty})]$ (用 μ 非 stochastic z_sem)
- $L_\text{HSIC} = \text{HSIC}(\mu_\text{sem}, z_\text{sty})$

### 2.3 Total Loss

**方案 A**:
$$L_A = L_\text{CE} + L_\text{aug} + \lambda_\text{IB}(r) \cdot L_\text{VIB} + \lambda_\text{orth} L_\text{orth} + \lambda_\text{hsic} L_\text{HSIC} + \lambda_\text{sem} L_\text{InfoNCE}$$

**方案 B**:
$$L_B = L_\text{CE} + L_\text{aug} + \lambda_\text{IB}(r) \cdot L_\text{VIB} + \lambda_\text{orth} L_\text{orth} + \lambda_\text{hsic} L_\text{HSIC} + \lambda_\text{sup} L_\text{SupCon}$$

### 2.4 4 个 Blocker Fix (Round-2 Codex review addressed)

#### Fix 1: EMA-lagged stop-grad prior (解 chicken-and-egg)
```python
with torch.no_grad():
    prototype_ema = β · prototype_ema + (1-β) · new_prototype_round_r
    # β = 0.99

# 训练时
prior_mu = prototype_ema[y].detach()  # stop gradient
```
**逻辑**: prior 滞后当前 batch 的 μ,避免 self-reinforcing collapse。

#### Fix 2: σ-head localization (FedBN-style)
```python
# Server._init_agg_keys
if 'log_var_head' in key:
    private_keys.add(key)          # 不参与 FedAvg
if 'log_sigma_prior' in key:
    private_keys.add(key)          # 不参与 FedAvg
```
**逻辑**: σ 反映 domain-conditional uncertainty,FedAvg 会污染。

#### Fix 3: Learnable per-class σ_prior + anti-lookup guard
```python
self.log_sigma_prior = nn.Parameter(torch.zeros(K))  # init σ=1
prior_log_var = 2 * self.log_sigma_prior[y]

# 训练时监控
intra_class_z_std = z_sem.per_class_std().mean()
if intra_class_z_std < 0.1 and kl_mean < 0.1:
    logger.warn("COLLAPSE: z_sem → prototype lookup")
```
**逻辑**: 固定 σ_prior 允许 degenerate solution `z_sem = prototype[y]`;可学 σ + intra-class std 监控防止。

#### Fix 4: L_HSIC redundancy ablation
加 ablation run (去 HSIC) 验证 VIB 独立贡献 — 见实验矩阵。

---

## 3. Diagnostic Suite (50+ metrics)

### 3.1 Layer-1: Train-time (每 5 轮)

**VIB (A/B 共用)**:
| Metric | Definition | Healthy | Alarm |
|--------|-----------|:-------:|:----:|
| `KL_mean` | $\mathbb{E}[\text{KL}(q\|p)]$ | 0.5-2.0 | >5 or <0.05 |
| `sigma_sem_mean` | $\mathbb{E}[\bar{\sigma}_q]$ | 0.1-1.0 | →0 or >10 |
| `sigma_sem_max` | $\max \sigma_q$ | <2 | >10 (explode) |
| `rate_R` | $\mathbb{E}[\text{KL}]$ | stable | diverge |
| `distortion_D` | $L_\text{CE}$ | monotonic ↓ | bounce |
| `z_sem_to_prior_cos` | $\cos(\mu_i, p_{y_i})$ | >0.7 | <0.3 |

**SupCon (B only)**:
| Metric | Definition |
|--------|-----------|
| `pos_sim_mean` | same-class cos avg |
| `neg_sim_mean` | diff-class cos avg |
| `alignment_WI` | $\mathbb{E}[\|z_i - z_j\|^2]$ same class |
| `uniformity_WI` | $\log \mathbb{E}[e^{-2\|z_i-z_j\|^2}]$ |
| `n_positive_avg` | batch avg positive count |

**Prototype**:
| Metric | Definition |
|--------|-----------|
| `prototype_drift_L2` | $\|p_c^{(r)} - p_c^{(r-1)}\|_2$ |
| `intra_class_z_std` 🔥 | per-class std of $z_\text{sem}$ (collapse detector) |
| `inter_class_proto_min_cos` | $\min_{c\ne c'}\cos(p_c, p_{c'})$ |

**Round-2 新增**:
| Metric | Definition |
|--------|-----------|
| `R_per_domain` | KL avg per domain |
| `R_std_across_domains` | std of R_d across domains |
| `irm_grad_var` | $\text{Var}_d[\nabla_\theta L_\text{CE}^d]$ (IRM-style) |
| `kl_collapse_alert` | bool: KL<0.1 AND std<0.1 |

**梯度诊断**:
| Metric | Definition |
|--------|-----------|
| `grad_cos(CE, VIB)` | gradient cos angle |
| `grad_cos(CE, SupCon)` | (B only) |
| `grad_norm_{CE, VIB, SupCon}` | L2 norm ratio |

### 3.2 Layer-2: Train-time Proxy Probe (每 20 轮, 500 子集)

冻结当前 encoder,sklearn 训 probe 在 held-out:

| Probe | Classifier | Target |
|-------|-----------|-------|
| `proxy_probe_sty_class_linear` | LogReg | $z_\text{sty} \to y$ |
| `proxy_probe_sty_class_MLP64` | MLP-64 | $z_\text{sty} \to y$ |
| `proxy_probe_sty_class_MLP256` | MLP-256 | $z_\text{sty} \to y$ |
| `proxy_probe_sem_class_linear` | LogReg | $z_\text{sem} \to y$ |
| `proxy_probe_sem_domain_linear` | LogReg | $z_\text{sem} \to c$ |
| `proxy_probe_sty_domain_linear` | LogReg | $z_\text{sty} \to c$ |

**关键**: 子集 500 + 每 20 轮,不拖慢训练 (~45 min 额外 overhead)。

### 3.3 Layer-3: Post-hoc Moyer Sweep

Frozen encoder R200 后:
- **Probe depths**: 0 (linear) / 1 (hidden=64) / 2 (hidden=64,64) / 3 (hidden=64,64,64)
- **Protocol**: Adam lr=0.001, BN, absolute-error loss (Moyer 2018 原设置)
- **Targets**: (sty→class, sem→class, sty→domain, sem→domain)
- **+ Hewitt-Liang selectivity**: probe 在 real y 上 acc - probe 在 random y 上 acc (防过拟合作弊)

### 3.4 Health snapshot 每 100 轮输出:

```
=== Round 100 Health ===
Task:     CE=0.42  acc(val)=0.78  proto_active=7/7
VIB:      KL=0.87  σ̄=0.34  μ_to_prior=0.81  ✓
SupCon:   pos=0.73 neg=0.15 align=0.45     ✓  [B only]
Geom:     cos(sem,sty)=0.02 z_sty_norm=0.45 ✓
Proxy probes (sty→class):
          linear=0.27 ↓✓ MLP-64=0.48 ↓✓ MLP-256=0.62 ↓✓
Proxy probes (sem→class): linear=0.81 (ref)
Proxy probes (sty→domain): linear=0.76 ✓ (sty is not dead)
Grads:    cos(CE,VIB)=+0.12  grad_norm_ratio(VIB/CE)=0.34 ✓
IRM:      grad_var_across_domains=0.008
Verdict:  ON TRACK
```

---

## 4. Experimental Design

### 4.1 2×2 Ablation Matrix (Round-2 review 重设计)

| | 无 VIB | 有 VIB |
|---|:---:|:---:|
| InfoNCE (原) | M0 orth_only (已有 EXP-109) | **M4 FedDSA-VIB** 🆕 |
| SupCon | **M6 orth+SupCon** 🆕 | **M5 FedDSA-VSC** 🆕 |

**Identifiable contributions**:
- VIB effect: $\Delta_\text{VIB} = \frac{1}{2}[(M_4 - M_0) + (M_5 - M_6)]$
- SupCon effect: $\Delta_\text{SupCon} = \frac{1}{2}[(M_6 - M_0) + (M_5 - M_4)]$
- Interaction: $\iota = M_5 - (M_0 + \Delta_\text{VIB} + \Delta_\text{SupCon})$

### 4.2 Seeds: 10 seeds (not 3)

Round-2 review 指出 orth_only σ=1.46 需要 ~33 seeds 检测 Δ=1pp。妥协方案:
- 10 seeds: {2, 15, 333, 42, 7, 100, 201, 500, 777, 999}
- 主卖点改为 **probe 大 effect (5-seed 已够)** + accuracy **non-inferior test**

### 4.3 Run Budget

| Method | Dataset | Seeds | Runs |
|--------|---------|:---:|:---:|
| M0 补跑 (已 3 补 7) | PACS, Office | 7 × 2 | 14 |
| M4 FedDSA-VIB | PACS, Office | 10 × 2 | 20 |
| M5 FedDSA-VSC | PACS, Office | 10 × 2 | 20 |
| M6 orth+SupCon | PACS, Office | 10 × 2 | 20 |
| **Total new** | | | **74** |

GPU: 4090 × 6 concurrent × 3h/run = ~37h wall.

### 4.4 Statistical Tests

- **Paired t-test** on 10-seed results (method-method pair)
- **Non-inferiority test**: $H_0: \text{acc}_\text{ours} - \text{acc}_\text{orth} < -2\text{pp}$ (拒绝=过)
- **Probe effect test**: one-sided paired t-test on `MLP-256 probe` (一侧,我方预期更低)

---

## 5. Baseline Comparison

### 5.1 Primary Baselines (同 seed,同 config)
| Method | Source | Status |
|--------|--------|:-------:|
| M0 orth_only | EXP-109/110 | ✅ 3 seed, 补到 10 |
| M1 CDANN | EXP-108 | ✅ 3 seed, 不补 (论文 table 对照) |

### 5.2 External Baselines (论文 number + 可能重跑)
| Method | Source | Status |
|--------|--------|:-------:|
| FedAvg | CLAUDE.md 参考 | 用论文数字 |
| FedBN | EXP-102 | 用 EXP-102 数字 |
| FedProto | EXP-065 | 用 EXP-065 数字 |
| FPL | RethinkFL/ | 用论文数字 |
| FDSE | FDSE_CVPR25/ | 可复现 (if 时间允许) |
| FediOS | arXiv 2311.18559 | 可复现 (if 时间允许) |

---

## 6. Implementation Status

### 6.1 已完成 (Phase 5c-1/2/3/4) ✅

| 文件 | 内容 | 测试 |
|------|------|:---:|
| `FDSE_CVPR25/algorithm/common/vib.py` | VIBSemanticHead + closed-form KL + EMA prototype + lambda_ib_schedule | **15/15** ✅ |
| `FDSE_CVPR25/algorithm/common/supcon.py` | SupCon loss + Wang-Isola diagnostics | **9/9** ✅ |
| `FDSE_CVPR25/algorithm/common/diagnostic_ext.py` | KL-collapse / R_per_domain / irm_gradient_variance | **11/11** ✅ |
| `FDSE_CVPR25/tests/test_vib.py` | VIB unit tests | |
| `FDSE_CVPR25/tests/test_supcon.py` | SupCon unit tests | |
| `FDSE_CVPR25/tests/test_diagnostic_ext.py` | ext diag unit tests | |

**Total: 35/35 tests green**

### 6.2 剩余 (Phase 5c-5/6) 🔄

- [ ] `algorithm/feddsa_vib.py` — 方案 A Client/Server (基于 feddsa_sgpa.py)
- [ ] `algorithm/feddsa_vsc.py` — 方案 B
- [ ] `algorithm/feddsa_supcon.py` — M6
- [ ] `config/{pacs, office}/feddsa_{vib, vsc, supcon}.yml` (6 files)
- [ ] `tests/test_integration_smoke.py` — 8 样本 1 round 全流程

### 6.3 部署 Gate (Phase 5d/e/f) 📋

| Gate | 内容 |
|------|------|
| 5d Codex code review | Critical / Important 全改,含 FedAvg private_keys 白名单正确性 |
| 5e Smoke test | 8 样本 1 round 全流程,所有诊断 metric 有合理值 |
| 5f 用户确认 | 用户最终 approve 后部署 |

---

## 7. Verdict Decision Tree

| Outcome | Next |
|---------|------|
| MLP-256 probe ≤ 0.50 AND PACS acc ≥ 80.0 AND 10-seed std < 1.5 | ✅ **全胜**, paper draft |
| MLP-256 probe ≤ 0.55 BUT acc ∈ [79, 80] | ⚠️ 调 λ_IB 降低,或 σ_prior ablate |
| MLP-256 probe > 0.65 | ❌ **方案失败**,pivot 到诊断论文 (Codex R2 推荐) |
| KL-collapse alert triggered | ⚠️ σ_prior 扫描 + 降 λ_IB |
| `irm_grad_var` 高 | ⚠️ 客户端分布过异构,需 FedProx 式正则 |

---

## 8. References

- **VIB**: Alemi et al. "Deep Variational Information Bottleneck", ICLR 2017
- **Moyer VIB**: Moyer et al. "Invariant Representations without Adversarial Training", NeurIPS 2018
- **SupCon**: Khosla et al. "Supervised Contrastive Learning", NeurIPS 2020
- **von Kügelgen**: "Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style", NeurIPS 2021 (inspired VIB+prototype combination)
- **Wang-Isola**: "Understanding Contrastive Representation Learning through Alignment and Uniformity", ICML 2020
- **Hewitt-Liang**: "Designing and Interpreting Probes with Control Tasks", EMNLP 2019
- **DSN**: Bousmalis et al. "Domain Separation Networks", NeurIPS 2016 (architecture inspiration)
- **IRM**: Arjovsky et al. "Invariant Risk Minimization", arxiv 2019 (IRM grad var diagnostic)
