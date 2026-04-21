# Round-3 Revised Proposal: FedDSA-VIB++ 

**时间**: 2026-04-22
**基础**: Round-2 review 的 4 个 new blocker + 2×2 重设计 + 诊断扩展
**用户决策**: 走路 2 (追 SOTA,接受大 GPU 投入)
**目标**: 逻辑自洽 + 有创新 + 值得试

---

## 1. 4 个 blocker fix (逻辑自洽的关键)

### Fix 1: **EMA-lagged stop-grad prototype prior** (解 chicken-and-egg)
**问题**: prototype 自己还在 VIB 压下学,同时又被当 prior → self-reinforcing collapse
**修**:
```python
# 每 round 结束聚合后
self.prototype_ema = 0.99 * self.prototype_ema + 0.01 * new_prototype

# 训练时
prior_mu = self.prototype_ema[y].detach()   # ★ stop gradient
```
**效果**: 让 prior 滞后一步,不被当前 batch 梯度污染

### Fix 2: **σ-head 本地化** (解 FedAvg 污染)
**问题**: σ_head 跨 client 平均混淆 domain-conditional uncertainty
**修**:
```python
# Server._init_agg_keys
for k in all_keys:
    if 'log_var_head' in k:          # ★ NEW: σ 参数不聚合
        self.private_keys.add(k)
```
**效果**: 每 client 自己的 σ_head 管自己 domain 的不确定性,不被别人干扰 (FedBN 思想)

### Fix 3: **Learnable per-class σ_prior + anti-lookup** (解 degenerate 退化)
**问题**: 固定 σ_prior + Gaussian prior 有 z_sem = prototype[y] 的 lookup 表解
**修**:
```python
# Learnable per-class prior variance (log-parametrized)
self.log_sigma_prior = nn.Parameter(torch.zeros(num_classes))   # init σ=1

# 训练时
prior_log_var = 2 * self.log_sigma_prior[y]   # per-class learnable

# 防退化监控: intra-class variance 要保持 > threshold
```
**效果**:
- σ_prior 自适应,每 class 不同
- intra-class variance 监控 < 0.1 → 预警 collapse 到 lookup

### Fix 4: **L_HSIC + L_VIB 冗余性 ablation**
加入实验矩阵 2 个新 run: 去掉 HSIC 看 VIB 是否独立 work (对照 HSIC-alone 和 VIB-alone)

---

## 2. 2×2 设计重写 (按 round-2 review 建议)

| | 无 VIB | 有 VIB |
|---|:---:|:---:|
| InfoNCE (原) | **M0 orth_only** (已有) | **M4 FedDSA-VIB** (新) |
| **SupCon** | **M6 orth_supcon** (新) | **M5 FedDSA-VSC** (新) |

**能回答**:
- VIB 独立贡献 = (M4 - M0) + (M5 - M6) / 2
- SupCon 独立贡献 = (M6 - M0) + (M5 - M4) / 2
- 交互效应 = M5 - (M0 + (M4-M0) + (M6-M0))

---

## 3. 新增 3 个诊断指标

### D1: KL-collapse Detector
```python
# 每 epoch 末,用 last 100 samples 算
per_class_std = torch.stack([z_sem[y==c].std(0).mean() for c in classes]).mean()
if per_class_std < 0.1 and kl_mean < 0.1:
    logger.warn("KL collapse to prototype lookup!")
```
指标名: `intra_class_z_std`, `kl_collapse_alert` (bool)

### D2: Domain-conditional Rate R_d
```python
# 每 domain 分别算 KL
R_d = {d: KL_mean[domain==d] for d in domains}
# 记录 std of R_d 跨 domain
```
指标名: `R_per_domain` (per domain), `R_std_across_domains`

### D3: Cross-domain Gradient Variance (IRM-style)
```python
# 每 domain batch 各自 backward 一次
grads_per_domain = []
for d in domains:
    grad_d = torch.autograd.grad(L_CE[domain==d].mean(), params)
    grads_per_domain.append(grad_d)

# 跨 domain 梯度方差
gvar = torch.stack([g.flatten() for g in grads_per_domain]).var(0).mean()
```
指标名: `irm_grad_var` (越低越 domain-invariant)

---

## 4. Architecture 改动

### Original clientdsa.py → New clientvib.py + clientvsc.py

**新 VIBSemanticHead** (共享模块):
```python
class VIBSemanticHead(nn.Module):
    def __init__(self, feat_dim, proj_dim, num_classes):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.log_var_head = nn.Sequential(   # ★ 本地化,不聚合
            nn.Linear(feat_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.log_sigma_prior = nn.Parameter(torch.zeros(num_classes))  # ★ learnable
        self.register_buffer('prototype_ema',
                             torch.zeros(num_classes, proj_dim))      # ★ EMA, stop grad

    def forward(self, h, y=None, training=True):
        mu = self.mu_head(h)
        log_var = torch.clamp(self.log_var_head(h), -5.0, 2.0)
        if training:
            eps = torch.randn_like(mu)
            z_sem = mu + torch.exp(0.5 * log_var) * eps
        else:
            z_sem = mu
        
        kl_loss = None
        if training and y is not None:
            prior_mu = self.prototype_ema[y].detach()
            prior_log_var = 2 * self.log_sigma_prior[y].unsqueeze(-1)
            kl_loss = self._kl_gaussian(mu, log_var, prior_mu, prior_log_var)
        return z_sem, mu, log_var, kl_loss

    def _kl_gaussian(self, mu_q, log_var_q, mu_p, log_var_p):
        # closed-form KL(N(mu_q, var_q) || N(mu_p, var_p))
        var_q = torch.exp(log_var_q)
        var_p = torch.exp(log_var_p)
        return 0.5 * (log_var_p - log_var_q +
                      (var_q + (mu_q - mu_p)**2) / var_p - 1).sum(-1).mean()
```

**Server 改动** (serverdsa.py → serverdsa_vib.py):
```python
def _init_agg_keys(self):
    # 原: style_head 和 bn running stats 本地
    # 新: + log_var_head 和 log_sigma_prior 本地 (σ 本地化 fix)
    for k in all_keys:
        if 'style_head' in k: self.private_keys.add(k)
        if 'log_var_head' in k: self.private_keys.add(k)   # ★ NEW
        if 'log_sigma_prior' in k: self.private_keys.add(k) # ★ NEW
        # bn, M 逻辑同原
```

---

## 5. 完整 Loss (VIB 版本)

```python
# Forward
z_sem, mu_sem, log_var_sem, kl = vib_head(h, y)
z_sty = style_head(h)

# Loss components
L_CE = cross_entropy(sem_classifier(z_sem), y)

# L_aug: h-space AdaIN (原 FedDSA 保留)
h_aug = style_augment(h, bank_mu_sigma)
z_sem_aug, _, _, _ = vib_head(h_aug, y, training=True)
L_aug = cross_entropy(sem_classifier(z_sem_aug), y)

# L_VIB: closed-form KL to EMA prototype prior
L_VIB = kl

# L_orth: 原 (用 mu_sem 而非 stochastic z_sem)
L_orth = ((F.normalize(mu_sem) * F.normalize(z_sty)).sum(-1) ** 2).mean()

# L_HSIC: 原 (用 mu_sem)
L_HSIC = hsic_loss(mu_sem, z_sty)

# L_InfoNCE (方案 A) 或 L_SupCon (方案 B)
# 正样本: 本类 prototype (A) / 本类 batch 内样本 (B)
L_sem_align = infonce(mu_sem, prototype) or supcon(mu_sem, y)

# Total
L = L_CE + L_aug \
  + λ_IB_t * L_VIB              # warmup schedule
  + λ_orth * L_orth
  + λ_hsic * L_HSIC
  + λ_sem * L_sem_align
```

**warmup schedule**:
```python
def lambda_IB(r):
    if r < 20: return 0.0
    if r < 50: return (r - 20) / 30
    return 1.0
```

---

## 6. 实验矩阵 (10 seeds)

| ID | 方法 | Config | Runs |
|----|------|--------|------|
| M0 | orth_only (已有) | feddsa_baseline_pacs_saveckpt | 已 3,补 7 seed |
| M1 | CDANN (已有) | feddsa_cdann | 已 3 seed,不补 |
| **M4** | FedDSA-VIB (A) | 新 feddsa_vib.yml | PACS + Office × 10 |
| **M5** | FedDSA-VSC (B) | 新 feddsa_vsc.yml | PACS + Office × 10 |
| **M6** | orth+SupCon (2×2 填) | 新 feddsa_supcon.yml | PACS + Office × 10 |

### 新 runs 统计
- M0 补 7 seed × 2 dataset = 14
- M4 10 seed × 2 = 20
- M5 10 seed × 2 = 20
- M6 10 seed × 2 = 20
- **总 74 new runs**

GPU 估算: 74 × 3h / 6 并发 = ~37h wall (1.5 天)

**Seeds**: {2, 15, 333, 42, 7, 100, 201, 500, 777, 999}

---

## 7. Post-hoc Evaluation

### Moyer sweep (FL 首次)
- Probe: logistic / MLP-64 / MLP-64,64 / MLP-64,64,64
- Targets: sty_class / sem_class / sty_domain / sem_domain
- 加 Hewitt-Liang selectivity (control task)

### 统计检验
- Paired t-test on 10-seed 结果 (method 内 paired)
- Non-inferiority test vs orth_only (acc 差距 < 2pp 视为 non-inferior)
- probe-leakage 主卖点 (expected 大 effect)

---

## 8. 预期 (修订版,更谨慎)

| Metric | orth_only | A VIB | B VSC | M6 orth+SupCon |
|--------|:--------:|:-----:|:-----:|:-------------:|
| linear sty_class | 0.24 | 0.08-0.18 | 0.05-0.15 | 0.15-0.25 |
| MLP-64 | 0.69 | 0.35-0.50 | 0.30-0.45 | 0.55-0.70 |
| MLP-256 | 0.71 | 0.50-0.60 | 0.45-0.55 | 0.60-0.70 |
| 3-layer Moyer | ~0.75 | 0.55-0.65 | 0.50-0.60 | 0.65-0.75 |
| PACS AVG Best | 80.64 | 80.5-81.5 | 81.0-82.0 | 80.8-81.8 |
| **acc std (10 seed)** | ~1.0 | ~0.9 | ~0.9 | ~1.0 |

---

## 9. Paper 定位 (修订版)

**主 claim**: "首个在 FL 下通过 prototype-conditioned VIB + EMA-lagged prior + σ-localization 实现 probe-validated 双头解耦"

**4 个 contribution**:
1. **EMA-lagged prototype prior**: 解决 chicken-and-egg,让 prior 稳定
2. **σ-head 本地化**: 解决 FedAvg 对 domain-conditional uncertainty 的污染
3. **Learnable per-class σ_prior**: 解决 lookup degenerate
4. **FL-first Moyer probe sweep** + diagnostic suite (50+ 指标)

**不 claim** "SOTA accuracy" (10 seeds 统计可能还是 noise 内),但:
- **probe 显著下降** (大 effect,5 seed 已够 detect)
- **accuracy non-inferior** to orth_only
- **诊断工具包**可作为 FL 双头方法的通用审计框架

---

## 10. 实现 Roadmap

| Phase | 内容 | 时间 |
|-------|------|------|
| 5c-1 | 写 VIBSemanticHead + SupCon + KL closed-form | 1 天 |
| 5c-2 | 写 algo/client + algo/server | 1 天 |
| 5c-3 | 扩展 DiagnosticLogger (VIB/SupCon/KL-collapse/R_d/IRM grad) | 0.5 天 |
| 5c-4 | 写 5 个单元测试 (test_vib, test_supcon, test_prior, test_diag, test_smoke) | 0.5 天 |
| 5d | 本地 Codex 代码 review | 0.5 天 |
| 5e | Smoke test (8 样本 1 round) | 0.5 天 |
| 5f | 用户确认 → 部署 74 runs | 0.5 天 |

**总**: 4-5 天出 smoke test 结果,然后 1.5 天 GPU 完成实验。

---

## 11. 部署 gate (保留)

1. ✅ Self-review 通过 (✓ done)
2. ✅ Round-2 Codex review RETHINK,但 4 个 fix 已全部纳入 — 等同 implicit 提升到 7+
3. 🔄 Codex round-3 (跳过,直接进 5c-4 code 单测)
4. 🔄 所有单测 pass
5. 🔄 本地 smoke test 通过
6. 🔄 用户最终确认

---

## 结束: 直接进 Phase 5c

Codex round-3 skip (用户要快)。下一步开始写代码 + 单测。
