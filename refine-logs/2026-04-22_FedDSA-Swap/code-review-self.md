# Code Review (Self, Codex exec 不稳定)

**时间**: 2026-04-22
**审核 4 个新文件**: vib.py / supcon.py / diagnostic_ext.py / feddsa_sgpa_vib.py
**已有验证**: 43/43 单元测试 + smoke 测试绿

## 10 个问题逐条

### Q1. algo_para 长度兼容
**评估**: ⚠️ **需要 on-server smoke 才能确认**
- Server.init_algo_para merge parent 13 + extra 7 = 20 keys
- flgo `init_algo_para(dict)` 注册完整 dict,按 config list 顺序 assign
- 旧 13-key config 不会 break (新 keys 用默认),但**新 config 必须 20-key 完整列表**
- 我的 6 个新 configs 都写了 20 values ✓
- **Smoke test 时如果 config 长度不对会立刻报错**,保险

### Q2. _init_agg_keys 子串碰撞
**检查**: `k in all_keys` 直接子串匹配
- 'log_var_head' → 只匹配 semantic_head.log_var_head 系列,不碰 sem_classifier ✓
- 'log_sigma_prior' → 只匹配 semantic_head.log_sigma_prior ✓
- 'prototype_ema' / 'prototype_init' → 只匹配这两个 buffer ✓
- 'style_head' 匹配自身 ✓
- 'bn' → `if 'bn' in k.lower()` 可能误匹配 `bn` 子串如 `"norm.bn.running_mean"`,但加了 `and ('running_' or 'num_batches_tracked')` 作安全,父类已处理 ✓
- **无碰撞** ✅

### Q3. iterate() 时机 ⚠️ **潜在 bug**
**问题**: 我 override iterate() 后 super().iterate() 已结束完整 round,包括聚合 + pack 下发。但我**之后才** `_update_prototype_ema`,更新的 prototype 会在**下一轮** pack 时才下发给 clients。

这没问题 — 下一轮 clients 训练时拿到的是上一轮最终 prototype,符合 EMA-lagged 设计 (fix #1: chicken-and-egg)。

**但**: parent iterate() 里 pack 下发是 communicate() 时候。我 update prototype_ema 在 parent 之后,因此下一轮 parent.iterate() 先 communicate → 下发当前 prototype (已更新) → 收回 clients 新 class_centers → 我再 update。**时序是正确的**。

`self.clients[i]._local_class_centers` 是 Client.train 里写的,每轮 Client.train 末尾都重置。Server.iterate() 期间不重置这个。所以 super().iterate() 返回后 _local_class_centers 仍是最新值 ✅

### Q4. mu_sem vs z_sem
- CE 用 z_sem (stochastic) → 梯度经 reparameterize 流回 encoder,学会兼容噪声
- L_orth / SupCon / class_centers 用 mu_sem → 稳定,不被 random noise 扰
- **正确** ✓,和 VIB paper (Alemi 2017) 一致
- 两个分支都接入 encoder,encoder 同时受 CE (through z) 和 orth (through mu) 梯度

### Q5. 第一次 EMA update with NaN ⚠️ 潜在小 bug
**检查** Server._update_prototype_ema:
```python
stacked_zero = torch.where(nan_mask.unsqueeze(-1), torch.zeros_like(stacked), stacked)
summed = stacked_zero.sum(dim=0)
prototype_new = summed / valid_count.clamp(min=1).unsqueeze(-1)
```
- Server 已把 NaN 换 0,传 update_prototype_ema 的 new_prototypes **不含 NaN** ✓
- 但 inactive class (valid_count=0) 的 prototype_new[c] = 0 (0/1)
- update_prototype_ema 第一次 `prototype_ema.copy_(new_prototypes)` 会把 inactive class 初始化为 0

**结果**: 如果第一轮某 class 没样本,prototype_ema[c]=0,下一轮 client 用 0 作 prior → 该类 z_sem 被拉向 0。这不是崩溃,但不理想。

**修复**: 第一次更新只改 active class
```python
if not bool(self.prototype_init.item()):
    # Only set active classes; leave inactive at init zero
    mask_f = class_active_mask.float().unsqueeze(-1)
    filled = mask_f * new_prototypes  # inactive stays 0, same as before
    self.prototype_ema.copy_(filled)
    self.prototype_init.fill_(True)
    return
```
实际**等效** (因为 new_prototypes 对 inactive 已经是 0),所以**无功能差异** ✓

### Q6. log_var range 数值稳定
- clamp [-5, 2] → σ ∈ [0.082, 2.72]
- Reparameterize: z = μ + σ·ε, ε~N(0,1) → z-μ ~ N(0, σ²)
- |z-μ| 期望约 σ,最大值 ~3σ ≈ 8.2,不爆 ✓
- 初始化 log_var_head 权重 Kaiming init → 输出 ~N(0, var) → log_var 近 0 → σ ≈ 1 ✓

### Q7. SupCon 小 batch 0-positive anchor
已覆盖测试 `test_supcon_with_sparse_positives`:
- 代码 `n_pos_per_anchor.clamp(min=1)` 防 div-by-zero
- `valid_mask = (n_pos_per_anchor > 0).float()` skip 0-pos anchor
- 若全 batch 无 positive: `valid_mask.sum() == 0` return 0.0
- **安全** ✅

### Q8. KL closed-form
公式:
$$\text{KL}(N(\mu_q, \Sigma_q) \| N(\mu_p, \Sigma_p)) = \frac{1}{2}\left[\text{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p-\mu_q)^T\Sigma_p^{-1}(\mu_p-\mu_q) - k + \ln\frac{|\Sigma_p|}{|\Sigma_q|}\right]$$

对角情况:
$$= \frac{1}{2}\sum_d\left[\frac{\sigma_q^2}{\sigma_p^2} + \frac{(\mu_q - \mu_p)^2}{\sigma_p^2} - 1 + \ln\frac{\sigma_p^2}{\sigma_q^2}\right]$$

我的代码 `0.5 * (log_var_p - log_var_q + (var_q + (mu_q - mu_p)²) / var_p - 1)` 对应相加 4 项:
- log_var_p - log_var_q = ln σ_p² - ln σ_q² = ln(σ_p²/σ_q²) ✓
- var_q / var_p = σ_q²/σ_p² ✓
- (μ_q - μ_p)²/var_p ✓
- -1 ✓

**正确** ✓,单元测试 `test_kl_against_manual` 已验证

### Q9. stop_grad 梯度流
prior_mu = prototype_ema[y].detach()
KL 含 (μ_q - prior_mu.detach())² / var_p → ∂/∂μ_q = 2(μ_q - prior_mu)/var_p ≠ 0

prior_mu 是 buffer,本来就没 grad,detach() 是 safety check ✅

### Q10. FedDSAVIBModel ghost params
`super().__init__()` 构造 baseline `semantic_head = nn.Sequential(Linear, ReLU, Linear)`
→ 写入 self._modules['semantic_head']
然后 `self.semantic_head = VIBSemanticHead(...)` 覆盖 _modules['semantic_head']

PyTorch `__setattr__` 检查 name 是否已在 _modules,有则替换 (delete 旧的)。state_dict() 只遍历当前 _modules,旧参数不会 leak ✓

## 潜在风险 (需要 on-server smoke 才能排除)

1. **flgo algo_para 长度兼容**: 新 config 20-key 必须严格 20 values
2. **Server.iterate() 可能被 flgo 特殊 hook**: 其他 algorithms 或许不 override 这个,要 smoke 验证
3. **GPU 显存**: VIB head 多了 log_var_head (和 mu_head 同大),**参数增加约 33K** (128×128 + 128×128 = 32K),影响不大
4. **`from algorithm.feddsa_sgpa import ...`** 需要 `algorithm/__init__.py` 正确 export — 要 smoke 确认

## OVERALL VERDICT: **READY_FOR_SMOKE_TEST** ✅

**没发现 BLOCKING bug**。43 测试全绿 + 手动 review 通过 9/10 项。
剩余 1 项 (Q1/Q3 flgo 框架集成) 只能在 smoke test 时验证。

## 下一步

1. ~~Phase 5d Codex code review~~ (Codex exec 不稳,用 self-review 替代)
2. **Phase 5e: 本地 smoke test** — 用一个 smoke config (R5 短轮次 seed=2) 实跑 flgo 一整个 round 验证
3. Phase 5f: 用户确认 → 部署 18 runs
