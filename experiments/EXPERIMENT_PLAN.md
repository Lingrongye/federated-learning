# FedDSA-Adaptive 实验方案

> 版本: v2.0 | 日期: 2026-04-14
> 基于: 15篇文献调研 + 71组历史实验 + GPT-5.4 Round 1审稿修订
> Round 1评分: 5/10 → 本版修订所有关键问题

---

## 一、问题诊断

### 1.1 当前FedDSA最佳结果 (EXP-069f, R500)

| 数据集 | FedDSA | FDSE | 差距 |
|--------|--------|------|------|
| PACS AVG | 80.96% | 82.17% | **-1.21%** |
| Office AVG | 90.23% | 91.58% | **-1.35%** |
| DomainNet AVG | 72.05% | 74.50% | **-2.45%** |

### 1.2 核心瓶颈定位

通过EXP-020~070的消融分析，定位到两个核心问题：

**问题1: Style Bank对低gap域有害**
- EXP-052 (NoAug) PACS=80.57% vs EXP-069f (有Aug) PACS=80.96%
- 增强仅提升+0.39%，远低于预期
- 原因：Beta(0.1,0.1)混合是U型分布 → Photo域被注入Art/Sketch风格 = 噪声
- 佐证：EXP-060~063所有style-side修改均失败（gate、softbeta、regime等）

**问题2: 全局原型"语义稀释"**
- 将Photo-dog和Sketch-dog的语义特征平均 → 模糊原型
- InfoNCE对齐到模糊原型 → 域间冲突
- EXP-071尝试域感知原型（结果待确认）

### 1.3 根因分析

```
当前style augmentation: alpha ~ Beta(0.1, 0.1)
→ alpha大概率≈0或≈1 → 要么不增强，要么完全替换
→ 没有"适度增强"的中间状态
→ 更关键：所有域用同一个alpha分布，不区分域gap大小

需要: 域gap大(Art/Sketch) → 强增强
      域gap小(Photo/DSLR) → 弱增强或不增强
```

---

## 二、技术方案: FedDSA-Adaptive (v2, 审稿修订版)

### 2.0 审稿关键修订

Round 1 (GPT-5.4, 5/10) 指出的核心问题与修订：

| 审稿问题 | 修订 |
|---------|------|
| ~~M2(概率采样)丢失"Share"claim~~ | **砍掉M2**，保留style bank dispatch（保证Share完整性） |
| gap度量用backbone h含语义混合 | **改用style head z_sty的统计量**作为gap信号 |
| 4客户端divide-by-max不稳定 | **改用EMA z-score归一化** |
| aug_strength太刚性(确定性) | **加随机抖动**: α = α_mean + ε, ε~N(0,0.05) |
| 缺固定α基线 | **新增EXP-072a/b/c**: 固定α=0.2/0.5/0.8 |
| 缺M3-only基线 | **新增EXP-072d**: M3单独测试 |
| 仅seed=2不可信 | **每个实验至少3 seeds** (2, 333, 42) |

### 2.1 修订后方案：两个改进点

| 改进 | 名称 | 来源论文 | 解决的问题 |
|------|------|---------|-----------|
| **M1** | 自适应增强强度 | AdaIN(ICCV'17) + FedFA(ICLR'23)启发 | Style bank伤害低gap域 |
| **M3** | 域感知原型对齐 | FedDAP(CVPR'26) | 全局原型语义稀释 |

> ~~M2(概率采样)~~ 已删除 — 审稿指出它将"Share"退化为"Perturb"，与论文核心claim矛盾。

### 2.2 M1: 自适应增强强度 (Adaptive Augmentation Strength)

**动机**: AdaIN原论文(Huang & Belongie, ICCV 2017)的content-style trade-off参数α，当前FedDSA未利用。

**域Gap度量 (修订: 使用style head z_sty，非backbone h)**：

```python
# 服务器在聚合后计算域gap
# 关键修订: 使用 z_sty 的统计量（纯域信号），不用 h（混合语义+域）
# 客户端上传: (mu_sty, sigma_sty) = z_sty的batch均值和标准差

# 全局中心
mu_sty_global = EMA(mean({mu_sty_1, ..., mu_sty_K}))  # EMA平滑
sigma_sty_global = EMA(mean({sigma_sty_1, ..., sigma_sty_K}))

# Per-client gap (L2距离)
raw_gap_i = ||mu_sty_i - mu_sty_global||^2 + ||sigma_sty_i - sigma_sty_global||^2

# 归一化 (修订: 用EMA z-score, 非divide-by-max)
gap_mean = EMA(mean(raw_gaps))
gap_std  = EMA(std(raw_gaps))
gap_normalized_i = clamp((raw_gap_i - gap_mean) / (gap_std + 1e-6), 0, 1)
```

**自适应α (修订: 加随机抖动)**：

```python
# 域gap → 增强强度 (含抖动)
alpha_mean = aug_min + (aug_max - aug_min) * gap_normalized
alpha = clamp(alpha_mean + torch.randn(1) * 0.05, aug_min, aug_max)

# 保留原style bank dispatch: 外域风格仍从真实style bank采样
idx = np.random.randint(0, len(self.local_style_bank))
mu_ext, sigma_ext = self.local_style_bank[idx]

# AdaIN变换
h_adain = sigma_ext * (h - mu_local) / sigma_local + mu_ext

# Trade-off: α控制注入强度
h_aug = alpha * h_adain + (1 - alpha) * h
```

**与原方案(Beta混合)的关键区别**:
- 原: `alpha ~ Beta(0.1,0.1)` → U型，无中间态，所有域相同
- 新: `alpha = f(gap) + noise` → gap大的域更强增强，gap小的域保护

### 2.3 M3: 域感知原型对齐 (Domain-Aware Prototype Alignment)

（审稿评为"最强最有理论依据的改进"）

**保持EXP-071实现**，核心改动：

```python
# 服务器保留 per-(class, client_id) 原型 (不做全局平均)
domain_protos[(class, client_id)] = proto_i^c

# 客户端用 SupCon 多正例 InfoNCE 对齐:
# 正例 = 同类的所有域原型
# 负例 = 异类的所有域原型
L_align = SupConInfoNCE(z_sem, y, domain_protos)
```

**代码**: 复用 `feddsa_domain_aware.py` 的 `_infonce_domain_aware()`

---

## 三、实验矩阵 (修订版)

### 3.1 Claim-Driven 实验设计

| Claim | 实验 | 验证方式 | Seeds |
|-------|------|---------|-------|
| C1: 自适应α比固定Beta更好 | EXP-072 vs 069f | PACS AVG, 各域准确率 | 2,333,42 |
| C1a: 自适应α比固定α更好 | EXP-072 vs 072a/b/c | 证明自适应必要性 | 2,333,42 |
| C2: 域感知原型比全局平均更好 | EXP-072d vs 069f | 解耦M3的贡献 | 2,333,42 |
| C3: M1+M3叠加优于单独 | EXP-073 vs 072/072d | 正交叠加验证 | 2,333,42 |
| C4: 最终方案超过FDSE | EXP-073 vs FDSE | 三数据集全面对比 | 2,333,42 |

### 3.2 实验运行计划 (按优先级)

#### Phase A: 基线对照 (必须先跑)

**EXP-072a/b/c: 固定α基线** (证明自适应必要性)
```
算法文件: feddsa_adaptive.py (fixed_alpha mode)
配置: α固定=0.2 / 0.5 / 0.8, 其余与069f一致
数据集: PACS, seeds=2,333,42
目的: 如果固定α=某值就够好 → 不需要自适应 → 方案不成立
```

**EXP-072d: M3-only** (域感知原型，无自适应增强)
```
算法文件: feddsa_domain_aware.py (已有)
配置: style augment保持原Beta混合, 改对齐为SupCon多正例
数据集: PACS, seeds=2,333,42
目的: 分离M3的独立贡献
```

#### Phase B: 核心验证

**EXP-072: M1 自适应增强强度**
```
算法文件: feddsa_adaptive.py (adaptive mode)
改动: gap用z_sty, EMA z-score归一化, 随机抖动
配置: aug_min=0.05, aug_max=0.8, 其余同069f
数据集: PACS, seeds=2,333,42
预期: Photo域提升(保护), Sketch不降(仍强增强)
```

#### Phase C: 叠加验证

**EXP-073: M1+M3 Full**
```
算法文件: feddsa_adaptive.py (full mode)
改动: 自适应增强 + 域感知原型对齐
数据集: PACS + Office, seeds=2,333,42
预期: 正交叠加增益
```

#### Phase D: 扩展验证 (如Phase C成功)

**EXP-074: 三数据集完整对比**
```
数据集: PACS + Office + DomainNet, seeds=2,333
完整基线表: FedAvg, FedBN, FedProto, FPL, FDSE, FedDSA-base, FedDSA-Adaptive
```

### 3.3 消融实验矩阵

| ID | 配置 | 对比 | 验证的claim |
|----|------|------|-----------|
| 072a | 固定α=0.2 | vs 072 | 自适应 vs 弱固定 |
| 072b | 固定α=0.5 | vs 072 | 自适应 vs 中固定 |
| 072c | 固定α=0.8 | vs 072 | 自适应 vs 强固定 |
| 072d | M3-only (域感知原型) | vs 069f | M3独立贡献 |
| 072e | 关闭低gap域增强 | vs 072 | 是否直接关闭更好 |
| 072 | M1 (自适应α) | vs 069f | M1独立贡献 |
| 073 | M1+M3 | vs 072, 072d | 叠加效果 |

### 3.4 超参数

| 参数 | 默认值 | 消融范围 | 说明 |
|------|--------|---------|------|
| aug_min (最小增强) | 0.05 | {0, 0.05, 0.1} | 低gap域最低增强 |
| aug_max (最大增强) | 0.8 | {0.6, 0.8, 1.0} | 高gap域最高增强 |
| noise_std (α抖动) | 0.05 | {0, 0.05, 0.1} | 随机性 |
| ema_decay (gap平滑) | 0.9 | {0.8, 0.9, 0.99} | gap归一化的EMA系数 |
| warmup_rounds | 10 | 不变 | 自适应在warmup后启动 |
| 其余超参 | 与069f一致 | 不变 | lr=0.1, tau=0.1, λ_orth=1.0 |

---

## 四、代码架构设计

### 4.1 新文件: `feddsa_adaptive.py`

基于 `feddsa.py` + `feddsa_domain_aware.py` 合并：

```
feddsa_adaptive.py
├── AlexNetEncoder (不变)
├── FedDSAModel (不变)
├── Server (修改)
│   ├── initialize(): 新增 aug_min, aug_max, ema_decay, use_domain_protos
│   ├── _compute_gap_metrics(): 从style bank中z_sty统计量计算gap
│   ├── pack(): 下发 gap_normalized + style_bank + domain_protos
│   ├── iterate(): 聚合后计算gap
│   └── _store_domain_protos(): 保留域原型 (if M3)
└── Client (修改)
    ├── _style_augment(): 自适应α (M1, 替代原Beta混合)
    ├── _infonce_domain_aware(): SupCon多正例 (M3)
    ├── _infonce_global(): 原始单原型InfoNCE (fallback)
    └── train(): 统一训练流程
```

### 4.2 Server端 — gap计算

```python
def _compute_gap_metrics(self):
    """Compute per-client domain gap using style head statistics (z_sty)."""
    if len(self.style_bank) < 2:
        return

    # style_bank stores (mu_sty, sigma_sty) from z_sty head
    all_mu = torch.stack([s[0] for s in self.style_bank.values()])
    all_sigma = torch.stack([s[1] for s in self.style_bank.values()])

    mu_center = all_mu.mean(dim=0)
    sigma_center = all_sigma.mean(dim=0)

    # Raw gaps
    raw_gaps = {}
    for cid, (mu, sigma) in self.style_bank.items():
        gap = ((mu - mu_center)**2).sum() + ((sigma - sigma_center)**2).sum()
        raw_gaps[cid] = gap.item()

    # EMA z-score normalization (robust to outliers, stable across rounds)
    gap_vals = list(raw_gaps.values())
    batch_mean = np.mean(gap_vals)
    batch_std = np.std(gap_vals) + 1e-8

    if not hasattr(self, '_ema_gap_mean'):
        self._ema_gap_mean = batch_mean
        self._ema_gap_std = batch_std
    else:
        self._ema_gap_mean = self.ema_decay * self._ema_gap_mean + (1-self.ema_decay) * batch_mean
        self._ema_gap_std = self.ema_decay * self._ema_gap_std + (1-self.ema_decay) * batch_std

    self.client_gaps = {}
    for cid, g in raw_gaps.items():
        z = (g - self._ema_gap_mean) / (self._ema_gap_std + 1e-8)
        self.client_gaps[cid] = float(np.clip(z * 0.5 + 0.5, 0.0, 1.0))  # sigmoid-like to [0,1]
```

### 4.3 Client端 — 自适应增强

```python
def _style_augment(self, h):
    """Adaptive AdaIN with gap-dependent strength + stochastic jitter."""
    # 仍然从style bank采样外域风格 (保持Share语义)
    idx = np.random.randint(0, len(self.local_style_bank))
    mu_ext, sigma_ext = self.local_style_bank[idx]
    mu_ext, sigma_ext = mu_ext.to(h.device), sigma_ext.to(h.device)

    mu_local = h.mean(dim=0)
    sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)

    # Full AdaIN transform
    h_norm = (h - mu_local) / sigma_local
    h_adain = h_norm * sigma_ext + mu_ext

    # Adaptive strength: gap_normalized → alpha
    alpha_mean = self.aug_min + (self.aug_max - self.aug_min) * self.gap_normalized
    alpha = float(np.clip(alpha_mean + np.random.normal(0, self.noise_std), self.aug_min, self.aug_max))

    # Content-style trade-off (from AdaIN paper)
    h_aug = alpha * h_adain + (1 - alpha) * h
    return h_aug
```

---

## 五、评估方案

### 5.1 数据集与基线

| 数据集 | 域数 | 类数 | 基线方法 |
|--------|------|------|---------|
| PACS | 4 | 7 | FedAvg, FedBN, FedProto, FPL, FDSE, FedDSA-base |
| Office-Caltech10 | 4 | 10 | 同上 |
| DomainNet(子集) | 6 | 10 | 同上 (Phase D) |

### 5.2 评估指标

1. **各域准确率**: 验证低gap域(Photo)不再被伤害
2. **AVG准确率**: 所有域简单平均 (3 seeds均值±std)
3. **域间方差**: std(各域准确率) — 越低越好
4. **vs FDSE差距**: 目标≥0

### 5.3 成功标准

| 等级 | 条件 |
|------|------|
| ✅ 达标 | PACS AVG ≥ 82%, Office AVG ≥ 91% |
| ⭐ 优秀 | 三数据集全面超过FDSE |
| 🎯 理想 | 超过FDSE + 域间方差降低 |

---

## 六、风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| z_sty统计量在warmup期不稳定 | 中 | warmup期间固定alpha=0.5 |
| EMA z-score在前几轮不准 | 中 | 前20轮用固定alpha fallback |
| 自适应不如最佳固定α | 低 | 072a/b/c会暴露此情况，则切换策略 |
| 域感知原型内存开销大 | 低 | PACS仅4域7类=28原型，可接受 |
| 服务器无GPU可用 | 高 | 优先lab-lry; 如不行等SC恢复 |

---

## 七、时间线

| 步骤 | 内容 | 依赖 | 状态 |
|------|------|------|------|
| Step 1 | 写方案文档 | - | ✅ v2.0 |
| Step 2 | GPT-5.4审稿 Round 1 | Step 1 | ✅ 5/10, 已修订 |
| Step 3 | GPT-5.4审稿 Round 2 | Step 2 | 🔄 进行中 |
| Step 4 | 实现 feddsa_adaptive.py | Step 3通过 | ⬜ |
| Step 5 | 跑 EXP-072a/b/c (固定α基线) | Step 4 | ⬜ |
| Step 6 | 跑 EXP-072d (M3-only) | Step 4 | ⬜ |
| Step 7 | 跑 EXP-072 (M1自适应) | Step 4 | ⬜ |
| Step 8 | 分析Phase A/B, 决定叠加 | Step 5-7 | ⬜ |
| Step 9 | 跑 EXP-073 (M1+M3) | Step 8 | ⬜ |
| Step 10 | 三数据集完整对比 | Step 9 | ⬜ |

---

## 附录A: Round 1 审稿原文

<details>
<summary>GPT-5.4 审稿 (5/10)</summary>

**Score: 5/10.** Good debugging instinct, but this is **not yet paper-grade**.

1. **Technical soundness**: M1 reasonable but weak theory (gap on backbone h mixes semantic); M2 weakest (DSU-style perturbation, not real cross-client transfer); M3 strongest.

2. **Novelty**: M1/M2 incremental engineering. M3 borrowed but meaningful. Reads more like repairing FedDSA than adding contribution.

3. **Experiment design**: Not sufficient. seed=2 only not credible. Missing: M3-only, fixed α baselines, DSU baseline, "style bank off for low-gap" sanity.

4. **M1+M2 feasibility**: Double-count augmentation strength. M2 already scales via variance; M1 scales again via gap.

5. **Issues**: Mixed gap metric; M2 loses "Share"; deterministic aug_strength too rigid; 4-client normalization weak.

6. **Missing baselines**: M3-only, DSU, FedFA-style, fixed-alpha, disable-aug-for-low-gap.

7. **Verdict**: Not ready. Ready only for M3-first + simpler M1 sanity. Do not invest in M2 until "Share" claim preserved.

</details>
