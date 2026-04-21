# Self-Review: FedDSA-VIB (A) + FedDSA-VSC (B) 方案

**扮演角色**: 最严格的 NeurIPS 二审 reviewer。不做"其实没问题"式的过场。

---

## 1. Novelty 被打点 (Codex round-2 大概率挑的问题)

### Blocker 1: "VIB + class-conditional prior 不是你的发明"
**攻击**:
- Moyer 2018 已经做了 VIB for invariance
- Class-conditional VAE (CVAE, Sohn 2015) 也是 p(z|x,y) 结构
- Information Dropout (Achille & Soatto 2018) 是 VIB + domain 判别

**我们的反击**:
- **FL + VIB + class-conditional prior 的组合**: Moyer 是 central,我们分布式
- **Prototype prior 来源于多 client 聚合**: Moyer 用单机 class mean,我们用 FedAvg prototype
- **组合首次**: FedDP / FedSeProto 用 MI bottleneck 但没用 class-conditional prior

**Self-verdict**: **Novelty 存在但不强**。1 句话 novelty: "First FL disentanglement with prototype-centered VIB, aligning Moyer-style minimal sufficient invariance to FL prototype aggregation"。够不够 top venue? **中等**,需要评估 novelty (Moyer sweep) 补强。

### Blocker 2: "SupCon 是 Khosla 2020 现成工具"
**攻击**: SupCon 本身没 novelty, FL 场景只是应用。
**反击**: **不声称 SupCon 是新**,只是作为 InfoNCE 的升级变体。方案 B 主 novelty 还是 VIB。SupCon 是 ablation,证明 prototype pull 升级的边际收益。

### Blocker 3: "A 和 B 差别小,何必做 2 个"
**攻击**: 方案 A 和 B 只差 InfoNCE vs SupCon,如果 A 已赢 orth_only,B 只是小 increment。
**反击**: **两个方案的差异正是 SupCon 的价值检验**。如果 B-A 差异在 1pp 内 (seed 方差内) → InfoNCE 已够,SupCon 是 noise。如果 B-A > 2pp → SupCon 升级有用。**这就是一个 ablation,不是两个竞争方案**。

---

## 2. 理论层面被打点

### Blocker 4: "Prior σ² 怎么选?"
**攻击**: `L_VIB = KL(N(μ_sem, σ²_sem) || N(proto_y, σ_prior²))`。σ_prior² 是 magic number,是否 ablation 过?
**反击**: 
- 需要 **σ_prior ∈ {0.1, 0.5, 1.0, 2.0}** 扫描 (方案 A 加 ablation)
- 太小 → KL 强制 z_sem = proto_y,z_sem 失去区分能力
- 太大 → KL 几乎为 0,VIB 不起作用
- **Moyer 原文用 1.0,我们默认也 1.0 + 小扫描**

### Blocker 5: "Prototype 还没收敛就做 prior 会崩"
**攻击**: 初期 prototype 是噪声,此时 KL 把 z_sem 拉向噪声 → 训练乱。
**反击**: 加 **prior warmup**:
- R0-20: λ_IB = 0 (纯 CE + L_aug 热身 prototype)
- R20-50: λ_IB linear 0 → 1.0
- R50+: full λ_IB

**但这增加一个超参**,可能 Codex 挑。

### Blocker 6: "Stochastic encoder 和 deterministic FedAvg 聚合兼容吗?"
**攻击**:
- encoder 参数 μ_head + σ_head 都参与 FedAvg
- σ_head 可能跨 client 漂移大 → 聚合后 σ 奇怪
**反击**: FedAvg 对 σ_head 参数平均,在参数空间合理,实测会验证 `sigma_sem_max` 不爆炸。**但若发现 σ 不稳,需要加 sigma_head 本地化 (FedBN 风格)**。

---

## 3. 实验层面被打点

### Blocker 7: "orth_only PACS std=1.46,你预期 +1pp 在噪声内"
**数字**: orth_only 3-seed = 80.64 ± 1.46
**攻击**: 方案 A 预期 80.5-81.5, B 81.0-82.0 — 都在 1 std 内,**statistically insignificant**。
**严重程度**: **高**。如果 paper 表格 A/B/orth_only 差异不显著,venue 过不了。

**应对**:
1. 跑 **5 seeds** 而非 3 (降低 std 估计误差)
2. Office seed std 更稳 (EXP-110 std=0.83),可能在 Office 上显著
3. 主 story 不只看 acc,**probe 降才是核心 contribution** — 即使 acc 持平,probe 降是真 signal

### Blocker 8: "FDSE/FediOS 复现难度"
**FDSE (CVPR 2025)**: 源码在 `FDSE_CVPR25/`,本项目已有
**FediOS (ML 2025)**: arxiv 2311.18559 有 code,需要集成到 PFLlib

**应对**: Priority 1 只做 M4+M5 (12 runs), Priority 2 的 FDSE/FediOS 用论文数字先顶上,代码复现作为时间允许时补。

### Blocker 9: "Proxy probe 10 轮一次拖慢训练"
**估算**: 每次 proxy probe = feature extract (1 min) + 6 个 probe fit (5 min) = **6 min**
R200 训练约 3h = 180 min,proxy 每 10 轮 × 20 次 = 120 min 额外
→ **训练 180 min + probe 120 min = 5h**,慢了 **70%**

**应对**:
- **减采样**: probe 训用 200-500 samples 子集 (小 dataset 够用)
- **减频率**: 每 20 轮一次 (10 次共 60 min,慢 33%)
- **async**: probe 在 CPU worker 异步跑 (复杂但零拖慢)
- **策略**: 默认 **每 20 轮 + 子集 500** → 额外 ~45 min,慢 25%

---

## 4. 潜在 failure mode

### Failure 1: **VIB KL 一直降不下来**
**原因**: σ_prior 选小,KL 永远大,λ_IB 起不来
**诊断**: `KL_mean > 3` 在 R100 后仍 high
**修复**: σ_prior 调大,或 λ_IB warmup 延长

### Failure 2: **σ_sem collapse**
**原因**: VIB weight 太大,encoder 把 σ 压到 0 → 退化为 deterministic
**诊断**: `sigma_sem_mean < 0.01`
**修复**: 加 `log_var` clamp ([-5, 2])

### Failure 3: **prototype drift 太大**
**原因**: 跨 client 原型漂移太大 → prior 不稳
**诊断**: `prototype_drift_L2 > 0.1` 持续
**修复**: EMA smoothing β=0.99

### Failure 4: **proxy_probe_sty_class MLP-256 不下降**
**原因**: VIB 机制未在 z_sem 真压住信息,z_sty 仍能从主干读
**诊断**: R100 后 `proxy_probe_sty_class MLP-256 > 0.65`
**决定**: 如果持续 → **方案失败,等同 orth_only**,需 rethink

### Failure 5: **acc 掉**
**原因**: VIB 压 z_sem 过头 → 丢 class 信号
**诊断**: val acc R100 < 78 (orth_only baseline)
**修复**: 降 λ_IB 或扩 σ_prior

---

## 5. 已覆盖 Round-1 Codex 建议 ✓

| Codex 建议 | 我们处理 |
|-----------|--------|
| 放弃 L_swap | ✅ 不做,改 VIB |
| 不要降 L_orth | ✅ 保持原权重 (预期 0.5-1.0,非 0.1) |
| Prototype pull | ✅ 保留 InfoNCE (A),SupCon 升级 (B) |
| **不** 用 adversary | ✅ 两方案都不加 GRL |
| lean Moyer minimal sufficient | ✅ VIB = 核心 |
| lean Prototype factorization | ✅ VIB prior = semantic prototype |
| 别 cite von Kügelgen | ✅ 重写 theory 段,不 cite Thm 4.4 |

---

## 6. Self-Scoring (7 dims)

| 维度 | A | B | 改进点 |
|------|:-:|:-:|--------|
| Problem Fidelity | 8 | 8 | ok |
| Method Specificity | **7** | **7** | VIB 公式完整,但 σ_prior + warmup 需 ablate |
| Contribution Quality | **6** | **6.5** | Novelty 是 combination,不是 component,**风险** |
| Frontier Leverage | 7.5 | 7.5 | lean Moyer 2018 正确 |
| Feasibility | 8 | 7.5 | A 简单,B 多一个 loss |
| Validation Focus | 9 | 9 | 50+ 诊断 + Moyer sweep 很严 |
| Venue Readiness | **6** | **6.5** | Novelty 中等,需 paper 叙事强调 "first FL + VIB prior" |

**Weighted**:
- A: 8×0.15 + 7×0.25 + 6×0.25 + 7.5×0.15 + 8×0.10 + 9×0.05 + 6×0.05 = **7.0**
- B: 8×0.15 + 7×0.25 + 6.5×0.25 + 7.5×0.15 + 7.5×0.10 + 9×0.05 + 6.5×0.05 = **7.2**

**Self-verdict**: **REVISE** (不是 RETHINK,但也不是 READY)

**真 blocker**:
1. Novelty 是 "组合新" 不是 "组件新" — 需要 paper narrative 强化
2. orth_only std 1.46 可能淹没 +1pp 提升 — 需要 5 seeds 或 Office 重点看
3. σ_prior 和 λ_IB warmup 2 个 magic numbers

**真 confidence**:
1. ✅ 50+ 诊断 + Moyer sweep 已经是 evaluation-level contribution
2. ✅ L_aug h-space 保留 (EXP-059 证据)
3. ✅ 不重蹈 CDANN 陷阱 (不加 adversary)

---

## 7. 改进建议 (进 round-2 前)

### 必改
1. **加 σ_prior 扫描** {0.3, 1.0, 3.0}: ablation 证明 robust
2. **加 λ_IB warmup**: R20-50 linear
3. **承认 novelty 是 combination**: paper 叙事强调 "FL-first VIB with prototype-conditional prior"
4. **Proxy probe 子集 500 + 每 20 轮**: 训练速度可控

### 建议加
5. **5 seeds 而非 3**: 提高统计显著性
6. **加 σ_head 本地化 (FedBN 风格) 的 ablation**: 防漂移
7. **prior variance ablation 加入实验矩阵**

### 不改
- ❌ 不加 adversary (Codex 明禁)
- ❌ 不做 swap (Round-1 否)
- ❌ 不降 L_orth (Codex 要求保留)

---

## 8. 决定

**基于 self-review,得出**:
- A/B 方案**可行,但 novelty 中等**
- 必须加诊断 + 多 seed 才能拿到显著 signal
- **推荐走 B 方案为主** (VIB + SupCon),A 作为 ablation
- 预期 round-2 Codex 评分 7.0-7.5 (REVISE/READY 边界)

下一步: 写 Codex round-2 prompt,请它评审本 proposal,目标得分 ≥7.5。
