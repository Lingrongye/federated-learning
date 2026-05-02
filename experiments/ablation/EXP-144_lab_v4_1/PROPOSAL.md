# EXP-144: LAB v4.2 (Loss-Aware Boost) Aggregation 方案

> **Status**: PROPOSAL (codex-passed, ready to start P0 offline replay)
> **Date**: 2026-05-01
> **Owner**: changdao
> **Replaces**: F2DC 框架的 Domain-Aware Aggregation (DaA, Eq 10/11)
> **Codex review history**: v1 → v2 → v3 → v4 → v4.1 → **v4.2 (this)**, 集成所有 codex guardrail

---

## TL;DR

LAB v4.2 = "client 自己测 global model 的 val cross-entropy loss → 上传 (loss_sum, val_n) → server sample-weighted 聚合 + ReLU underfit boost → bounded simplex projection (bisection) 严格保证强域 ratio ≥ 0.80 弱域 ratio ≤ 2.00 → 域内等权分给 cli。预注册 λ=0.15, val_loss 测于聚合后给下一轮 LAB 用。"

---

## 一、动机

### 1.1 现状问题

F2DC 框架的 DaA 在跨 dataset 表现极不稳定：

| 数据集 | DaA 增量 | 结果 |
|---|:---:|:---:|
| **PACS** | **-2.14 ~ -2.36pp** | ❌ 输 vanilla |
| **Office** | +2.89 ~ +2.99pp | ✅ 赢 vanilla |
| **Digits** | +0.36pp | ≈ 持平 |

PACS -2.14pp 净亏使 F2DC + DaA 论文版无法稳定超 vanilla baseline。

### 1.2 改进目标

1. PACS AVG_B ≥ vanilla 72.02（**必须翻盘**）
2. Office AVG_B ≥ max(vanilla, DaA - 0.5) = 64.19（**目标 ≥ 64.5**）
3. Digits AVG_B ≥ max(vanilla, DaA - 0.2) = 93.55
4. 跨 dataset 稳定 + 可诊断 + paper-grade 严谨

---

## 二、观察到的现象（数据驱动诊断）

基于 8 个 +DaA run × 5 round 快照 × 全部 cli = 500 对 dispatch ratio + 100-round per-domain accuracy trajectory。

### 现象 1: DaA 名为 "Domain-aware"，实为 "sample-size 倒数加权"

ratio 标准差**全为 0**。同一个 sample_share 跨 round/seed/算法永远得到同一个 ratio。DaA 公式输入只有 `n_k`，没有 domain 标签也没有训练信号。

### 现象 2: DaA 在 PACS 误砍"最强 + 主力域"

| 域 | 占总样本 | DaA ratio | vanilla acc |
|---|:---:|:---:|:---:|
| photo | 12.8% | ×1.51 | 67.5 |
| art | 23.7% | ×1.24 | 58.2 |
| cartoon | 18.0% | ×1.10 | 78.5 |
| **sketch** | **45.4%** | **×0.69** ⚠️ | **83.8（最高）** |

sketch 是 PACS 最强 + 占 45% 数据，被 DaA 误砍 → -16.6pp 暴跌。

### 现象 3: DaA 加权方向 → acc 涨跌方向**强对齐**（13/20）

DaA 不是无效，传导是机械有效的。问题是**选错了升砍对象**：sample 多 ≠ 应该被压制。

### 现象 4: 每个 dataset 都有"结构性 underdog"，必须升权

| Dataset | Underdog | 100 round 中低于均值频率 | vanilla 行为 |
|---|---|:---:|---|
| PACS | art | 99% | 慢爬升 R10→R100 +29pp |
| Office | dslr | 99% | **vanilla s15 完全卡死 R10=R100=40.0** ⚠️ |
| Office | webcam | 100% | 慢爬升 |

### 现象 5: DaA 胜负取决于"被砍域已经学得多好"

| Dataset | 被砍最大域 | vanilla acc | 状态 | 净结果 |
|---|---|:---:|---|:---:|
| PACS | sketch | 83.8% | 强 + 主力 | ❌ 输 |
| Office | caltech | 66.1% | 中等 | ✅ 赢 |
| Digits | mnist | 97.2% | saturated | ≈ 平 |

### 现象 6: partition 闲置数据可做 val（**精确数字由脚本生成，此处仅 ≈**）

代码 `partition_*_domain_skew_loaders` 是"每 cli 从 remaining pool 不放回抽 percent × ini_len"。注意 partition 含 **class coverage repair**（line 209-228），所以最终 used set 可能跟简单 `selected_idx` 不一致。**精确 unused pool 必须由 P0 脚本从 `used = union(all selected_idx)` 反推**。

| Dataset | train pool | 已用 ≈ | 闲置 ≈ |
|---|:---:|:---:|:---:|
| PACS | ~9991 | ~7784 | **≈ 2200** (22%) |
| Office | ~2533 | ~1271 | **≈ 1260** (50%) |
| **Office dslr** | **~157** | **~93** | **≈ 64**（紧）|
| Digits | ~188k | ~9k | ≈ 179k (95%) |

---

## 三、改进方案：LAB v4.2

### 3.1 核心设计原则

| 现象 | LAB 对策 |
|---|---|
| 1. DaA 输入只有 sample 数 | **改用 per-domain val cross-entropy loss 当信号** |
| 2. 误砍主力强域 | **ReLU 只升 + bounded simplex projection (bisection) 硬保 ratio ∈ [0.80, 2.00]** |
| 3. Sample 数 ≠ 该不该升 | **直接看哪个域 val_loss 高** |
| 4. Underdog 必须升权 | **保留升权机制（λ=0.15 预注册）** |
| 5. Ratio 极差大易过拟合 | **硬上界 2.00，温和升权** |
| 6. partition 闲置数据 | **client-held val + 上传 (loss_sum, val_n) scalar** |

### 3.2 数据流改造

```
当前:
  train_dataset (per domain)
    ├── selected_idx → client 训练
    └── not_select_index → 丢弃

LAB v4.2:
  train_dataset (per domain)
    │
    ├── selected_idx (各 cli) → client 训练 (原样不变)
    │
    ├── domain_val_pool (50 张/域)
    │     │ 来源: unused = all_train_indices - union(all selected_idx for this domain)
    │     │      ↑ 用最终 used 反推, 防止 class coverage repair 泄漏
    │     │ stratified per-class, 最多 5 张/类
    │     │ deterministic eval transform (无 random crop/flip)
    │     │ shard 给该域所有 cli
    │     │ 永不参与训练
    │     │ 仅来自 train_index_list, 绝不碰 test_index_list
    │     └── seed 固定 42 (跟 train seed 解耦)
    │
    └── 剩余 → 丢弃
  test_dataset → server 评估 global model（不动）
```

**Codex guardrail #1 落地点**: val pool 不直接读 `not_used_index_dict`（partition 里的 class repair 可能让该 dict 不准），改成：

```python
used_for_train = set()
for k in clients_in_domain[d]:
    used_for_train.update(client_selected_idx[k])
unused_for_val = set(all_train_indices[d]) - used_for_train
val_pool[d] = stratified_sample(unused_for_val, per_class=5, max_total=50)
```

### 3.3 Val 集大小（domain-level shard）

实际 val 大小受 3 个 cap 约束:
- `val_size_per_dom`：默认 50（用户传参）
- `C × per_class`：stratified per-class 硬约束（不可破坏，否则丢失均衡性）
- `len(unused_pool)`：物理上限

最终单域 val 大小 = `min(val_size_per_dom, num_classes × per_class, len(unused))`

| 域 | unused pool ≈ | C × 5 cap | **val 总量** | 各 cli 持有 | 占用率 ≈ |
|---|:---:|:---:|:---:|:---:|:---:|
| **PACS photo** (C=7, 2 cli) | 668 | **35** | **35** | 18/17 | 5% |
| **PACS art** (C=7, 3 cli) | 205 | **35** | **35** | 12/12/11 | 17% |
| **PACS cartoon** (C=7, 2 cli) | 938 | **35** | **35** | 18/17 | 4% |
| **PACS sketch** (C=7, 3 cli) | 393 | **35** | **35** | 12/12/11 | 9% |
| Office caltech (C=10, 3 cli) | 451 | 50 | 50 | 17/17/16 | 11% |
| Office amazon (C=10, 2 cli) | 574 | 50 | 50 | 25/25 | 9% |
| Office webcam (C=10, 2 cli) | 177 | 50 | 50 | 25/25 | 28% |
| **Office dslr** (C=10, 3 cli) | **64** | 50 | **min(50, unused) ≈ 50** | shard | **~78%** |
| Digits 各域 (C=10) | 6k+ | 50 | 50 | shard | < 1% |

**修订（codex 三轮 Important #2）**: PACS val 实际 **35/域**（受 7 类 × 5 限制），不是 50。这是 stratified 的硬约束，不破坏 per-class=5 的均衡。如果需要 50/域可以改 `--lab_val_per_class 8`（7×8=56 ≥ 50）但会降低 stratified 严格性。

**特殊保护**: `val_size = min(val_size_per_dom, C × per_class, len(unused))`，Office dslr 自动收缩到 unused 余量内。

### 3.4 通信流程（明确时序，**Codex guardrail #2: val 测于聚合后**）

```
═══════════════════════════════════════════════════════════════
                    Round r 时序
═══════════════════════════════════════════════════════════════

[1] Server 算 LAB 权重 (基于上一轮 val_loss)
       ┌──────────────────────────────────────────────┐
       │  if r == 1:                                   │
       │      w[d] = sample_share[d]    # 退化 FedAvg  │
       │  else:                                        │
       │      w[d] = LAB( val_loss_ema[d, r-1] )       │
       └──────────────────────────────────────────────┘

[2] Server 下发 global_model_{r-1} 给所有 client

[3] Client 本地训练 (跟现在一致)
       ┌──────────────────────────────────────────────┐
       │  client model ← global_model_{r-1}            │
       │  for E epochs:                                │
       │      train on local_train_split               │
       │      (val_split 完全不参与训练)              │
       └──────────────────────────────────────────────┘

[4] Client 上传: model_params (NOTHING related to val 这步)

[5] Server 用 LAB 权重 w[d] 聚合 → global_model_r

[6] (Codex guardrail #2 关键步骤) Server 让所有 client 用新 global_model_r 测 val_loss
       ┌──────────────────────────────────────────────┐
       │  Server 下发 global_model_r 给 client (轻量)  │
       │  Client k:                                    │
       │      with deterministic eval transform:       │
       │          loss_sum_k = sum(CE_loss for batch   │
       │                           in client_val_k)    │
       │          val_n_k    = len(client_val_k)       │
       │      上传 (loss_sum_k, val_n_k)               │
       │  Server 算 (sample-weighted):                 │
       │      val_loss[d, r] = sum(loss_sum_k for      │
       │                            k in dom d)        │
       │                       / sum(val_n_k for k     │
       │                              in dom d)        │
       │      val_loss_ema[d,r] = 0.3*val_loss[d,r]    │
       │                          + 0.7*val_loss_ema   │
       │                            [d,r-1]            │
       └──────────────────────────────────────────────┘

[7] val_loss_ema[d, r] 存起来，给 round r+1 用 (回到 [1])

═══════════════════════════════════════════════════════════════
```

**关键性质**：
- val_loss[d, r] 是**当前 round 聚合后**的 global_model 的 loss
- LAB(r+1) 用 val_loss[d, r]（**不是当前 round 自己的**）
- 因果链清晰：LAB(r) ← val_loss(r-1) ← global_model(r-1)
- raw val 数据**永远不出 client**
- 通信开销：每 cli 每 round 多 **2 个 float** (loss_sum, val_n)

### 3.5 LAB 算法（5 步 + bisection projection）

```python
# ===== Step 1: 聚合 per-domain val_loss (sample-weighted) =====
val_loss[d] = sum(loss_sum[k] for k in dom d) / sum(val_n[k] for k in dom d)

# ===== Step 2: EMA 平滑 =====
loss_ema[d] = 0.3 * val_loss[d] + 0.7 * loss_ema_prev[d]

# ===== Step 3: ReLU underfit-only 目标分布 =====
loss_mean = mean(loss_ema)
gap[d] = max(0, loss_ema[d] - loss_mean)
q[d]   = gap[d] / sum(gap)   if sum(gap) > 0 else 0   # 全域学得差不多 → 退化 FedAvg

# ===== Step 4: λ-mix + bounded simplex projection (BISECTION) =====
λ = 0.15
w_raw[d] = (1-λ) * sample_share[d] + λ * q[d]

ratio_min = 0.80
ratio_max = 2.00
w_min[d] = ratio_min * sample_share[d]   # 强域硬下界
w_max[d] = ratio_max * sample_share[d]   # 弱域硬上界

# Codex guardrail #3: 用 bisection 找 tau s.t. sum(clip(w_raw + tau, w_min, w_max)) = 1
# 标准简洁实现, 数值稳定:
def bounded_simplex_projection(w_raw, w_min, w_max, target=1.0, tol=1e-9, max_iter=64):
    lo = -1.0  # tau 下界 (使 sum 最小)
    hi = +2.0  # tau 上界 (使 sum 最大)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        s = sum(clip(w_raw[d] + mid, w_min[d], w_max[d]) for d in domains)
        if abs(s - target) < tol:
            break
        if s > target:
            hi = mid
        else:
            lo = mid
    return [clip(w_raw[d] + mid, w_min[d], w_max[d]) for d in domains]

w_proj = bounded_simplex_projection(w_raw, w_min, w_max)
# 结果严格保证: sum(w_proj) = 1, 且每个 w_proj[d] ∈ [w_min[d], w_max[d]]

# ===== Step 5: 域内等权分给 cli =====
p_k = w_proj[ dom(k) ] / num_cli_in_dom[ dom(k) ]

# ===== Step 6: 真实 boost 用于 ROI 诊断 =====
positive_delta[d] = max(0, w_proj[d] - sample_share[d])  # 真实加成量
```

### 3.6 Bounded simplex projection 单测覆盖（Codex guardrail #3）

```python
def test_projection_sum_to_one():
    """sum(w_proj) ≈ 1.0"""

def test_projection_in_bounds():
    """所有 w_proj[d] ∈ [w_min[d], w_max[d]]"""

def test_projection_degenerate_q_zero():
    """q 全 0 → w_raw = sample_share → projection ≈ FedAvg"""

def test_projection_small_dom_hits_max():
    """sample_share 最小域，q 高 → 触发 ratio_max=2.00 上界"""

def test_projection_large_dom_hits_min():
    """sample_share 最大域，q 低 → 触发 ratio_min=0.80 下界"""

def test_projection_pacs_realistic():
    """用 PACS PG-DFC vanilla 真实 acc 模拟，验算 final_ratio
       期待: photo×1.14, art×1.32, cartoon×0.85, sketch×0.85
       tolerance ±0.02"""
```

### 3.7 公式核心保证

| 性质 | 保证机制 |
|---|---|
| 强域永不被砍超过 20% | `w_proj[d] ≥ 0.80 × sample_share[d]` (bisection 输出严格满足) |
| 弱域永不被升超过 100% | `w_proj[d] ≤ 2.00 × sample_share[d]` (同上) |
| 总权重和为 1 | bisection 收敛到 tol=1e-9 |
| 已学好的域不参与重权 | ReLU(loss - mean) = 0 |
| 信号干净 | client-held val + EMA + CE loss + global model |
| FL 数据隔离 | raw val 不出 client，仅上传 2 个 scalar |
| Train/val 隔离 | val pool 用 `all_train_idx - union(used)` 反推 |
| Train/test 隔离 | val pool 仅来自 train_index_list |

### 3.8 用真实 vanilla 数据预演（PACS, λ=0.15）

| 域 | val_loss proxy | gap (ReLU) | q | w_raw | **w_proj** | **final ratio** | DaA ratio |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| photo | 0.325 | 0.045 | 0.246 | 0.146 | 0.146 | **×1.14** | ×1.51 |
| art | 0.418 | 0.138 | 0.754 | 0.314 | 0.314 | **×1.32** | ×1.24 |
| cartoon | 0.215 | 0 | 0 | 0.153 | 0.153 | **×0.85** | ×1.10 |
| **sketch** | 0.162 | 0 | 0 | 0.386 | 0.386 | **×0.85** ✓ | **×0.69** ⚠️ |

**sketch ratio 从 ×0.69 (DaA 砍 31%) 改善到 ×0.85 (LAB 被动稀释 15%)**，预期 sketch acc 损失从 -16.6pp 缩到 -3~5pp。

### 3.9 预期跨 dataset 效果

| 数据集 | DaA 实测 | **LAB 预期** | 反转 |
|---|:---:|:---:|:---:|
| PACS AVG_B | -2.14pp ❌ | **+1~3pp** ✅ | **+3~5pp** |
| Office AVG_B | +2.89pp | **目标 +1.5~2.5pp（≥ 64.5）** | -0.5pp（接受） |
| Digits AVG_B | +0.36pp | **≈ +0.2~0.4pp** | ≈持平 |

---

## 四、Assumptions & Constraints

### Codex guardrail #4: Full participation 假设

> **All main experiments use full participation (online_ratio = 1.0), consistent with the current F2DC setup.**
>
> LAB 的 domain-level 权重 (sample_share[d], val_loss[d]) 和域内等权分配 (p_k = w_proj[d] / num_cli_in_dom[d]) 都依赖 full participation 假设。如果未来扩展到 partial participation，所有依赖 sample_share / dom(k) / val_loss 的计算必须按 online clients 重算（仅对当 round 在线的 cli 计 share，仅在线 cli 上传 val_loss）。本方案不在 partial participation 范围内。

### 其他默认假设

- Fixed allocation (PACS photo:2/art:3/cartoon:2/sketch:3 等)
- Single-domain per client (每个 cli 持单一 domain 数据)
- Server 拿 global model 做 test eval（标准 FL 做法）
- val seed = 42 跟 train seed 解耦（跨实验 val 集固定一致）

---

## 五、诊断指标系统

### 5.1 浪费的三层定义

| Level | 触发条件 | 含义 |
|---|---|---|
| **L1 单 round** | EMA(Δacc) < -1.0pp 且连续 ≥ 3 round | 反向 |
| **L2 短期** | window_roi < 0.5 持续 ≥ 2 个 20-round window | 信号有但模型不响应 |
| **L3 全程** | cumulative_roi < 0.5pp / pp_boost | 终局浪费 |

### 5.2 每 round 必须 dump 的字段（19 + Codex 加 1 = 20）

#### 核心 8 个

| 字段 | 含义 |
|---|---|
| `acc_dom[d, r]` | per-domain test acc（仅 sanity log，不进 LAB） |
| `val_loss_dom[d, r]` | client-held val sample-weighted CE loss |
| `gap_dom[d, r]` | ReLU(loss_ema - mean) |
| `boost_dom[d, r]` = `positive_delta[d]` | max(0, w_proj - sample_share) |
| `w_dom[d, r]` = `w_proj[d]` | 投影后的最终域权重 |
| `final_ratio[d, r]` = w_proj / sample_share | 严格 ∈ [0.80, 2.00] |
| `delta_acc[d, r]` | 单 round Δacc |
| `cum_boost[d, r]` | 累计加成 |

#### EMA + 边界 5 个

| 字段 | 用途 |
|---|---|
| `signal_source` | 固定 "client_val_ce_loss" |
| `signal_round` | 标 R-1（不是当 R） |
| `loss_ema[d]`, `gap_ema[d]` | EMA 平滑 |
| `ratio_clipped[d]` | bounded simplex 命中边界（`@max` / `@min` / `None`） |
| `effective_contrib[d] = Σ p_k × grad_l2[k]` | 真实模型贡献 |

#### Val 元数据 6 个 + Codex 加 1

| 字段 | 用途 |
|---|---|
| `val_n_per_cli[k]` | client 持的 val 大小 |
| `val_n_per_dom[d]` | domain 总 val 大小（≤ 50） |
| `val_loss_sum_per_cli[k]` | raw 上传值 |
| `val_seed` | 固定 42 |
| `val_transform` | "deterministic_eval" |
| `val_pool_origin` | "train_index_unused_via_used_complement" |
| **`val_class_counts[d, c]`** ⭐ | **Codex guardrail #5: per-class val 样本数（解释 dslr 等小域 loss 抖动）** |

### 5.3 诊断输出三种形式

#### 形式 A：每 round stdout

```
[LAB R 30] val_loss_ema={photo:0.31, art:0.41, cartoon:0.21, sketch:0.16}
           ratio={photo×1.14, art×1.32, cartoon×0.85, sketch×0.85}
           clip={None}
           per_dom_test_acc={photo:70.1, art:58.4, cartoon:79.2, sketch:81.5} (sanity only)
```

#### 形式 B：每 10 round 浪费报警

```
⚠️ [LAB WASTE WARNING R 60] dom=art cum_boost=18.5%
   window_roi=0.17 (< 0.5 = wasted)
```

#### 形式 C：训练后自动 report (`analyze_lab.py`)

```
================================================================
  LAB 诊断报告
================================================================
【全程加权累计】(用 positive_delta 算 ROI)
  domain   total_boost  total_acc_gain  ROI    判定
  photo    +12.3%       +5.5pp          0.45    ⚠️ 边际
  art      +35.1%       +5.5pp          0.16    ❌ 浪费
  cartoon   +0.0%       -1.2pp          ---     未升权
  sketch    +0.0%       -3.5pp          ---     未升权

【浪费分析】
  累计发出 boost: 47.4%
  浪费率: 35.1/47.4 = 74.1%   ❌ 超阈值 30%

【边界触发】
  ratio_clipped@max: dslr 触发 12 次 (R8-R20)
  ratio_clipped@min: 无

【Val class coverage】
  Office dslr per-class val counts: [3,5,5,5,5,4,5,5,5,3]  ← 不均衡, 解释 noise

【验收结果】
  Gate 1 (PACS sketch acc ≥ 79.0):    实际 80.5  ✅ PASS
  Gate 2 (浪费率 < 30%):              实际 74.1% ⚠️ WARN
  Gate 3 (PACS AVG_B ≥ 72.02):       实际 73.2  ✅ PASS
  Gate 4 (final_ratio ∈ [0.80,2.00]): 100% ✅ PASS (algo 正确)
  Gate 5 (clip_rate < 30%):           实际 12% ✅ PASS
================================================================
```

---

## 六、Dataset-specific 验收门

### PACS

| Gate | 阈值 | 类型 |
|---|:---:|:---:|
| AVG_B ≥ vanilla | ≥ 72.02 | 硬 |
| sketch acc | ≥ 79.0 | 硬 |
| sketch ratio R10-R100 mean | ≥ 0.80 | 硬 |

### Office（**目标提高，不放宽 gate**）

| Gate | 阈值 | **目标** | 类型 |
|---|:---:|:---:|:---:|
| AVG_B ≥ max(vanilla, DaA - 0.5) | ≥ 64.19 | **≥ 64.5** | 硬 |
| dslr final ≥ DaA baseline | ≥ 53.3 | ≥ 53.3 | 硬 |

### Digits

| Gate | 阈值 | 类型 |
|---|:---:|:---:|
| AVG_B ≥ max(vanilla, DaA - 0.2) | ≥ 93.55 | 硬 |

### 全局

| Gate | 条件 | 类型 |
|---|---|:---:|
| Stability | 3-seed paired delta ≥ 2/3 非负 | 硬 |
| Algorithm correctness | 所有 final_ratio ∈ [0.80, 2.00] | 硬 |
| Waste | total_waste_ratio < 30% | WARN |
| Clip rate | clip_rate > 30% | **WARN（cap 在工作可能正常）** |

---

## 七、实验计划

### P0：Offline replay (0.5h, 无 GPU)

**纯 Python 脚本，不动训练代码**。用现有 vanilla cold path npz（PACS s15/s333、Office s2/s15/s333）模拟 LAB 权重轨迹。

**实现**:
- 读 `experiments/cold_path_analysis/diag_pgdfc_pacs_s15/round_*.npz` 等
- 用 `1 - per_domain_test_acc` 当 val_loss 近似 proxy（offline 没法重新跑训练拿真 val_loss）
- 跑 LAB 公式 + bisection projection
- 输出 ratio 时间序列、ratio_clipped 频率、boost 分布

**判定通过条件**:
- 所有 final_ratio ∈ [0.80, 2.00]（algorithm correctness check，必过）
- bisection projection 数值稳定（无 NaN, 收敛 < 64 iter）
- ratio_clipped 频率 < 30%
- 没有 domain 100% rounds 都饱和

**P0 局限性**: 用 1-acc 当 val_loss proxy，**仅验证权重轨迹合理性，不能证明效果**。真实 val_loss 在 P1 编码后才能拿到。

### P1：编码 + 单 seed smoke (10h GPU)

**预注册 λ=0.15**，不做 sweep 选 λ。

跑：
- PACS s=15 单 seed × λ=0.15
- Office s=2 单 seed × λ=0.15
- Digits s=15 单 seed × λ=0.15

**判定（不依赖 test acc 选参，但 test 作 sanity log）**:
- val_loss 趋势：boost 是否流向 underfit 域？
- ratio_clipped 频率 < 30%？
- waste warning 频率 < 30%？
- **test sanity log**：PACS sketch acc 不得 < 60，Office AVG_B 不得 < 60（异常触发 debug，**不触发 λ 调整**）

### P3：3-seed × 3-dataset 主对照 (24h GPU)

只在 P0/P1 通过的情况下跑。**这才是产论文数据的步骤**。

预期主表新行：

| Method | PACS AVG_B | Office AVG_B | Digits AVG_B |
|---|:---:|:---:|:---:|
| PG-DFC vanilla | 72.02 | 61.79 | 93.39 |
| PG-DFC + DaA | 69.88 | 64.69 | 93.75 |
| **PG-DFC + LAB** | **≥ 73.5** | **≥ 64.5** | **≥ 93.7** |

### P2：λ ablation (appendix only, 6h GPU)

**仅在 P3 主对照通过后跑**。λ ∈ {0.05, 0.10, 0.15, 0.20}，PACS s=15 一个 seed。论文 appendix 用，证明 LAB 对 λ 不敏感。

### P4：失败回退（条件触发）

如果 P3 任何 gate 失败 → 升级 saturation detector 或回退双因子方案。

---

## 八、风险与回退

| 风险 | 影响 | 应对 |
|---|---|---|
| Office dslr val 50 张噪声大 | LAB 信号不稳 | EMA + ablation 验证 val_size ∈ {30, 50} 不敏感 |
| dslr per-class 分布不均 | val_loss 抖动 | dump `val_class_counts[d,c]` 解释 |
| art 持续被 boost 但 ROI 低 | PACS 收益打折 | 加 saturation detector |
| LAB 在 Digits 信号方向跟 DaA 不一致 | Digits 退步 | P1 smoke 早发现 |
| LAB 跑出来 PACS 仍亏 vanilla | 方案失败 | P4 回退双因子方案 |
| Class coverage repair 让 not_used_index 不准 | val/train 泄漏 | 用 `used = union(selected_idx)` 反推 unused |

---

## 九、Novelty 表述（Paper 用）

> **"Domain-level validation-loss-driven aggregation for fixed source-domain federated domain generalization. Each client maintains a class-stratified held-out validation split (sharded from the unused portion of the training partition, recovered via index complement against final used training indices, never from the benchmark test split, evaluated with deterministic transforms) and uploads only two scalars (loss_sum, val_n) per round. The server aggregates domain-level cross-entropy loss via sample-weighted averaging, applies ReLU-based underfit-domain boost, and projects the final domain weights onto a bounded simplex {ratio ∈ [0.80, 2.00], sum = 1} via bisection, providing strong-domain protection by construction. All experiments use full client participation."**

跟现有工作的差异：

| 现有工作 | 信号 | LAB 的差异 |
|---|---|---|
| Context-aggregator | val loss + class imbalance | 我们: domain-level + bounded simplex 强域保护 |
| FedPLA | val loss | 我们: ReLU underfit-only + 显式 ratio cap (bisection) |
| FedLBW | val-loss weighting | 我们: domain-level sample-weighted aggregation |
| FedDF | server-side public set | 我们: client-held in-domain val + 不上传 raw |

---

## 十、最终参数表

| 参数 | 默认值 | 注释 |
|---|:---:|---|
| `λ` (混合系数) | **0.15**（预注册主方法） | sweep 仅 appendix |
| `ratio_min` | **0.80** | bounded simplex 下界 |
| `ratio_max` | **2.00** | bounded simplex 上界 |
| `EMA α` | **0.3** | val_loss 平滑 |
| `val_size_per_dom` | **min(50, C × per_class, len(unused))** | PACS C=7 → 35, Office/Digits C=10 → 50 |
| `val_per_class` | **5** | stratified per-class |
| `val_seed` | **42** | 跟 train seed 解耦 |
| `val_transform` | **deterministic_eval** | 无 random crop/flip |
| `val_pool_origin` | **all_train - union(used)** | Codex guardrail #1 |
| `signal_source` | **client_val_ce_loss** | 论文表述 |
| `signal_round` | **R-1** | LAB(r) 用上一轮 val_loss |
| `comm_overhead` | **+2 floats/cli/round** | (loss_sum, val_n) |
| `bisection_tol` | **1e-9** | projection 收敛精度 |
| `bisection_max_iter` | **64** | projection 收敛上限 |
| `window_size` | **20** | ROI window |
| `waste_threshold` | **0.5** | window_roi < 0.5 = 浪费 |
| `total_waste_threshold` | **0.30** | WARN（不硬 fail）|
| `clip_rate_threshold` | **0.30** | WARN（cap 在工作可能正常）|
| `participation` | **full (online_ratio=1.0)** | Codex guardrail #4 |

---

## 十一、实施顺序

1. **(0.5h)** P0 offline replay
   - 写 `experiments/ablation/EXP-144_lab_v4_1/p0_offline_replay.py`
   - 读现有 vanilla cold path npz
   - 用 1-acc 当 val_loss proxy 模拟 LAB
   - 检查 ratio 严格 ∈ [0.80, 2.00]、bisection 数值稳定、saturation 频率合理
   - 输出 `p0_replay_report.md` + 4 张 figure
2. **(决策点)** P0 通过 = bisection 算法正确 + ratio 严格满足约束 + 无 NaN
3. **(2.5h)** 编码（**P0 通过后才动训练代码**）
   - partition 改造：`val_pool = train_idx - union(selected_idx)` + stratified per-class + eval transform + train/test 隔离
   - bounded_simplex_projection (bisection) + 6 个单测
   - client val_loss 上传通道（loss_sum, val_n）
   - server LAB 公式 + 20 诊断字段（含 val_class_counts）
   - round 时序：聚合后才测 val_loss
4. **(0.5h)** Codex review + sanity test 全过
5. **(0.5h)** Commit + push
6. **(10h)** P1 单 seed smoke (PACS + Office + Digits, λ=0.15)
7. **(决策点)** val 信号合理 + 没饱和 + clip_rate 低 + test sanity 通过
8. **(24h)** P3 3-seed × 3-dataset 主对照
9. **(决策点)** 三 dataset 全过 gate
10. **(6h)** P2 λ ablation (appendix)
11. **(1h)** 回填 NOTE.md + 主表 + paper draft

**总时间**: 编码 ~3.5h，GPU ~40h，撰文 ~1h。

---

## 十二、Codex Review 历史

| 版本 | Codex 反馈 | 修订 |
|---|---|---|
| v1 | 没有诊断 | 加 8 字段 |
| v2 | server 持 raw val 破坏 FL | 改 client-held val + scalar |
| v3 | ratio clip 不严格 + 用 1-acc 太粗 | bounded simplex + CE loss |
| v4 | dslr val 数量不可能 + 时序模糊 | domain-level shard + 明确时序 |
| v4.1 | 5 个 guardrail 待落地 | (本版整合) |
| **v4.2 (this)** | **All passed** | **集成 5 个 guardrail，pass to P0** |

### v4.2 的 5 个 Codex Guardrail

1. ✅ **Val pool 来源**: `unused = all_train - union(used)`，避免 class repair 泄漏
2. ✅ **Bisection projection**: 标准 bisection 算法 + 6 个单测
3. ✅ **新增字段**: `val_class_counts[d, c]`
4. ✅ **Full participation 写明**: assumption 明文
5. ✅ **数字用 ≈**: 现象 6 标 ≈，精确数字由 P0 脚本生成

---

## 一句话定位

> **LAB v4.2 = "client 自己测 global model 的 val cross-entropy loss → 上传 (loss_sum, val_n) → server sample-weighted 聚合 + ReLU underfit boost → bounded simplex projection (bisection) 严格保证强域 ratio ≥ 0.80 弱域 ratio ≤ 2.00 → 域内等权分给 cli。预注册 λ=0.15, val_loss 测于聚合后给下一轮 LAB 用。Codex 5 个 guardrail 全部集成。"**
