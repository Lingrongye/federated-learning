# EXP-067 | FedDSA-RegimeGated: Style Graph Dispatch + Regime-Gated Server SAM

## 基本信息
- **日期**:2026-04-12
- **算法**:`feddsa_regime_gated`
- **核心改进**:用 decoupled style bank 做**两件事**
  1. **Style graph neighbor dispatch**:每客户端收到 K 个最近邻的风格(不再随机)
  2. **Regime-gated server-side SAM**:在 low style-gap regime 下,对 consensus QP 之后的全局模型做一次 SAM look-ahead 扰动(FedGloSS 风格,零额外通信)
- **状态**:✅ 代码 + 17/17 单元测试 PASS(本地 pfllib 环境) — 等待启动

## 动机(为什么这样做)

### EXP-064 + EXP-066 的关键发现
- Consensus QP(照抄 FDSE Eq.8)在 PACS s=2 突破到 **83.04 > baseline 82.24 > FDSE 80.81**
- Consensus QP 在 Office **3-seed 均值 89.83 (+0.70 vs baseline 89.13)**,std 从 2.42 降到 0.40 — 主要是稳定性提升
- KL BN 一致性正则(照抄 FDSE C 模块)在 Office 上 seed 敏感性极高:seed=2/15 掉分 1~3.7,seed=333 暴涨 +5.17 — 不可控

### GPT-5.4 两轮 novelty check 的关键 insight

**第一轮(评审 3 个方向 A/B/C)**:
- A (SGA-SAM) 4/10 — FedSCAM 直接撞车
- B (PGO) 6/10 — FedSOL/FedORGP 部分撞车
- **C (RGHA) 7/10 — 唯一被推荐,但必须 "radically simplify"**
- 替代方向 D (Style-Manifold Semantic Stability) — GPT-5.4 自己提的

**第二轮(评审方向 D)**:
- D 5/10 — FedCCRL (2024) 已做 "original vs augmented sample consistency"
- 只有 D5(prototype stability on style manifold)可能 MEDIUM novel

**结论**:走**简化版 C**,不走 D。并采纳 GPT-5.4 对 C 的 positioning 建议:
> 不要卖 "hybrid of QP + SAM + ALA",要卖 **"decoupled style bank 提供免费的 regime signal 和 graph 结构"**。

## 方法(single clean policy)

### 核心观察:style bank 提供两个免费信号

从全局 style bank `{(μ_i, σ_i)}_{i=1..N}` 可以免费计算:

```
d_ij = ||μ_i - μ_j||_2 + ||log σ_i - log σ_j||_2
```

这给我们两个结构:
1. **regime score** `r = mean_{i<j}(d_ij)` — 标量,反映当前 federation 是 high / low gap
2. **style graph** `G = (V={clients}, E={d_ij})` — 图结构,反映客户端间的风格邻接关系

### Policy 1: Style Graph Neighbor Dispatch(替换随机分发)

**原版 FedDSA**:服务器随机从 style bank 里选 K 个风格发给每个客户端。

**EXP-067**:服务器按 `d(self, j)` 升序,发 K 个**最近邻**的风格。

```python
def _dispatch_knn_styles(client_id, available, k):
    my_style = bank[client_id]
    scored = [(dist(my_style, bank[j]), j) for j in available]
    scored.sort()
    return [bank[c] for _, c in scored[:k]]
```

**为什么这样做**:
- **PACS**(high gap):最近邻仍然是其他域(因为所有域都够远),保持 diversity
- **Office**(low gap):最近邻非常接近,gentle augmentation,避免 off-manifold noise
- 不需要新超参
- 是 **on-manifold 的**,在两个 regime 下都合理

### Policy 2: Regime-Gated Server-Side SAM(附加 consensus 之后)

**核心流程**(每轮服务器 aggregation):
```
1. snapshot w_old
2. run Consensus QP aggregation (inherited from feddsa_consensus)
   → mutates self.model to w_new_consensus
3. compute current aggregated pseudo-gradient d_t = w_new_consensus - w_old
4. compute regime score r from style bank
5. IF r < regime_threshold AND prev_pseudo_grad is not None:
       # apply SAM look-ahead using previous round's aggregated direction
       w_new_consensus += sam_rho * prev_pseudo_grad / ||prev_pseudo_grad||
6. save d_t as prev_pseudo_grad for next round
```

**关键点**:
- 始终先做 Consensus QP(high regime 已验证有效)
- **仅在 low regime 下叠加 SAM look-ahead**(寻找 flat minima)
- SAM 方向用**前一轮**的 aggregated pseudo-gradient,零额外通信(FedGloSS 风格)
- `regime_threshold=0.0` 初始值 → **第一轮实验让 SAM 永不触发**,纯看 graph dispatch 对 consensus 的影响。这样可以**分离变量**:
  - 先看 graph dispatch 单独贡献(regime_threshold=0.0)
  - 然后根据日志里观察到的 PACS/Office 的典型 r 值,再调 threshold 启用 SAM

## 与其他方法的机制差异(reviewer 防御)

| 方法 | 聚合 | 风格分发 | 风格来源 | 正则 |
|---|---|---|---|---|
| FedDSA 原版 | FedAvg | 随机 | global bank | 正交 + HSIC + InfoNCE |
| FDSE | Consensus QP | — | — (擦除) | KL BN 一致性 |
| FedGloSS | SAM | — | — | — |
| FedSOL | FedAvg | — | — | proximal 正交 |
| CCST | FedAvg | 随机 (per-class) | server bank | — |
| FedCCRL | FedAvg | 随机 | client stats | contrastive + JS alignment |
| FedAWA | client-vector adaptive | — | — | — |
| FedDisco | label discrepancy weights | — | — | — |
| **EXP-067** | **Consensus + regime-gated SAM** | **KNN graph** | **decoupled style bank** | 正交 + HSIC + InfoNCE |

核心差异:
- **CCST / FedCCRL** 有 bank 但**随机分发**,我们是 **graph-based KNN**
- **FedGloSS** 有 SAM 但**固定 ρ,无 regime awareness**,我们是 **regime-gated**
- **FedDisco / FedAWA** 有 adaptive weights 但**信号不是 style**,我们是 **decoupled style-distance 驱动**

## 代码结构

```
FDSE_CVPR25/
├── algorithm/
│   └── feddsa_regime_gated.py              ← 新增, 继承 feddsa_consensus.Server
├── config/
│   ├── pacs/feddsa_regime_gated.yml        ← 新增, 照抄 feddsa_consensus.yml
│   └── office/feddsa_regime_gated.yml      ← 新增
experiments/ablation/EXP-067_regime_gated/
├── NOTE.md                                  ← 本文档
└── test_regime_gated.py                     ← 17 个 unit tests, 17/17 pass
```

## 超参数

| 参数 | 值 | 备注 |
|---|---|---|
| algo_para (继承自 feddsa) | [1.0, 0.0, 1.0, 0.1, 50, 5, 128] | 与 EXP-064 相同 |
| `regime_threshold` | 0.0 (第一轮) | 第一轮永不触发 SAM,用于隔离 graph dispatch 贡献。后续根据 log 调整 |
| `sam_rho` | 0.05 | FedGloSS 默认范围 |
| PACS lr / E / R | 0.1 / 5 / 200 | 同 EXP-064 |
| Office lr / E / R | 0.05 / 1 / 200 | 同 EXP-064 |

## 单元测试(17/17 PASS)

运行命令:
```bash
cd FDSE_CVPR25 && python ../experiments/ablation/EXP-067_regime_gated/test_regime_gated.py
```

测试覆盖:

**Regime score (6)**
- `regime_score_empty_bank` — warmup 返回 None
- `regime_score_single_client` — 单客户端返回 None
- `regime_score_two_identical` — r=0 精确验证
- `regime_score_two_different` — r=5.3863 与手算一致
- `regime_score_mean_of_pairs` — 3 客户端对的平均
- `pacs_vs_office_ordering` — **合成数据 r_pacs=24.58 vs r_office=0.52, 47 倍差异** ✅

**SAM look-ahead (3)**
- `sam_lookahead_no_prev` — 无 prev_pseudo_grad 时 no-op
- `sam_lookahead_applies_perturbation` — ρ/||g|| 缩放正确
- `sam_lookahead_direction_matches_prev` — 方向和符号正确

**Regime gate (1)**
- `regime_gate_logic` — high/low r + prev None/存在 4 种 case 的 gate 决策

**Inheritance (1)**
- `method_inheritance` — 继承 consensus QP / quadprog / graph helpers

**Style graph dispatch (6)**
- `knn_dispatch_picks_nearest` — 最近 k 个的选择
- `knn_dispatch_excludes_self_via_available` — self 排除由 caller 处理
- `knn_dispatch_fallback_no_self_bank` — 新客户端 fallback 到随机
- `knn_dispatch_k_larger_than_bank` — k > N 时优雅降级
- `style_graph_edges_count` — N*(N-1)/2 edge 数正确
- `knn_dispatch_agrees_with_graph_edges` — KNN 结果与 graph 距离一致

## 运行命令

```bash
# SC2 (seed=2 先验证)
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

for T in PACS_c4 office_caltech10_c4; do
    nohup $PY run_single.py --task $T --algorithm feddsa_regime_gated --gpu 0 \
        --config ./config/$(echo $T | cut -c1-4 | tr '[:upper:]' '[:lower:]')/feddsa_regime_gated.yml \
        --logger PerRunLogger --seed 2 \
        > /tmp/exp067_${T}_s2.out 2>&1 &
done
```

## 预期 / 成功判据

### seed=2 一阶段验证
1. PACS AVG Best ≥ EXP-064 Consensus 的 83.04(**graph dispatch 不伤 PACS**)
2. Office AVG Best > baseline 89.95(**graph dispatch 改善 Office**)
3. log 中 r_PACS, r_Office 两组数值 → 用于校准 regime_threshold

### 二阶段(根据 seed=2 结果调 threshold)
- 如果 seed=2 的 log 显示 `r_PACS >> r_Office`,设 `regime_threshold = (r_PACS + r_Office)/2` 重跑
- 期望:Office 在 SAM 启用后进一步提升到 ≥ 90

### 最终成功判据
- 3-seed 均值:PACS ≥ baseline + 0.5,Office ≥ baseline + 0.5
- **同一套超参同时超 baseline on both datasets**(FDSE 自己做不到的 "one method fits both")

## 下一步 / Future work

- **EXP-067 若成功 + 考虑 graph-paper 升级**:把 "KNN dispatch" 扩展为 "graph geodesic style transport",把 "regime score" 扩展为 "graph-level structural signal"
- **若不够,再谈方向 D 的 prototype stability 重构**(GPT-5.4 推荐但 novelty 只有 5/10)

## 结果 (3-seed COMPLETE)

### Office-Caltech10 (✅ DONE)

| seed | AVG Best | Last | vs baseline 89.13 | vs Consensus 89.83 |
|---|---|---|---|---|
| 2 | 88.91 | 87.79 | -0.22 | -0.92 |
| 15 | 90.03 | 89.55 | +0.90 | -0.08 |
| 333 | 90.22 | 88.28 | +1.09 | +0.23 |
| **Mean ± Std** | **89.72 ± 0.58** | | **+0.59** | **-0.11** |

### PACS (✅ DONE)

| seed | AVG Best | Last | vs baseline 81.29 | vs Consensus 80.74 |
|---|---|---|---|---|
| 2 | 81.11 | 70.49 | -1.18 | -1.93 |
| 15 | 78.70 | 72.62 | -2.59 | -0.69 |
| 333 | 78.57 | 74.27 | -2.72 | -1.23 |
| **Mean ± Std** | **79.46 ± 1.17** | | **-1.83** | **-1.28** |

### Regime score r (backbone feature space, not discriminative!)

| Dataset | round 10 | round 20 | round 100 | round 200 |
|---|---|---|---|---|
| PACS avg | ~2.4 | ~2.0 | - | ~2.0 |
| Office avg | ~2.8 | ~2.8 | ~2.3 | ~2.1 |

**关键发现**: PACS r ≈ Office r (均 ~2.0-2.2), backbone BN 后的激活统计量不 discriminative。
**结论**: KNN (nearest) dispatch 在 PACS 有害 (-1.28), Office 中性 (-0.11)。问题不在 dispatch。
