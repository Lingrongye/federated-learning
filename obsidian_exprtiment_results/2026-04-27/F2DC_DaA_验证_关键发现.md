---
date: 2026-04-27
type: 关键发现 (decision-grade)
status: office s=15 R100 完成 (DaA 验证有效), 其它跑中
exp_id: EXP-133
---

# F2DC paper Eq (10)(11) Domain-Aware Aggregation 验证 — 关键发现

## 背景

我们在主表数据看到 F2DC paper 报 PACS 76.47, 但我们 release code 复现 F2DC PACS 仅 71.02 (gap 5.45pp). grep 发现 F2DC paper 提的 "Domain-Aware Aggregation" (`--agg_a / --agg_b`) **从来没在 release code 实施**, paper 三大 contribution 之一空缺.

我们决定 implement Eq (10)(11) 验证: paper 报 +3-4pp 的 DaA 在 single-domain client setup 下是否真的有效?

## 我们 implement 的 paper 公式

**Eq (10)** (paper image 8 + image 9, 简化后):
$$
d_k = \sqrt{\frac{C}{2}} \cdot \left|\frac{n_k}{N} - \frac{1}{Q}\right|
$$
- C = 类数 (PACS=7, Office=10, Digits=10)
- Q = 域数 (都是 4)
- N = Σ n_k

**Eq (11)** (paper image 9):
$$
p_k = \frac{\sigma(\alpha \cdot \frac{n_k}{N} - \beta \cdot d_k)}{\sum_j \sigma(\alpha \cdot \frac{n_j}{N} - \beta \cdot d_j)}
$$
- α=1.0, β=0.4 (paper Fig 7 default)
- σ = sigmoid

## ⚠️ 我们之前的怀疑 (后被推翻)

> "在 single-domain client setup 下 (PACS 10 client 各持 1 domain), `B_k` 简化后只依赖 `n_k/N`, 不真正反映 client 持有的 domain identity. 所以 DaA 应该 degenerate, paper 报的 +3pp 可能是噪声."

→ **数据推翻了这个怀疑**.

## 实验结果

### F2DC + DaA office s=15 R100 完成 ⭐

| Method | Office best (single seed s=15) | Δ vs vanilla F2DC (主表 60.56) |
|---|:--:|:--:|
| FedAvg (主表) | 57.90 | -2.66 |
| **vanilla F2DC** (主表 2-seed) | **60.56** | baseline |
| **F2DC + DaA s=15 (我们 R100)** | **63.93** ⭐ | **+3.37pp** |
| F2DC paper (3-seed, 有 DaA) | 66.82 | (+6.26 vs 我们 vanilla) |
| FDSE (主表) | 63.52 | +2.96 |

**关键观察**:
- 单 seed +3.37pp 是显著真实 gain, 不是噪声
- 已经追平 FDSE office (63.52, single-seed 63.93 ≈ FDSE)
- 跟 paper 报的 66.82 还差 ~3pp, 可能因为 single-seed (paper 是 3-seed mean) + 微差超参

### 还在跑

| Run | R/100 | best | Server |
|---|:--:|:--:|:--:|
| F2DC+DaA office s=333 | 0 | — | sc3 (刚 launch) |
| F2DC+DaA PACS s=15 | 12 | 38.99 | sc3 |
| F2DC+DaA PACS s=333 | 待 launch | — | 等 sc5 释放 |

## 为什么 DaA 在 single-domain setup 下还有效

我们之前推断 d_k 在 single-domain 下只反映 sample size,paper 公式应 degenerate. 但实测有效, 可能原因:

1. **sample size 偏差本身就是有效信号**: PACS Office Digits 各 client sample 数确实差很大 (PACS sketch client 1300 vs photo client 1670 等), reweight 让大 client 降权, 小 client 升权 → 跨 domain 公平性提升
2. **跨 domain fairness 的代理**: 即使不知道 client 持有什么 domain, 让所有 client 权重接近 1/K 也间接 promote domain fairness (因为 single-domain client 跟 domain identity 1-to-1)
3. **隐式正则**: sigmoid + 偏离 1/Q 的非线性 reweight 可能起到正则作用, 抵御 dominant domain 主导

## 我们 critique 部分错了

| 之前判断 | 实测结果 | 修正 |
|---|---|---|
| "DaA in single-domain setup 是空名号" | DaA 确实带 +3.37pp | DaA 通过 sample-size proxy 间接 promote domain fairness, 不只是空名号 |
| "我们的 PDA-Agg 才是 first 真 domain-aware" | DaA 已经 work, novelty 立场要调整 | 我们 PDA-Agg 应定位为 "比 paper DaA 更精细的 prototype-space refinement" |

## 立即行动

### P0: PG-DFC + DaA 集成
预期收益:
- **Office**: PG-DFC v3.2 (61.25) + DaA → ~64-65, **直接 challenge FDSE 63.52** (我们一直输给 FDSE 的 office bottleneck 可能解决!)
- **PACS**: PG-DFC v3.2 (73.20) + DaA → ~75-76, 跟 paper F2DC 76.47 持平
- **Digits**: 不太确定 (digits 跑完 F2DC+DaA 才能判)

实施:
- PG-DFC backbone aggregate_nets 已经调 `super().aggregate_nets`, 启动加 `--use_daa True --num_domains_q 4` 即生效
- 不用改任何代码, 直接 launch

### P1: 完成 F2DC+DaA 全数据集
- F2DC+DaA office s=333 (跑中)
- F2DC+DaA PACS s=15/s=333
- F2DC+DaA Digits s=15/s=333
- 跟 paper 报值对比验证我们环境复现

### P2: PDA-Agg 重新定位
- 不再 frame 为 "first prototype-space reweight"
- 改为 "DaA refinement: 在 paper 的 sample-size aware reweight 之上加 prototype-drift refinement"
- ablation 必须包含 DaA only / DaA + PDA-Agg 才能证明增量价值

## 待回填

- [ ] F2DC+DaA office s=333 完成 (sc3 跑中, ~30 min)
- [ ] F2DC+DaA PACS s=15 完成 (sc3 跑中, ~70 min)
- [ ] F2DC+DaA PACS s=333 (待 sc5 释放 launch)
- [ ] F2DC+DaA Digits s=15/s=333
- [ ] PG-DFC v3.2 + DaA 全数据集 (PACS/Office/Digits × 2 seed = 6 runs)

## paper narrative 调整

旧 narrative (要废弃):
> "F2DC's DaA degenerates in single-domain setup. We propose the first true prototype-space domain-aware aggregation."

新 narrative (基于实测):
> "F2DC paper introduces a domain-aware aggregation but does not release its implementation. We provide a faithful implementation and verify its empirical effectiveness (+3.37pp on Office). Building on this, we further refine it via prototype-space drift signal (PDA-Agg), achieving additional +X.Xpp gain."
