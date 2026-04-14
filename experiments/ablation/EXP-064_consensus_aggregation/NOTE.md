# EXP-064 | Consensus-Aware Aggregation (核心 fix 候选)

## 基本信息
- **日期**:2026-04-11
- **算法**:feddsa_consensus
- **状态**:✅ 3-seed 全部完成 (SC2)

## 动机

### EXP-060~063 的教训
所有 4 个 style-side fixes 在 Office 上都**更差**:
| Variant | Office AVG | Delta |
|---|---|---|
| baseline | 89.95 | — |
| Gated | 87.98 | -1.97 |
| NoAug | 88.58 | -1.37 |
| SoftBeta | 88.39 | -1.56 |
| AugSched | 88.43 | -1.52 |

**H1 (style aug 是 Office 负迁移源头) 被证伪**:
- 禁用 aug 反而更差 → style aug 即使在 Office 也有价值
- 问题不在 style aug

### 新假设 (H2)
**Office 输 FDSE 的真正原因是 aggregation conflict**,不是 style aug。

证据 (GPT-5.4 review Q2):
- FDSE Table 3: consensus maximization 单项贡献 > similarity-aware personalization
- Office 核心矛盾: subtle client drift → FedAvg destructive averaging
- 各 domain 都是 real photos,shared semantic signal 很强,需要的是避免 aggregation 把它弄坏

## 方法

**只改一处**: Server.iterate() 的 shared parameters 聚合逻辑
- 原: FedAvg (weighted mean by data volume)
- 新: **Consensus-maximization** (FDSE Eq. 8, multi-gradient descent)

```
对每个 shared layer l:
    d_k = (theta_k - theta_global) / ||theta_k - theta_global||
    find λ ∈ simplex minimizing ||sum(λ_k * d_k)||²
    new_theta = theta_global + mean(||Δ||) * sum(λ_k * d_k)
```

保留 FedDSA 所有其他机制:
- ✅ 双头解耦 (encoder + semantic_head + style_head)
- ✅ orth loss 正交约束
- ✅ 全局风格仓库 + AdaIN 增强
- ✅ InfoNCE 原型对齐
- ✅ style_head 私有
- ✅ BN 私有

**只改聚合,不改训练**。代码位置:`feddsa_consensus.py` Server._aggregate_shared_consensus()

## 单元测试结果 (23/23 PASS)

| Test | 结果 |
|---|---|
| QP solver: identical vectors | ✓ λ=[0.33, 0.33, 0.33] |
| QP solver: conflicting vectors | ✓ λ=[0.5, 0.5, 0.0], combined norm 0.0002 |
| End-to-end aggregation | ✓ conv1 weight changes by 0.019 |
| style_head privacy | ✓ unchanged (delta 0) |
| Forward pass after agg | ✓ no NaN, finite |
| Consensus ≠ FedAvg | ✓ diff 0.00046 |
| Method inheritance | ✓ all methods callable |

## 运行命令
```bash
# Office (期望提升)
nohup python run_single.py --task office_caltech10_c4 --algorithm feddsa_consensus --gpu 0 \
  --config ./config/office/feddsa_consensus.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp064_office.out 2>&1 &

# PACS (sanity check, 不能破坏)
nohup python run_single.py --task PACS_c4 --algorithm feddsa_consensus --gpu 0 \
  --config ./config/pacs/feddsa_consensus.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp064_pacs.out 2>&1 &
```

## 对照基线 (seed=2)

| 数据集 | FedDSA baseline | FDSE | 目标 |
|---|---|---|---|
| PACS AVG | 82.24 | 80.81 | ≥ 81 (不破坏) |
| Office AVG | 89.95 | **92.39** | **≥ 91** (接近 FDSE) |

## 假设验证

**如果 Office AVG ≥ 91 (接近 FDSE)** → H2 成立,aggregation 是关键
**如果 Office AVG 显著提升 (≥ 90.5)** → H2 部分成立,需要组合其他机制
**如果 Office AVG < 90** → H2 也不成立,问题可能更深(FDSE 的 layer decomposition?)

## 结果 (seed=2)
| 数据集 | AVG Best | Last | Gap | vs baseline |
|---|---|---|---|---|
| PACS | **83.04** | 67.58 | 15.46 | **+0.80** ✅ (最好! > FDSE 80.81) |
| Office | **89.40** | 88.20 | 1.20 | -0.55 (最接近 baseline 的变体) |

## seed=2 观察
- **PACS 反超 baseline 和 FDSE!** 83.04 > baseline 82.24 > FDSE 80.81
- Office 仍然 -0.55, 但是所有 6 个变体(Gated/NoAug/SoftBeta/AugSched/Cons/Cons+KL)里**最好的**
- 与假设 H2 一致:aggregation conflict 是关键
- 但 gap 较大 (PACS 15.46),last 严重掉落 → 最后期间训练不稳

## 3-seed 完整结果 (✅ COMPLETE)

### PACS

| seed | AVG Best | Last | vs baseline 81.29 |
|---|---|---|---|
| 2 | **83.04** | 67.58 | +1.75 |
| 15 | 79.39 | 75.89 | -1.90 |
| 333 | 79.80 | 70.37 | -1.49 |
| **Mean ± Std** | **80.74 ± 1.63** | | **-0.55** |

### Office-Caltech10

| seed | AVG Best | Last | vs baseline 89.13 |
|---|---|---|---|
| 2 | 89.40 | 88.20 | +0.27 |
| 15 | **90.11** | 89.10 | +0.98 |
| 333 | **89.99** | 88.10 | +0.86 |
| **Mean ± Std** | **89.83 ± 0.40** | | **+0.70** |

## 结论 (multi-seed verified)

- **PACS seed=2 的 83.04 是单 seed 假象**:s15/s333 均 < baseline。Multi-seed mean 80.74 < baseline 81.29
- **Consensus 在 PACS 上实际有害**:强制"最小冲突聚合"限制了各域自由发展
- **Office 上 Consensus 大幅提升稳定性**:std 从 2.42 → 0.40 (6x 降低), mean +0.70
- **H2 结论**:Consensus 解决了 Office 的 aggregation 不稳定,但不是 PACS 的问题
