# EXP-060 | Distance-Gated Style Dispatch

## 基本信息
- **日期**:2026-04-11
- **类型**:核心方法改进 (GPT-5.4 research review 的 top priority)
- **算法**:feddsa_gated (只改 Server.pack 的 dispatch 逻辑)
- **状态**:⏳ 待执行

## 动机 (来自 research review)

### 核心发现
FedDSA 在 PACS 赢是因为 sketch/art_painting 风格差异大, AdaIN 扩张 support;
在 Office 输是因为所有域都是 real photos, 风格差异小, AdaIN 反而制造 off-manifold noise。

**相同方法在两个数据集表现相反,原因是"unconditional raw style sharing"**:
- PACS: 大风格差 → 风格共享 = support expansion (好)
- Office: 小风格差 → 风格共享 = noise injection (坏)

### 文献证据
- **CCST (WACV 2023)**: OfficeHome 上 DG methods 提升 < 1%, 作者明确说"domain style discrepancy 小"
- **FedCCRL (arXiv 2024)**: Art/Clipart +1-5%, Product/Real +0.2% — 和我们模式完全一致
- **StyleDDG**: VLCS gains << PACS, 因为 style gap 更小

## 方法

### 只改一处: Server.pack() 的 dispatch 逻辑

**原版**: 从 style_bank 中**随机**选 5 个外部风格,不管距离
**新版**: 计算当前 client 的 style 与 bank 中其他 client 的距离,只 dispatch 距离超过阈值的风格

距离公式:
```
d(i, j) = ||μ_i - μ_j||₂ + ||log(σ_i) - log(σ_j)||₂
```

Gate 逻辑:
- 如果 max_d < threshold → 不 dispatch 任何风格 (no style aug 这轮)
- 否则 → 取 top-K 最远的风格, 且都必须超过 threshold

### 预期行为
- **PACS**: 所有 client pair 距离大 → 正常 dispatch 5 个 → 保持原有增益
- **Office**: 所有 client pair 距离小 → 可能 dispatch 0 个 → 退化为 no style aug → 减少负迁移

### 超参
- `dist_threshold=1.0` (需要看实际分布调整, code 会 log 每轮的 max/mean/min 距离)

## 与之前实验的区别

| 实验 | 改动位置 | 区别 |
|---|---|---|
| EXP-024 soft_aug | Beta 参数 | 改**采样强度**, 所有 style 都发 |
| EXP-047A augdown | loss_aug 权重 | 改**何时起作用** |
| EXP-047D noaug_late | hard stop | 改**何时停** |
| EXP-058 detach | 梯度流 | 改梯度,不改 dispatch |
| EXP-059 stylehead_bank | 存什么 | 改**存什么**, 不改**选哪个** |
| **EXP-060 gated** | **选哪个 + 是否发** | **首次改 dispatch 选择逻辑** |

## 运行命令

```bash
# PACS (expect 维持原有性能, 因为所有 style 距离都大)
nohup python run_single.py --task PACS_c4 --algorithm feddsa_gated --gpu 0 \
  --config ./config/pacs/feddsa_gated.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp060_pacs.out 2>&1 &

# Office (期望提升, 因为大多数 style 会被 gate 掉)
nohup python run_single.py --task office_caltech10_c4 --algorithm feddsa_gated --gpu 0 \
  --config ./config/office/feddsa_gated.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp060_office.out 2>&1 &
```

## 对照基线 (seed=2)

| 数据集 | Baseline (原版 FedDSA) | FDSE | 目标 |
|---|---|---|---|
| PACS AVG | 82.24 | 80.81 | ≥ 80 (不能掉) |
| Office AVG | 89.95 | 92.39 | > 89.95, 最好 ≥ 91 |

## 验证假设
1. **假设A**: PACS 上距离大,几乎所有风格通过 gate → 性能不降
2. **假设B**: Office 上距离小,很少风格通过 gate → 减少负迁移 → 性能提升
3. **假设C**: gated 后 Office 至少和 FedBN (88.99) 相当

## 结果
| 数据集 | ALL Best | AVG Best | AVG Last | Gap | vs baseline |
|---|---|---|---|---|---|
| PACS | | | | | |
| Office | | | | | |

## 诊断 log
log 会输出每轮 style distance 的 max/mean/min,看:
- PACS 和 Office 的距离分布差多少倍
- threshold 设置是否合理

## 结论
