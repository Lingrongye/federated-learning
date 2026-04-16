# EXP-075 | Gradual Training — Fix Warmup Collapse

## 基本信息
- **日期**: 2026-04-16
- **算法**: feddsa_gradual
- **服务器**: SC4 GPU0
- **状态**: 准备中

## 动机

所有 FedDSA 变体在 warmup 后性能下降 (10/11 实验)。FDSE 从头训练所有组件,持续上升。
根因: 硬 warmup 突然激活新 loss + 特征层 AdaIN 破坏语义空间 + CE/InfoNCE 梯度冲突。

## 三个关键修复

### Fix 1: 分离式 Sigmoid Ramp-Up
- w_aug(t) = sigmoid((t - t_mid_aug) / tau_w_aug) — 增强权重
- w_align(t) = sigmoid((t - t_mid_align) / tau_w_align) — 对齐权重
- 增强早开(t_mid=15), 对齐晚开(t_mid=40)

### Fix 2: 浅层特征增强
- 把 AdaIN 从 encoder 输出 h(1024d) 移到 conv3 输出(384 channels)
- 避免破坏已学好的高层语义特征

### Fix 3: 梯度冲突诊断
- 每 10 round 测量 grad_CE vs grad_InfoNCE 的余弦相似度
- 验证: 渐进式训练是否减少梯度冲突

## 实验设计 (4 configs × 3 seeds = 12 runs)

| Config | aug_level | t_mid_aug | t_mid_align | 目的 |
|---|---|---|---|---|
| **gradual_shallow** | 1 (conv3) | 15 | 40 | 主实验: 全部修复 |
| **gradual_deep** | 0 (h-space) | 15 | 40 | 控制: ramp-up 但 AdaIN 位置不变 |
| **gradual_noaug** | 2 (无) | - | 40 | 控制: 只有对齐,无增强 |
| **gradual_syncramp** | 1 (conv3) | 30 | 30 | 消融: 同步 ramp vs 分离 ramp |

Seeds: 2, 333, 42

## 预期结果

| Config | 预期 R200 Final | 预期 Peak→Final Drop | 理由 |
|---|---|---|---|
| gradual_shallow | **≥81%** | **<1%** | 三重修复应最稳定 |
| gradual_deep | ~79-80% | 1-3% | ramp-up 帮助但 h-space 增强仍有害 |
| gradual_noaug | ~80% | <1% | 无增强噪声,但缺风格多样性 |
| gradual_syncramp | ~80% | <2% | 同步 ramp 不如分离 |

## 对照基线
- FedDSA 原版 (warmup=50, tau=0.2): 80.93% (R200)
- FDSE: ~80.36% (R200), ~82.17% (R500)
- FedBN: ~79.47%

## 结果 (SC4, 更新至 R144-145)

### PACS 3-seed (R144-145)

| Config | s | R | max | last | drop |
|---|---|---|---|---|---|
| shallow | 2 | 144 | 79.5 | 79.3 | -0.1 |
| shallow | 333 | 145 | 78.8 | 70.5 | **-8.3** |
| shallow | 42 | 145 | 79.8 | 79.2 | -0.6 |
| deep | 2 | 144 | 80.7 | 80.4 | -0.2 |
| deep | 333 | 145 | 79.9 | 73.2 | **-6.7** |
| deep | 42 | 144 | 81.6 | 80.2 | -1.5 |
| **noaug** | **2** | **145** | **81.7** | **51.2** | **-30.4 崩了!** |
| noaug | 333 | 144 | 77.7 | 69.6 | **-8.1** |
| noaug | 42 | 145 | 80.2 | 80.2 | 0.0 |
| syncramp | 2 | 144 | 81.6 | 80.2 | -1.4 |
| syncramp | 333 | 144 | 78.4 | 73.6 | **-4.8** |
| syncramp | 42 | 145 | 81.0 | 80.9 | -0.1 |

### 梯度冲突日志 (shallow s=2, 完整)

| Round | cos_sim | Acc | 分析 |
|---|---|---|---|
| R10 | +0.716 | 77.2 | 对齐 |
| R20 | +0.500 | 77.8 | 对齐 |
| R30 | +0.367 | 79.0 | 渐弱 |
| R40 | +0.215 | 79.0 | 弱对齐 |
| **R50** | **-0.010** | 79.2 | **穿零！冲突开始** |
| **R60** | **-0.235** | 78.8 | **明确冲突** |
| R70 | +0.692 | 74.6 | 崩溃后 spike |
| R80 | +0.306 | 77.5 | 恢复 |
| R90 | +0.228 | 77.6 | 衰减 |
| R100 | +0.153 | 77.5 | 趋向 0 |

### R200 最终结果

| Config | s=2 max/last | s=333 max/last | s=42 max/last | 3s mean last |
|---|---|---|---|---|
| noaug | 81.7/78.1 (-3.5) | 77.7/74.9 (-2.8) | 80.7/77.3 (-3.4) | **76.8** |
| deep | 80.7/77.8 (-2.8) | 79.9/70.0 (-9.9) | 81.6/76.6 (-5.1) | 74.8 |
| shallow | 80.8/74.2 (-6.6) | 78.8/77.2 (-1.6) | 80.2/75.9 (-4.3) | 75.8 |
| syncramp | 81.6/77.5 (-4.1) | 78.8/78.7 (-0.1) | 81.3/77.0 (-4.4) | 77.7 |

### 最终结论

1. **全面失败** — 所有配置 3-seed mean last < 78%，远低于 FDSE 80.36%
2. noaug 之前 R100 看似稳定，到 R200 也崩了 — sigmoid ramp 只是延迟不是解决
3. **梯度冲突日志是本实验最大价值** — 证明 CE/InfoNCE 在 R50 穿零的因果链
4. **结论: 调度策略（ramp/bell/cutoff）无法修复根因，需要安全阀** → EXP-077/078
