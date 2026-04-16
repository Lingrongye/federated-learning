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

## 结果 (SC4, r52-53/200 interim)

### PACS 3-seed (interim @ round 53)

| Config | s=2 max | s=333 max | s=42 max | 稳定性 (last vs max) |
|---|---|---|---|---|
| **gradual_noaug** | **81.19** (last=81.06) | 77.45 | **80.11** (last=80.11) | **极稳 -0.13** |
| gradual_deep | 80.54 (last=79.98) | **79.23** | **81.53** (last=80.78) | 稳定 -0.75 |
| gradual_shallow | 79.48 (last=78.78) | 78.34 | 79.80 (last=78.68) | 轻微降 -1.0 |
| gradual_syncramp | **81.60** (last=79.63) | 78.42 (last=**58.09!**) | 80.45 | **s333 崩了!** |

### 关键发现 (interim)

1. **gradual_noaug (无增强,只有渐进对齐) 最稳定**: s2 max=81.19, last=81.06, drop 仅 0.13%
2. **syncramp s333 崩了**: 同步 ramp 不如分离 ramp, 暴跌到 58%
3. **AdaIN 增强是不稳定源**: noaug > deep > shallow > syncramp (增强越少越稳)
4. **deep(h-space) > shallow(conv3)**: 意外发现, 可能 conv3 384-ch 统计量太粗糙

### 梯度冲突日志

| Config | r10 cos_sim | r30 cos_sim | r50 cos_sim | r100 cos_sim | 趋势 |
|---|---|---|---|---|---|
| (待收集 — 需等实验完成后从 log 中提取) |

### 梯度冲突日志

| Config | r10 cos_sim | r30 cos_sim | r50 cos_sim | r100 cos_sim | 趋势 |
|---|---|---|---|---|---|
| gradual_shallow | - | - | - | - | - |
| gradual_deep | - | - | - | - | - |
