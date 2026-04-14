# EXP-069 | FedDSA 超参调优 Grid Search

## 基本信息
- **日期**:2026-04-12
- **算法**:feddsa (原版 baseline)
- **目的**:从未调过客户端训练超参,可能是 PACS/Office 性能提升的最高 ROI 方向
- **服务器**:lab-lry GPU 1
- **状态**:⏳ 准备中

## 动机

EXP-060~068 的 9 个实验全部围绕**聚合机制**改进(Consensus/dispatch/SAM),但:
- PACS baseline (81.29) 已经赢 FDSE (80.36)
- 所有聚合变体反而在 PACS 上更差
- 客户端训练参数 (lambda_orth, lambda_sem, tau, warmup) 一直用默认值

**假设**: 调 lambda/tau 可能比改聚合更有效。

## Grid 设计

基准: `algo_para = [1.0, 0.0, 1.0, 0.1, 50, 5, 128]` = [lambda_orth, lambda_hsic, lambda_sem, tau, warmup, dispatch_num, proj_dim]

| ID | 变化参数 | 值 | 预期效果 |
|---|---|---|---|
| 069a | lambda_orth=0.5 | 弱正交约束 | 更多语义-风格信息流 |
| 069b | lambda_orth=2.0 | 强正交约束 | 更纯净分离但可能过度约束 |
| 069c | lambda_sem=0.5 | 弱 InfoNCE | 更自由的本地表示,可能利于 PACS |
| 069d | lambda_sem=2.0 | 强 InfoNCE | 更强对齐,可能利于 Office |
| 069e | tau=0.05 | 尖锐 InfoNCE | 更强判别但可能过拟合 |
| 069f | tau=0.2 | 平滑 InfoNCE | 更温和学习 |

## 实验 Phase 1: PACS seed=2 (6 runs)

先跑 PACS seed=2 找 top 2-3 配置,再跑 Office + multi-seed。

## 结果 (✅ COMPLETE, PACS seed=2, 201 rounds, lab-lry GPU1)

| ID | 变化参数 | PACS s=2 Max | Last | vs baseline 82.24 |
|---|---|---|---|---|
| **baseline** | 默认 1.0/1.0/0.1 | **82.24** | — | — |
| 069a | orth=0.5 | 80.41 | 73.44 | **-1.83** |
| 069b | orth=2.0 | 79.51 | 74.68 | **-2.73** |
| 069c | sem=0.5 | 78.29 | 77.07 | **-3.95** (最差) |
| 069d | sem=2.0 | 79.64 | 74.00 | **-2.60** |
| 069e | tau=0.05 | 80.06 | 73.26 | **-2.18** |
| 069f | tau=0.2 | 79.55 | **79.50** | -2.69 (last 最稳) |

## 结论

1. **默认超参 [1.0, 0.0, 1.0, 0.1] 已是最优!** 所有 6 个变体都比 baseline 差 1.8-3.9%
2. **lambda_sem 最敏感**: sem=0.5 (069c) 最差 (-3.95%), 说明 InfoNCE 对齐是核心
3. **lambda_orth 不敏感**: 0.5 和 2.0 差不多 (80.41 vs 79.51), 正交约束有用但不是瓶颈
4. **tau=0.2 过拟合最小**: last=79.50 最接近 max, 但 max 本身不高
5. **结论: 超参调优不是提升路径, baseline 就是 sweet spot**
