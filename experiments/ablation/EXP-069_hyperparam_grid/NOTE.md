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

## 结果(待填)

| ID | PACS s=2 AVG Best | vs baseline 82.24 | 发现 |
|---|---|---|---|
| baseline | 82.24 | — | |
| 069a (orth=0.5) | - | - | |
| 069b (orth=2.0) | - | - | |
| 069c (sem=0.5) | - | - | |
| 069d (sem=2.0) | - | - | |
| 069e (tau=0.05) | - | - | |
| 069f (tau=0.2) | - | - | |
