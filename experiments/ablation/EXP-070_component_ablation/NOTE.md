# EXP-070 | FedDSA 组件消融 (Component Ablation)

## 基本信息
- **日期**:2026-04-13
- **算法**:feddsa 各变体
- **服务器**:lab-lry GPU 1
- **状态**:⏳ 启动中

## 动机

GPT-5.4 review 指出缺少干净的组件消融 (攻击点 #2: "bag of tricks")。
需要验证 Decouple/Share/Align 三个模块各自的贡献。

## 消融设计

| 变体 | Decouple (orth) | Share (AdaIN aug) | Align (InfoNCE) | 配置 |
|---|---|---|---|---|
| Decouple only | ✅ lambda_orth=1.0 | ❌ warmup=9999 | ❌ lambda_sem=0 | feddsa_ablation_decouple.yml |
| +Share | ✅ | ✅ warmup=50 | ❌ lambda_sem=0 | feddsa_ablation_share.yml |
| +Align | ✅ | ❌ warmup=9999 | ✅ lambda_sem=1.0 | feddsa_ablation_align.yml |
| **Full FedDSA** | ✅ | ✅ | ✅ | baseline (已有: 82.24) |

## Phase 1: PACS seed=2

## 结果(待填)

| 变体 | PACS s=2 AVG Best | vs Full (82.24) | 贡献 |
|---|---|---|---|
| Decouple only | - | - | - |
| +Share | - | - | Share 模块的增量 |
| +Align | - | - | Align 模块的增量 |
| **Full FedDSA** | **82.24** | — | 基准 |
