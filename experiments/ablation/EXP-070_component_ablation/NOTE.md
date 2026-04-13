# EXP-070 | FedDSA 组件消融 (Component Ablation)

## 基本信息
- **日期**:2026-04-13
- **算法**:feddsa 各变体
- **服务器**:lab-lry GPU 1
- **状态**:✅ 完成 (201 rounds)

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

## 结果 (✅ COMPLETE, PACS seed=2, 201 rounds)

| 变体 | PACS s=2 AVG Best | vs Full (82.24) | 解读 |
|---|---|---|---|
| **Decouple only** | **81.07%** | -1.17 | 解耦本身有价值,是最强单组件 |
| +Share (no align) | 79.15% | -3.09 | Share 无 Align 引导反而有害 |
| +Align (no share) | 78.99% | -3.25 | Align 无 Share 增强也有害 |
| **Full FedDSA** | **82.24%** | — | **三者协同效应 > 任何子集** |

## 关键发现

1. **Decouple 是基础**: 单独解耦(81.07%)已接近 FDSE(80.36%),证明双头正交分离本身就有价值
2. **Share 和 Align 单独都有害**: 风格增强没有语义对齐引导 → 噪声; 语义对齐没有风格增强 → 缺多样性
3. **三者协同才是关键**: Full(82.24%) > Decouple(81.07%) > +Share(79.15%) > +Align(78.99%)
4. **不是 "bag of tricks"**: 单独加任何一个模块都比不加更差,只有三者结合才产生 +1.17% 的协同增益

## 消融故事线(paper 可用)

> "FedDSA 的三个组件(Decouple, Share, Align)存在强协同效应。
> 单独的风格共享(+Share)在缺乏语义对齐引导时反而引入噪声(-3.09%)。
> 单独的语义对齐(+Align)在缺乏风格多样性支撑时也无法发挥作用(-3.25%)。
> 只有三者结合——解耦提供纯净的语义/风格空间,共享提供跨域增强素材,
> 对齐引导语义特征稳定收敛——才能产生协同增益(+1.17% over Decouple only)。"
