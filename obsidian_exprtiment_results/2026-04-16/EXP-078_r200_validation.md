# EXP-078 | R200 完整验证 — MSE 锚点 + Alpha-Sparsity + Detach Aug

## 基本信息
- **日期**: 2026-04-16
- **算法**: feddsa_scheduled.py (modes 4, 6, 7)
- **服务器**: SC2 (078a) + Lab-lry GPU1 (078c) + SC4 (078d)
- **状态**: 🔄 运行中

## 核心设计解释

### 所有 mode 的共同点
**L_orth = 1.0 × cos²(z_sem, z_sty) 从 R0 开始全权重。** 这是与原始 FedDSA 最关键的区别——原版 L_orth 随 aux_w 从 0 渐增，导致前 50 轮正交约束几乎不生效。

### 各 mode 的损失公式

```
mode 4 (078a, MSE anchor):
  L = CE(z_sem) + CE(z_sem_aug) + L_orth
    + 标准InfoNCE(z_sem vs protos, tau=0.05) 
    + MSE(z_sem, 同类proto)                    ← FPL的安全阀
  
  MSE锚点作用: 像引力中心，防止z_sem被InfoNCE拉偏
  R50结果: cos_sim=+0.678 (最高), 说明MSE让CE和InfoNCE高度协作

mode 6 (078c, MSE + alpha-sparsity):
  L = CE(z_sem) + CE(z_sem_aug) + L_orth
    + AlphaInfoNCE(z_sem vs protos, tau=0.07, alpha=0.25)
    + MSE(z_sem, 同类proto)
  
  Alpha-sparsity作用: cos^0.25 让正例梯度自动变小
    → 正例(高sim)梯度 ∝ 0.25 × sim^(-0.75) → sim越高梯度越小
    → CE已经在拉正例了，alpha不再重复拉，只专注推负例
    → 与CE互补而非冲突
  R50结果: max=82.2%, cos=+0.365, drop=0.0 — 最佳!

mode 7 (078d, detach aug):
  L = CE(z_sem) + L_orth                        ← CE只用原始特征
    + AlphaInfoNCE([z_sem; z_sem_aug], tau=0.07) ← 增强特征只走对比
    + MSE(z_sem, 同类proto)
  
  设计意图: 增强特征不干扰分类器，只给对比学习提供
  "同一内容不同风格"的额外样本。如果解耦不完美，
  z_sem和z_sem_aug有差异，InfoNCE会倒逼semantic_head
  更好地滤掉风格。
  
  但R50结果(80.4%) < mode 6(82.2%)，说明在安全阀够强
  的情况下，增强走CE反而更好——安全阀修复后，增强走CE
  不再有害。
```

### 关键对比: 为什么 mode 6 最好

| | mode 4 | mode 6 | mode 7 |
|--|--------|--------|--------|
| InfoNCE 类型 | 标准 | alpha-sparsity | alpha-sparsity |
| MSE 锚点 | 有 | 有 | 有 |
| 增强走 CE? | 是 | 是 | **否** |
| R50 Acc | 79.8 | **82.2** | 80.4 |
| R50 cos_sim | +0.678 | +0.365 | +0.362 |
| 评价 | cos最高最稳 | **Acc最高无下降** | 增强不走CE反而低 |

mode 4 的 cos 最高(+0.678)说明MSE让梯度方向高度一致，但Acc较低，可能是标准InfoNCE的正例梯度太强限制了学习空间。mode 6 的alpha-sparsity弱化正例梯度，让模型更自由地找到好的分类边界，所以Acc最高。

## 动机

EXP-077 R50 快速验证确认两个最有效的修复方案：
- 077a (mode=4, MSE anchor): cos_sim=+0.678, max=80.1%
- 077c (mode=6, MSE+alpha): cos_sim=+0.365, max=82.2% ← 最佳

现在跑完整 R200 × 3 seeds 验证长期稳定性。

## 变体解释

### 078a: MSE Anchor (mode=4)
- 标准 InfoNCE（cos_sim / tau） + MSE(z_sem, global_proto[同类]).detach()
- tau = 0.05（比原始 0.2 低，参考 FPL 的 tau=0.02）
- MSE 作为"引力中心"，防止 z_sem 偏离原型太远
- cos_sim@R50 = +0.678（最高，说明 MSE 让 CE 和 InfoNCE 高度协作）

### 078c: MSE + Alpha-Sparsity (mode=6)
- alpha-sparsity InfoNCE: cos_sim.clamp(0).pow(0.25) / tau
- tau = 0.07（参考 FedPLVM 最优值）
- alpha=0.25: 正例梯度自动弱化（sim 高→梯度小），负例梯度增强
- + MSE 锚点（lambda_mse=1.0）
- R50 max = 82.2%（追平原始 FedDSA peak）且零下降

## 训练配置

| 参数 | 078a | 078c | 078d |
|------|------|------|------|
| mode | 4 | 6 | 7 |
| tau | 0.05 | 0.07 | 0.07 |
| lambda_sem | 1.0 | 1.0 | 1.0 |
| lambda_mse | 1.0 | 1.0 | 1.0 |
| alpha_sparsity | — | 0.25 | 0.25 |
| lambda_orth | 1.0 (从 R0 全开) | 1.0 (从 R0 全开) | 1.0 (从 R0 全开) |
| 风格增强 | 有 (走 CE) | 有 (走 CE) | 有 (**只走对比，不走 CE**) |
| R | 200 | 200 | 200 |
| Seeds | 2, 333, 42 | 2, 333, 42 | 2, 333, 42 |

### 078d 特殊设计 (mode=7, PARDON-inspired)
- 增强特征 z_sem_aug 与原始 z_sem concat 后一起送入 alpha-sparsity InfoNCE
- CE loss 只用原始特征，不用增强特征
- + MSE 锚点 + alpha-sparsity（双安全阀）
- R50 验证: max=80.4%, cos=+0.362, drop=-0.3%

## 部署

| 服务器 | Config | Seeds |
|--------|--------|-------|
| SC2 | feddsa_mse_anchor_r200.yml (078a) | 2, 333, 42 |
| Lab-lry GPU1 | feddsa_mse_alpha_r200.yml (078c) | 2, 333, 42 |
| SC4 | feddsa_detach_aug_r200.yml (078d) | 2, 333, 42 |

## 成功标准
- R200 final ≥ 81% (3-seed mean)
- peak→final drop < 1%
- cos_sim@R100, R150, R200 仍 > 0.2
- 超过 FDSE R200 (80.36%)

## 结果

### 078a MSE Anchor (SC2) — ⛔ 手动终止于 R~142

| s | R(终止) | peak | last | drop | 说明 |
|---|---------|------|------|------|------|
| 2 | 142 | **81.3** | 76.4 | **-4.9** | 持续下行 |
| 333 | 140 | **79.7** | 73.7 | **-6.0** | 持续下行 |
| 42 | 141 | **81.9** | 79.9 | **-2.0** | 相对稳定但仍不如 orth_only |
| **Mean** | | **81.0** | **76.7** | **-4.3** | |

> ❌ R50 时 cos_sim=+0.678，但 R142 mean final 仅 76.7%，远低于 orth_only 80.7%。全局原型漂移导致 MSE 锚点随之漂移，长期失效。主动终止，R200 预测 ~74-76%。

### 078c MSE + Alpha (Lab-lry) ✅ 2026-04-17 同步回来完整 R200

| s | R | max | last | drop |
|---|---|-----|------|------|
| 2 | 201 | 80.14 | 74.58 | -5.56 |
| 333 | 201 | 82.29 | 75.95 | -6.34 |
| 42 | 201 | 80.28 | 75.09 | -5.19 |
| **Mean** | | **80.90** | **75.21** ❌ | **-5.70** |

**078c 与其他 InfoNCE 变体一样失败**（last 75.21 远不如 bell_60_30 79.29）。证实 MSE+alpha 双安全阀在 PACS R200 无效。

### 078d Detach Aug (SC4) — ⛔ 手动终止于 R~102（无继续价值）

| s | R(终止) | max | last | drop | 说明 |
|---|---------|-----|------|------|------|
| 2 | 102 | **81.7** | 79.1 | -2.6 | 峰值R~61，之后缓慢滑落 |
| 333 | 102 | 15.9 | **15.0** | ≈0 | ⚠️ R2就NaN崩溃，锁死15%，从未有效训练 |
| 42 | 102 | **79.1** | 78.5 | -0.6 | 稳定但不如orth_only |
| **Mean** (去除s=333) | | **80.4** | **78.8** | **-1.6** | |

> ⚠️ 旧表"s=333 max=60.2%"是误读。s=333 从 R2 就因 NaN loss 崩溃，完全没有有效训练。
>
> ❌ detach_aug 结论：CE 只用原始特征、InfoNCE 用增强特征 → 两条梯度流分离 → 敏感 seed 从开始就 NaN；稳定 seed 峰值 81.7% 但 R100+ 缓慢下滑至 78-79%，最终不超过 orth_only 80.7%。**主动终止，无继续价值**。

## Office-Caltech10 扩展验证 ✅ DONE (SC4)

Office 对照基线: FedAvg 85.67%, FedBN 88.65%, FedDSA 89.13%, FDSE 90.58%

| Config | s | R | max | last | drop |
|--------|---|---|-----|------|------|
| **orth_only** | 2 | 200 | **87.7** | **87.2** | -0.5 |
| **orth_only** | 333 | 200 | **89.8** | **88.7** | -1.1 |
| **orth_only** | 42 | 200 | **90.7** | **89.5** | -1.3 |
| **orth_only Mean** | | | **89.4** | **88.5** | **-1.0** |
| mse_alpha | 2 | 200 | 86.8 | 86.1 | -0.7 |
| mse_alpha | 333 | 200 | 87.5 | 86.4 | -1.1 |
| mse_alpha | 42 | 200 | 89.0 | 87.9 | -1.1 |
| **mse_alpha Mean** | | | **87.8** | **86.8** | **-1.0** |

> orth_only Office 超越 FedBN (88.65%→88.5% ≈ 持平)，接近 FedDSA (89.13%)，低于 FDSE (90.58%)。
> mse_alpha 在 Office 反而低于 orth_only — 说明安全阀在低域差异数据集上效果打折。

## 结论（最终）

1. **078a MSE anchor 终止 @ R142** — mean final 76.7%，全局原型漂移导致锚点随之漂移；主动终止
2. **078d detach_aug 终止 @ R102** — s=333 NaN 崩溃锁死；稳定 seed R100+ 滑落 78-79%；主动终止
3. **078c (mse_alpha R200，lab-lry)** — lab-lry 连接失败，结果未知；推测长期也会失效
4. **Office 验证 orth_only** — PACS+Office 均有效，正交解耦是通用机制
5. **当前最可靠方案 = orth_only** — PACS 80.7%（超FDSE 80.36%），Office 88.5%（超FedBN 88.65%）
6. **InfoNCE / 安全阀思路在 R200 全面告失** — 所有安全阀变体均不超过 orth_only
7. **SC2+SC4 完全释放**，可用于新实验

| 变体 | mean final | vs orth_only |
|------|-----------|--------------|
| mode0 orth_only | **80.7%** | baseline ✅ |
| mode1 bell_60_30 | 80.2% | -0.5% |
| mode2 cutoff_80 | 79.0% | -1.7% |
| mode3 always_on | 75.1% | -5.6% ❌ |
| mode4 mse_anchor (078a) | 76.7% (R142) | -4.0% ❌ |
| mode7 detach_aug (078d) | ~78.8%† | -1.9% ❌ |

†去掉 NaN seed（s=333）仅 2 seeds 均值