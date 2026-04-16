# EXP-078 | R200 完整验证 — MSE 锚点 + Alpha-Sparsity

## 基本信息
- **日期**: 2026-04-16
- **算法**: feddsa_scheduled.py (modes 4, 6)
- **服务器**: SC2 (078a) + Lab-lry GPU1 (078c)
- **状态**: 🔄 刚启动

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

| 参数 | 078a | 078c |
|------|------|------|
| tau | 0.05 | 0.07 |
| lambda_sem | 1.0 | 1.0 |
| lambda_mse | 1.0 | 1.0 |
| alpha_sparsity | — | 0.25 |
| lambda_orth | 1.0 (从 R0 全开) | 1.0 (从 R0 全开) |
| 风格增强 | 有 (AdaIN, warmup 后) | 有 |
| R | 200 | 200 |
| E | 5 | 5 |
| LR | 0.1 | 0.1 |
| Seeds | 2, 333, 42 | 2, 333, 42 |

## 部署

| 服务器 | Config | Seeds |
|--------|--------|-------|
| SC2 | feddsa_mse_anchor_r200.yml (078a) | 2, 333, 42 |
| Lab-lry GPU1 | feddsa_mse_alpha_r200.yml (078c) | 2, 333, 42 |

## 成功标准
- R200 final ≥ 81% (3-seed mean)
- peak→final drop < 1%
- cos_sim@R100, R150, R200 仍 > 0.2
- 超过 FDSE R200 (80.36%)

## 结果

### 078a MSE Anchor (SC2)
| s | R | max | last | drop | cos_sim |
|---|---|-----|------|------|---------|
| 2 | | | | | |
| 333 | | | | | |
| 42 | | | | | |

### 078c MSE + Alpha (Lab-lry)
| s | R | max | last | drop | cos_sim |
|---|---|-----|------|------|---------|
| 2 | | | | | |
| 333 | | | | | |
| 42 | | | | | |

## 结论
> 待实验完成后填入
