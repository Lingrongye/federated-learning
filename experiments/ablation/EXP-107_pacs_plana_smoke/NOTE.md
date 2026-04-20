# EXP-107: PACS Plan A smoke — 验证 whitening 磨风格假设

**日期**: 2026-04-20 部署
**算法**: `feddsa_sgpa` (所有新机制关闭, 等价 Plan A + diag 监控)
**服务器**: seetacloud2 GPU 0 (单 run)
**状态**: 🟡 DEPLOY

## 目的 (大白话)

EXP-098 PACS Linear+whitening 诊断出 **z_sty_norm R10→R200: 3.12→0.15 (-95%)**, 我推断是 whitening 在 PACS 把风格磨成 0.

**但这只是推断**. 可能是:
1. whitening 磨的 (我们的假设) ← 如果关掉 whitening 后 z_sty 保持, 假设成立
2. 训练本身就磨 z_sty (与 whitening 无关) ← 如果关掉后 z_sty 还是塌缩, 假设失败

这个 smoke 用 **同一算法 feddsa_sgpa 但关掉 whitening + centers + ETF**, 看 PACS 上 z_sty_norm 的轨迹.

## Claim (验证数据集边界)

| Claim | 判定 | 失败含义 |
|-------|------|---------|
| **z_sty 保持** | R200 z_sty_norm ≥ 1.5 (vs EXP-098 Linear+whitening 的 0.15) | whitening 不是磨风格的原因, 诊断结论要改 |
| **精度接近 Plan A** | AVG Best ≈ 81.69 (EXP-080 mean ± 2pp) | 代码等价性有问题 |
| **几何到极限** | R200 intra ≥ 0.99 inter ≤ -0.16 | Plan A 本身几何已满 |

## 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Task | PACS_c4 (7 类, 4 outlier) | |
| Backbone | AlexNet + 双 128d 头 | |
| algo_para | [1.0, 0.1, 128, 10, 1e-3, 2, **1**, **0**, **0**, **0**, 0, 0] | diag=1, use_etf=0, use_whitening=0, use_centers=0 |
| R / E / LR | 200 / 5 / 0.05 | PACS 惯例 |
| λ_orth | 1.0 | 正交解耦 (Plan A 唯一保留机制) |
| Seeds | {2} (smoke, 单 seed 先验证诊断) | |
| Config | `FDSE_CVPR25/config/pacs/feddsa_plana_pacs_r200.yml` | |

## 对照实验 (已有)

| 实验 | 配置 | AVG Best | z_sty R200 |
|------|------|----------|-----------|
| EXP-080 PACS Plan A (feddsa) | orth_only | **81.69** | 未监控 |
| EXP-098 PACS Linear+whitening (feddsa_sgpa) | use_etf=0 uw=1 uc=1 | 80.20 | **0.15** 🔴 |
| EXP-098 PACS Hard ETF (feddsa_sgpa) | use_etf=1 uw=1 uc=1 | 78.96 | 0.27 |
| **EXP-107 PACS Plan A smoke (本实验)** | use_etf=0 uw=0 uc=0 | 待填 | **待填** |

## ⚠️ 已知风险

1. **文件名过长**: 12 algo_para 拼接预计 278 字节 > 255, record JSON 可能保存失败
   - 对策: rescue 脚本已就绪 (`scripts/rescue_record_from_log.py`)
2. **单 seed 噪声**: smoke 用 seed=2, 若诊断数据明确则够, 若边界 fuzzy 需扩 seed

## 📊 实验统计

- **总 runs**: 1 (seed=2)
- **预估 wall**: ~2.5h (E=5 R200 单卡)
- **启动**: 2026-04-20 待部署
- **完成**: 待填

## 📎 相关文件

- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (use_whitening=0 路径)
- Config: `FDSE_CVPR25/config/pacs/feddsa_plana_pacs_r200.yml`
- 诊断分析: `obsidian_exprtiment_results/2026-04-20/诊断指标分析_数据集边界证据.md`
- Log (将产生): `task/PACS_c4/log/*2026-04-20-*|1|0|0|0|0|0M*_S2_*.log`
- Diag (将产生): `task/PACS_c4/diag_logs/R200_S2_plana/`
