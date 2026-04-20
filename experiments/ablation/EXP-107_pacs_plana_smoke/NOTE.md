# EXP-107: PACS Plan A smoke — 验证 whitening 磨风格假设

**日期**: 2026-04-20 12:13 部署 / 2026-04-20 ~19:30 完成 (R200 ~7h wall)
**算法**: `feddsa_sgpa` (所有新机制关闭, 等价 Plan A + diag 监控)
**服务器**: seetacloud2 GPU 0 (与其他 run 共享)
**状态**: ✅ **完成. 🔴 假设证伪: whitening 不是磨风格的原因, 训练本身就磨 (和 baseline 一样塌 95%)**

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

## 🏆 完整结果 (2026-04-20 回填)

### AVG / ALL accuracy (seed=2, R200 完成)

| 指标 | 值 | 说明 |
|------|-----|------|
| N rounds | 201 | ✅ 完整完成 |
| AVG Best | **80.47** @R83 | |
| AVG Last (R200) | **79.88** | |
| ALL Best | **82.24** @R186 | |
| ALL Last | **82.14** | |

### z_sty_norm 轨迹 (🔴 关键发现)

| Round | z_sty_norm |
|-------|-----------|
| R10 | **3.1244** (启动稳定) |
| R50 | 1.3000 (已降 58%) |
| R100 | 0.4512 (降 86%) |
| R150 | 0.2319 |
| R200 | **0.1461** (降 **95.3%**) |

## 对照实验汇总

| 实验 | 配置 | AVG Best | z_sty R10 | z_sty R200 | z_sty 塌缩 % |
|------|------|----------|-----------|------------|-------------|
| EXP-080 PACS Plan A (feddsa) | orth_only | **81.69 (3-seed mean)** | 未监控 | 未监控 | — |
| EXP-098 PACS Linear+**whitening** | use_etf=0 uw=1 uc=1 | 80.20 | 3.12 | **0.146** | **-95%** 🔴 |
| EXP-098 PACS Hard ETF | use_etf=1 uw=1 uc=1 | 78.96 | 3.12 | 0.27 | -91% |
| **EXP-107 PACS Plan A smoke (本, NO whitening)** | use_etf=0 **uw=0** uc=0 | **80.47** (seed=2) | **3.12** | **0.1461** | **-95%** 🔴 |

## 🔍 Verdict

**Claim 1 "z_sty 保持"**: ❌ **证伪** — 关掉 whitening 后 z_sty 依然塌 95% (R200=0.146, 与 EXP-098 Linear+whitening R200=0.146 几乎完全一致)

**Claim 2 "精度接近 Plan A"**: ✅ **成立** — EXP-107 AVG Best 80.47 ≈ EXP-080 Plan A 81.69 (在 ± 2pp 范围内, 且 EXP-107 仅 seed=2 单 seed 噪声)

**Claim 3 "几何到极限"**: 待查 (需要提 intra/inter_cls_sim R200)

## 📌 重大意义

**原 anchor claim "whitening-induced style collapse" 需要修正**:
- 假设 whitening 是罪魁祸首 → 实际 whitening 只是无辜的归一化操作
- **真正的 collapse 来源**: PACS 训练过程中 encoder 自然地将 z_sty 压缩至 0 (因为 CE loss 只在 z_sem → sem_classifier 起作用, L_orth 只要求正交不约束 z_sty 绝对 norm, 模型倾向于 "解耦即灭活 z_sty" 以简化优化)

**对 EXP-108 CDANN 的影响**:
- CDANN 的 z_sty_norm R100=10+ 证据仍成立, 但解释改为 "**CDANN 的正向 L_dom_sty 监督阻止了 PACS 训练自然的 z_sty 压缩**"
- 新 one-sentence novelty: "**Positive domain supervision on z_sty prevents training-induced style collapse in PACS, where style carries class signal.**"
- 不再依赖 "whitening 是元凶" 这个错误假设, paper 叙事更诚实

## ⚠️ 已知风险

1. **文件名过长**: 12 algo_para 拼接预计 278 字节 > 255, record JSON 可能保存失败
   - 对策: rescue 脚本已就绪 (`scripts/rescue_record_from_log.py`)
2. **单 seed 噪声**: smoke 用 seed=2, 若诊断数据明确则够, 若边界 fuzzy 需扩 seed

## 📊 实验统计

- **总 runs**: 1 (seed=2)
- **实际 wall**: ~7h (GPU 共享多 runs, 比单卡独占慢 3×)
- **启动**: 2026-04-20 12:13
- **完成**: 2026-04-20 ~19:30 (R200, log 记录完整)

## 📎 相关文件

- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (use_whitening=0 路径)
- Config: `FDSE_CVPR25/config/pacs/feddsa_plana_pacs_r200.yml`
- 诊断分析: `obsidian_exprtiment_results/2026-04-20/诊断指标分析_数据集边界证据.md`
- Log (将产生): `task/PACS_c4/log/*2026-04-20-*|1|0|0|0|0|0M*_S2_*.log`
- Diag (将产生): `task/PACS_c4/diag_logs/R200_S2_plana/`
