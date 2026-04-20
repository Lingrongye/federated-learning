# EXP-104: Plan A + whitening + centers, **diag=0** Office R200 3-seed

**日期**: 2026-04-20 启动 / 待完成
**状态**: 🟡 部署中

## 大白话

> 跟 EXP-100 Linear+whitening 完全一样 (都启用 whitening + centers), 唯一区别: **不开诊断框架**. 看看 diag 本身是不是 gain 的隐藏来源.

## Claim

| Claim | 判定 |
|-------|------|
| diag 无副作用 | 本实验 ≈ EXP-100 88.75% |
| diag 有副作用 | 本实验 明显低于 EXP-100 (Δ ≥ 1%) |

## 配置

| use_etf | use_whitening | use_centers | diag |
|---------|---------------|-------------|------|
| 0 | 1 | 1 | **0** ← vs EXP-100 这里是 1 |

## 🏆 结果 (待回填)

| 配置 | AVG Best |
|------|---------|
| Plan A | 82.55 |
| EXP-100 (diag=1) | **88.75 ± 0.86** |
| **本实验 (diag=0, full uw=1 uc=1)** | **88.77 ± 0.67** |

**Per-seed (本实验)**:
| seed | ALL Best | AVG Best | AVG Last |
|------|---------|---------|---------|
| 2 | 80.57 | 87.90 | 85.85 |
| 15 | 83.34 | 88.87 | 86.21 |
| 333 | 86.10 | 89.54 | 87.84 |
| **mean** | 83.33 ± 2.26 | **88.77 ± 0.67** | 86.63 ± 0.87 |

**解读**: 本实验 88.77 vs EXP-100 88.75 → **Δ = +0.02, 几乎完全一致**, 证明 **diag=1 对训练过程无副作用** (无可训练梯度泄漏, 无 RNG 污染). ✅ **diag 安全, 未来 CDANN 等实验可放心开启 diag=1**

## 📎 相关

- Config: `FDSE_CVPR25/config/office/feddsa_full_nodiag_office_r200.yml`
