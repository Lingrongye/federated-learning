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
| EXP-100 (diag=1) | **88.75** |
| **本实验 (diag=0)** | 待填 |

**解读**: 若本实验 ≈ 88.75 → diag 无副作用, 安心用 diag; 若 < 88 → diag 引入可训练梯度 / RNG 污染 / 其他副作用, 需要审查 diag 代码

## 📎 相关

- Config: `FDSE_CVPR25/config/office/feddsa_full_nodiag_office_r200.yml`
