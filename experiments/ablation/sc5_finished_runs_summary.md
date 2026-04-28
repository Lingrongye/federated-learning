---
date: 2026-04-28
type: sc5 实验完成情况汇总 (kill 前快照)
status: sc5 已 kill 全部 4 个进程, 数据冻结
---

# sc5 已完成 / 部分完成实验汇总

## ⚠️ 重要说明: 全部 sc5 实验**均无 diag dump**

sc5 上所有实验都在 diagnostic hook 上线 (commit `b553edb`, 2026-04-28) **之前** launch。
这些数据**只能用于主表 final acc 回填, 不能用于 cold path 衍生指标分析** (cos sim trajectory / DaA dispatch / t-SNE 等)。

如需 paper-grade 诊断分析, 必须重跑 (V100/sc6 上有 diag dump 的实验)。

---

## EXP-133: PG-DFC + DaA (sc5 部分)

| Run | R/100 | best | last (R99/R42/R40) | 状态 |
|---|:--:|:--:|:--:|:--:|
| **pgdfc_daa_office_s15** | **100 ✅** | **63.80** | 59.29 | **完成** ⭐ |
| pgdfc_daa_pacs_s15 | 43 | 66.72 | 65.96 | 已 kill (重复 sc6) |
| pgdfc_daa_pacs_s333 | 41 | 60.44 | 58.93 | 已 kill (重复 sc6 greedy 排队) |

### 唯一完成的: PG-DFC+DaA office s=15
- best = **63.80** (R~ in middle)
- last (R99) = 59.29 — best vs last gap 4.5pp (训练后期 drift, 跟我们之前观察 office gap 5.2pp 一致)

### 对比基准 (主表数据)
| Method | Office best (s=15) | Δ vs vanilla PG-DFC 61.25 |
|---|:--:|:--:|
| vanilla PG-DFC office (主表) | 61.25 | baseline |
| **PG-DFC + DaA office s=15 (sc5)** | **63.80** | **+2.55pp** ⭐ |
| F2DC + DaA office s=15 (sc3) | 63.93 | +3.37 vs vanilla F2DC 60.56 |
| FDSE office (主表) | 63.52 | (PG-DFC+DaA 已超 FDSE +0.28pp) |

→ **DaA transfer 到 PG-DFC 成功** (推翻之前 R67 中间 snapshot 的悲观判断)

### 缺 cross-seed (s=333)
- 主表 PG-DFC+DaA office 还差 s=333 (kill 时未完成)
- 等 V100 后 launch s=333 凑 2-seed mean

---

## EXP-132: Baselines (FedBN/FedProx/FedProto)

**全部 R100 完成**:

| Method | PACS s=15 | PACS s=333 | Office s=15 | Office s=333 | Digits s=15 | Digits s=333 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| FedBN | 63.54 | 60.39 | 54.57 | 57.71 | 91.19 | 90.43 |
| FedProx | 63.16 | 63.15 | 55.37 | 59.49 | 92.17 | 91.72 |
| FedProto | 62.59* | 60.13* | 59.18 | 60.62 | 92.81 | 92.26 |

*FedProto PACS s=15/s=333 R=41 时被 kill (memory leak 卡死), 用 best so far 数字

**这些已经回填到主表 PG-DFC对比基线主表_完整结果.md 的 Table 1/2/3 baseline 行**, 不需要重新回填。

---

## 总结

| 类别 | 完成 | 部分完成 | 失败 |
|---|:--:|:--:|:--:|
| EXP-132 baselines | 10 个 R100 | 2 个 (fedproto pacs) | 0 |
| EXP-133 PG-DFC+DaA | 1 (office s=15) | 2 (pacs s=15/s=333 kill 时 R=41/43) | 0 |
| **total** | **11** | **4** | **0** |

## 已 kill 进程 (释放 sc5)

PID 90385 / 90387 / 91020 / 91451 全部 kill, sc5 GPU 完全闲置 (24GB free).

## 后续

- ✅ EXP-132 数据已在主表
- ⭐ **EXP-133 PG-DFC+DaA office s=15 (63.80) 需要补到主表**, 标注 +2.55pp vs vanilla PG-DFC
- ⏳ 等 sc6 + V100 完成的 PG-DFC+DaA PACS 数据 + office s=333 后, **整体重写主表"PG-DFC+DaA"行**
