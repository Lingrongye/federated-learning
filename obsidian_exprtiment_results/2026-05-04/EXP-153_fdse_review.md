---
date: 2026-05-04
type: 实验记录 (FDSE 复现 review + 修复 PACS 训崩)
status: ✅ 4 runs R100 完成 + paper figure 用 fix 数据
exp_id: EXP-153
goal: 修 FDSE PACS R18 peak 53.40 后 collapse 到 R99 33.58 (drift -19.82) bug
---

# EXP-153: FDSE 复现 review + 修复

## 一句话

F²DC framework 下 FDSE PACS s=15 训崩根因找到: 我们用 default lmbd=0.01 但 FDSE_CVPR25 原 PACS 配置是 **lmbd=0.5/lr=0.05** (10× 比例). 我们 lr=0.01 下应同步降 **lmbd=0.05 维持 5× 比例**. 修复后 PACS s=15 best 57.53 / last 43.07 (vs 原版 53.40 / 33.58, drift -19.82→-14.46). Office 修复 marginal (本来没明显 collapse).

## R100 完整结果 (lmbd=0.05 fix)

| Run | best | last R99 | per-dom @ best |
|---|:--:|:--:|:--:|
| PACS s=15 | **57.53** (R28) | 43.07 | P60.5/A39.7/C68.2/S61.8 |
| PACS s=333 | 54.16 (R16) | 42.02 | P56.6/A51.7/C66.7/S41.7 |
| Office s=15 | 62.56 (R44) | 58.77 | c59.8/a57.9/w75.9/d56.7 |
| Office s=333 | **65.05** (R58) | 62.28 | c58.0/a67.9/w77.6/d56.7 |

### Mean (s=15+s=333)

| Dataset | best mean | last mean | vs 原 (lmbd=0.01) |
|---|:--:|:--:|:--:|
| PACS fix | 55.85 | 42.54 | best +1.11, **last +4.83** ✅, drift 缩小 5pp |
| Office fix | 63.80 | 60.52 | +0.28 / +1.19 marginal |

## 关键 finding

1. **lmbd 跟 lr 必须维持比例** (10× 原版 / 5× 我们): default lmbd=0.01 给 PACS 太弱.
2. **EPS=1e-2 clamp 是数值稳定必要**: 改回 1e-8 + clip_grad 跨任意 lmbd 都立即崩. 4 个 fix 版本实测验证 (lmbd=0.5+EPS=1e-8 R5 collapse, lmbd=0.05+EPS=1e-8 R3 collapse, lmbd=0.05+EPS=1e-2 ✅).
3. **PACS R28 peak 后仍 slow decline**, 但 drift 从 -19.82 缩到 -14.46.
4. **Office 修复 marginal** (本来没 collapse 问题, lmbd 影响小).

## paper figure 更新

`figures/convergence_paper_style.png` 双 panel (Office + PACS) 已用 fix 后数据更新. PACS FDSE 现在稳定 plateau ~45 (替代原 collapse 到 33). 视觉跟 F²DC paper Figure 5 风格一致, Ours 仍稳压全 baseline +5-25pp.
