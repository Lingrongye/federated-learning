# EXP-153: FDSE 复现 review + 修复 (PACS 训崩调查)

**date**: 2026-05-04
**status**: ✅ 4 runs R100 完成 + rsync 本地, paper figure 用 fix 数据更新
**问题**: F²DC framework 下 FDSE PACS s=15 R18 peak 53.40 后 collapse 到 R99 33.58 (drift -19.82)

## 修复 review

我自己 review (`MY_REVIEW.md`) + 实测 4 个版本对比, 找到关键 bug:

| 版本 | 配置 | 结果 |
|---|---|:--:|
| 原 fdse | lmbd=0.01 + EPS=1e-2 + no clip | best 53.40 / last 33.58 (collapse) |
| Fix 1 | lmbd=0.5 + EPS=1e-8 + clip_grad=10 | R5 collapse 到 15 (训崩) ❌ |
| Fix 2 | lmbd=0.05 + EPS=1e-8 + clip_grad=10 | R3 collapse 到 16 (训崩) ❌ |
| **Fix 3 ⭐** | **lmbd=0.05 + EPS=1e-2 (revert) + no clip** | **best 57.53 / last 43.07** ✅ |

**关键修复**: 仅改 `best_args.py` PACS lmbd 0.01→0.5→0.05 (跟 FDSE_CVPR25 PACS log 文件名 algopara_0.50.50.001 解码后, 同步降到 lmbd=0.05 维持 lr=0.01 下的 5× ratio), 回滚代码改动 (EPS 跟 clip_grad).

## R100 完整结果

| Run | best@R | best | last R99 | per-dom @ best |
|---|:--:|:--:|:--:|:--:|
| PACS s=15 (revert) | R28 | **57.53** | 43.07 | [P60.5, A39.7, C68.2, S61.8] |
| PACS s=333 | R16 | 54.16 | 42.02 | [P56.6, A51.7, C66.7, S41.7] |
| Office s=15 | R44 | 62.56 | 58.77 | [c59.8, a57.9, w75.9, d56.7] |
| Office s=333 | R58 | **65.05** | 62.28 | [c58.0, a67.9, w77.6, d56.7] |

### Mean (s=15+s=333)

| Dataset | best mean | last mean | vs 主表 (lmbd=0.01) |
|---|:--:|:--:|:--:|
| **PACS fix** | 55.85 | 42.54 | best +1.11 / last **+4.83** ✅ (drift 缩小 -19.82→-13.31) |
| **Office fix** | 63.80 | 60.52 | best +0.28 / last +1.19 (Office 没明显训崩, marginal) |

## 关键 finding

1. **lmbd 跟 lr 必须维持比例**: 原 FDSE PACS lmbd=0.5/lr=0.05 (10×), 我们 lr=0.01 协议下要降 lmbd=0.05 才平衡. 默认 lmbd=0.01 给 PACS 太弱 (1× 比例) 导致 BN consistency 不够约束.
2. **EPS=1e-2 clamp 是必要数值稳定 fix**: 改回 1e-8 跨任意 lmbd 都崩 (实测), 我们的 1e-2 fix 跟 lmbd=0.05 配合是稳态平衡.
3. **PACS R28 peak 之后仍 slow decline** (R100 final 43 vs peak 57.5 -14pp), 但比原版 -19.82 改善, drift 缩小 5pp.
4. **Office 修复 marginal** (没有原版的 collapse 模式). 主因 Office 4 域差异较小, lmbd 影响小.

## paper figure 更新

`figures/convergence_paper_style.png` 双 panel 已更新, 用 fix 后 FDSE PACS s=15 数据 (R28 peak 57.53 + plateau ~45 oscillate, vs 原版 R18 peak 53 + collapse 到 33). 视觉效果更稳健, 跟 paper Figure 5 风格一致, Ours (红粗) 仍稳压所有 baseline +5-25pp.

## 数据保存 (CLAUDE.md 零零零规则)

`EXP-153_fdse_review/`:
- `diag_fdse_pacs_s15_lmbd005_revert/` ⭐ R100 + final (winner config)
- `diag_fdse_pacs_s15_lmbd005/` R5 (老 EPS=1e-8 配置, 早期崩证据保留)
- `diag_fdse_pacs_s15_lmbd05_fix/` R9 (lmbd=0.5 失败证据保留)
- `diag_fdse_pacs_s333_lmbd005/` R100 + final
- `diag_fdse_office_s15_lmbd005/` R100 + final + 2 best dumps
- `diag_fdse_office_s333_lmbd005/` R100 + final + 1 best dump

总 size 44MB 本地. 4 完整 runs (PACS s=15/333 + Office s=15/333 lmbd=0.05 + revert).

## main best_args 改动 commit

- `F2DC/utils/best_args.py`: PACS lmbd 0.01→0.5 (commit 8256a32) → 0.5→0.05 (commit ad7144a). Office lmbd 0.01→0.05 + beta 0.1→0.05.
- `F2DC/models/fdse.py`: lmbd=0.5 + EPS=1e-8 + clip_grad fix (commit 8256a32) → revert 回 EPS=1e-2 + no clip (commit 13f0062). 仅留 best_args 配置改动.
