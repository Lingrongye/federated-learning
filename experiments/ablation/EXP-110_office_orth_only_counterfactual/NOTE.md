# EXP-110: Office orth_only (ca=0) 反事实 probe 3-seed

**创建日期**: 2026-04-21

## 一句话

把 EXP-108 Office CDANN 配置关掉 CDANN (ca=1→0),其他一致,跑 3-seed saveckpt。拿新 checkpoint 做 probe 对比,定位 Office 上 CDANN 是否对 probe_sty_class (MLP=0.91) 有任何压制作用。

## 背景

EXP-108 Office CDANN s=2 MLP probe = 0.906 / s=15 = 0.908。Codex 判 CDANN 无法结构性解决泄漏。需对照组证实。

## 配置

```
config: FDSE_CVPR25/config/office/feddsa_whiten_only_saveckpt_office_r200.yml
algo_para: lo=1.0 te=0.1 pd=128 wr=10 es=1e-3 mcw=2 dg=0 ue=0 uw=1 uc=0 se=1 lp=0 ca=0
- Office 用 uc=0 (与 EXP-108 CDANN Office config 一致)
- E=1 (Office 惯例)
R200, E=1, B=50, LR=0.05
seeds: 2 (PID 255844), 15 (PID 256308), 333 (PID 256522)
启动: 2026-04-21 03:00-03:05
```

与 EXP-108 Office CDANN 唯一差异:`ca=1→ca=0` + `dg=1→dg=0` (诊断关,不影响 probe)。

## 判据

同 EXP-109 类比:
- MLP probe ≈ 0.91 (±2pp) → CDANN 对 probe 无贡献
- 显著 > 0.91 → CDANN 压了一点

## 附带

- AVG Best 对照 EXP-102 whiten_only 89.26±0.83 看 3-seed 是否复现
- 对照 EXP-108 Office CDANN 89.54±0.49 看有无差异

## 结果 (待回填)

| Seed | AVG Best | ALL Best | A | C | D | W | probe_sty_class lin/mlp | probe_sem_class lin/mlp |
|------|---------|----------|---|---|---|---|------------------------|------------------------|
| 2 | TBD | | | | | | | |
| 15 | TBD | | | | | | | |
| 333 | TBD | | | | | | | |
