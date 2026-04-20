# EXP-109: PACS orth_only (ca=0) 反事实 probe 3-seed

**创建日期**: 2026-04-21

## 一句话

把 EXP-108 的 CDANN config 关掉 CDANN (ca=1 → ca=0),其他全部一致,跑 3-seed。拿到 checkpoint 后和 EXP-108 CDANN 做 probe 对比,回答"CDANN 是不是真的压了 probe_sty_class 的非线性泄漏"。

## 背景

EXP-108 发现 Office CDANN `probe_sty_class MLP ≈ 0.91`。Codex R2 预言:由于主干共享 (shared-trunk),CDANN 的 GRL on z_sem 只是做了一层"皮",不能把类别信息从主干挤走。MLP probe 仍然能从 z_sty 读出 class,这和 Codex 预言一致。

但缺失对照:**如果完全没有 CDANN,probe_sty_class 是多少?** 如果也是 0.91,说明 CDANN 没贡献任何 probe 压制,完全是双头架构造成的。

EXP-107 smoke 曾跑过 PACS Plan A smoke (lo=1 whitening, se=0) 得到 z_sty_norm = 0.1461。但没保存 checkpoint,无法 probe。

## 配置

```
config: FDSE_CVPR25/config/pacs/feddsa_baseline_pacs_saveckpt_r200.yml
algo_para: lo=1.0 te=0.1 pd=128 wr=10 es=1e-3 mcw=2 dg=1 ue=0 uw=1 uc=1 se=1 lp=0 ca=0
- ue=0 (Linear classifier)
- uw=1 (pooled whitening on)
- uc=1 (use_centers on)
- ca=0 ★ CDANN OFF (vs EXP-108 ca=1)
- se=1 ★ 保存 checkpoint
R200, E=5, B=50, LR=0.05, WD=1e-3, lr_decay=0.9998
seeds: 2 (PID 247627, 启动 03:00 之前), 15 (PID 255317, 03:00), 333 (PID 255604, 03:00)
```

与 EXP-108 CDANN config 唯一差异:`ca=1 → ca=0`。其他 100% 一致。

## 目的判据 (probe 跑完后)

| 判据 | 结论 |
|------|------|
| orth_only MLP probe_sty_class ≈ CDANN 的 0.91 (±2pp) | **CDANN 对 probe 毫无贡献**,所有泄漏都是双头架构 + 主干共享造成 |
| orth_only MLP 显著 > CDANN (如 0.95+) | CDANN 压了一点点非线性泄漏 (但不够) |
| orth_only MLP 显著 < CDANN | 不符合预期 (不应发生,除非 CDANN 注入了噪声) |

## 附带数据

- AVG Best: 和 EXP-108 CDANN 比较看 CDANN 是否有 accuracy 增益
- EXP-080 orth_hparam_sweep 里 PACS orth_only lo=1 single-seed 是 81.69。3-seed mean 会是多少?

## 结果 (待回填)

| Seed | AVG Best | ALL Best | Sketch | Art | Photo | Cartoon | probe_sty_class lin/mlp | probe_sem_class lin/mlp | 备注 |
|------|---------|----------|--------|-----|-------|---------|------------------------|------------------------|------|
| 2 | TBD | | | | | | | | PID 247627 (最长,已 3h+) |
| 15 | TBD | | | | | | | | PID 255317 |
| 333 | TBD | | | | | | | | PID 255604 |
