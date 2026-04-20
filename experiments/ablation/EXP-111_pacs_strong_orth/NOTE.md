# EXP-111: PACS 强正交 λ_orth ∈ {3, 10} × seed=2

**创建日期**: 2026-04-21

## 一句话

加强双头解耦的正交损失权重,从 lo=1 → 3 → 10,看 probe_sty_class 非线性泄漏能不能被压下去。如果能压 → 强正交是可行方向;如果不能 → 主干共享是结构瓶颈,正交损失无力。

## 背景

EXP-108 CDANN + EXP-109 反事实将告诉我们"额外的 CDANN 损失"没法压 probe 泄漏。那"加强已有的正交损失"能不能?

L_orth = E[cos²(z_sem, z_sty)] 是双头分开的唯一机制。现在 lo=1。**如果 lo=10 仍然压不住 probe,那就证实了"loss-based branch purification is weak"**。

## 配置

```
PACS config: lo=3.0 (feddsa_pacs_orth3_saveckpt_r200.yml)
            lo=10.0 (feddsa_pacs_orth10_saveckpt_r200.yml)
其余同 EXP-109 (ca=0, uw=1, uc=1, se=1)
seeds: 2
启动: 2026-04-21 03:00-03:10
PIDs: orth3 256060, orth10 256736
```

## 判据

| lo=10 的 probe_sty_class MLP | 结论 |
|------------------------------|------|
| 明显 < 0.80 | **强正交有效,方向明确**。下一步: 3-seed 稳健性 + lo 扫描上限 |
| 0.80 - 0.90 | 有部分效果,但离"纯化"还远 |
| ≈ 0.91 (同 lo=1) | **正交损失无力压非线性泄漏**。confirm Codex "shared-trunk" 诊断 |

同时看 accuracy:
- lo=10 可能过度抑制 z_sem 类信息 → accuracy 掉很多
- 如果 lo=10 probe 掉且 acc 不掉 → 理想情况 (但 unlikely)
- 如果 lo=10 probe 不掉且 acc 掉 → 最差情况 (确认无路)

## 结果 (待回填)

| λ_orth | AVG Best | ALL Best | Sketch | probe_sty_class lin/mlp | probe_sem_class lin/mlp | cos_sim(z_sem, z_sty) |
|-------|---------|----------|--------|------------------------|------------------------|----------------------|
| 1 (EXP-109 s=2) | TBD | | | | | 约 0? |
| 3 | TBD | | | | | |
| 10 | TBD | | | | | |
