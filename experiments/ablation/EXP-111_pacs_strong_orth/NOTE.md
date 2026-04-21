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

## 结果 (accuracy ✅, probe 待回填)

### Accuracy (seed=2)

| λ_orth | AVG Best | Round | Last | Photo | Art | Cartoon | Sketch |
|-------|---------|-------|------|-------|-----|---------|--------|
| 1 (EXP-109 s=2) | 0.8223 | R181 | 0.8141 | 0.667 | 0.889 | 0.838 | 0.895 |
| **3** | **0.8133** | R166 | 0.8103 | 0.672 | 0.850 | 0.838 | 0.893 |
| **10** | **0.8103** | R180 | 0.8031 | 0.632 | 0.850 | 0.850 | 0.908 |
| Δ (lo=10 − lo=1) | -1.2pp | | -1.1pp | -3.5pp | -3.9pp | +1.2pp | +1.3pp |

**趋势**: 加强正交 → accuracy 单调轻伤 (82.23 → 81.33 → 81.03),大约 1pp 每 10× lo。
**per-client**: Art 和 Photo 客户端掉得较多,Sketch 和 Cartoon 小幅改善。

### Probe Results (capacity probe,待回填完成)

checkpoints:
- orth3 s=2: `sgpa_PACS_c4_s2_R200_1776735738`
- orth10 s=2: `sgpa_PACS_c4_s2_R200_1776735958`

probe 批处理 `/tmp/probe_all_new_pacs.sh` 已启动 (2026-04-21 早上),结果会自动存到 `capacity_probes/orth3_s2.json` 和 `capacity_probes/orth10_s2.json`。
