# EXP-125 OCSD Verify 1 — Summary

**Checkpoint**: `/root/fl_checkpoints/sgpa_PACS_c4_s2_R200_1776795770`
**Config**: seed=2, VIB, use_whitening=1, use_centers=1
**Model**: FedDSAVIBModel (VIBSemanticHead, eval mode = mu deterministic)

## Statistics (PACS Art, 2048 samples)

- Overall accuracy: 1894/2048 = **92.48%**
- Over-conf (>0.8) correct: 1829 (89.3%)
- Over-conf (>0.8) wrong: **51** (**2.5%**) ← OCSD target set

## Confusion (True → Predicted)

| True → Pred | Count |
|---|:-:|
| person → **dog** | 8 |
| dog → **elephant** | 5 |
| person → **elephant** | 3 |
| elephant → **dog** | 3 |
| person → **horse** | 3 |
| dog → **horse** | 3 |
| horse → **elephant** | 3 |
| dog → **person** | 2 |
| guitar → **person** | 2 |
| horse → **dog** | 2 |
| guitar → **dog** | 2 |
| dog → **guitar** | 2 |
| house → **person** | 2 |
| giraffe → **horse** | 1 |
| horse → **giraffe** | 1 |
| giraffe → **house** | 1 |
| guitar → **horse** | 1 |
| guitar → **house** | 1 |
| elephant → **house** | 1 |
| giraffe → **person** | 1 |
| horse → **person** | 1 |
| person → **house** | 1 |
| person → **giraffe** | 1 |
| guitar → **giraffe** | 1 |

## Next — 人眼看 50 张图

1. Open `top50_images/` — 逐张看
2. Fill `judgement_sheet.md` — A/B/C 标注
3. Count 分布决定 OCSD 方向生死:
   - **A 主导 (>60%)** → style shortcut 假设成立, **继续 OCSD**
   - **A ≈ B (40-60 each)** → 部分成立, OCSD 期望降低
   - **B/C 主导** → shortcut 假设失败, **放弃 OCSD 切 RCA-GM**
