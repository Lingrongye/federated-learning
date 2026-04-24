# EXP-125 OCSD Verify 1 v2 (test only)

**Checkpoint**: /root/fl_checkpoints/sgpa_PACS_c4_s2_R200_1776795770
**Client 0 test split**: 204 samples (consistent with training split)

## Stats

- Acc: 124/204 = **60.78%**
- Over-conf correct: 97 (47.5%)
- **Over-conf WRONG**: **24** (11.8%)

## Confusion

| True → Pred | Count |
|---|:-:|
| person → **dog** | 5 |
| dog → **elephant** | 3 |
| guitar → **person** | 2 |
| horse → **elephant** | 2 |
| horse → **dog** | 2 |
| guitar → **dog** | 2 |
| house → **person** | 2 |
| giraffe → **horse** | 1 |
| horse → **giraffe** | 1 |
| guitar → **horse** | 1 |
| elephant → **house** | 1 |
| person → **giraffe** | 1 |
| person → **horse** | 1 |

## Decision

人眼看 top 50 → 统计 A/B/C:
- A 主导 (>60%) → shortcut 假设成立, 继续 OCSD
- A ≈ B → 部分成立
- B/C 主导 → 放弃 OCSD, 切 RCA-GM
