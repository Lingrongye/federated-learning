# EXP-124 | FedDSA-PCH | Per-Cell Hardness Pilot

## 基本信息
- **日期**: 2026-04-24 启动 (基于 EXP-123 Stage B 诊断)
- **算法**: `feddsa_pch` (extends feddsa_scheduled sm=0 orth_only base)
- **服务器**: lab-lry GPU 1 (RTX 3090 24GB, 当前空闲)
- **状态**: 🟡 代码完成 + 单测通过, 等 push 到 lab-lry smoke + pilot run

## 这个实验做什么 (大白话)

**验证 "per-cell hardness re-weight" 方向**:
- EXP-123 Stage B 诊断: FDSE 的 +2.31pp 优势 ~90% 来自 3 个 hard cells
  - (Art, guitar): FDSE 赢 FedBN +24.33pp
  - (Art, horse): +15.52pp
  - (Photo, horse): +17.50pp

**假设**: 如果我们让 orth_only 训练时对这些 hard cells 的样本给 **2x CE loss weight**, model 会在这些 cell 上学更多 → 接近 FDSE 水平

**为什么是 pilot (不是 adaptive)**:
- 硬编码 hard cells 就是"完美先验知识" — 如果这都打不过 baseline, 说明 per-cell 方向错了
- 硬编码过了再做 adaptive (online tracker + dynamic weight)

## 方法细节

### Hard cells 映射 (from EXP-123 Stage B)

```python
HARD_CELLS = {
    0: {3, 4, 6},  # Client 0 = Art: guitar, horse, person
    2: {4},        # Client 2 = Photo: horse
}
```

PACS 顺序:
- Client 0 = Art (art_painting)
- Client 1 = Cartoon (no hard cells, 正常 CE)
- Client 2 = Photo
- Client 3 = Sketch (no hard cells)

### Loss 改动

```python
# 在 feddsa_scheduled.Client.train() 的 self.loss_fn(output, y) 处
# 替换为 per-sample weighted CE:
if cid in HARD_CELLS:
    hard = HARD_CELLS[cid]
    w = ones(B)
    for c in hard: w[labels == c] = hw  # hw=2.0
    loss = (F.cross_entropy(logits, y, reduction='none') * w).mean()
else:
    loss = F.cross_entropy(logits, y)  # 正常 CE
```

### 超参

- `hw = 2.0` (hard weight multiplier, 保守默认值)
- 其他全继承 feddsa_scheduled orth_only (sm=0, lo=1, tau=0.2, pd=128)

## 预期

| 场景 | orth_only 变化 | 判断 |
|:--:|:-:|:-:|
| (Art, guitar) 43.94 → 55+ | Art domain +0.5pp | 方向成立 |
| (Art, horse) 50.85 → 55+ | Art +0.3pp | ↑ |
| (Photo, horse) 55.86 → 60+ | Photo +0.3pp | ↑ |
| **AVG Best 涨 >+0.8pp** | 79.95 → **80.7+** | 🟢 验证方向 |
| 涨 < +0.3pp | 几乎没变化 | 🟡 per-cell 方向弱 |
| 下降 | hw 过大或诊断错 | 🔴 killed |

## 部署计划

1. ✅ 代码写完 (algorithm/feddsa_pch.py, 46 行, 继承 feddsa_scheduled)
2. ✅ 单测 7/7 通过 (weighted CE 正确性 + 梯度流)
3. ⏳ git commit + push
4. ⏳ lab-lry git pull → smoke R=5 (1 seed, ~5min 验证不崩)
5. ⏳ pilot R=200 × seed=2 on lab-lry GPU 1 独占 (~2.5h)
6. ⏳ 分析 seed=2 结果:
   - 若 AVG_B ≥ 80.5 → 扩展 3 seeds (seed=15, 333)
   - 若 AVG_B < 80 → kill + 调整 hw 或方向

## 相关文件

- `FDSE_CVPR25/algorithm/feddsa_pch.py` — 算法实现
- `FDSE_CVPR25/config/pacs/feddsa_pch_hw2_r200.yml` — 主 config
- `FDSE_CVPR25/config/pacs/feddsa_pch_smoke_r5.yml` — smoke config
- `FDSE_CVPR25/tests/test_feddsa_pch.py` — 7 单测
- 上游诊断: `experiments/ablation/EXP-123_art_diagnostic/stageB_full/ANALYSIS.md`

## 胜负判决

**pilot seed=2 R=200 判决**:
- AVG_B ≥ 80.5: 方向成立, 扩展 3 seeds
- AVG_B 80.0-80.5: 模糊, 看 Art/Photo per-class 是否涨
- AVG_B < 80.0: kill, 调 hw 或换方向

**3 seeds 完成后判决** (如扩展):
- 3-seed mean AVG_B > FDSE 81.54: 🏆 超 FDSE
- 3-seed mean AVG_B > orth 79.95: 🟢 per-cell 方向有效
- 3-seed mean AVG_B ≈ orth: 🔴 方向失败
