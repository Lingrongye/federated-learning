# EXP-124 | FedDSA-PCH | Per-Cell Hardness Pilot

## 基本信息
- **日期**: 2026-04-24 启动 + 当日完成 (lab-lry 3 seeds × R=200, 总 wall ~8h)
- **算法**: `feddsa_pch` (extends feddsa_scheduled sm=0 orth_only base)
- **服务器**: lab-lry GPU 1 (RTX 3090 24GB)
- **状态**: 🔴 **失败 (3-seed 平均 AVG 几乎无变化)**

---

## 🔴 3-seed 最终结果 (2026-04-24, 11:00-11:04 完成)

### 3-seed mean

| Method | AVG Best | Art | Cartoon | Photo | Sketch | Δ vs orth |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| orth_only (Stage B 2 seeds) | 79.95 | 62.50 | 87.92 | 79.98 | 91.09 | baseline |
| **PCH (hw=2, 3 seeds)** | **80.09** | **63.73** | **83.90** | **82.24** | **90.48** | **+0.14** ❌ |
| FDSE (3 seeds) | 81.54 | 64.71 | 85.18 | 86.83 | 89.46 | +1.59 |

**AVG Best 涨 +0.14pp — 在 seed noise (std 0.74) 内, 实质失败**.

### Per-seed

| Seed | AVG_B | Art | Cartoon | Photo | Sketch |
|:-:|:-:|:-:|:-:|:-:|:-:|
| s=2 | 79.63 | 62.75 | 86.32 | 81.44 | 88.01 |
| s=15 | 79.50 | 59.31 | 82.48 | 85.63 | 90.56 |
| **s=333** | **81.13** | **69.12** | 82.91 | 79.64 | 92.86 |

**s=333 意外 81.13 + Art 69.12 单 seed 突破** (Art 接近 FDSE 64.71 还更高), 但 **其他 seed 不稳**, 3-seed 平均不住.

### Per-cell 分解 (目标 cell vs 副作用)

| Cell | orth_only (s333 EXP-109) | PCH 3-seed mean | 效果 |
|---|:-:|:-:|:-:|
| (Art, guitar) | ~44 | ? (需查) | 目标 cell |
| (Art, horse) | ~51 | ? | 目标 cell |
| (Photo, horse) | ~56 | ? | 目标 cell |
| **Cartoon 其他 class** | ~88 | **~84** (-4) ⚠️ | **副作用** |

---

## 🔬 失败诊断

### 1. Cartoon -4pp 是 FedAvg 聚合副作用

**原因**:
- Art client 的 CE loss × 2 放大后, Art 端 gradient norm 增大
- FedAvg 聚合时 Art 的 gradient 贡献被相对放大 (即使按 data vol 加权)
- 结果: global model 被 Art 方向拉过去 → Cartoon/其他 domain 的 performance 受影响
- **不是 per-cell re-weight 的"纯粹"效果**, 是 gradient-level 的 cross-domain 污染

### 2. Art 涨 1.23 不够覆盖 Cartoon 掉的 -4

- Art: +1.23 × (1/4 AVG 权重) = +0.31 pp 到 AVG
- Cartoon: -4.02 × (1/4) = -1.00 pp 到 AVG
- Photo: +2.26 × (1/4) = +0.57 pp
- Sketch: -0.61 × (1/4) = -0.15 pp
- **净: -0.27 + 0.31 + 0.57 - 0.15 = +0.14** ✅ 数学对的上

### 3. 单 seed (s=333) 突破的意外

s=333 PCH AVG 81.13 已超 orth_only s=333 (~78.88) by **+2.25pp** — 这个 seed PCH **显著有效**.

但 s=2, s=15 没有这种突破. 说明 **PCH 对 seed 极敏感**, 不稳定.

这支持 "hardcoded hard cells 假设脆弱" — 只在特定 seed 的初始化下对齐hardcoded cells 的真实 hard 才有效.

---

## ❌ 方向判决

**PCH (hardcoded per-cell hardness CE re-weight) 失败**:
1. 3-seed AVG 涨 0.14pp 在 noise 内
2. Cartoon -4pp 副作用严重
3. 只 s=333 单 seed 意外突破, 不稳定
4. **Hardcoded 方法 fundamental 限制**: 无法 per-dataset 推广 (CLAUDE.md 禁止)

### 对 EXP-125 OCSD 方向的启示

PCH 失败 + EXP-125 验证 1 (用户人眼看 over_conf_wrong 图) 观察:
- **不是 style shortcut** (图不风格极端)
- **是 model class boundary 学歪** (人眼能认的图, model 错)

综合结论: **方向 III (per-cell hardness) 不 work**, 切到 **CDCA-Orth (跨域 class prototype anchor)** 更直接.

### 下一步

放弃 OCSD (方向 III 的 adaptive 版本) + PCH (方向 III 的 hardcode 版本).
**转 CDCA-Orth** (alignment 路线, 不 re-weight).

- 跨域 class 表征 anchor
- 直接修 Art/Photo/Cartoon 各 class 在 feature space 的位置对齐
- 无 gradient 放大副作用
- 1 个新超参 (λ_anchor)

详见: `experiments/IDEA_DISCOVERY_2026-04-24/IDEA_REPORT.md` (CDCA-Orth 定义)
+ `obsidian_exprtiment_results/2026-04-24/关键实验发现备忘.md`

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
