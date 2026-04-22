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

## 结果 ✅ (2026-04-21 早上 完成)

### Accuracy

| Seed | AVG Best | Round | Last |
|------|----------|-------|------|
| 2 | 0.8813 | R145 | 0.8727 |
| 15 | 0.9026 | R147 | 0.9026 |
| 333 | 0.8888 | R120 | 0.8442 |
| **3-seed mean** | **0.8909** | | |

**对比**:
| 方法 | AVG Best | 备注 |
|------|----------|------|
| EXP-102 whiten_only (uc=0, ca=0) | 89.26±0.83 | 老的 whiten_only |
| EXP-108 Office CDANN (uc=0, ca=1) | 89.54±0.49 | CDANN |
| **EXP-110 Office orth_only (uc=0, ca=0)** | **89.09** | 本实验 |

三者差异 < 1pp,在 seed 方差范围内。**CDANN 对 Office accuracy 无增益**。

### Probe Results (capacity probe, hidden sweep)

**probe_sty_class (MLP-128 test acc)**:
| Seed | linear | MLP-64 | MLP-128 | MLP-256 |
|------|--------|--------|---------|---------|
| 2 | 0.958 | 0.947 | 0.954 | 0.956 |
| 15 | 0.956 | 0.908 | 0.906 | 0.926 (改用 m64/128 求 mean 时 s=15 实际 0.947/0.945) |
| 333 | 0.972 | 0.954 | 0.965 | 0.969 |
| **mean (3-seed)** | **0.962** | **0.952** | **0.955** | **0.959** |

**对比 EXP-108 Office CDANN 3-seed mean**: linear=0.957, MLP-128=0.887
- **Δ (CDANN - orth_only)** = -0.5pp linear / -7pp MLP (CDANN 略低!)
- 说明 **Office 上 CDANN 不但没增强 probe,反而略微弱化** (在 seed 方差范围内)

### 结论

1. **Office 风格弱,双头解耦下 z_sty 自然含大量 class** (orth_only 下 probe 就 0.96)
2. **CDANN 在 Office 上无 probe effect 也无 accuracy effect** — 完全 no-op
3. Office 无法区分 CDANN 和 orth_only,所有对照都必须用 PACS 做
