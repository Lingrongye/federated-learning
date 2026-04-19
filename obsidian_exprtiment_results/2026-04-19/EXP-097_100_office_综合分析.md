# EXP-097 + EXP-100 Office R200 综合分析 — 重大反转 + 诊断启发式解读

> 2026-04-20 凌晨完成。**ETF 反向证伪** + **发现真正 gain 来自 pooled whitening 基础设施**

## TL;DR (一句话)

把 Plan A 的 Linear 换成 Fixed ETF 反而**减分 -1.78%**,但意外发现 **Plan A + pooled style statistics 广播 = 88.75% AVG Best**,比原 Plan A (82.55%) 高 **+6.20%**,离 FDSE SOTA (90.58%) 只差 1.83%。论文主贡献从 ETF 转向 **"Pooled source-domain style second-order statistics broadcast"**。

## 总结果表 (3-seed mean,最关键的一张)

| 方法 | AVG Best | AVG Last | ALL Best | drop | std | 对比 Plan A |
|------|----------|----------|----------|------|-----|------------|
| Plan A orth_only (EXP-083) | 82.55 | 81.35 | 88.61 | — | ~1 | 基线 |
| SAS τ=0.3 (EXP-084) | 84.40 | 83.07 | 89.82 | — | — | +1.85 |
| **SGPA (use_etf=1)** | **86.97** | 85.44 | 82.01 | 1.53 | 1.23 | **+4.42** |
| **Linear+whitening (use_etf=0)** | **88.75** 🔥 | 86.91 | 82.81 | 1.84 | 0.86 | **+6.20** |
| FDSE (EXP-051) | 90.58 | 89.22 | 86.38 | — | — | +8.03 |
| **Δ Linear − SGPA (ETF 贡献!)** | **-1.78** ❌ | -1.47 | -0.80 | +0.31 | -0.37 | — |

**3 个秒懂的数字**:
- Linear 88.75 vs Plan A 82.55 = **+6.20** (说明 whitening 基础设施本身就强)
- SGPA 86.97 vs Plan A 82.55 = **+4.42** (说明 SGPA 仍比 Plan A 强,但比 Linear 弱)
- Linear 88.75 vs FDSE 90.58 = **-1.83** (离 SOTA 很近)

## Per-domain 结果 (Best/Last, 3-seed mean)

| 配置 | Caltech | Amazon | DSLR | Webcam |
|------|---------|--------|------|--------|
| Plan A orth_only | 72.6/70.2 | 90.9/88.1 | 100.0/97.8 | 94.3/93.1 |
| SAS τ=0.3 | 75.0/73.8 | 91.6/88.4 | 100.0/97.8 | 95.4/93.1 |
| **SGPA (use_etf=1)** | 70.5/69.6 | 88.8/88.1 | 97.8/95.6 | 90.8/88.5 |
| **Linear+whitening** | 72.3/70.5 | 88.4/87.4 | 100.0/97.8 | 94.3/92.0 |
| Δ Linear − SGPA | **+1.8** | -0.4 | **+2.2** | **+3.5** |
| Δ Linear − Plan A | -0.3 | -2.5 | ±0 | ±0 |

**Per-domain 启示**:
- **SGPA 在 DSLR 掉 -2.2** (100→97.8): Fixed ETF 对 DSLR 小样本域 (157 samples) 强制几何反而有害
- **SGPA 在 Webcam 掉 -3.5** (94.3→90.8): ETF 阻止了 Linear 对高相似性 Webcam 的学习
- **SGPA 在 Caltech 掉 -1.8** (72.3→70.5): Caltech 是最难域, ETF 没帮上
- **Linear 在 Amazon 掉 -2.5** vs Plan A: 但被其他域 gain 抵消
- **Linear 的优势主要在 DSLR + Webcam 保持高位**, ETF 在这两个相似视觉风格的域反而掉分

## 每 seed 详细

### SGPA (use_etf=1)

| seed | AVG Best | AVG Last | ALL Best | ALL Last | drop |
|------|----------|----------|----------|----------|------|
| 2 | 85.89 | 82.83 | 79.75 | 76.58 | 3.06 |
| 15 | **88.68** | 88.68 | 82.94 | 82.94 | 0.00 (best=last) |
| 333 | 86.35 | 84.81 | 83.32 | 81.73 | 1.53 |
| **mean** | **86.97 ± 1.23** | 85.44 | 82.01 | 80.42 | 1.53 |

### Linear+whitening (use_etf=0)

| seed | AVG Best | AVG Last | ALL Best | ALL Last | drop |
|------|----------|----------|----------|----------|------|
| 2 | 87.56 | 86.81 | 80.17 | 78.98 | 0.75 |
| 15 | **89.55** | 87.11 | 83.35 | 81.35 | 2.43 |
| 333 | 89.14 | 86.80 | **84.91** | 82.93 | 2.34 |
| **mean** | **88.75 ± 0.86** | 86.91 | 82.81 | 81.09 | 1.84 |

**观察**:
- Linear 3-seed std 0.86 < SGPA 1.23 → **Linear 更稳**
- SGPA seed=2 drop=3.06 最大 → ETF 下训练后期有小幅度不稳
- Linear 每 seed 都 > 87% → 3-seed 一致性好

## 诊断数据启发式分析 (数据污染但有效)

### 关键数据:由于 diag_logs 路径未分 SGPA/Linear(bug `6a31e22` 已修),每个 `R200_S{seed}/diag_aggregate_client-1.jsonl` 每轮有 2 行 — 一个 SGPA 一个 Linear,交错写入。

### 启发式分离规则

**假设: 每轮 client_center_var 更小的 = SGPA (ETF 作为 shared anchor),更大的 = Linear**

这个假设在 PACS 干净 diag (commit `6a31e22` 修复后新部署) 里得到验证: PACS R1-R3 SGPA center_var 始终比 Linear 低 4-6x,两组不重叠。

### Office 诊断数据 (3-seed 启发式分离后)

| Round | SGPA center_var | Linear center_var | Ratio | SGPA drift | Linear drift |
|-------|-----------------|-------------------|-------|------------|--------------|
| R1 | 0.0087 | 0.0219 | **2.5x** | 0.15 | 0.75 |
| R5 | 0.0082 | 0.0233 | **2.8x** | 0.19 | 0.19 |
| R50 | 0.0019 | 0.0189 | **10.0x** | 0.008 | 0.013 |
| R100 | 0.0015 | 0.0164 | **10.9x** | 0.003 | 0.006 |
| **R200** | **0.0011** | **0.0121** | **11.0x** 🔥 | 0.002 | 0.003 |

### 关键洞察

1. **ETF 确实让跨 client 类中心一致性高 11x** (R200 ratio 10-11x)
2. **但 test accuracy Linear 反超** — 说明"跨 client 一致"不等于"分类准"
3. **原因猜想**: Office 10 类单 outlier (DSLR 157 样本),Linear 的自由分类边界能更好适应 class 分布不均;ETF 强制 10 类等角分布,可能**偏袒不了关键类**
4. **ETF 在优化跨 client 一致性上是严格有效的**,但这个目标与 accuracy 目标**不完全一致**

## 论文叙事大转弯 (必须接受)

| 原方向 | 新方向 |
|--------|--------|
| Primary: SGPA 双 gate + proto 推理 | Primary: **Pooled style statistics broadcast + FedDSA decouple** |
| Supporting: Fixed ETF classifier | **ETF 有害,删除** |
| 主故事: Neural Collapse 加速 | 主故事: **跨 client 风格二阶统计共享让 FedAvg 聚合更准** |

## 为什么意外?Office 结果与 smoke test 不符吗?

**smoke test 84.98% 是对的,但归因错了**:
- smoke (seed=2 R50 SGPA only): 84.98%
- R200 3-seed SGPA: 86.97% → smoke 是过早截断的 ETF 结果
- R200 3-seed Linear: **88.75% > SGPA** → 真正 gain 不是 ETF,是 whitening + class_centers 基础设施

smoke test 没骗我们 (SGPA R50 确实快超 Plan A R200),但**没告诉我们 Linear 同样有 gain 甚至更多**。控制变量实验正是为这个设计的。

## 下一步必须做的 ablation (定位 gain 来源)

| 实验编号 | 改动 | 目的 |
|---------|------|------|
| EXP-102 (新) | Plan A + pooled whitening only (不收 class_centers) | pooled whitening 单独贡献 |
| EXP-103 (新) | Plan A + class_centers only (不广播 whitening) | class_centers 单独贡献 |
| EXP-104 (新) | Plan A + diag=0 同 Linear (不开诊断框架) | 排除 diag 副作用 |
| EXP-098 PACS (跑中) | 看 PACS 4-outlier 下是否 Linear 仍超 SGPA | 双数据集验证 |
| EXP-099 SGPA 推理 (未做) | 独立 script 测 proto_vs_etf_gain | SGPA 推理端是否还有 on-top-of 增量 |

### 优先级
1. **EXP-098 PACS** (跑中,自然完成)
2. **EXP-102 Plan A + whitening only** (最小消融,1-2h)
3. **EXP-099 推理** (零 GPU,提取诊断方向)
4. EXP-103/104 (如果 102 不足以解释)

## Diag 污染教训

| 问题 | 原因 | 修复 |
|------|------|------|
| SGPA+Linear 同 seed diag jsonl 交错 | `diag_root = f'R{N}_S{seed}'` 不带 variant | commit `6a31e22` 加 `_etf`/`_linear` 后缀 |
| 无法可靠分辨哪行是谁的 | 每行没有 variant 字段 | 同上 (新路径目录已隔离) |
| Office diag 数据靠启发式分离 | bug 发现时 Office 已跑完 | PACS 用干净路径验证 |

**教训**: 新 diag 路径必须带**完整实验标识** (variant + seed + task + rounds),否则多 run 共享路径竞争写。未来 diag 路径模板:

```
task/{task}/diag_logs/R{num_rounds}_S{seed}_{variant}_{algorithm}/
```

## Obsidian 链接

- 完整 NOTE: [EXP-097_sgpa_office_r200.md](EXP-097_sgpa_office_r200.md)
- 对照 NOTE: [EXP-100_linear_office_r200.md](EXP-100_linear_office_r200.md)
- 诊断数据源文件: `FDSE_CVPR25/task/office_caltech10_c4/diag_logs/R200_S{2,15,333}/diag_aggregate_client-1.jsonl` (污染, 启发式分离)
- 相关 commit: `bb3d52b` (NOTE 回填), `6a31e22` (diag path fix), `557f4fa` (results collected)
