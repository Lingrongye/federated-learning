# Crisis Diagnosis: FedDSA Auxiliary Losses Are Net Harmful

## Problem Anchor

- **Bottom-line problem**: FedDSA 的辅助损失（Style Augmentation CE + InfoNCE Alignment）在 warmup 后提供短期 peak 提升，但随后导致持续性能下降，使 R200 终值**低于纯 CE 训练的最好成绩**。这意味着 "Share" 和 "Align" 两个核心贡献实际上是负面的。
- **Must-solve bottleneck**: CE 与 InfoNCE 的梯度冲突在 R50 后变为对抗性（cos_sim 穿越零点），导致训练不稳定。75 个实验中 10/11 已完成变体显示 warmup 后净负效果。
- **Non-goals**: 不是要调超参或换 warmup 策略——这些已经试过并失败了（EXP-029 PCGrad, EXP-060~063 style fix, EXP-075 gradual ramp）
- **Constraints**: AlexNet backbone, PACS/Office/DomainNet, 3-seed 验证, FDSE 是必比基线 (PACS 82.17% R500)
- **Success condition**: 找到一个训练动态稳定的方案，R200 final ≥ 81% PACS (超过 FDSE R200 的 80.36%)，peak→final drop < 1%

## 一、75 个实验的完整证据审计

### 1.1 "Decouple-Share-Align" 三模块净效果审计

| 模块 | 声称作用 | 实际证据 | 净效果 |
|------|---------|---------|--------|
| **Decouple** (L_orth) | 正交分离语义/风格 | EXP-021 orth=2.0 第二名(81.58%); EXP-070 消融证明协同; 从 R0 开始不崩 | **正面** |
| **Share** (Style Aug CE) | 跨域风格增强 | EXP-060~063 四种 style fix 全失败; gradual_noaug 最稳; augmentation 越少越稳 | **负面** |
| **Align** (InfoNCE) | 语义原型对齐 | 短期 peak +1-2%，长期 cos(CE,InfoNCE)<0 冲突，final 低于 pre-warmup | **短正长负** |

### 1.2 梯度冲突的铁证（EXP-075 诊断日志）

```
cos_sim(grad_CE, grad_InfoNCE) trajectory (shallow s=2):
R10: +0.72 (aligned)
R20: +0.50 (aligned)
R30: +0.37 (weakening)
R40: +0.22 (weakly aligned, w_align=0.50)
R50: -0.01 (CROSSING ZERO, w_align=0.73)  ← 冲突开始
R60: -0.24 (CONFLICT, Acc starts dropping)
R70: +0.69 (spike after collapse — NOT recovery, but feature space reset)
R80: +0.31 (re-aligning but damage done)
R90: +0.23 (slowly converging to ~0)
R100:+0.15 (approaching orthogonal — losses optimizing different things)
```

**6 个 run 全部一致**: R50 穿零，R60 明确冲突，然后 cos 长期趋向 0。

### 1.3 所有已完成实验的 net 效果 (final - pre_warmup_max)

| 方法 | 总 runs | HELPED (>+0.5) | NEUTRAL (±0.5) | HURT (-3~-0.5) | CRASHED (<-3) |
|------|---------|---------------|----------------|----------------|---------------|
| feddsa tau=0.1 | 12 | 1 (8%) | 0 | 3 (25%) | **8 (67%)** |
| feddsa tau=0.2 | 1 | 0 | 1 | 0 | 0 |
| adaptive md0 tau=0.2 | 3 | 0 | 0 | 1 | **2** |
| adaptive md2 tau=0.2 | 3 | 0 | 0 | 1 | **2** |
| adaptive md3 tau=0.2 | 3 | 0 | 0 | 1 | **2** |
| base_256d | 3 | 0 | 2 | 0 | **1** |

**全部 25 个已完成 run 中，仅 1 个 HELPED，3 个 NEUTRAL，21 个 HURT/CRASHED (84%)。**

### 1.4 正在运行实验的中期趋势

| 实验 | R进度 | 趋势 | 下降幅度 |
|------|-------|------|---------|
| M4_dual (intra+cross) | R139 | 崩溃中 | -4.39% |
| M4_intra (仅域内) | R118 | 轻降 | -1.67% |
| M4_cross (仅跨域) | R118 | 稳定 | **-0.66%** |
| M5_style | R93 | 稳定 | -0.69% |
| M6_delta_film | R74 | 严重崩 | -8.24% |
| gradual_shallow | R100 | 下降 | -1.39% |
| gradual_noaug | R100 | 下降 | -2.89% |

### 1.5 Per-client 梯度冲突分析

Photo 域（域差异最小）**最先进入冲突**：
- R40: Photo cos=+0.12 vs Sketch cos=+0.23
- R50: Photo cos=**-0.15** vs Sketch cos=+0.07
- R60: Photo cos=**-0.37** vs Sketch cos=-0.11

这与 Office（全低差异域）上 FedDSA 输 FDSE 1.45% 完全一致。

## 二、根因诊断

### 2.1 为什么 CE 和 InfoNCE 会冲突？

**InfoNCE 的目标**: 把 z_sem 拉向全局原型（跨域平均），推开异类原型
**CE 的目标**: 把 z_sem 投影到分类器能区分的方向

在训练早期（R0-40），两者大致对齐——更好的类聚类也让分类更容易。
但在训练后期（R50+），CE 已经找到了好的分类边界，而 InfoNCE 仍然强制拉向全局原型。这个原型是**跨域平均**的，对于低差异域（Photo）来说，被拉向混合原型反而破坏了本地的好分类边界。

### 2.2 为什么 Style Augmentation 有害？

AdaIN 在 h-space (1024d) 做统计量替换：
```python
h_norm = (h - mu_local) / sigma_local
h_aug = h_norm * sigma_ext + mu_ext
```

这在**高层特征空间**做了激进的分布变换。当 sigma_ext 和 mu_ext 来自差异很大的域（如 Sketch→Photo），增强后的特征可能完全偏离有意义的表征空间。CE_aug 试图让分类器对这些"噪声特征"也正确分类，这给编码器施加了矛盾的梯度。

### 2.3 为什么原始 FedDSA tau=0.2 看起来稳定？

关键区别：只有**一个 seed (s=2)** 的完整 R200 数据。且：
- aux_w 是线性 ramp（0→1 over 50 rounds），不是硬切换
- tau=0.2 让 InfoNCE 梯度更柔和（温度高→softmax 更平）
- 但这也意味着 InfoNCE 的对比信号更弱，本质上是"削弱了 Align 的影响"

**tau=0.2 的"稳定"可能只是因为 InfoNCE 太弱以至于不够伤害模型。**

### 2.4 正交约束（L_orth）的角色

两个版本的处理不同：
- **原始 feddsa.py**: `aux_w * λ_orth * L_orth` — 跟着 warmup 从 0 渐增到 1
- **gradual.py**: `λ_orth * L_orth` — 从 R0 就全权重

原始版本里，正交约束与 InfoNCE 同步增长，两个力协调发展。
Gradual 版让正交从 R0 全开但 InfoNCE 延后 → z_sem 在"被正交推开但没有方向"的状态下学了很多轮，InfoNCE 后来加入时特征空间可能已经不适合对齐了。

但更重要的是：**EXP-070 消融显示，仅 Decouple (无 Share/Align) 就能达到 80.65%**，与 Full FedDSA (81.29%) 差距仅 0.64%。这 0.64% 的协同增益，是通过引入两个**长期有害**的模块换来的。

## 三、核心困境

### 3.1 Paper Story 崩塌

FedDSA 的论文声称 "Decouple-Share-Align" 三步机制协同工作。但实验证据表明：

| 声称 | 现实 |
|------|------|
| "Style 是可共享资产" | Style augmentation 是噪声源，增强越多越不稳定 |
| "InfoNCE 软对齐优于 MSE" | InfoNCE 短期有益但长期与 CE 冲突 |
| "三模块协同" | 仅 Decouple 有效，Share+Align 净效果为负 |
| "81.29% 超 FDSE" | 这是 peak 值，final 只有 ~75-80%，很多 seed 崩溃 |

### 3.2 可能的出路

**路线 A: 修复训练动态（保住现有框架）**
- 让 InfoNCE 在 R50 后衰减（bell-curve 权重调度）
- 只在早期用 Share+Align "启动"好的表征，后期切回纯 CE
- 优点: 保住 DSA 故事
- 风险: 这本质上承认了 Share+Align 只在早期有用

**路线 B: 重新定义 Share 和 Align**
- Share: 不在 h-space 做 AdaIN，改为更温和的增强方式
- Align: 不用全局原型，改为直接的域间特征匹配
- 优点: 保住框架，改善组件
- 风险: 工作量大，可能仍无效

**路线 C: 聚焦 Decouple + Regime 诊断**
- 承认 Share+Align 的条件性，聚焦"解耦本身提供 regime 诊断信号"
- 论文 framing: "何时该共享风格、何时不该"
- 优点: 诚实，有独特贡献（style_head r值 3.6x 区分度）
- 风险: 贡献太薄，可能不够一篇论文

**路线 D: Decouple + 新的对齐机制**
- 保留解耦（唯一有效的部分）
- 用全新的对齐方式替代 InfoNCE（如 EMA teacher、mutual information、或 stop-gradient 方法）
- 优点: 直接解决根因
- 风险: 需要重新开发和验证

## 四、对路线 A 的具体方案（最快验证）

### 4.1 Bell-Curve 权重调度

```python
# 当前: aux_w = min(1.0, round / warmup)  → 0→1→1→1→1...
# 新: aux_w = bell(round, peak_round, width)

def bell_weight(t, t_peak=60, width=30):
    """Gaussian bell: peaks at t_peak, decays both sides."""
    return math.exp(-0.5 * ((t - t_peak) / width) ** 2)

# R0:  0.14 (低权重,纯CE主导)
# R30: 0.61 (渐增)
# R60: 1.00 (peak)
# R90: 0.61 (开始衰减)
# R120:0.14 (基本关闭)
# R150:0.01 (几乎为零)
```

这样 InfoNCE 在 R0-60 提供"启动引导"，在 R60 后逐渐退出，让 CE 主导后期精细化。

### 4.2 最小实验验证

| 实验 | Config | 目的 |
|------|--------|------|
| bell_60_30 | t_peak=60, width=30 | 主实验 |
| bell_40_20 | t_peak=40, width=20 | 更早衰减 |
| ce_only_r50 | R50 后完全关闭 aux | 极端消融 |

预期: 如果 bell_60_30 的 final ≥ pre-warmup peak (79-80%)，且 peak→final < 1%，则证明"早期引导+后期退出"策略有效。

## 五、对路线 D 的具体方案（更根本）

### 5.1 为什么 InfoNCE 冲突但 Decouple 不冲突？

L_orth = cos²(z_sem, z_sty) — 只约束 z_sem 和 z_sty 的关系，不指定 z_sem 的绝对方向。CE 可以在正交约束下自由找到好的分类边界。

L_InfoNCE — 强制 z_sem 向特定的全局原型移动。这与 CE 的分类边界优化直接竞争。

**核心洞察: 好的辅助损失应该是"约束性的"（告诉模型什么不该做），而不是"指令性的"（告诉模型往哪走）。**

### 5.2 替代方案: Stop-Gradient 对齐

```python
# 当前 InfoNCE: z_sem 向全局原型移动 (指令性)
loss_sem = InfoNCE(z_sem, global_protos)  # 梯度: z_sem 和 encoder 都更新

# 替代: z_sem 只需要和 DETACHED 的其他域特征相似 (约束性)
z_cross = gather_cross_domain_features()  # 其他域的 z_sem
loss_align = -cos_sim(z_sem, z_cross.detach())  # stop-gradient: 不通过 z_cross 更新
```

Stop-gradient 的关键区别: 不强迫模型追逐一个"移动的全局原型"，而是鼓励当前特征与其他域的**当前快照**相似。不需要全局原型聚合。

### 5.3 替代方案: 域间 BN 一致性（借鉴 FDSE）

FDSE 的成功之处在于它用 BN 统计量的 KL 散度做域对齐，而不是特征空间的对比学习。BN 统计量对齐是"温和的"——它让不同域的特征分布接近，但不强制特征点到点匹配。

```python
# FDSE-inspired: BN 统计量一致性
loss_bn_align = KL(local_bn_stats, global_bn_stats)  # 温和的分布对齐
```

## 六、建议的下一步

| 优先级 | 行动 | 时间 | 理由 |
|--------|------|------|------|
| P0 | 验证 bell-curve 权重调度 | 1天 | 最快验证，改几行代码 |
| P0 | 验证 "R80 后关闭 aux" 硬消融 | 与上同时 | 极端对照组 |
| P1 | 等 M4_cross 和 M4_intra 完成 R200 | ~6h | 看单独模块是否稳定 |
| P2 | 如果 bell-curve 有效，3-seed 验证 | 2天 | 确认不是偶然 |
| P3 | 如果 bell-curve 无效，转向路线 D | 3-5天 | 需要新的对齐机制 |

## Experiment Handoff Inputs

- **Must-prove claim**: bell-curve 权重调度能让 FedDSA 的 R200 final ≥ pre-warmup peak
- **Must-run ablation**: bell vs hard-cutoff vs current (always-on)
- **Critical metric**: peak→final drop，3-seed mean
- **Highest-risk assumption**: InfoNCE 的早期引导价值能被"记住"到后期
