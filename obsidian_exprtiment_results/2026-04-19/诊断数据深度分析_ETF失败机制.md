# 诊断数据深度分析 — ETF 失败的真正机制 + 下一步方向

> 2026-04-20 凌晨. 基于 Office 污染 jsonl 启发式分离 + PACS 干净 diag (R55)
> 回答: "为什么 whitening + Linear > whitening + ETF"? 下一步该做啥?

## TL;DR (大白话)

ETF 强制特征对齐 10 个"钉死的标准方向",**对齐是对齐了 (align=0.95),但每类内部特征变松散了** (intra 0.91 vs Linear 0.98). Linear 让每类**自由找最紧的位置** (不一定对准 ETF 方向). 当前 Office 单 outlier 场景下,Linear 的紧密度赢了 ETF 的标准化.

**根因**: ETF 把 classifier 的几何**从"贴合数据分布"强行掰成"理论最优单纯形"**,数据不买账.

## 一、三层诊断数据完整对比

### PACS seed=2 client 0 Layer 1 (R5 → R55, ETF vs Linear)

| Round | ETF etf_align | ETF intra | ETF inter | Lin etf_align | Lin intra | Lin inter |
|-------|--------------|-----------|-----------|----------------|-----------|-----------|
| R5 | 0.705 | 0.625 | 0.101 | -0.008 | 0.863 | 0.019 |
| R10 | 0.822 | 0.784 | -0.062 | -0.018 | 0.878 | -0.029 |
| R20 | 0.896 | 0.842 | -0.127 | -0.016 | 0.919 | -0.048 |
| R30 | 0.920 | 0.867 | -0.142 | -0.018 | 0.942 | -0.069 |
| R40 | 0.935 | 0.883 | -0.149 | -0.019 | 0.960 | -0.088 |
| R50 | 0.946 | 0.895 | -0.154 | -0.020 | 0.973 | -0.103 |
| **R55** | **0.950** | 0.900 | **-0.156** | **-0.020** | **0.978** | **-0.110** |

**PACS 12-sample mean (4 clients × 3 seeds @ R55)**:
| | ETF | Linear | Δ |
|---|-----|--------|---|
| etf_align | **0.951** ± 0.006 | -0.001 ± 0.031 | ETF 达理论上限 **95%** |
| **intra_cls_sim** | 0.909 ± 0.015 | **0.980** ± 0.004 | **Linear 类内紧密 +7.7%** |
| **inter_cls_sim** | **-0.160** ± 0.008 | -0.126 ± 0.016 | **ETF 类间分离 +27%** (理论下界 -0.167) |
| orth | 0.0000 | 0.0000 | 都彻底解耦 |

### Office 污染 jsonl 启发式分离 (client 0, S2)

| R | ETF align | ETF intra | ETF inter | Lin align | Lin intra | Lin inter |
|---|-----------|-----------|-----------|-----------|-----------|-----------|
| R5 | 0.272 | 0.636 | 0.652 | 0.005 | 0.512 | 0.457 |
| R25 | 0.766 | 0.822 | -0.045 | -0.003 | 0.934 | 0.130 |
| R50 | 0.825 | 0.845 | -0.078 | -0.002 | 0.923 | 0.113 |
| R100 | 0.871 | 0.867 | -0.097 | -0.006 | 0.945 | 0.081 |
| **R195** | **0.888** | 0.858 | -0.093 | -0.002 | **0.941** | 0.051 |

### Layer 2 (Server aggregate, PACS seed=2)

| Round | ETF center_var | Lin center_var | Ratio (Lin/ETF) |
|-------|---------------|----------------|-----------------|
| R1 | 0.0180 | 0.1031 | **5.7x** |
| R20 | 0.00118 | 0.01381 | **11.7x** |
| R40 | 0.00030 | 0.00740 | **24.7x** |
| **R56** | **0.00010** | 0.00481 | **48.1x** 🔥 |

| Round | ETF drift | Lin drift |
|-------|-----------|-----------|
| R1 | 0.839 | 0.523 |
| R56 | 0.0105 | 0.0183 |

## 二、决定性洞察 (3 条)

### 💡 洞察 1: ETF vs Linear 的根本 trade-off

**ETF 赢: 跨客户端一致性 + 类间分离**
- etf_align 0.95 (理论上限 1)
- center_var 比 Linear 低 **48 倍** (PACS R56)
- inter_cls -0.160 (理论下界 -0.167 的 **96%**)

**Linear 赢: 类内紧密度**
- intra_cls_sim 0.978 vs ETF 0.909 (+7.7%)
- 每类特征**自由聚集到最紧方向**, 不被 ETF 顶点约束

**机制**: ETF 把 K 个类方向**钉在 simplex 上**, 但实际类中心**不一定在这些位置**. 强制对齐 → 类内特征被**拉散** (tradeoff 每类聚合度).

### 💡 洞察 2: 为什么 Linear 在 Office 反超,但 PACS 可能翻转

**Office 10 类 + 单 outlier DSLR 157 样本**:
- 10 类在 128d 空间容易自然分离 (即使没 ETF 约束)
- Linear 类内紧密 (0.941) 的优势 > ETF 标准化的优势
- 结果: Linear **88.75** > ETF **86.97** (-1.78)

**PACS 7 类 + 4 outlier (全域)**:
- 4 个域**视觉风格差异巨大** (照片/卡通/素描/艺术)
- 跨客户端一致性 (ETF 48x 更好) 可能**比类内紧密更重要**
- **预测**: ETF 可能在 PACS 翻身 (需 R200 验证)

### 💡 洞察 3: pooled whitening 的机制

**whitening 广播实际在做什么**:
- 每客户端传自己的 (μ_sty, Σ_sty) → server 做 **global covariance 估计**
- broadcast Σ_inv_sqrt 给所有客户端 = **跨客户端风格归一化坐标系**
- 客户端自己的 z_sty 在这个共同坐标系里对齐

**为什么比 class_centers 更强 (+0.49% 单独 89.26 vs 88.77)**:
- whitening 在**整个特征分布**层面做 normalization (second-order)
- class_centers 只在**每类中心**层面传信息 (first-order)
- 加了 centers 反而**干扰了 whitening 的分布对齐**

**这是一种 "cross-client BN"**:
- 普通 FedBN: 每 client 私有 BN → 客户端分布独立
- whitening broadcast: 共享 global second-order moment → 隐式对齐
- 效果: encoder 通过 z_sty 路径被 "cross-client normalized"

## 三、为什么 Office 结果还"不够好"

**当前: 89.26% 离 FDSE 90.58% 差 1.32%**

**差距来源分析**:
1. **FDSE 有新 trainable 参数** (DFE+DSE 层分解), 我们零新参数 — 这是故意的
2. **FedAvg 本身限制**: whitening 是辅助,核心聚合还是 FedAvg
3. **没用 SGPA 推理** (Layer 3 13 指标还没激活) — EXP-099 待测

## 四、下一步 4 个方向 (按潜力 + 成本排序)

### 🔥 方向 1: 跑 EXP-099 SGPA 推理 (0 GPU,高潜力)

**Hypothesis**: Linear+whitening 89.26 已经把 "训练端" 用到极限,但**推理端双 gate + proto** 可以免费 on-top 再提分 0.5~1%.

**具体**: 改一个带 se=1 的 EXP-105 (单 seed R200 Office) → checkpoint 保存 → run_sgpa_inference.py → 看 proto_vs_etf_gain.

成本: 1 个 Office R200 (~20min GPU) + script 运行 (~5min CPU).

### 🟡 方向 2: Hybrid 策略 (Linear + ETF 分配策略)

**Hypothesis**: 
- 用 Linear classifier (训练端紧密)
- 但 inter_cls_sim 附近 **软性 regularize 向 ETF 方向** (拿到部分跨客户端一致性)

成本: 代码改动 ~50 行 + 几个 R200 调 lambda.

### 🔶 方向 3: 改进 pooled whitening (可能突破 FDSE)

当前用的是 **uniform pooled** (平均各 client (μ, Σ)),但各 client 样本数不同. 改 **sample-weighted pooled** 可能带来 0.3~0.5% gain.

成本: 代码 10 行改动 + 跑 3 seeds.

### ⚠️ 方向 4: 引入少量新参数 (违反"零可训参数"承诺)

如 FedALA 逐元素加权聚合 (AAAI'23), 可能让 FedAvg 更准. 但引入 trainable. 只作为备选.

## 五、推荐下一步执行

**立即 (0 GPU)**:
1. 等 PACS 6 runs 完成 (~1h)
2. 如果 PACS ETF > Linear → **hybrid 策略价值大增**, ETF 在某些 regime 是好的
3. 如果 PACS Linear > ETF → **论文就是 FedDSA-Plus = Plan A + whitening**, 简化叙事

**平行 (1 GPU, 20 min)**:
- 部署 **EXP-105**: Linear+whitening se=1 (Office R200 seed=2) → 生成 checkpoint
- 跑 EXP-099 推理 script → 报 proto_vs_etf_gain

**继续 (3 GPU-h)**:
- 方向 3 sample-weighted pooled whitening (消融)
- 方向 2 hybrid (探索)

## 六、写论文的时机

**如果 PACS 也 Linear > ETF**: 
- 论文主轴 = "FedDSA-Plus: Pooled Style Whitening Broadcast" 
- 数据齐全 (Office 89.26 / PACS TBD), 开始写

**如果 PACS ETF > Linear**: 
- 论文主轴 = "Domain-Aware Hybrid: ETF for feature-skew heavy, Linear for label-skew heavy"
- 故事更复杂但更有意思,需要方向 2 hybrid 对照

## 七、核心数据可视化 (待做)

9 张诊断图 (之前规划但没实现):
1. 训练端: etf_align / intra_cls / inter_cls vs round (ETF vs Linear)
2. 训练端: orth vs round
3. 聚合端: center_var Plan A vs ETF vs Linear
4. 聚合端: param_drift
5-9. 推理端 (EXP-099 出数据后)

可以等 PACS 完成后集中画,放论文 Fig 2.
