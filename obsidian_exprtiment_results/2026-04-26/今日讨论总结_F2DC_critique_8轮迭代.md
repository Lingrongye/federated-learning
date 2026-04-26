---
date: 2026-04-26
type: 对话总结 / decision log
duration: 约 8 小时讨论
status: 方案敲定 (PG-DFC),等 F2DC sanity 复现完成后实施
---

# 今日讨论总结 — F2DC critique 8 轮迭代,从 MC-FL 到 PG-DFC

> 本日跟 Claude 围绕"如何在 F2DC 基础上做出毕设可发的创新"做了 8 轮深度讨论
> Claude 多次提出方案被用户质疑后推翻,最终敲定 PG-DFC (Prototype-Guided DFC) 小改动方案
> 整个过程暴露了 5 个重要错误,值得记下来避免再犯

---

## 一、今日时间线 + 讨论主线

### 阶段 1: F2DC 源码深读 (上午)

读了 F2DC 官方 GitHub 代码 (`/Users/changdao/联邦学习/F2DC/`):
- `models/f2dc.py`: 训练循环 + 三路 loss (DFD_dis1/dis2/sep + DFC)
- `backbone/ResNet_DC.py`: DFD/DFC 模块 (GumbelSigmoid + 2 层 conv)
- `models/utils/federated_model.py`: 聚合用 vanilla weighted FedAvg
- `utils/best_args.py`: 超参 (lr=0.01, lambda1=0.8, lambda2=1.0, gum_tau=0.1, tem=0.06)
- `utils/training.py`: communication=100, local_epoch=10, parti_num=10

**核心发现**:
- F2DC paper 写的 Domain-Aware Aggregation 在代码里**根本没实现** (agg_a/agg_b 是死参数)
- DFD mask 是 (B,C,H,W) per-pixel 但 paper 说"feature units" - 实现跟描述有 gap
- DFC 是 2 层 conv 残差,只用本地 nr_feat,无外部参考

### 阶段 2: 用户提"毕设不能抄" (中午)

讨论焦点从"复现 F2DC"转到"基于 F2DC 改进 + 创新":
- 不能 fork F2DC 整体复制
- 在我们 flgo 框架下重写实现,只参考思想
- 引用规范明确

### 阶段 3: 提出 MC-FL 方案 (下午早段) ❌ 被推翻

**Claude 提议**: Mask Consensus FL (MC-FL) — server 维护 channel-level robustness consensus,让 client 的 DFD mask 跨 client 对齐

**用户质疑**: "DFD 解耦真的是一个跨 client 共识问题吗?怎么判断?"

**深入分析后发现**:
- 50/50 — 没数据支持 mask consensus 是真问题
- FedAvg 已经强制 backbone 对齐,mask 可能也自动对齐
- F2DC > FedBN 说明 sparse mask 比 full mask 好 (即便不一致)
- **决议**: 不能预设方向,必须先做 1 周诊断

### 阶段 4: 讨论 wrong-high label 动机 (下午中段)

**Claude 第一版**: "wrong-high 在 PACS/Office 多类小数据集会失效,top-2 跟 top-1 是同语义簇"

**用户质疑**: "他不是每个 client 只去训练自己的模型吗?怎么可能 top2 跟 top1 是属于不同 domain 的"

**Claude 承认错误**:
- 把"top2 vs top1"跟"domain 差异"混在一起说了
- top-1/top-2 是**类别**层面 (同 client 内),不是 domain 层面
- Office-Caltech 没有 keyboard / computer-keyboard 同时存在,例子是瞎编的

**用户继续逼问**: "你得结合数据集来考虑 office 里面有什么类 什么相似 还有 pacs 有什么类"

**Claude 补充分析**:
- Office-Caltech 10 类: 显示设备组 (monitor/laptop/projector ★高混淆) + 桌面输入组 (keyboard/mouse/calculator) + 独立类
- PACS 7 类: 4 动物 (dog/horse/elephant/giraffe ★★ 极高混淆) + 3 独立类
- R1 风险在 PACS 上 (4/7=57% 类 + 跨 sketch 域压平差异) 比 Office (60%, 但 4 个清晰域) 严重

### 阶段 5: 用户提议看原 paper 动机 (下午后段)

**用户**: "得先好好去考虑一下他当初这样设计就是预测 top2 错误的动机是什么"

**Claude 推演 5 个层次的动机**:
1. Hard negative mining
2. Curriculum learning
3. 信息互补 (粗细粒度分离 — **后来证明这是脑补的**)
4. 避免 nr 跟 r 学一样
5. Self-distillation 变种

**用户质疑**: "感觉不太对 看一下原论文里面是怎么说的"

**真去读 paper 原文 (2603.14238 line 269-301)**:
- Paper 真实说法: nr_feat 学的是 "**client-specific domain context that may cause wrong decisions**" — 域上下文,不是细粒度
- Paper 自己承认: DFD 切不干净, f⁻ 是 "**domain artifacts + valuable class-relevant clues 的混合**"
- DFC 的设计动机: **抢救混入 nr 的 class clue**

**Claude 老实承认**:
- 我之前两次 critique 都误读了 paper
- 不是"细粒度 vs 粗粒度"的世界观,是"class signal vs domain context"
- DFC 不是 fancy 装饰,是配套的"切+缝"系统
- "wrong-high label 不是 critique 的好切入点" — paper 动机清晰
- 真正切入点是 paper 自己承认的 "DFD 切不干净, DFC 抢救有限"

### 阶段 6: 用户质疑 F2DC vs FDSE 数字 (下午晚段) ❌ Claude 数据错

**Claude 之前说**: "F2DC PACS 76.47 vs FDSE 83.81 → F2DC 输 FDSE 7.3pp,设计有问题"

**用户质疑**: "确定吗 这个数据哪里的 有差这么多吗"

**真去读 F2DC paper Table 1 (line 521-530)**:
- F2DC PACS = 76.47 ✓
- F2DC paper 报的 **FDSE = 73.13** (在 F2DC setup: K=10, ResNet-10, R=100)
- 不是 83.81 — 83.81 是 **FDSE 自己 paper 报的** (FDSE setup: K=4, ResNet-18, R=200)
- 两个 setup 完全不同, 数字不能跨 paper 比

**Claude 承认错误**:
- 张冠李戴, 跨 setup 借数字
- 真实情况: F2DC 在自己 setup 下**赢 FDSE +3.34pp** (PACS) / +3.64pp (Office)
- F2DC 是 paper-validated SOTA, 不是被 baseline 打败的方法
- 我之前所有"F2DC 设计有问题"的 critique 都站不住脚

### 阶段 7: 用 research-lit 综合搜索 (晚段)

按用户指示用 research-lit skill 综合搜索 (web + arxiv + 本地 PDF):
- Q1: domain feature decoupling calibration federated
- Q2: prototype guided feature recovery disentangled
- Q3: cross-client feature consensus alignment

找到 3 篇高相关 paper:
- **A** [2005.13947] Disentanglement Then Reconstruction (UDA, 2020) — prototype × reconstructor 框架
- **B** [2301.12798] RFedDis (FL, 2023) — 双 head + evidential uncertainty
- **C** [2501.15486] FedAlign (FedDG, 2025) — cross-client style transfer

### 阶段 8: 提出 PG-DFC 方案 → 用户质疑没读源码 (深夜)

**Claude 提议**: PG-DFC = A 的 prototype reconstruction + B 的 evidential + C 的 cross-client

**用户质疑**: "你确定已经精读了源码吗?下载下来先去读一下看看 看看他们的实验具体是什么样的 实验配置如何 是否跟我们是契合的"

**真 clone 源码到 `baselines/RFedDis/` 和 `baselines/FedALign/` 后发现**:

| Paper | 实际 setup | 契合度 |
|---|---|:--:|
| **A: Dis+Recon** | UDA centralized, ResNet-50, Office-31 | ☆ 完全不是 FL |
| **B: RFedDis** | K=4 1-1 domain, AlexNet, R=300, E=1 | ★★★★★ 跟我们一样 |
| **C: FedAlign** | FedDG leave-one-out, MobileNet, R=10, E=3 | ★ 完全不是 personalization |

**Claude 承认错误**:
- 没读源码就凭 abstract 推方案
- A 不是 FL, "prototype reconstruction" 不能直接搬
- C 是 FedDG, "cross-client style transfer" 不是我们 setup
- 真正能借的只有 B
- "三个工作合三为一" 是粗糙拼凑

### 阶段 9: 提出 EU-DFC → 用户继续质疑 (深夜后)

**Claude 提议 EU-DFC**: F2DC + RFedDis evidential

**用户质疑**: "真的能契合吗 告诉我你的方案是什么样的"

**Claude 重新审视**:
- RFedDis 的 evidential 假设两路都学正确, F2DC 的 nr 学 wrong-high — 数学上不通
- "在 F2DC 三路上加 evidential" 实际上要 skip nr 路, 只用 (r, rec)
- 这只是"小改动",不是大 contribution

**承认真相**: "在没有 Phase 0 实验数据下,任何具体 method 都是猜测"

### 阶段 10: 提出"双 head + prototype" → 用户说不要重构 (凌晨)

**Claude 提议**: 借 RFedDis 双 head 框架重写 F2DC,prototype 替代 KL repel

**用户**: "不行我已经开始复现 F2DC 了 我们还是要在这个方向上去改一下优化一下就行吧"

**Claude 调整策略**: 提出 4 个小改动 (A: channel-level mask / B: soft suppression / C: 补 DA-Agg / D: prototype-guided DFC)

### 阶段 11: 锁定 D 方案 (凌晨)

**用户**: "感觉 D 应该确实能够有点作用给 解释清楚一点 D 在干什么"

**Claude 详细展开 PG-DFC**:
- 直觉: 蒙眼画师 vs 给画师字典
- 完整代码 (DFC_PG 类 + EMA + 聚合)
- 4 个关键设计决策 (用 r_feat / channel-level / dot-product attention / warmup)
- 跟 F2DC paper 弱点对应

### 阶段 12: 讨论 EMA 共识 (凌晨末)

**用户**: "为什么现在联邦学习原型都用上一个 EMA 这是共识吗"

**Claude 澄清**:
- 不是共识,是惯例 (common practice)
- FedProto / FPL 经典工作其实**不用 EMA**,用 round-end snapshot
- EMA 是 MoCo/BYOL 带火的,2023 后 follow-up 大量沿用
- 用 EMA 的 4 个真理由: 小 batch 噪声 / 跨 round 稳定性 / bootstrapping / 实现简单
- 3 个真问题: α 调参敏感 / LR 大时拖后腿 / class imbalance 时 0 prototype 污染
- 我们用 EMA + warmup 是稳妥选择, 也可以退回 round-end snapshot

### 阶段 13: 写方案到 obsidian (最终)

写 `PG-DFC方案_Prototype引导的F2DC校准.md` (~750 行 14 章节 2 附录)

---

## 二、最终敲定的方案 — PG-DFC v6

### 一句话
> 在 F2DC 基础上,把"盲修复 DFC"改成"prototype-guided 校准 DFC"。给 DFC 输入加上 server 跨 client 聚合的 class prototype bank,通过 attention 让 DFC 知道 nr_feat 该往哪个 class 方向救。

### 改动量
- 约 200 行代码
- 完全兼容 F2DC (proto_weight=0 时退化成原版)
- 不破坏 DFD/wrong-high label/三路监督等任何原设计

### 预期增益
- PACS 76.47 → 78+ (+1.5pp)
- Office 66.82 → 68+ (+1.5pp)
- 实际可能 -0.5 ~ +2pp,必须实测

### 4 个增益来源
1. 消除 F2DC DFD/DFC 梯度冲突: +0.3-0.5pp
2. Prototype 正向 pull 比 wrong-high 反向 push 稳: +0.3-0.5pp
3. 跨 client prototype 信号 (FedProto 增益): +0.5-0.8pp
4. warmup 设计避免破坏 F2DC: 不退步

---

## 三、推翻的论点 / 错误清单 (诚实记录)

| # | Claude 提议 | 错在哪 | 用户/源数据如何揭穿 | 后果 |
|---|---|---|---|---|
| 1 | "wrong-high label 在 PACS top-2 是另一动物会失效" | 把"类相似"跟"域相似"混了说 | 用户问 "client 只训自己模型,怎么是不同 domain" | 修正后严重度从 ★★★★ 降到 ★★★ |
| 2 | "细粒度 vs 粗粒度世界观对立" | 这是 Claude 脑补的,paper 没说 | 用户让看原 paper, 真实是"class signal vs domain context" | 整个 framing 推翻重写 |
| 3 | "F2DC 输给 FDSE 7.3pp,设计有问题" | 数据张冠李戴,跨 setup 借数字 | 用户问 "确定吗 这个数据哪里的" | F2DC 真实是 SOTA 赢 +3pp,所有 critique 重写 |
| 4 | "MC-FL: mask 跨 client consensus 是真问题" | 没数据支持,可能伪问题 | 用户问 "DFD 解耦真的是一个跨 client 共识问题吗" | 改为先做 1 周诊断再说 |
| 5 | "三家 paper 合三为一做 PG-DFC v1" | 没读源码就推方案,A/C 不能搬 | 用户让 clone 源码看 setup | A 是 UDA 不是 FL,C 是 FedDG 不是 personalization,只有 B 能借 |
| 6 | "EU-DFC: F2DC + evidential" | nr 学 wrong-high 跟 evidential 假设冲突 | 用户问 "真的能契合吗" | 改为 skip nr 路只用 r/rec 加 evidential — contribution 量小 |
| 7 | "双 head + prototype 大改" | 改动太大,F2DC 已在跑 | 用户说 "F2DC 已经开始复现了" | 改为小改动,锁定 D 方案 |
| 8 | "EMA 是 FL prototype 共识" | 不是共识,是惯例 | 用户问 "这是共识吗" | 澄清 FedProto/FPL 不用 EMA,EMA 是 2023 后 trend |

---

## 四、真正确认的事 (今日有效产出)

### 关于 F2DC 本身
1. F2DC 是 paper-validated SOTA: PACS 76.47 (vs FDSE 73.13), Office 66.82 (vs FDSE 63.18)
2. F2DC 设计自洽: wrong-high label 是 hard mining, DFC 是配套修复, paper 动机清晰
3. F2DC paper 自己承认的 limitation:
   - DFD 切不干净 (f⁻ 含 class clue)
   - τ 调小反而效果差 (Fig 6)
   - DFC 校准能力有限 (Tab 7)
4. F2DC 实现 vs paper 差异:
   - DA-Agg 写了但代码用 vanilla FedAvg
   - mask 是 per-pixel 但 paper 说 "feature units"

### 关于现有工作
1. RFedDis (B) setup 跟我们一致 (K=4 Office AlexNet) — 唯一可直接复现的参考
2. RFedDis 的"disentangling"实际是双 head + KL repel,不是 mask 切
3. RFedDis 的 evidential CE + DS_Combin 在 PyTorch 实现简单
4. FedAlign 是 FedDG,不能直接借
5. Disentanglement-Then-Reconstruction 是 UDA,prototype reconstruction 思路只能借 idea 不能借代码

### 关于我们的 setup
1. 我们 K=4 1-1 domain personalization (PACS_c4 / Office_c4)
2. R=200 E=5 跟 F2DC 原 setup (R=100 E=10) 不同
3. AlexNet (PACS) / ResNet-18 (Office) 跟 F2DC 原 setup (ResNet-10) 不同
4. 我们 orth_only PACS 80.64 / Office 89.09 — 比 F2DC 论文报数 (76.47/66.82) 高,但 setup 不同不可直接比

### 关于 EMA 在 prototype 中的使用
1. 不是共识,是惯例
2. FedProto/FPL 经典工作不用 EMA
3. EMA 主流但不唯一,可用 round-end snapshot 替代
4. 用 EMA 必须配 warmup,否则早期噪声拖累

---

## 五、5 个重要教训 (避免再犯)

### 教训 1: 不能凭印象引数据
- 二版 critique 引 "F2DC 输 FDSE 7.3pp" 是张冠李戴
- 必须查 paper Tab 原文,确认 setup 一致后才能比

### 教训 2: 不能凭 abstract 推方法
- 三版 critique 借 paper A 的 prototype reconstruction 但 A 是 UDA
- 必须 clone 源码读 setup + method 实现细节

### 教训 3: 不能跨 setup 借鉴
- UDA / FedDG / personalization FL 各有独特约束
- 跨 setup 借鉴需要重新论证可行性

### 教训 4: critique 要诚实
- 不能为了 critique 而 critique
- 不能预设 paper 有缺陷然后找证据
- paper-validated SOTA 通常有它的道理,critique 切入点要从 paper 自己承认的 limitation 找

### 教训 5: 真正的方案必须基于实测数据 + 兼容已有 framework
- 八版 critique 演化到第六版才到位
- 前五版都是凭推理给方案,被用户每次质疑推翻
- 最终方案 (PG-DFC) 是小改动 + 完全兼容 F2DC + 等 sanity 复现完才动手

---

## 六、立即下一步

1. **等 EXP-130 F2DC sanity 复现完成** (已在跑,8 个 bug 都修了)
2. **拿到 F2DC baseline 真实数字** (PACS / Office / Digits 各 3 seed)
3. **W2 Day 1**: 在 F2DC repo 创建 branch `pg-dfc-v1`,加 DFC_PG 类
4. **W2 Day 5**: 跑 PACS 1-seed sanity 验证退化等价 (proto_weight=0 时跟 F2DC 数字一致 ±0.3pp)
5. **W3 Day 1**: 跑 PACS 1-seed full (proto_weight=0.3 + warmup=30) — 拿第一个 PG-DFC 数字

---

## 七、文件清单 (今日产出)

| 文件 | 内容 |
|---|---|
| `2026-04-26/PG-DFC方案_Prototype引导的F2DC校准.md` | 方案主文档,~750 行 14 章节 |
| `2026-04-26/关键实验发现备忘.md` (追加发现 9) | PG-DFC 方案敲定记录 |
| `2026-04-26/今日讨论总结_F2DC_critique_8轮迭代.md` | 本文档 — 今日时间线 + 推翻清单 + 教训 |
| `baselines/RFedDis/` | clone 的 RFedDis 源码 |
| `baselines/FedALign/` | clone 的 FedAlign 源码 |
| `papers/2005.13947.pdf` | Disentanglement Then Reconstruction (UDA, 2020) |
| `papers/2301.12798.pdf` | RFedDis (FL, 2023) |
| `papers/2501.15486.pdf` | FedAlign (FedDG, 2025) |

---

## 八、给未来自己的话

今天 8 轮迭代下来, 最大的感受是:

**Claude 倾向于"给完整方案"而不是"承认不确定"**,每次被推翻都重新提一个看起来更完整的方案。但真正的研究是先做实验拿数据,再设计方案。下次如果 Claude 又开始"凭推理给方案",直接问它:

1. "你这个方案的数据依据是什么?"
2. "你读源码了吗?setup 跟我们一致吗?"
3. "如果 Phase 0 数据反过来,这方案还成立吗?"

这三个问题能挡掉 80% 的虚浮提议。

**最终敲定的 PG-DFC v6 方案是经过 8 轮 critique 沉淀的版本** — 改动量小 (200 行)、F2DC 兼容、有明确 paper 弱点对应、跟我们已有工作衔接好。但**仍然只是预期方案,实际效果必须等 W2-W3 实测才知道**。
