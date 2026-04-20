# EXP-108 CDANN 完整流程 — 大白话版

**对应技术版**: [EXP-108_CDANN流程_完整技术版](EXP-108_CDANN流程_完整技术版.md)
**一句话**: 给解耦加个"警察", 告诉模型什么是语义什么是风格, 修 PACS whitening 把风格类信号误擦的毛病.

---

## 🤔 这个实验为什么要做?

### 背景(不用看懂可以跳过)

我们之前的方案 **FedDSA-SGPA** 在不同数据集表现**打架**:

| 数据集 | 我们方法 | 只做 Plan A (基础) | 差距 |
|-------|---------|-------------------|------|
| **Office** (普通相机照片) | **88.75** 🔥 | 82.55 | **+6.20pp** ✅ |
| **PACS** (油画/素描/卡通/照片) | 80.20 ⚠️ | 81.69 | **-1.49pp** ❌ |

**同样一个方法**, Office 涨 6 分, PACS **反而掉**! 这叫**"数据集边界"**.

### 诊断发现了啥

我们看了训练过程的数据 (z_sty_norm, 即"风格信号强度"):

- **Office Linear+whitening**: z_sty 从 R10=3.12 → R200=2.21 (**只磨掉 2%**)
- **PACS Linear+whitening**: z_sty 从 R10=3.12 → R200=0.15 (**磨掉 95%!**)

**PACS 的风格被 whitening 当成噪声擦掉了**, 但它的风格 (素描的线条/油画的纹理) **本身就是识别类别的关键信号**. 擦掉 = 丢了类信号 = 精度掉.

### 根本问题

我们原来的解耦约束 `cos²=0 + HSIC=0` 只是说**"让 z_sem 和 z_sty 不相关"**, 但**没告诉模型谁是谁**. 模型可能把类判别信号**错放到 z_sty 里**, 然后被 whitening 擦掉.

**比喻**: 两个人站在房间里不干扰彼此(统计独立), 但没人告诉他们谁是警察谁是小偷. 结果他们可能站反了位置 — 你把"小偷"当噪声赶出去, 其实把警察也赶了.

---

## 💡 我们的方案 CDANN

### 核心思路 (一个比喻)

加一个**"裁判 dom_head"** (小 MLP, 9K 参数), 它永远**正向**猜"你这个特征来自哪个客户端/域". 然后**让 encoder 学会**:

- **z_sem 必须让裁判猜不出域** (通过 GRL 反向梯度, 把 z_sem 磨成 domain-blind)
- **z_sty 必须让裁判 100% 猜对域** (正向梯度, 把 z_sty 加强成 domain-专家)

**关键细节**: 裁判 dom_head **本身不对抗**, 两路都是 "我想猜对" (standard CE). **对抗发生在 encoder 端** — encoder 看到"裁判用 z_sem 猜对了", 通过 GRL 反转梯度就**反向学习** "我要让 z_sem 更混乱".

### 精确架构 (一张图胜千言)

```
                                                  ┌── sem_classifier(z_sem) → 预测 y   [正向: encoder 学类别]
                                                  │
 x → encoder → feature ─┬─ sem_head → z_sem ──────┤
                        │                          └── dom_head(GRL(z_sem, λ)) → 预测 d  [GRL: encoder 被反向推]
                        │
                        └─ sty_head → z_sty ──── dom_head(z_sty) → 预测 d              [正向: encoder 被正向推]

 dom_head 自己: 两路都 minimize CE (普通分类器, 不对抗)
 asymmetry 只在 encoder 的梯度方向
```

### 一句话记住

> **把"让 z_sem 和 z_sty 统计独立"升级为"让 z_sem 变域盲 + z_sty 变域专家", 用一个共享裁判 + 非对称梯度方向实现, 成本 9K 参数, 专治 PACS 里 whitening 擦风格类信号的病.**

---

## 🛠️ 我们做了什么 (8 大步, 大白话版)

### ① 问题发现 (已完成)

跑 EXP-098 PACS + EXP-102 Office, 诊断数据暴露 "PACS z_sty_norm 塌 95%" 问题.

### ② 文献调研 (已完成, 30 篇论文)

看了 2024-2026 顶会相关工作 (文件 `LITERATURE_SURVEY_DANN_AND_DECOUPLING.md`):
- **跟擦除派的区别**: FedPall/ADCOL/FDSE 擦掉域信息, **我们保留** (z_sty 专门装域信息)
- **跟监督派的区别**: Deep Feature Disentanglement SCL (2025) 是对称双头非 FL, **我们非对称 + FL**

### ③ 头脑风暴 (已完成, 12 个方案)

`IDEA_REPORT_2026-04-20.md`, 最后选 I2 CDANN (novelty 9/10, risk 最低).

### ④ 5 轮 Codex gpt-5.4 精炼 (已完成)

分数演化: **R1 7.1 → R2 8.35 → R3 8.4 → R4 8.75 → R5 8.75** (proposal-complete).

R5 审稿人原话: **"near review-time ceiling, novelty ceiling 是内在的, 只有实验结果能再提分"**.

5 轮改了啥:
- **R1**: 合并两个 heads 为 1 个 shared head (不用 2 个模块)
- **R2**: 精确表述 "dom_head 自己非对抗, asymmetry 在 encoder 梯度方向"
- **R3**: 加 probe_sty_class (z_sty→class) 直接证明 anchor
- **R4**: 写法微调, 锁定 one-sentence novelty

### ⑤ 写代码 + 单测 (已完成, +120 行 + 14 新测试)

`FDSE_CVPR25/algorithm/feddsa_sgpa.py`, 新增:
- `GradientReverseLayer` (梯度反转)
- `dom_head` MLP (128→64→4)
- `lambda_adv` 三段式 schedule (R0-20=0, 20-40 ramp, 40+=1)
- CDANN 专属 5 个诊断指标

**测试 55/55 全绿** ✅

### ⑥ Codex 代码审查 + 修复 (已完成)

Codex 找出 1 个 CRITICAL + 2 IMPORTANT + 1 MINOR:
- **CRITICAL**: warmup 写错了 — R<20 不是真 baseline, L_dom_sty 仍在训练. **改**: 加 warmup gate 彻底跳过 CDANN loss
- **IMPORTANT**: task heuristic 不匹配 runtime client 数 → Server 动态 rebuild dom_head
- **IMPORTANT**: `ca=1+ue=1` 没拦 → raise ValueError

全部修完, 单测仍 55/55 绿.

### ⑦ 部署训练 (待执行)

**6 runs 并行** (PACS 3 seeds + Office 3 seeds), seetacloud2 单 4090, 预计 12h wall.

### ⑧ 回填 NOTE + 诊断分析 (待完成)

训练完跑 `run_frozen_probes.py`, 填 NOTE 主表 + 诊断表 + probe 表, 同步到 Obsidian.

---

## 🔬 诊断指标 — 我们到底测了啥

**核心**: **两层诊断, 一层训练中, 一层训练后**.

### 第一层: 训练端 (每 5 轮自动记)

**21 个原有 + 5 个新 CDANN = 26 个指标**, 全部写到 jsonl 文件.

**最关键的 3 个看点**:

1. **`z_sty_norm`** (风格信号强度) — 生死线
   - Baseline PACS: R10=3.12 → R200=**0.15** (被 whitening 磨成 0)
   - CDANN PACS 期望: R200 ≥ **1.5** (保留下来)
   - 这个指标看懂了, 方案成败立判

2. **`loss_dom_sty`** (z_sty 预测域准不准)
   - 期望从 R40 后 **下降到接近 0** (dom_head 能从 z_sty 100% 猜出域)
   - 证明: **正向监督成功**, z_sty 真的变成 domain-专家

3. **`loss_dom_sem`** (z_sem 预测域难不难)
   - 期望从 R40 后 **上升到 ≈ log(4) = 1.39** (dom_head 猜不出, 接近随机)
   - 证明: **GRL 对抗成功**, encoder 让 z_sem 变 domain-blind

### 第二层: 训练后 (单独跑一次 `run_frozen_probes.py`)

训练完后**冻结整个模型**, 然后**重新训 3 个超简单的 Linear probe**:

| Probe | 干啥 | 期望 | 证明啥 |
|-------|------|------|--------|
| `probe_sem_domain` | 从 z_sem 猜 domain | ≈ **25%** (4 个 client 里瞎猜 1/4) | GRL 把 domain 从 z_sem 里挤干净了 |
| `probe_sty_domain` | 从 z_sty 猜 domain | ≥ **95%** | 正向监督让 z_sty 充满 domain 信息 |
| **`probe_sty_class`** 🔥 | **从 z_sty 猜 class** | **PACS ≥ 40% (baseline 15%)** | **anchor 核心**: z_sty 里有类信号 (之前被 whitening 擦掉, 现在被 CDANN 保住了) |

### 为啥要两层

- **训练端**: 看机制有没有**在 train 时**按预期工作 (比如 GRL 到第 40 round 是否真的起效)
- **训练后 probe**: 看**最终 representation 本身**是否真的如我们声称的那样分工. 训练端 dom_head acc 是 jointly trained 的, 有点"自测自考试"嫌疑; 冻结后重训 probe 才是客观的 representation 质量评估.

---

## 🎯 成败判定 (Verdict Decision Tree)

```
(A) PACS AVG Best ≥ 82.2 + Office ≥ 88.0 + PACS probe_sty_class ≥ 40%
    → ✅ 全胜, CDANN 方案 validated, 开始写论文

(B) PACS Best ∈ [80, 82) 但 probe_sty_class 确实 ≥ 35%
    → ⚠️ 机制对了 (representation 证据), 但 empirical gain 不足
    → 可能需要调 λ_adv schedule, pilot 再试 seed=2

(C) PACS Best < 80 (掉基线)
    → ❌ 方案 fail, 检查 dom_head 是否正确聚合 / GRL 方向
    → Pivot 到 Option I1 (Selective Whitening)

(D) Office Best < 87 (伤 Office > 1.75pp)
    → ⚠️ CDANN 在 style-weak 数据集有副作用
    → λ_adv 降到 0.5, 或只 PACS 启用
```

---

## 📂 相关文件

- 技术版 (含完整公式): [EXP-108_CDANN流程_完整技术版](EXP-108_CDANN流程_完整技术版.md)
- 最终 proposal: `refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md`
- 5 轮 review 全记录: `refine-logs/2026-04-20_FedDSA-CDANN/REVIEW_SUMMARY.md`
- 文献综述: `LITERATURE_SURVEY_DANN_AND_DECOUPLING.md`
- 代码主文件: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (+120 行)
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (55/55 绿)
- Configs: `FDSE_CVPR25/config/{office,pacs}/feddsa_cdann_*_r200.yml`
- Frozen probe 脚本: `FDSE_CVPR25/scripts/run_frozen_probes.py`
- 实验 NOTE: `experiments/ablation/EXP-108_cdann_office_pacs_r200/NOTE.md`

---

## 一句话总结当前状态

**从问题发现到代码完成, 写了 4600+ 行文档 + 130 行代码 + 14 个新测试 + 5 轮 Codex 精炼 (8.75/10) + 1 轮 Codex 代码审查 (REVISE 全修). 诊断两层全就位 (26 训练 + 3 frozen probe = 29 指标). 剩下就是部署跑 12h + 回填数据了.**
