# OCSD 方向:观察现象、机制假设与前期验证计划

> **文档性质**: 方向评估文档 (pre-implementation)
> **目的**: 在写代码之前, 系统梳理观察到的现象、OCSD 的核心假设、所有可能的失败模式, 以及最小成本的验证实验设计
> **创建日期**: 2026-04-24
> **决策属性**: 文档结论决定 OCSD 方向是否值得投入开发资源

---

## 一、观察到的核心现象 (来自 Stage B 诊断数据)

### 1.1 跨方法一致的 "over-confident wrong" 现象

在 PACS Art 域上, **三个机制完全不同的方法**都存在相同比例的"错但自信"样本:

| 方法 | Art over_confidence_wrong 比例 | Art ECE |
|---|:-:|:-:|
| FedBN | 13-15% | 0.190 |
| orth_only | 13-15% | 0.180 |
| FDSE | 13-15% | 0.177 |

**关键观察**: 这三个方法的 mechanism 差异巨大:
- FedBN 只保留本地 BN, 不解耦
- orth_only 用 cos² 正交解耦
- FDSE 用层级 DFE+DSE 迭代擦除

**但都犯同样比例的错**. 这说明问题**不是方法选择**的问题, 而是**训练范式本身**的问题.

### 1.2 FDSE 是"高方差赢法", 不是"方法更好"

FDSE vs FedBN 的 +2.31pp 优势, 分布极不均匀:

```
赢的 cells: +24.33 (Art guitar)  +15.52 (Art horse)  +17.50 (Photo horse)
输的 cells: -13.36 (Art dog)     -8.89 (Art giraffe) -7.56 (Art elephant)
其他 cells: 基本持平
```

**策略性含义**: 打赢 FDSE 不需要做同样极端的事. **在 25/28 cells 各涨 0.3-0.8pp**, 净收益会超过 FDSE 的 +2pp, 且方差更低.

### 1.3 方法收敛速度差异暗示 shortcut 动力学

| 方法 | R_best 分布 (s=2, 15, 333) |
|---|---|
| FedBN | 133, 87, **37** |
| orth_only | 168, 130, ~180 |
| FDSE | 188, 119, 182 |

- FedBN 早熟 (s=333 时 R=37 达峰后下滑 100+ 轮) → 快速学 shortcut 然后过拟合
- FDSE 慢热 (R=180+ 才达峰) → 擦除机制延迟了 shortcut 形成
- orth_only 居中

**推测**: shortcut 学习在训练早期 (R < 50) 快速形成, 后期的 acc 下降可能就是 shortcut 在测试集上失效的表现.

---

## 二、机制假设: 为什么会出现这个现象

### 2.1 核心因果链假设

```
Domain-skewed FL setup (每 client 单域)
         ↓
本地训练时, style 在域内是完美判别器
(Art 里凡是油画笔触 + 大物体 = 大概率吉他 / 大象等)
         ↓
模型学 shortcut: "style pattern → class" 比学 semantic 更快
         ↓
训练 confidence 在这些样本上快速饱和 (>0.95)
         ↓
测试时: 跨域样本 (Photo 的吉他) 没有 Art 笔触
         ↓
shortcut 失效, 模型高 confidence 错成别的类
         ↓
over_conf_wrong 13-15%
```

### 2.2 为什么现有所有方法都解决不了

FDSE / F2DC / I2PFL / FedCCRL / FedAlign 这些方法都做 **"style invariance"**, 但都是**样本无差别**的:

| 方法 | 对所有样本做什么 |
|---|---|
| FDSE | 层级擦除 style 信号 |
| F2DC | Decouple → Calibrate → Merge |
| I2PFL | Feature-level MixUp + MSE 对齐 |
| FedCCRL | Cross-client MixStyle + SupCon + JS |
| FedAlign | MixStyle + SupCon + JS + MSE |

**共同盲点**: 真正依赖 style shortcut 的样本只有 **13-15%**, 其他 85% 靠 semantic 学对. 对全体施加 invariance 会:
- 纠正 13% 偷懒的 (有益)
- 压制 85% 认真的 (浪费梯度 / 可能有害)
- 净效果是"某些 cells 大涨 + 某些 cells 反而掉" (= FDSE 的高方差 pattern)

### 2.3 OCSD 的核心点子

**只对 shortcut 样本施加 invariance, 其他样本放过**.

检测 shortcut 样本的方法: **反事实置信度下降**.

```
对样本 i, 做两次 forward:
  p_orig_i = 原始 forward 的置信度
  p_mix_i  = 在浅层做 MixStyle 扰动后的置信度

shortcut_weight_i = ReLU(p_orig_i - p_mix_i)
```

- 真正依赖 semantic 的样本: style 扰动后预测几乎不变, `shortcut_weight ≈ 0`
- 依赖 shortcut 的样本: style 扰动后预测崩塌, `shortcut_weight` 在 0.3-0.8

**关键性质**: 这个权重**自适应**, 无需阈值超参, 天然在 [0, 1].

### 2.4 和现有方法的 novelty 差异

| Weighting 机制 | 方法 |
|---|---|
| 降权已学会样本 `(1-p)^γ` | Focal Loss |
| 高 loss 样本 | OHEM |
| Uniform | Label Smoothing / FedAlign JSD / I2PFL APA |
| **反事实置信度下降 (counterfactual stability)** | **OCSD (我们)** |

**"Counterfactual confidence drop under style perturbation"** 作为 sample-level detector 在 FL domain shift 文献中为空白. 这是 novelty 的数学 / 概念来源.

---

## 三、可能让 OCSD 失败的 6 个根本性风险

### 风险 1 (最致命): 因果链本身可能不对

**我们假设的链条**:
```
Domain skew → style shortcut → over_conf_wrong 13-15%
```

**但可能实际是**:
```
PACS Art 本身有 13-15% 样本 = 标注歧义 / 本质难分 / 画风极端 → 与 shortcut 无关
```

Art 域里有些"抽象画吉他"可能画得根本不像吉他. 即使训练再好, 测试时也会错, 且因为模型见过类似的训练样本, 错得有 confidence.

**如果 over_conf_wrong 主要是本质难样本**, MixStyle 扰动对它们没用 — 它们错是因为**内容**难, 不是**风格**误导. OCSD 的整个机制前提就不成立.

### 风险 2: MixStyle 扰动太猛, adaptive weighting 退化为 uniform

`shortcut_weight = ReLU(p_orig - p_mix)` 的前提是大部分样本 p_orig ≈ p_mix, 只有少数掉 confidence.

如果 MixStyle α 太大, **所有样本**的 p_mix 都掉 0.1-0.2. `shortcut_weight` 全体 > 0, 权重的区分度消失. OCSD 退化为普通 uniform KL consistency loss, novelty 消失, 和 FedAlign 的 L_JS 没区别.

### 风险 3: OCSD 梯度和 CE 梯度冲突

两个 loss 通过共享的 encoder 后半部分 (bn2-5 / fc6-7) 反向传播:
- L_CE 让 `logits_orig` 拟合 label
- L_OCSD 让 `logits_mix` 向 `logits_orig.detach()` 靠拢

**冲突点**: OCSD 梯度反向流过整个 encoder, 可能干扰 CE 学 label 的方向.

**可观测信号**: 加 OCSD 后 L_CE 不降反升, 或 L_orth 变乱.

### 风险 4: 循环论证 — 用 style 扰动检测 style shortcut

MixStyle 扰动后 p 掉有两种可能:
- (a) 样本依赖 style, 扰动破坏 shortcut (我们想捕捉的)
- (b) bn1 层的 low-level 纹理被改, 下游 conv 没见过这种组合, feature 整体 noisy (干扰)

如果 (b) 占主导, OCSD 检测的**不是 shortcut 样本, 是"对扰动敏感的弱样本"**. 这是 paper 审稿人**一定会问**的问题.

### 风险 5: Art 样本少, 统计噪声大

PACS Art 只有 ~2,000 样本, 4-client 下每 client ~500, 每 batch 50 个里 Art 只占 ~12.

`shortcut_weight` 的 per-batch 估计方差大, 训练不稳定. 可能这轮 shortcut_weight=0.3, 下一轮同批样本 =0.05.

### 风险 6: Stage B 诊断数据本身可能不稳定

- 13-15% 这个数字是 best-round 快照还是平均?
- 不同 seed 下这个数字一致吗?
- 不同 round 下这个数字波动多大?

如果 13-15% 本身是高方差点估计, **OCSD 整个 motivation 的基础就不稳**.

---

## 四、前期验证计划: 6-8 小时筛掉高风险假设

### 验证 1: 挖出 over_conf_wrong 样本的真实身份 (1-2 小时, 最关键)

**目的**: 检验风险 1 和风险 6 — over_conf_wrong 是 style shortcut 还是本质难样本?

**做什么**:
1. 用已训完的 orth_only 模型, 对 PACS Art 测试集, 对每个样本记录:
   - 原图文件名
   - 真 label、预测 label
   - 对真 label 的置信度 p_true
   - 对预测 label 的置信度 p_pred

2. 筛出 `p_pred > 0.8 且 预测 != label` 的样本 (这就是 over_conf_wrong 集合)

3. **人眼看这些图 50 张左右**, 做三类判断:

| 判断类别 | 含义 | 下一步 |
|---|---|---|
| 50 张都是"风格极端 Art 图" (油画感强、笔触重) | shortcut 假设成立 | 继续 OCSD 方向 |
| 一半风格极端, 一半"画得不像" | 假设部分成立 | OCSD 能修一半问题, 可继续但期望降低 |
| 主要是"标注歧义 / 画得根本不像" | **假设失败** | **放弃 OCSD, 换 RCA-GM** |

4. 跑 3 个 seed 分别筛, 看 over_conf_wrong 样本集的 **IoU**:
   - IoU > 60% → 结构性问题, 稳定且可解
   - IoU < 30% → 错得随机, OCSD 方向不靠谱

**这一步是 OCSD 方向的生死线**. 花 1.5 小时看 50 张图, 比花 8 小时写代码后发现方向错有价值 10 倍.

### 验证 2: MixStyle 扰动强度校准 (1 小时)

**目的**: 检验风险 2 — shortcut_weight 是否能区分 shortcut vs semantic 样本?

**做什么**:
在一个**已训好**的模型上 (不改训练, 只 inference), 对 Art 测试集:

1. 记录 p_orig (原始 forward)
2. 记录 p_mix 在四种扰动设置下:
   - MixStyle at bn1, α=0.3
   - MixStyle at bn1, α=0.1
   - MixStyle at bn2, α=0.3
   - MixStyle at bn2, α=0.1

3. 画**直方图**:
   - x 轴: shortcut_weight = ReLU(p_orig - p_mix)
   - y 轴: 样本数
   - 两组颜色: 绿色 = over_conf_correct, 红色 = over_conf_wrong

**理想结果**:
- 红色 (wrong) 样本的 shortcut_weight 明显右偏, 均值 > 0.3
- 绿色 (correct) 样本集中在 0 附近, 均值 < 0.1

**失败结果**:
- 两组分布完全重叠 → shortcut_weight 无区分能力 → OCSD 检测器失效

**决策**: 哪组超参让两个直方图分离最明显, 用那个. 如果没有任何超参能分离, **放弃 OCSD**.

### 验证 3: 梯度冲突检查 (1 小时)

**目的**: 检验风险 3 — OCSD 梯度是否和 CE 梯度打架?

**做什么**:
不做完整训练, 只用 2 个 batch:

1. 用已训好的 orth_only 模型
2. 在同一批数据上分别计算:
   - g_CE = ∇ L_CE
   - g_OCSD = ∇ L_OCSD
3. 计算 `cos(g_CE, g_OCSD)`:

| cos 值 | 含义 | 行动 |
|---|---|---|
| > 0.3 | 梯度方向一致 | 安全, 继续 |
| -0.3 到 0.3 | 基本独立 | 安全, 继续 |
| < -0.3 | 反向打架 | 加 stop-grad 或换 MixStyle 层位置 |

4. **更严格**: 算每一层 conv 的分层梯度 cos, 找出冲突最严重的层.

**如果怎么改都冲突**: OCSD 和 CE 架构不兼容, 放弃.

### 验证 4: 循环论证自检 (30 分钟)

**目的**: 检验风险 4 — MixStyle 扰动的效果是 style-specific 还是 general noise?

**做什么**:
造一个**控制实验**, 用 Gaussian noise 替代 MixStyle:
```python
h_noisy = h + torch.randn_like(h) * std
```

分别看:
- MixStyle 扰动下, over_conf_wrong vs over_conf_correct 的 shortcut_weight 分布差异
- Gaussian noise 下, 同样的分布差异

**理想结果**:
- MixStyle 能 specifically 区分两组 (机制干净)
- Gaussian noise 对两组都掉置信度 (说明 MixStyle 是 style-specific 的)

**失败结果**:
- Gaussian noise 和 MixStyle 效果一样 → OCSD 检测的是"对扰动敏感的弱样本", 不是 shortcut 样本
- Novelty 受质疑, 但仍可能有效果

**审稿人防御**: 这个控制实验**一定会被问**, 提前跑好存图.

### 验证 5: 小规模端到端 (2-3 小时)

前四步都通过了, 才做这步.

**做什么**:
只跑 **PACS seed=2** 的完整 R200, 和 orth_only baseline 对比.

**观察指标**:

| 指标 | 期望 | 含义 |
|---|---|---|
| AVG Best | 涨 ≥0.3pp | 初步效果信号 |
| L_CE 曲线 | 和 baseline 相似下降 | OCSD 没打断主学习 |
| P1: 平均 shortcut_weight | 先升后降 (R 30-80 升, R 100+ 降) | 机制: 先学 shortcut, 后被压制 |
| P2: over_conf_wrong ratio | 从 13-15% 降到 <10% | 测试时 shortcut 被修好 |
| P3: Art ECE | 从 0.18 降到 <0.12 | 校准改善 |

---

## 五、决策树 (验证结果 → 下一步行动)

```
验证 1 (看 50 张图)
├─ over_conf_wrong 主要是标注歧义 → 放弃 OCSD, 换 RCA-GM
├─ 主要是风格极端 Art 图 → 继续
└─ 3-seed IoU < 30% → 样本集不稳定, 换方向
         ↓
验证 2 (扰动强度)
├─ 直方图两组分离明显 → 继续 (记录最佳 α 和层位置)
├─ 重叠 → 调参再试
└─ 怎么调都重叠 → 放弃 OCSD
         ↓
验证 3 (梯度冲突)
├─ cos ∈ [-0.3, 0.3] → 继续
├─ cos < -0.3 但可缓解 (stop-grad / 换层) → 缓解后继续
└─ 无法缓解 → 放弃 OCSD
         ↓
验证 4 (循环论证)
├─ MixStyle 有 specific 效果 → novelty 成立, 继续
└─ Gaussian noise 和 MixStyle 效果一样 → 机制可疑但可试
         ↓
验证 5 (端到端 1 seed)
├─ acc 涨 + P1/P2/P3 都动 → 成功, 跑 3-seed 大实验
├─ acc 涨但 P 不动 → 机制不对但有效, 当 engineering trick
├─ acc 不涨且 P 不动 → 失败, 放弃
└─ acc 不涨但 P 动 → 机制对但 leverage 不够, 叠加 RCA-GM
```

---

## 六、实施细节的 5 个关键陷阱 (真要做时别踩)

### 陷阱 A: BN running stats 污染

AlexNet 的 BN 层在 train 模式会更新 running mean/var. 第二次 forward (MixStyle 扰动后) 如果也在 train 模式, 会**用扰动数据污染 BN 统计量**, 推理性能会莫名变差.

**解决**: 第二次 forward 前把 BN 切到 eval 模式, 或用 `torch.no_grad()` 包住不更新的部分.

**这个坑很难 debug**: 训练 loss 正常, 测试精度莫名其妙. 必须注意.

### 陷阱 B: MixStyle 的统计量维度

```python
# 正确: 只在 spatial 维度求统计, channel 独立
mu = feat.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

# 错误: 跨 channel 求统计, 会把不同通道的风格混了
mu = feat.mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
```

### 陷阱 C: batch 内 class 稀疏

batch 里某 class 可能只有 1 个样本, MixStyle 采样可能采到异类风格.

**选择**: MixStyle 随机采样不区分 class — 可接受, 因为 MixStyle 混的是**风格** (低层统计), 不是 content.

### 陷阱 D: KL 方向搞错

```python
# 正确: logits_orig 作参考 (detach), 让 mix 向 orig 靠
kl_div(log_softmax(logits_mix), softmax(logits_orig.detach()))

# 错误: 反过来会让模型保持被扰动状态, 训练崩溃
kl_div(log_softmax(logits_orig), softmax(logits_mix.detach()))
```

### 陷阱 E: 计算成本

双 forward = 2x encoder 前传. AlexNet 前传 ~30ms, 训练时间增加 50-80%.

**如果时间紧**: 每 2 步做一次 OCSD forward (stochastic OCSD), 成本 +25-40%.

---

## 七、方向 B (RCA-GM) 的简短描述 (作为 plan B)

如果 OCSD 的验证 1 就失败了, 直接切换到 RCA-GM:

**核心思想**: 现有所有聚合方法 (I2PFL、F2DC、FedProto) 都用**均值**作跨域锚点. 在 PACS 里 Art 是 outlier 域, 任何以均值为基准的方法都在**主动把 Art 朝其他域拉**. 改用 **geometric median** (理论上 50% breakdown point) 作锚点, Art 保留自己的 class 结构.

**机制**:
- Client 上传 per-class z_sem 均值 (每轮 K×128 floats)
- Server 用 Weiszfeld 迭代算 geometric median
- Client loss 加 Huber anchor (对 outlier 鲁棒)

**为什么值得作 backup**:
- Novelty 在 FL domain shift 零应用 (理论漂亮)
- 实施独立于 OCSD, 不共享失败风险
- 预期 Art 涨 1-2pp, Office AVG +1-1.5pp

**OCSD 失败的原因可能恰好是 RCA-GM 的理由**: 如果 over_conf_wrong 是 Art 本质难样本, 那说明 Art 域本身结构就不同, 用鲁棒锚点保护 Art 结构可能更对症.

---

## 八、时间规划建议

| 时间 | 活动 | 交付物 |
|---|---|---|
| Day 1 上午 | 验证 1: 挖样本 + 看 50 张图 | 判断 OCSD 方向生死 |
| Day 1 下午 | 验证 2: 扰动强度校准 | 直方图 + 最佳超参 |
| Day 2 上午 | 验证 3: 梯度冲突检查 | cos 值报告 |
| Day 2 下午 | 验证 4: 循环论证自检 | Gaussian vs MixStyle 对比图 |
| Day 3 | 验证 5: PACS seed=2 端到端 | acc + P1/P2/P3 曲线 |
| Day 4 | 决策: 3-seed 大实验 / 叠加 RCA-GM / pivot | 下一步 plan |

---

## 九、一页纸总结 (给你自己复习用)

**现象**: PACS Art 在 FedBN / orth / FDSE 三个机制不同的方法上都有 13-15% over-conf-wrong, ECE 都是 0.18.

**假设**: 这 13% 是 style shortcut 样本 — 本地训练时 style 是完美判别器, 模型偷懒学 style 而非 semantic, 测试时跨域失效.

**方法**: OCSD 用"反事实置信度下降"作 sample-level detector, 只对 shortcut 样本施加 invariance loss, 其他样本放过.

**novelty**: Counterfactual confidence drop as shortcut detector, 在 FL domain shift 文献零应用.

**6 个风险**:
1. 13% 可能是本质难样本不是 shortcut
2. MixStyle 太猛会让权重退化为 uniform
3. OCSD 和 CE 梯度可能打架
4. 循环论证: 用 style 扰动检测 style shortcut
5. Art 样本少, 统计噪声大
6. Stage B 的 13% 数字本身可能不稳定

**5 步验证** (按顺序, 任一步失败立即止损):
1. 人眼看 50 张 over_conf_wrong 图 (1.5h) — 生死判定
2. 扰动强度校准, 画直方图 (1h)
3. 梯度冲突 cos 值检查 (1h)
4. Gaussian noise 控制组 (0.5h)
5. PACS seed=2 端到端 (3h)

**Plan B**: 如果验证 1 就失败, 切 RCA-GM (geometric median + Huber anchor).

**核心纪律**: **不要有 sunk cost 陷阱**. 验证 1 失败立即放弃, 不自我欺骗.
