# 大白话_FedDSA-BiProto方案

> 对应学术版: `refine_logs/2026-04-24_feddsa-biproto-asym_v1/FINAL_PROPOSAL.md`
> 创建日期: 2026-04-24
> 来源: 4 轮 GPT-5.4 xhigh review 后最终方案 (8.75/10 near-READY)

---

## 一句话记住

> **把"风格"从"要擦掉的噪音"升级为"要好好保留的档案"**, 在服务器上给每个地区(域)建一张身份卡(Pd), 然后强制让"类别方向"和"风格方向"在几何上互相看不见.

---

## 一、这事儿到底在解决啥?

### 场景 (联邦学习 + 跨域)
- 有 4 个公司 (client), 每个公司手里有一批图
- 公司 A 的图是**照片**, 公司 B 的图是**素描**, 公司 C 是**卡通**, 公司 D 是**画作** — 这叫"跨域"(不同 domain)
- 数据**不能汇总** — 隐私, 只能参数合并 (叫 federated learning)
- 任务: 让一个模型学会"这是狗", 不管它是照片狗还是素描狗

### 战场现状
- 对手 = FDSE (CVPR 2025 的论文方法)
- 两个数据集: PACS (4 个域, 7 类) + Office-Caltech10 (4 个域, 10 类)
- 我们目前的基线叫 **orth_only**:
  - PACS 上: 我们 80.64, FDSE 79.91 → **我们赢 +0.73** ✅
  - Office 上: 我们 89.09, FDSE 90.58 → **我们输 −1.49** ❌
- **任务: Office 必须补回 −1.49, 再涨 +0.5; PACS 不能退**

---

## 二、现在大家怎么做? 为啥不够?

用**图书馆比喻**:

| 流派 | 怎么处理"风格"? | 比喻 | 缺陷 |
|---|---|---|---|
| **擦除派** (FDSE, FedDP) | 当噪音擦掉 | 图书管理员把所有书脊的花纹抹掉, 只留内容 | **浪费信息** — 风格其实是有用的信号 |
| **私有派** (FedSTAR, FedBN) | 每家公司自己留着, 不共享 | 每个图书馆把自己地区特色书单藏起来, 不给别家看 | **错失跨馆互补** — 甲馆从没见过乙馆的素描风格 |
| **混着用派** (FISC, MixStyle) | 跨公司直接换风格 | 照片书糊上素描书的封面再给你看 | **乱迁移** — 顺手把内容也换了, 越学越差 |
| **对抗派** (CDANN) | 用 GRL 强制让"内容"表征不含域信息 | 管理员拿小鞭子抽内容书架, 不许它带任何地区口音 | **我们实验证伪** (EXP-108): 跟没训一样, 还把类别信息挤到了风格那边 |
| **我们** (BiProto) | **建一个"域身份档案馆"** | 每个地区派代表去中央档案馆录一张身份卡, 卡上写"我这儿是素描口音" | 首次把"域"当一等公民, 不擦不私不乱 |

---

## 三、我们方案的 3 个核心零件 (全大白话)

### 零件 1: "档案馆" = Federated Domain Prototype Pd

- **比喻**: 中央档案馆有一个抽屉, 里面有 4 张身份卡 (4 个域), 每张卡 128 维浮点数 (就是每个域的"风格指纹")
- **每轮训练末**: 每个客户端算一下自己这一轮所有样本的"风格向量"平均值, 寄给服务器
- **服务器**: 把寄来的平均值更新进档案馆对应那张卡 (EMA 平滑更新, 新 10% + 旧 90%)
- **下一轮**: 所有客户端都拿到这张最新的档案卡

**为啥重要**: 这是**第一次**在 FL 里把"域"当成和"类别"平级的一阶对象. 现有方法要么擦掉域, 要么锁在本地, 没人把它做成"跨公司档案".

### 零件 2: "专门看风格的小脑子" = Asymmetric Statistic Encoder

- **比喻**: 识别一张画是不是"素描风格", 不用看画了啥, 只要看**整体颜色分布 (μ) 和颜色起伏 (σ)** 就够了 — 就像看书脊颜色就能猜是哪个出版社, 不用读内容
- **实现**: 从 AlexNet 前 3 个 conv 层取出每个通道的 μ, σ (均值和标准差), 拼起来送一个 2 层小 MLP (~1M 参数) 输出 z_sty
- **为啥非对称?**: 语义 encoder 是大脑袋 (60M AlexNet), 风格 encoder 是小脑袋 (1M MLP). **原版 DualEncoder 方案是两个大脑袋 (120M), 我们砍到 ~61M**, 省了一半参数
- **关键细节**: 风格 encoder 的输入是 `detach()` 过的, 梯度不回流到 AlexNet — 这样 AlexNet 的训练路径和原来 orth_only 一模一样, 不破坏已经赢了 PACS 的配方

### 零件 3: "内容和风格互相看不见" = Prototype-Level Geometric Exclusion

- **比喻**: 想象一个黑板, 横轴是"类别方向", 纵轴是"域方向". 我们强制让这两根轴**垂直**. 这样你沿着"类别轴"走的时候, 不会偷偷掺杂"域信息"
- **怎么做到**: 在低维 (128d) 的原型空间里, 算"类别原型 Pc" 和 "域原型 Pd" 的 cosine 角度, loss 把它推到接近 0 (垂直)
- **妙处**: 这个操作**只在 128 维的小空间里做**, 不是在整个 feature map 上做. 梯度冲突半径极小, 不会像历史上 HSIC 那样全局污染 (EXP-017 证伪过)

---

## 四、一个教训: Straight-Through 黑魔法

这里有个**最容易被审稿人抓的细节**, 我们踩了 2 轮坑才搞对:

### 问题
说好"用 Pd (中央档案馆那张卡)做互斥", 但 Pd 是 no-gradient buffer (不能反向传播), 那训练时梯度怎么到 encoder_sty?

### 方案 A (朴素,被 reviewer 否决)
直接用 **batch 本地的域平均** 代替 Pd 做 exclusion.
- 问题: **头条 (headline) 说用 federated Pd, 实际代码用 batch local** — 审稿人一眼识破, 大扣分

### 方案 B (straight-through 黑魔法, 最终采用)
```python
raw = Pd[d].detach() + bc_d - bc_d.detach()
domain_axis[d] = F.normalize(raw)
```

**翻译成大白话**:
- 前向 (forward): `raw = Pd − 0 = Pd` → 看起来用的就是 federated Pd ✅
- 反向 (backward): `grad = 0 + bc_d − 0 = bc_d 的梯度` → 梯度实际走 batch 平均回传到 encoder_sty ✅

**比喻**: "名义上我打的是中央档案馆那张正宗卡 (headline 对齐), 训练时实际改的是手上的临时便签 (gradient 有信号). 等训练久了 EMA 稳了, 临时便签和正宗卡会收敛成一样的东西, 所以这个小偏差会自己消失."

这就是审稿人一开始最不满、最后说"搞定了"的那个点.

---

## 五、我们加了什么"验证性实验"?

用户明确要求: 不能只看 accuracy, 要有 t-SNE 图证明"解耦真的做成了". 所以方案内嵌 **3 套可视化 evidence**:

### Vis-A: t-SNE 双面板 (最直观的图)

| 左半图 | 右半图 |
|---|---|
| z_sem 散点图, 颜色按**类别**涂 | z_sty 散点图, 颜色按**域**涂 |
| **预期**: 同类颜色聚一团, 跨域混合 (说明语义纯净) | **预期**: 同域颜色聚一团, 跨类混合 (说明风格纯净) |

如果 t-SNE 图长成这样, 就是"真解耦成功"的图证据. 如果两张图长得一样乱, 那叫"假解耦".

### Vis-B: Probe 梯子 (quantify 能读出多少信息)

训练一个小 MLP (线性 / 64 维 / 256 维) 去从 z_sem 或 z_sty 预测类别或域. **读不出来 = 真没编码进去**.

| 应该读得出 | 应该读不出 |
|---|---|
| z_sem → 类别 (>0.85, 因为语义 encoder 就是学类别的) | z_sem → 域 (<0.55) |
| z_sty → 域 (>0.95, 因为风格 encoder 学的就是域) | z_sty → 类别 (<0.50, **这是关键** — orth_only 当前能读到 0.81, 我们要压到 < 0.50) |

### Vis-C: 原型质量表

- 类别原型之间是不是两两远 (cosine off-diagonal 小)
- 域原型之间是不是两两远
- 类别原型 ⊥ 域原型 (互相垂直)
- z_sem / z_sty 的范数不坍缩到 0 (防止模型偷懒)

---

## 六、我们做了 4 轮 review 改了啥? (给自己 6 个月后看)

| 轮次 | 分 | 审稿人主要批评 | 我们怎么改 |
|:-:|:-:|---|---|
| R1 | 6.5 | "组件太多" — 你说有双原型但实际 L_sem_proto 和 L_sty_proto 是两条平行 contribution, 稀释焦点 | 删 L_sem_proto, 把 Pc 降级为辅助监控 |
| R2 | 7.8 | "**头条和实现不对齐**" — 说用 Pd 实际用 batch; "**C0 诊断冻死了**" — 整个预测路径都冻住, 根本测不到 add-on 的增量 | 搞了 straight-through 黑魔法对齐头条; C0 只冻 AlexNet, 保留 head 可训 |
| R3 | 8.25 | "D=K=4 (4 个公司 4 个域) 下, 'federated 档案馆' 和 '每公司自己一张卡' 实现上一样, 你怎么证明 Pd 真的有用?" | 加 `to our knowledge` 措辞, 加 Limitations 小节, 把 **−Pd ablation 升为 MANDATORY** (必做对比实验, 换成 batch-local 只有服务端 buffer, 看差多少) |
| R4 | 8.75 | "设计层面没问题了, 剩下是实验问题 — 你只能跑 −Pd 实验才能回答" | 把 4 个细节 AI 合并到 FINAL, 接受 "继续 refine 边际收益为零, 该去跑实验" |

**为啥停在 R4 不继续 R5?**

因为 reviewer 在 R4 明确说:
> "v4 的设计层面问题全解决了. 剩下唯一阻止 READY 的, 不是设计瑕疵, 是实证问题 — 只有跑 −Pd ablation 才能回答 Pd 在 D=K=4 下是否真有用."

换言之: 继续改文字没用, 分数上不去了. 该去写代码跑实验.

---

## 七、实际跑起来长啥样? (Stage-gated pilot)

| 阶段 | 干啥 | GPU-h | 要过关才进下一步 |
|:-:|---|:-:|---|
| **S0** | **先做诊断**: 用已有 Office checkpoint, 冻 AlexNet, 加上风格支路训 20-30 round, 看能不能涨 | 2 | ≥ +1.0pp 强信号; +0.3~+1.0 弱信号; < +0.3 **直接杀掉整个方案** |
| 实现 | 写代码 ~250 行 + 单元测试 + codex review | 0 | 测试全过才进 |
| **S1** | Office seed=2 跑满 R200 | 4 | AVG Best ≥ 90.0 |
| **S2** | Office 3-seed 跑满 | 20 | 3-seed mean ≥ 91.08 (**超 FDSE 才算赢**) |
| **S3** | PACS 3-seed 跑满 | 30 | 3-seed mean ≥ 80.91 (不得退) |
| **S4** | 做消融实验 (-Pd 必做, 其他推荐) | 40 | - |
| 出图 | 三套可视化 | 2 | - |
| **总计** | **≤ 98 GPU-h**, 但 Pilot (S0+S1+S2) 只要 **≤ 26**, 严格在 50 GPU-h 预算内 | | |

**最重要的是 S0**: 如果 S0 失败 (< +0.3), 说明 Office 的瓶颈根本不在"架构解耦不够", 而在"数据集本身特性 / 聚合方式 / Caltech 是 outlier 域", 那花 40 GPU-h 跑完整 BiProto 也没用, 赶紧回去做 Calibrator 兜底或者 SAS τ 调聚合.

---

## 八、和我们历史上别的方案什么关系?

| 方案 | 做了啥 | 结果 | 教训 |
|---|---|---|---|
| **orth_only** (基线) | 语义头 + 风格头, 加一个正交损失 cos² | PACS ✅ +0.73, Office ❌ −1.49 | 目前最好, 但 Office 不够 |
| SCPR (EXP-095) | 多原型 InfoNCE, style attention | **全线证伪** | 4 client 区分度不足, InfoNCE 没安全阀会崩 |
| SGPA (EXP-096-100) | Fixed ETF classifier + pooled whitening | Office 86.97 但被 Linear 对照 88.75 反超 | ETF 在 Office 反而减分; whitening 真 gain |
| CDANN (EXP-108) | GRL 对抗, 强制 z_sem 去 domain | probe 0.96 但 accuracy 0 | 对抗路线完全死, **不要再试** |
| VIB (EXP-113) | 信息瓶颈, 压缩风格 | Office +0.76 / PACS -0.72 | regime-dependent, 不通用 |
| Calibrator (兜底 A) | 在 z_sty 上加 MLP 提取 class 信号辅助分类 | 简单, 预期 +0.4~2.4pp, novelty 低 | 兜底保底 |
| DualEncoder (兜底 B 原版) | 2 份完整 AlexNet 独立跑 | 参数 2×, 通信 2× | 太贵, 被 reviewer 否 |
| **BiProto (本方案, 兜底 B 精简版)** | 非对称 encoder + Federated Pd + proto-level ST exclusion | **待实验** | Novelty 中上, 参数 +1.7%, 通信 +2% |

---

## 九、Paper 卖点一句话

> *"To our knowledge, we are the first to maintain a federated domain prototype object Pd — a server-side EMA centroid across participating clients — and use it as the forward anchor of low-dimensional prototype-space class-domain exclusion via a straight-through hybrid gradient axis, empirically validated on D=K=4 benchmarks (PACS, Office-Caltech10) with full-scope generalization left as future work."*

翻译: "据我们所知, 首次在联邦跨域学习里建一个**'域档案馆'**(Pd, 服务器级跨客户端 EMA), 用它作为低维原型空间里'类别-域几何互斥' 损失的前向锚点, 梯度通过 straight-through 黑魔法流到风格 encoder. 实证 scope 限定在'每客户端一个域' (D=K=4) 的 PACS 和 Office-Caltech10 benchmark, 更广的 D≠K 场景留作未来工作."

---

## 十、老生常谈的风险清单 (给 6 个月后的自己)

1. **Office 瓶颈可能根本不在架构** → S0 诊断先跑, 别盲目写代码
2. **z_sty 可能坍缩到 0** → 监控 z_sty_norm, < 0.3 警告
3. **InfoNCE 还是可能 R100 后崩** → Bell ramp-down 已内置, 真崩补 α-sparsity
4. **D=K=4 下 −Pd 可能和 full 差不多** → 预注册了 claim downgrade, 不要 post-hoc 改叙事
5. **Straight-through 有 bias** → Limitations 小节诚实声明, 通过 Vis-C Pd⊥bc_d trajectory 监测 bias vanishing

---

## 相关文件

- **学术版完整方案**: `refine_logs/2026-04-24_feddsa-biproto-asym_v1/FINAL_PROPOSAL.md`
- **Round-by-round 演化**: `refine_logs/2026-04-24_feddsa-biproto-asym_v1/REVIEW_SUMMARY.md`
- **详细报告 + pushback log**: `refine_logs/2026-04-24_feddsa-biproto-asym_v1/REFINEMENT_REPORT.md`
- **兜底方案对照**: `experiments/FALLBACK_PLAN_DualEncoder.md` (原 DualEncoder) / `experiments/FALLBACK_PLAN_FedDSA-Calibrator.md` (Calibrator)
