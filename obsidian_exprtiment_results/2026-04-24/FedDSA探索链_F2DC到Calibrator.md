# FedDSA 探索链 —— 从 F2DC 精读到兜底方案

**日期**: 2026-04-24
**会话目的**: 探索 F2DC 思想能否借鉴到 FedDSA 填上 Office -1.49 / PACS -1.5 差距
**最终结论**: F2DC 核心的"spatial mask"思想在我们 pooled 双头架构上**移植代价大**;**pooled 级 Calibrator** 作为兜底方案可行;更推荐**跨域 class prototype anchor**(不在本 session 展开)

---

## 一、起点 —— 我们的困境

| 数据集 | FDSE 本地 | orth_only 3-seed | 差距 |
|---|:-:|:-:|:-:|
| PACS AVG Best | 79.91 (EXP-081 旧) / **81.54** (EXP-123 新重跑) | 80.64 / 79.95 | 新配置下 **-1.13 pp** |
| Office AVG Best | 90.58 | 89.09 | **-1.49 pp** |

Stage B 诊断发现:FDSE 的 +2pp 几乎全来自 3 个 hard cell,`(Art, guitar) +24.33 pp`、`(Art, horse) +15.52`、`(Photo, horse) +17.5`。这些 cell 里**笔触(style)= 形状(class)**,是典型"风格就是判别信号"场景。

---

## 二、F2DC 精读要点

**论文**: Wang et al., CVPR 2026, "Federated Feature Decoupling and Calibration"

### 2.1 核心哲学:Calibrate 而非 Eliminate

- FDSE 的"擦除派":把 domain bias 当噪声删掉
- F2DC 的批评:**擦除时 class 信号和 domain 纠缠,一起被擦了**(Fig.2 长颈鹿鹿角被擦)
- F2DC 的主张:**domain-related feature 是"脏但有用的矿石",应该"洗"而不是"扔"**

### 2.2 Fig.1 维度塌缩(motivation)

Vanilla FedAvg 下每个 client 的特征协方差矩阵**大量奇异值趋近 0** → 表征塌缩到单一域的低维子空间。聚合成 global 也是塌缩的。F2DC 让奇异值分布更均匀。

### 2.3 Fig.2 长颈鹿漏鹿角(关键故事)

| 方法 | Cartoon 长颈鹿的 Grad-CAM |
|---|---|
| Vanilla | 看不到长颈鹿头 |
| FDSE | **看不到鹿角和头**(笔触被擦,鹿角的线条也被擦) |
| F2DC | 头、鹿角、身体都看到 |

### 2.4 三大模块

```
DFD (Domain Feature Decoupler)
  ├── A_D: 2 层 CNN,输入 feature map,输出 attribution
  ├── Gumbel-Concrete mask M (可微分 0/1 选择)
  ├── f+ = M ⊙ f  (domain-robust)
  └── f- = (1-M) ⊙ f  (domain-related)
  
  Loss: Separability (推开 f+/f-) + Discriminability (f+ 答对, f- 答指定错答案 ŷ)

DFC (Domain Feature Corrector) ★ 核心创新
  ├── A_C: 2 层 CNN
  └── f* = f- + (1-M) ⊙ A_C(f-)  残差校正
  
  Loss: CE(aux_cls(f*), y)  逼 corrector 真把 class 洗出来

DaA (Domain-aware Aggregation)
  └── 按 domain 分布偏差给 client 降权 (4 client 对称 setup 无用)
```

### 2.5 Table 7 关键数字(证据)

| 用哪个 feature 分类 | PACS AVG |
|---|:-:|
| 只用 f+ (干净的) | 75.13 |
| 只用 f- (脏的) | 57.87 |
| **只用 f\* (洗过的脏的)** | **73.49** (+15.62 vs f-) ⭐ |
| f+ + f* (最终) | 76.47 |

**f* 单用就能打 73.49**(vs f- 57.87),+15.62 pp 就是"calibration 哲学"的净收益证据。

---

## 三、F2DC 思想移植到 FedDSA 的三个选项

| 选项 | 做在哪层 | 约束什么 | 代价 | 判决 |
|---|---|---|:-:|:-:|
| A | 权重矩阵 `[128×1024]` | 两个 head 配方模式正交 | ~10 行 | **无效**(L_orth 已做到) |
| B | pooled 向量 `[1024]` | 两路读的 channel 集合互斥 | ~20 行 | **无效**(pool 已抹平空间) |
| **C** | feature map `[256, 6, 6]` | **空间格子物理分配** | ~200 行 | **真 F2DC,代价最大** |

**关键理解**:F2DC 强在**空间维度硬切**。pool 前的 feature map 上不同空间位置对应原图不同物理区域(鹿角 vs 背景)。pool 后的 1024d 向量已经把空间信息抹平,两路怎么切都没意义。

---

## 四、之前类似探索的失败案例(重要历史)

### 4.1 EXP-108 CDANN (2026-04-20)

**设计**:
- 加 shared `dom_head`,正向监督 `dom_head(z_sty) → domain`
- GRL 反向 `dom_head(z_sem) → domain`
- 期望:z_sem 变域盲,z_sty 变域专家,class 信号留 z_sem

**结果**:
| 指标 | orth_only | CDANN | Δ |
|---|:-:|:-:|:-:|
| PACS AVG Best | 80.64 | 80.08 | **-0.56 pp** |
| probe_sty_class (linear) | 0.240 | **0.962** | +72 pp |

**颠覆性**:CDANN **把 class 信号从 z_sem 挤到 z_sty**(probe 飙到 0.96),但**accuracy 反而跌**。
**教训**:任何给 z_sty **正向 loss** 的方案都可能重演 CDANN 失败(class 被挤过去,主任务受损)。

### 4.2 EXP-111 强正交 lo=3/10

强化 `L_orth` 到 lo=3 和 lo=10,看能否压 probe:

| lo | AVG Best | probe MLP-64 | probe MLP-256 |
|:-:|:-:|:-:|:-:|
| 1 | 82.23 | 0.694 | 0.813 |
| **3** | 81.33 | **0.201** 🏆 | 0.714 |
| 10 | 81.03 | 0.339 | 0.799 |

**发现**:强正交能压 MLP-64 probe 到 0.20(random),但 **MLP-256 还能读到 0.71** → class 信息**不是被删除,只是被非线性编码藏起来**。accuracy 还掉了 1 pp。

---

## 五、三个约束层面(衡量解耦的维度)

| 层面 | 测什么 | 我们数据 | 评价 |
|:-:|---|:-:|---|
| ① 特征 | `cos(z_sem, z_sty)` batch 内 | 0.025 (L_orth 训到) | ✅ 做到位 |
| ② 权重 | `W_sem·W_sty^T` 行向量 cos + channel 使用相关 | corr 0.025, pearson 0.04 | ✅ **意外地做到位** |
| ③ 信息 | MLP probe 从 z_sty 读 class | MLP-256 = 0.81 | ❌ **泄漏** |

**悖论**:前两个层面几乎完美,第三层面失败。

---

## 六、Decouple Probe 可视化验证(2026-04-24 关键实验)

### 6.1 方法
对 3 个 R200 ckpt 提取 `semantic_head.0.weight` 和 `style_head.0.weight`([128, 1024]),算:
- `corr_abs_mean` = 行向量 cos 绝对均值
- `channel_usage_pearson` = 两路对每个 trunk channel 依赖的 pearson
- `heavy_overlap` = top-25% 重度 channel 的重合数

脚本:`FDSE_CVPR25/scripts/visualize_decouple.py`

### 6.2 结果(3 个数据集完全一致)

| label | corr avg | pearson | heavy-overlap |
|---|:-:|:-:|:-:|
| PACS s=2 | 0.0254 | +0.040 | 64/256 (25%, random) |
| PACS s=15 | 0.0238 | −0.011 | 61/256 (24%) |
| Office s=15 | 0.0247 | −0.016 | 56/256 (22%) |

### 6.3 颠覆性结论

**L_orth 做得近乎完美**:
- 128 sem 行 × 128 sty 行的 16384 对 cos,平均 0.025(几乎全白)
- 两路对 1024 个 trunk channel 的依赖**统计独立**(pearson ≈ 0)
- 重度 channel 重合率 = 25% = **完美随机**

**但 probe 仍读出 0.81 class** → **不是两路共享 channel 的问题,是 trunk 每个 channel 都被 L_CE 训出了 class 信息** → 无论 z_sty 挑哪批 channel,每个 channel 里都有 class 残余。

**对三选项判决**:
- 选项 A(权重正交):corr 已 0.025,物理下限附近,**完全无效**
- 选项 B(channel mask):channel 使用已独立,**完全无效**
- 选项 C(spatial mask):唯一有希望的 head 路线,**但代价大**

图:`experiments/ablation/decouple_probe/figs/decouple_*.png`

---

## 七、PGA Pilot —— Feature Map 级 Attribution 的失败

### 7.1 设计
不改训练,**用已有 ckpt 做 post-hoc attribution**:
- 跨 4 个 client 聚合 per-location class prototype P ∈ [7, 256, 6, 6]
- 对测试图的 feature map f ∈ [256, 6, 6]:
  - `α[h, w] = max_c cos(f[:, h, w], P[c, :, h, w])`
- 上采样 α 到原图尺寸,叠加可视化

### 7.2 三个版本

**v1**(global-pooled prototype)
- Bug: FedBN 下所有 domain 用同一个 encoder 不对
- Bug: 用全局 pooled prototype,不是 per-location
- 结果:Cartoon 错位

**v2**(per-location prototype + per-domain encoder)
- 修了 FedBN bug 和 prototype spatial bug
- 但**测试图用 224×224 + ImageNet mean/std preprocess**,**训练用 256×256 + /255**
- **输入分布完全错配** → feature 空间不对齐 → attribution 反了

**v3**(preprocess 对齐训练)
- 修了 preprocess bug:Resize([256,256]) + PILToTensor + /255.0 → [0, 1]
- 加了 client→domain 映射 sanity check(✅ 字母序假设成立)
- 结果:
  | Domain | 结果 |
  |---|---|
  | Art | 🟡 部分命中 |
  | Cartoon | ✅ 红色集中在吉他垂直主体 |
  | **Photo** | ❌ **红色在背景,蓝色在黑色电吉他** |
  | **Sketch** | ❌ **红色散在背景,蓝色在吉他线条区** |

### 7.3 根因分析 —— cos-with-mean-prototype 的数学病

**合照类比**:100 张学生合照,叠起来求平均:
- **墙/桌椅**(每张都一样)→ 平均后**清晰**
- **学生**(每张姿势不同)→ 平均后**糊成一团**

新来一张合照,问每个像素和平均像多少:
- 墙位置 → 跟清晰的墙比 → **cos 高** → 红
- 学生位置 → 跟糊团比 → **cos 低** → 蓝

**对应 PGA**:
- 背景位置(每张 guitar 图都相似)→ prototype 在这里清晰 → 新图背景跟它像 → 红
- 主体位置(每张 guitar 姿态/颜色不同)→ prototype 在这里是模糊团 → 新图具体的 guitar 跟糊团不像 → 蓝

**所以红在背景、蓝在主体不是 bug,是"平均操作 + cos"的固有副作用**。

### 7.4 PGA 的方法论天花板

- 非负 feature(ReLU 后)+ cos 天然下限高(0.86+)→ 信号范围窄
- 跨 domain 平均 prototype 在"主体位置"被 wash out
- 静态像素逼近 static prototype **本质不可取**
- 要修就得做 gradient-based attribution(Grad-CAM)→ 等于放弃 PGA 本身

图:`experiments/ablation/pga_pilot/figs/pga_vs_freq_guitar.png`

---

## 八、用户的关键洞察(Session 后期)

> "用像素去逼近原型本来就不可取 —— 应该像 F2DC 那样**训练一个东西**,让这些特征能显性地靠真正的判别特征去评判哪些是判别主体,而不是因为位置改变而改变"

完全对。static pixel-to-mean-prototype 的问题是**位置依赖** —— guitar 在不同位置、不同姿势下 pixel 位置变了,feature 完全不同。
正确的思路:**learned attribution** —— 用一个训练好的模块(像 F2DC DFD)识别 class-discriminative pattern,不管它在哪个空间位置。

但这等于实现 F2DC 选项 C,~200 行,3 个新超参,不是 session 当下的 scope。

---

## 九、兜底方案(两个候选,优先级有序)

### ⭐ 兜底 B (首选 / 真·物理解耦):FedDSA-DualEncoder

**文档**: `experiments/FALLBACK_PLAN_DualEncoder.md`

**核心**: 给 semantic 和 style 各一个**完整独立的 AlexNet encoder**,权重完全分开,梯度不交叉。

- **代价**: 参数 2×,训练时间 1.5×,通信可选 1× 或 2×(encoder_sty 本地私有更省)
- **收益**: probe_sty_class **物理保证下降**(从 0.81 → < 0.3),诅咒源头被铲除
- **Loss**: L_CE(只训 encoder_sem)+ L_orth + L_sty_aux(domain 监督防坍缩,无 GRL)
- **FL 聚合**: 建议选项 Y(只聚合 encoder_sem,encoder_sty 本地私有,符合 "style 本地" 假设)
- **预期**: Office +0.9~2.9 pp,PACS +0.9~1.9 pp
- **Novelty**: FL 里独立双 trunk 少见(因为 2× 参数大家不愿做),有叙事空间

### ⚠️ 兜底 A (次选 / 接受泄漏并利用):FedDSA-Calibrator

**创建文档**: `experiments/FALLBACK_PLAN_FedDSA-Calibrator.md`

**核心思路**:
- 不试图"治"z_sty 的 class mirror(EXP-108/111 证明治不了)
- **利用它** —— 加 calibrator 从 z_sty 里显式提取 class 信号
- Final: `classifier(z_sem + λ·z_sty_cal)`

**架构改动**(~50 行):
```python
self.corrector = nn.Sequential(Linear(128,128), BN, ReLU, Linear(128,128))
self.aux_cls = nn.Linear(128, num_classes)

z_sty_cal = z_sty + self.corrector(z_sty)        # 残差校正
z_combined = z_sem + lam * z_sty_cal              # 合并
logits = self.sem_classifier(z_combined)          # 主分类
aux_logits = self.aux_cls(z_sty_cal)             # 辅助监督

L = CE(logits, y) + λ_orth · cos²(z_sem, z_sty) + λ_aux · CE(aux_logits, y)
```

**Warmup schedule**(防梯度冲突):
- R 0-20:λ=0, λ_aux=0(稳定 encoder)
- R 20-40:线性 ramp 到 λ=0.3, λ_aux=0.5
- R 40+:稳态

**定位**:**兜底**,不是主攻。在"跨域 prototype anchor"(主攻)和"spatial DFD"(次选)都失败时启用。

---

## 十、整个 Session 的关键数据汇总

| 项 | 数据 | 来源 |
|---|---|---|
| FDSE PACS | 81.54 (EXP-123) | 本地重跑 |
| FDSE Office | 90.58 | 本地复现 |
| orth_only PACS | 80.64 / 79.95 (R200 new) | EXP-109 |
| orth_only Office | 89.09 | EXP-113 |
| CDANN PACS | 80.08(-0.56) | EXP-108 |
| z_sty class probe (orth_only linear) | 0.240 | EXP-109 |
| z_sty class probe (orth_only MLP-256) | 0.813 | EXP-111 |
| z_sty class probe (strong orth lo=3, MLP-64) | 0.201 | EXP-111 |
| W_sem·W_sty 行正交度 | corr 0.025 | decouple_probe |
| Channel usage pearson | ≈ 0 | decouple_probe |
| (Art, guitar) orth_only vs FDSE | 43.94 vs 61.57 (-17.6) | Stage B |

---

## 十一、技术决策清单

### 已确认死的路

- ❌ **PGA (cos with mean prototype)** —— 数学本质反了,修不了
- ❌ **CDANN 类方向**(给 z_sty 正向监督)—— 挤 class 入 z_sty,主任务受损
- ❌ **强 L_orth (lo=3/10)** —— probe 压一点 accuracy 退一点
- ❌ **选项 A 权重行正交** —— L_orth 已做到,冗余
- ❌ **选项 B channel mask** —— pool 后 channel 意义模糊
- ❌ **FREQ pilot** —— Sketch/Cartoon 差异退化(更早 pilot)

### 待验证的路

- 🟡 **跨域 class prototype anchor**(主攻候选):不碰 z_sty,直接在 pooled 空间拉齐跨 domain z_sem
- 🟡 **F2DC 选项 C(spatial DFD)**:真 F2DC 思想,~200 行,3 新超参
- 🟡 **FedDSA-Calibrator(兜底)**:pooled 级 corrector,50 行,1 新超参

### 开放问题

- Office -1.49 差距是**本质问题**(Office 风格差异小导致 style-head 没东西学,对称泄漏更严重)还是**方法问题**?
- AlexNet 6×6 feature map 分辨率是不是**硬天花板**?换 ResNet-18 会不会更好?CLAUDE.md 要求 AlexNet...

---

## 十二、下一步决策建议

**从 ROI 角度优先级**:
1. **先做跨域 prototype anchor pilot**(50 行 + 3 seed × 3 GPU-hour),**最可能填 Office -1.49**
2. 如果失败 → 启动 **FedDSA-Calibrator**(兜底)
3. 如果还失败 → 评估是否**换 backbone**(违反 CLAUDE.md 但可能是 hard reality)
4. 最后选 → **放弃 head 路线**,做 **F2DC 选项 C spatial DFD**(大改,novelty 最高)

---

## 十三、Obsidian 交叉链接

- [[论文精读_F2DC]] —— F2DC 完整精读
- [[大白话_FedDSA-CDANN]] —— CDANN 方案大白话版
- [[FedDSA-CDANN_技术方案]] —— CDANN 技术方案
- [[EXP-108_cdann_results]] —— CDANN 实验结果
- [[EXP-109_pacs_orth_only_counterfactual]] —— orth_only 反事实
- [[EXP-111_pacs_strong_orth]] —— 强正交消融
- [[EXP-113_results]] —— VIB 实验
- [[关键实验发现备忘]] —— 2026-04-21 发现 4 (CDANN anchor 颠覆)
- [[../../experiments/FALLBACK_PLAN_FedDSA-Calibrator|FALLBACK_PLAN_FedDSA-Calibrator]] —— 兜底方案完整文档
- [[../../experiments/ablation/decouple_probe/NOTE|decouple_probe NOTE]] —— 诅咒验证实验
- [[../../experiments/ablation/pga_pilot/figs/pga_vs_freq_guitar.png|PGA v3 可视化]]

---

## 十四、一句话总结 Session

**F2DC 的"calibrate 而非 eliminate"哲学对我们的困境有启发,但其精髓(spatial feature-map mask)在我们 pooled 双头架构下移植代价大**。探索中发现 `L_orth` 在权重层面已经做得近乎完美(corr 0.025),**class 残余泄漏的根源在 trunk 每个 channel 都含 class 信息**,head 层面治不了。**PGA(cos with mean prototype)方法论本身有数学病**,preprocess 对齐后 Cartoon 能命中但 Photo/Sketch 仍反。**用户的洞察完全正确:应该做 learned attribution (F2DC DFD 那种)而非静态 pixel-to-prototype 匹配**。**兜底方案 FedDSA-Calibrator** 作为保险已写好完整设计文档,随时可启动。真正推荐的下一步是**跨域 class prototype anchor**,不在 z_sty 上纠结,直接在 pooled 空间拉齐跨 domain z_sem。
