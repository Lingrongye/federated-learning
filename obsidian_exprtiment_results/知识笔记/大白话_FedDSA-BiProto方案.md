# 大白话_FedDSA-BiProto方案

> 对应学术版: `refine_logs/2026-04-24_feddsa-biproto-asym_v1/FINAL_PROPOSAL.md`
> 创建日期: 2026-04-24
> 来源: 4 轮 GPT-5.4 xhigh review 后最终方案 (8.75/10 near-READY)

---

## 一句话记住

> **把"风格"从"要擦掉的噪音"升级为"要好好保留的档案"**, 在服务器上给每个地区(域)建一张身份卡(Pd), 然后强制让"类别方向"和"风格方向"在几何上互相看不见.

---

## 🚨 方法流程图 — 每一步在干嘛 + 每个 Loss 怎么设计

> 这一节假设你**完全没读过我们的方案**, 每个符号第一次出场都先解释清楚再用.

---

### 第 0 步: 先认识所有出场角色 (符号速查表)

> 看流程之前先把这张表过一遍, 后面就不会迷路.

| 名字 | 是什么 | 形状 / 例子 | 大白话 |
|---|---|---|---|
| **k** | client 编号 (从 0 数) | 0, 1, 2, 3 (4 个 client) | 第几个公司, 比如 k=0 是 Photo 公司 |
| **d** | domain (域) 编号 | 0, 1, 2, 3 | 第几种风格, 比如 d=0 是照片风格 |
| **x** | 输入图像 | shape [B, 3, 224, 224] | 一批图, B=50 张, 每张 RGB 224×224 |
| **y** | 类别标签 | shape [B] | 每张图属于哪个类 (狗/马/...) |
| **AlexNet (encoder_sem)** | 主干卷积网络 | ~14M 参数 | 通用的"看图找特征"网络 |
| **conv1 / conv2 / conv3** | AlexNet 的前 3 个卷积层 | — | 网络的浅层 (从图里拎出"边缘/纹理") |
| **中间层激活 (taps)** | conv1/2/3 的输出 feature map | shape [B, C, H, W] | 网络浅层算出来的特征图, "图刚被加工到一半的样子" |
| **(μ, σ)** | 每个 channel 的均值和标准差 | shape [B, C] | "这层每个频道的色彩平均亮度 + 起伏程度" — AdaIN 论文证明这就是"风格" |
| **detach()** | PyTorch 函数 | — | 把张量从计算图剥离, 之后的梯度不会回流 ("信息能用, 但不准学这部分") |
| **pooled** | AlexNet 最后输出的特征向量 | shape [B, 1024] | "看完图整张提炼出的 1024 维总结" |
| **encoder_sty** | 我们新加的小 MLP | ~1M 参数 | 专门吃 (μ,σ) 做出"风格 embedding"的小脑子 |
| **semantic_head** | 语义投影层 | 1024 → 128 | 把 pooled 压缩到 128 维 |
| **z_sem** | 语义向量 | shape [B, 128] | "这张图说的是什么 (不带风格)" |
| **z_sty** | 风格向量 | shape [B, 128] | "这张图是什么风格 (不带内容)" |
| **sem_classifier** | 分类器 | 128 → C 类 | 从 z_sem 预测 class |
| **logits** | 分类原始分数 | shape [B, C] | softmax 之前的预测 |
| **Pd** | "域档案馆" | shape [D=4, 128] | 服务器存的 4 张"风格身份卡", 每个 domain 一张 |
| **Pd[d]** | 第 d 个 domain 的卡 | shape [128] | 比如 Pd[0] 就是 Photo 风格的代表向量 |
| **Pc** | "类档案馆" | shape [C, 128] | 服务器存的 C 张"类别身份卡", 每个 class 一张 (本方案中只做监控用, 不是 loss target) |
| **EMA** | 指数滑动平均 | new = 0.9·old + 0.1·new_data | "档案不要剧烈跳变, 用旧档案 90% + 新数据 10%" |
| **FedAvg** | 联邦平均 | server 把 K 个 client 的权重加权平均 | 联邦学习最经典的合并方式 |

---

### 第 1 步: 整个流程发生在 1 个 round 内, 分 client 和 server 两块

> 一个 round = 所有 client 训一次 + server 合并一次. 训 200 round 就是这样转 200 圈.

```
═══════════════════════════════════════════════════════════════
              🏢 Client k 这一侧 (假设我们看 client 0 = Photo)
═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│  Step 1: 拿数据                                              │
│  ─────────────────────────────────────────────────────────── │
│  x = 这一批 50 张图 [B=50, 3, 224, 224]                      │
│  y = 50 个类别标签 [50]                                      │
│                                                              │
│  ⚠️ 注意: 这 50 张图全部是 Photo 风格 (因为 client 0 只有    │
│  Photo 数据). 整 batch 同一个 domain — 这是 bug 的根源.       │
└──────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 2: 把图喂进 AlexNet (encoder_sem) 走一遍                │
│  ─────────────────────────────────────────────────────────── │
│  x → conv1 → conv2 → conv3 → conv4 → conv5 → pooled          │
│                                                              │
│  ▼ 关键: 在 conv1, conv2, conv3 这 3 个浅层位置, 把它们输出  │
│    的 feature map 截出来 (这叫"中间层 taps").                │
│                                                              │
│  ▼ 风格指纹是怎么从 feature map 里算出来的 (一步步看):       │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  conv1 输出 feature map:  [50,  64,  55, 55]                 │
│                          (50 张图, 64 个 channel,            │
│                           每个 channel 是 55×55 的像素图)    │
│                                                              │
│    对每张图×每个 channel 在 55×55 像素上求均值               │
│      → μ₁ 形状 [50, 64]    "每张图过 64 种滤镜后的平均亮度" │
│    对每张图×每个 channel 在 55×55 像素上求标准差             │
│      → σ₁ 形状 [50, 64]    "每张图过 64 种滤镜后亮暗起伏度" │
│                                                              │
│  conv2 输出: [50, 192, 27, 27]                               │
│      → μ₂ [50, 192]      σ₂ [50, 192]                        │
│                                                              │
│  conv3 输出: [50, 384, 13, 13]                               │
│      → μ₃ [50, 384]      σ₃ [50, 384]                        │
│                                                              │
│  现在手上有 6 个矩阵:                                        │
│    μ₁ [50,64]   σ₁ [50,64]                                   │
│    μ₂ [50,192]  σ₂ [50,192]                                  │
│    μ₃ [50,384]  σ₃ [50,384]                                  │
│                                                              │
│  在第 1 维 (特征维) 把它们 concat 起来 (batch 维 50 保留):   │
│    taps_stats = concat([μ₁,σ₁,μ₂,σ₂,μ₃,σ₃], dim=1)            │
│                = 形状 [50, 64+64+192+192+384+384]            │
│                = 形状 [50, 1280]                             │
│                                                              │
│  ── 也就是说: 每张图都有自己的 1280 维"风格指纹" ──          │
│                                                              │
│  最终 Step 2 输出:                                           │
│    pooled:     [50, 1024]   ← AlexNet 最后的语义总结         │
│    taps_stats: [50, 1280]   ← 6 个 (μ,σ) 拼起来的风格指纹   │
│                                                              │
│  ▼ 为什么 (μ, σ) 就代表"风格"?                              │
│    AdaIN 论文 (Huang 2017) 的发现:                           │
│      "梵高风格 / 照片风格" 这种全局画风, 体现在每个 channel │
│      的整体亮度 μ 和起伏 σ.                                  │
│      "画的具体内容 (这是狗还是马)" 藏在像素的空间排列里,    │
│      跟 (μ, σ) 无关 (因为我们对空间维 H×W 求了均值/标准差,   │
│      把空间结构丢掉了).                                       │
│    → 所以 (μ, σ) 物理上只能抓风格, 丢掉了内容.               │
└──────────────────────────────────────────────────────────────┘
                          │
                          ↓
              ┌───────────┴───────────────┐
              │ pooled                     │ taps_stats (μ,σ)
              │ (走语义路径, 带梯度)        │ ← .detach() 切断梯度
              │                            │   (这些统计量"能看, 但不让 sty 侧的梯度回流到 AlexNet"
              │                            │    保证 AlexNet 还是 orth_only 那样训, 不被风格分支搅乱)
              ↓                            ↓
┌──────────────────────────┐    ┌──────────────────────────────┐
│ Step 3a: 语义路径         │    │ Step 3b: 风格路径             │
│ ───────────────────────── │    │ ──────────────────────────── │
│ semantic_head (MLP)       │    │ encoder_sty (1M MLP)          │
│ pooled[50,1024]           │    │ taps_stats[50, 1280]          │
│   → z_sem [50, 128]       │    │   → z_sty [50, 128]           │
│                           │    │                              │
│ z_sem[i] 表示: 第 i 张图  │    │ z_sty[i] 表示: 第 i 张图      │
│ "讲的是什么内容"           │    │ "是什么风格"                 │
└──────────────────────────┘    └──────────────────────────────┘
              │                            │
              ↓                            │
┌──────────────────────────┐               │
│ Step 4: 分类             │               │
│ sem_classifier           │               │
│ z_sem[50,128]            │               │
│   → logits [50, C]       │               │
└──────────────────────────┘               │
              │                            │
              ↓                            │
        预测对不对?                         │
              │                            │
              ↓                            │
       (后面算 L_CE, 见 Loss 1)            │
                                           │
              ┌────────────────────────────┘
              ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 5: 算 4 条 loss (具体见下面 Loss 详解)                 │
│  L_total = L_CE                                              │
│          + λ₁·L_orth          (z_sem 和 z_sty 互相看不见)    │
│          + λ₂·L_sty_proto     (z_sty 拉向 Pd[k])             │
│          + λ₃·L_proto_excl    (类方向 ⊥ 域方向)              │
└──────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 6: 反向传播 + 更新参数 (PyTorch 自动)                  │
│  ─────────────────────────────────────────────────────────── │
│  encoder_sem (AlexNet) 收到:  L_CE 的梯度 + L_orth 的梯度 +  │
│                              L_proto_excl 的梯度              │
│  encoder_sty (小 MLP) 收到:   L_orth + L_sty_proto +         │
│                              L_proto_excl 的梯度              │
│  ★ encoder_sem 不收到 L_sty_proto, 因为 taps 已 detach        │
└──────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 7: client 训练完一轮, 准备打包上传                     │
│  ─────────────────────────────────────────────────────────── │
│  上传给 server 的东西:                                        │
│  ① 训完的 encoder_sem + encoder_sty + heads + classifier 参数 │
│  ② z_sty 这一 client 所有 batch 平均 → 1 个 [128] 向量        │
│     (这个向量将用来更新 server 的 Pd[k] 档案)                 │
│  ③ z_sem 各 class 的平均 → C 个 [128] 向量                    │
│     (用来更新 Pc[c] 档案, 但本方案 Pc 只做监控不参与 loss)   │
└──────────────────────────────────────────────────────────────┘
                          │
                          ↓
═══════════════════════════════════════════════════════════════
                         ☁️ Server 这一侧
═══════════════════════════════════════════════════════════════
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 8: server 拿到 K 个 client 上传的东西后做合并          │
│  ─────────────────────────────────────────────────────────── │
│  ① 模型参数: FedAvg 加权平均 (按 client 数据量)               │
│  ② Pd 更新 (★ 关键, 每个 domain 一张卡):                     │
│       for d in 0..D-1:                                       │
│         如果有 client 属于 domain d:                          │
│           Pd[d] = 0.9·Pd[d] + 0.1·F.normalize(那个 client    │
│                                              上传的 z_sty 均值) │
│  ③ Pc[c] 类似 EMA 更新 (但只用于监控)                        │
└──────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 9: server 把聚合后的 (模型 + Pd + Pc) 广播回所有 client│
│         → 进入下一 round, 跳回 Step 1                        │
└──────────────────────────────────────────────────────────────┘
```

---

### 4 个 Loss 详细设计 (一个一个看, 每个先讲它在干嘛)

#### Loss 1: L_CE (主任务 — 老朋友, 任何监督学习都有)

```
┌──────────────────────────────────────────────────────────────┐
│  作用: 让模型学会"看图认 class"                              │
│                                                              │
│  公式: L_CE = CrossEntropy(logits, y)                        │
│        = -mean( log(softmax(logits)[正确 class]) )           │
│                                                              │
│  作用对象 (谁因为这个 loss 调参):                             │
│    encoder_sem (AlexNet) + semantic_head + sem_classifier    │
│                                                              │
│  权重: 始终 = 1                                              │
└──────────────────────────────────────────────────────────────┘
```

#### Loss 2: L_orth (z_sem 和 z_sty 互相垂直 — 继承自 orth_only baseline)

```
┌──────────────────────────────────────────────────────────────┐
│  作用: 强制 z_sem 和 z_sty 在 128 维向量空间里"互相看不见"   │
│       即两者点积尽量为 0 (角度 90°)                          │
│                                                              │
│  公式: 对每个样本 i 算                                       │
│         cos(z_sem[i], z_sty[i]) = z_sem[i]·z_sty[i] /        │
│                                   (||z_sem||·||z_sty||)      │
│       L_orth = mean over i ( cos² )                          │
│                                                              │
│  作用对象: encoder_sem + semantic_head + encoder_sty         │
│  权重: λ₁ = 1.0, 全程 on                                     │
│                                                              │
│  解读: cos² ∈ [0, 1], 越小越正交.                            │
│        理想: 两个 128 维向量永远 90°夹角.                    │
└──────────────────────────────────────────────────────────────┘
```

#### Loss 3: L_sty_proto (风格原型对齐) ⚠️ **这个 loss 是 bug 来源**

```
┌──────────────────────────────────────────────────────────────┐
│  设计意图 (脑子里想做的事):                                  │
│    让 encoder_sty 学到"区分不同 domain 的能力" — 通过把       │
│    z_sty 拉向自己 domain 的档案 Pd[d] (而推开别的 Pd).       │
│                                                              │
│  公式分两部分:                                               │
│    ─────────────────────────────────────────────────────     │
│    Part A: InfoNCE (类似分类损失)                            │
│      对 batch 里每个样本 i:                                  │
│        sim[i, d] = cos(z_sty[i], Pd[d]) / τ   for d=0..D-1   │
│        target = client 自己的 domain 编号 (恒为 k)           │
│        L_info = CrossEntropy(sim, target)                    │
│      ★★★ 注意: 这 batch 的 50 张图全是 client k 的, 所以      │
│        全部样本的 target 都等于 k. 没有"反例" pair.           │
│    ─────────────────────────────────────────────────────     │
│    Part B: MSE 锚点 (FPL 论文风格)                           │
│      L_mse = mean( ||z_sty[i] - stopgrad(Pd[k])||² )         │
│      ★★★ 显式把 z_sty 拉向 Pd[k]                             │
│    ─────────────────────────────────────────────────────     │
│    L_sty_proto = L_info + 0.5 · L_mse                        │
│                                                              │
│  作用对象: encoder_sty (单独训风格分支)                      │
│  权重: λ₂, Bell schedule (R0-50 关, R80-150 peak=0.5,        │
│       R150-200 ramp 关)                                      │
│                                                              │
│  实际效果 (现实): 见后面 "🐛 自指死循环"                     │
└──────────────────────────────────────────────────────────────┘
```

#### Loss 4: L_proto_excl (类方向和域方向几何互斥)

```
┌──────────────────────────────────────────────────────────────┐
│  设计意图: 让 "class 在 128 维里指的方向" 跟 "domain 在 128  │
│          维里指的方向" 互相垂直, 这样 z_sem (走 class 方向)  │
│          就不会沾染 domain 信息.                             │
│                                                              │
│  公式 (一步一步看):                                          │
│    ─────────────────────────────────────────────────────     │
│    ① 算"类方向": batch 里所有 y=c 的样本 z_sem 取均值再归一  │
│       class_axis[c] = F.normalize(                           │
│            mean of z_sem[i] for i where y[i]==c              │
│       )                                                      │
│       (有 |C| 个 class, 就有 |C| 个 class_axis 向量)         │
│    ─────────────────────────────────────────────────────     │
│    ② 算"域方向" — 用了 straight-through 黑魔法               │
│       bc_d = F.normalize( mean of z_sty over batch )         │
│         (整 batch 都属 domain k, 所以 bc_d 就是 client 自己   │
│          这一 batch 的 z_sty 均值, 带梯度)                   │
│                                                              │
│       domain_axis = F.normalize(                             │
│            Pd[k].detach()  +  bc_d  -  bc_d.detach()          │
│       )                                                      │
│                                                              │
│       这是个 trick: forward 时 bc_d - bc_d.detach() = 0,     │
│         所以 forward 值 = Pd[k] (server 那张档案卡)          │
│       backward 时 Pd[k].detach() 没梯度, 所以梯度只走 bc_d   │
│         → 流到 encoder_sty 上                                │
│       一句话: "前向报上去的值是 Pd, 但训练时改的是手上 bc_d" │
│    ─────────────────────────────────────────────────────     │
│    ③ 算互斥 loss (类方向和域方向的 cos² 越小越好)            │
│       L_proto_excl = mean over (c, d) pairs of               │
│            cos²(class_axis[c], domain_axis[d])               │
│                                                              │
│  作用对象: encoder_sem + semantic_head + encoder_sty         │
│           (encoder_sem 通过 class_axis 收梯度,                │
│            encoder_sty 通过 bc_d 收梯度)                     │
│  权重: λ₃ = 0.3, 全程 on                                     │
└──────────────────────────────────────────────────────────────┘
```

---

### 🐛 看完流程能直接发现的核心 bug: Pd 自指死循环

> 把上面流程里"红色危险路径"单独抽出来看, 你会看到一个**圈**.

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   z_sty (encoder_sty 输出, 形状 [B, 128])                  │
│       │                                                    │
│       │ ① client 训完 1 round, 把 z_sty 整 client 的平均   │
│       │    上传给 server (一个 [128] 向量)                 │
│       ↓                                                    │
│   server 收到这个均值, 用 EMA 更新档案:                    │
│      Pd[k] ← 0.9·Pd[k] + 0.1·(刚收到的均值)                │
│                                                            │
│      → 此时 Pd[k] 实际就是 "z_sty 历史均值的滑动版本"      │
│       │                                                    │
│       │ ② server 把更新后的 Pd 广播回所有 client            │
│       ↓                                                    │
│   client 拿到新的 Pd[k], 进入下一 round 训练                │
│       │                                                    │
│       │ ③ L_sty_proto = InfoNCE + MSE(z_sty 拉向 Pd[k])    │
│       ↓                                                    │
│   z_sty 被推向 Pd[k]                                        │
│                                                            │
│   但 Pd[k] = z_sty 自己的历史均值                          │
│   → z_sty 被推向自己的过去                                 │
│                                                            │
│       │                                                    │
│       └─────── 回到第 ① 步, 开始下一轮自我加强 ──────────► │
│                                                            │
│   每 round 自我强化一次, 经过 200 round:                   │
│     z_sty 整 client 收敛到一个常量点                       │
│     4 个 client (4 个 domain) × 1 个常量点 =               │
│     z_sty 实际只占 (4-1) = 3 维 → mode collapse           │
│                                                            │
│   实测: SVD effective rank 从期望 28 → 实际 2.7 (s=333)    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 🐛 第二个 bug: L_proto_excl 在 collapse 后变噪声梯度

```
domain_axis = F.normalize(Pd[k] + bc_d - bc_d.detach())
                            │
                            │ 因为 z_sty 已塌缩 → bc_d 也是塌缩值
                            │ Pd[k] 也是塌缩值
                            ↓
        domain_axis ≈ 一个固定的常量向量

L_proto_excl = cos²(class_axis, ≈常量)
                            │
                            │ 拿 class_axis 跟"几乎不变的常量"算互斥
                            │ → encoder_sem 收到的梯度方向几乎固定
                            │ → 这不是真信号, 是噪声
                            ↓
        encoder_sem 的训练被噪声梯度污染
                            │
                            ↓
        accuracy 比纯 orth_only (没这条 loss) 还低 ❌
```

**为什么必然 collapse**:
- 每个 client batch 全是同一 domain k → InfoNCE target 永远是 Pd[k] (没 negative pair 推力)
- MSE 显式拉 z_sty → Pd[k]
- 但 Pd[k] 又来自 client k 自己的 z_sty 均值
- → **z_sty 被拉向"自己的过去"** → 整个 client 的 z_sty 收敛到一个点
- 4 个 client × 4 个常量点 = z_sty rank-3 collapse (实测 SVD 验证)

### 🐛 第二个核心 bug: L_proto_excl 在 collapse 后变噪声

**正常情况**: domain_axis 应该编码"这个 domain 的真实风格方向"
**实际情况**: 
- domain_axis = F.normalize(Pd[k] + bc_d - bc_d.detach())
- forward 值 = Pd[k] (塌缩后是常量点)
- 跟"常量"做 cos² 互斥 → encoder_sem 收到固定方向梯度 → **不是真信号是噪声**
- 这个噪声梯度污染 encoder_sem → accuracy 反而退步

### 实测对照

| Seed | z_sty 塌缩程度 (ER) | Office accuracy | 解释 |
|:-:|:-:|:-:|---|
| s=333 | **2.73 (最严重)** | **92.46 (最高)** | 塌缩越彻底 L_proto_excl 越接近"常量梯度" → 越接近 orth_only |
| s=2 | 16.07 (中等塌缩) | 88.00 (低) | 塌缩不够彻底 → L_proto_excl 还有 noise 梯度 → 污染 encoder_sem |
| s=15 | 17.06 (中等塌缩) | 86.65 (最低) | 同上 |

**反 paradox**: 越塌缩 accuracy 越好 → 因为 BiProto "越失败"才"越接近无害" (退化为 orth_only)

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
