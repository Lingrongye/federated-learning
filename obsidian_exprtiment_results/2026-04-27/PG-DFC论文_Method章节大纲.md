---
date: 2026-04-27
type: paper drafting
status: 大纲完成,等 PACS Wave 2 数据齐后填实验数字
related:
  - obsidian/2026-04-26/PG-DFC方案_Prototype引导的F2DC校准.md
  - experiments/ablation/EXP-131_PG-DFC_v3.2/NOTE.md
  - experiments/ablation/EXP-130_F2DC_baseline_main_table/NOTE.md
---

# PG-DFC 论文 Method 章节大纲

> 基于"Cross-Client Decoupling"概念,把 PG-DFC 定位成新概念的首次实现 (而非 F2DC 的 fix)
> 核心 narrative: 现有 disentangle FL 都 client-local 校准,缺 cross-client 方向 → 我们用 prototype 做 directional anchor

---

## §1 Introduction

### §1.1 概念缺口 (拉高度,3-4 段)

**段 1: 跨域 FL 大背景**
- 联邦学习中 domain skew 是核心挑战 (引 FedAvg, FedBN, FDSE)
- 现有方法分两派:
  - 参数空间适应派 (FedAvg/FedProx/FedBN)
  - 特征解耦派 (F2DC, FDSE, FedSeProto, FedDP, FediOS)
- 解耦派当前 SOTA, 但有架构性 gap

**段 2: 解耦派的共同范式 + 共同缺陷**
- 所有 decouple FL 方法都遵循 "decouple → calibrate" 两步
  - F2DC: DFD 切 robust/non-robust → DFC conv 残差校准
  - FDSE: DFE 提取 + DSE 擦除
  - FedSeProto: MI 分离 → 丢弃域特征
  - FediOS: 正交投影 → personalized subspace
- **共同缺陷: 校准操作完全是 client-local 的, 没有利用 cross-client 结构**
- 这违背了 FL "协同利用分布式知识" 的初衷

**段 3: F2DC 作为代表案例 + 引用 paper 自承认 limitation**
- F2DC (CVPR 2026) 是当前 SOTA
- F2DC 论文 Sec 4.1 末段原话:
  > "the calibrator inevitably inherits a complex mixture of domain artifacts and valuable class-relevant clues"
- F2DC 自己承认: 校准器没有外部引导, 只能本地摸索
- 我们认为这不是 capacity 问题, 是**根本性架构缺失** — 校准器缺 directional anchor

**段 4: 我们的 motivation**
- Per-domain 实证发现:
  - PACS art 域 (单 client 492 sample) 在 F2DC 下 R49 后过拟合下降
  - Office dslr 域 (单 client 25 sample) 表现最弱
- Small domain 缺训练信号 → 本地校准器学不出准确方向 → cross-client 引导 most needed

**段 5: 方法 + Contribution**
- 提出 PG-DFC: 首次将 cross-client class prototype 作为 DFC 的 directional anchor
- 通过 cross-attention 让 DFC 自适应选择相关 prototype
- 在 F2DC 解耦框架不变的前提下, 实证 per-domain analysis 揭示小 domain 增益最大
- Contribution:
  1. **概念**: 首次定义 cross-client directional anchor for decoupled feature calibration (新框架 "Cross-Client Decoupling")
  2. **算法**: 首次将 prototype 用作 subnetwork 的 forward-pass input (vs FedProto/FPL 用作 loss/classifier)
  3. **实证**: per-domain analysis 揭示 prototype guidance 在 small domain 上的非对称巨大增益

---

## §2 Related Work

### §2.1 Federated Cross-Domain Learning
- FedAvg/FedProx baselines
- FedBN: 本地 BN
- 解耦派: F2DC, FDSE, FedSeProto, FedDP, FediOS
- 共同点: 都 client-local 操作 → 我们的 critique 落点

### §2.2 Federated Prototype Learning
- FedProto (AAAI'22): 单 prototype + MSE 替代分类器
- FPL (CVPR'23): FINCH 多 prototype + InfoNCE
- FedPLVM (NeurIPS'24): α-sparsity loss + 双层聚类
- MP-FedCL: k-means 多原型
- I2PFL (2025): 域内 / 域间 prototype + 重加权
- **共同点**: prototype 都用作 loss term (对比 / 正则) 或 classifier 替代
- **我们差异**: prototype 作为 subnetwork (DFC) 的 forward-pass input → 不同位置, 不同 role

### §2.3 Robust Aggregation in FL
- Krum / Multi-Krum: byzantine FL
- 我们的 outlier handling 借鉴这一思路, 但应用场景不同 (domain skew vs malicious)

---

## §3 Method: PG-DFC

### §3.1 Cross-Client Decoupling Framework (概念定义)

**核心思想**: 把 decouple FL 的两步操作提升到 cross-client 层级:
- Local Decoupling (已有, F2DC 提供): 本 client 内分 robust/non-robust
- **Cross-Client Calibration (本工作): 用跨 client class consensus 引导 non-robust 的修正方向**

**形式化**:
- Local: 每 client k 算 decoupled features $f_i^{+,k}, f_i^{-,k}$ via DFD (F2DC 提供)
- Cross-Client: server 维护 class prototype bank $\{\mu_c^{global}\}_{c=1}^C$
- $\mu_c^{global}$ = client local prototype 的 robust aggregation
- DFC 不再 conv 残差盲修, 而是 attention-pick 相关 prototype 引导

### §3.2 Cross-Client Class Prototype Bank

**Client 端 (per-round sample 累加)**:
$$\mu_c^k = \frac{1}{|S_c^k|} \sum_{i \in S_c^k} \text{AvgPool}(f_i^{+,k})$$
- $S_c^k$ = client k 当 round 内 GT class = c 的样本集合
- 不用 batch-mean EMA (避免噪声), 用 round 内 sample 累加 (FedProto/FPL 标准做法)

**Server 端聚合 (L2-norm + 等权)**:
$$\mu_c^{global} = \text{normalize}\left(\frac{1}{|K_c|} \sum_{k \in K_c} \frac{\mu_c^k}{||\mu_c^k||}\right)$$
- $K_c$ = 当 round 有 class c sample 的 client 集合
- L2-normalize 消除 client size 主导 (大 client 不淹没小 client)
- 等权聚合 → 真正 cross-domain consensus 方向

**Server 跨 round 平滑** (可选, 防投票成员变化跳跃):
$$\mu_c^{global,t} = \beta \cdot \mu_c^{global,t-1} + (1-\beta) \cdot \text{aggregated}_t$$
- $\beta = 0.8$, 在 raw space 做 EMA, 输出时再 normalize

### §3.3 Prototype-Guided DFC (Method Core)

替代 F2DC 原 DFC 的 conv 残差:

**原 F2DC DFC**:
$$f_i^* = f_i^- + (1 - M_i) \odot \text{Conv}(f_i^-)$$
- 只看本地 $f_i^-$, 没外部引导

**PG-DFC**:
$$\begin{aligned}
q_i &= W_q \cdot \text{AvgPool}(f_i^+)  \quad \text{// query from robust feature} \\
k_c &= W_k \cdot \mu_c^{global}, \quad v_c = W_v \cdot \mu_c^{global} \\
\alpha_{i,c} &= \text{softmax}_c\left(\frac{\text{cos}(q_i, k_c)}{\tau}\right) \\
\text{proto\_clue}_i &= \sum_c \alpha_{i,c} \cdot v_c \\
f_i^* &= f_i^- + (1 - M_i) \odot \left(\text{Conv}(f_i^-) + w \cdot \text{proto\_clue}_i\right)
\end{aligned}$$

**关键设计选择**:
1. **Query 来自 $f_i^+$ (robust)**, 不是 $f_i^-$ (non-robust)
   - 用已识别的 class signal 去查字典, 不用噪声特征
2. **Cosine attention + temperature** (避免 magnitude mismatch)
3. **Proto 路径上 mask.detach()** (防止 prototype 反传污染 mask 学习, 阻断 client-local F2DC mask 训练 dynamics)

### §3.4 训练流程

```
For round t in 1..R:
    Client k:
        Forward: standard F2DC + PG-DFC (using current μ_global)
        Loss: F2DC 原 loss (CE + DFD + DFC)  -- 完全保留 F2DC 原 loss, 不加新 loss
        Backward: standard
        End-of-round: 累加 sample 算 μ_c^k, 上传

    Server:
        Aggregate backbone: F2DC 原 weighted FedAvg
        Aggregate prototype: L2-norm + 等权 + 跨 round EMA β
        Distribute new μ_global to all clients
```

### §3.5 与 prior FL prototype 工作的对比

| 方法 | Prototype 用作什么 | 注入位置 |
|---|---|---|
| FedProto | 分类器替代 | logits 层 |
| FPL | InfoNCE anchor | feature loss |
| FedPLVM | α-sparsity loss | 损失正则 |
| MP-FedCL | 对比学习 anchor | logits |
| **PG-DFC (我们)** | **Subnetwork (DFC) 的 forward input** | **中间层 (DFC module 内)** |

**核心 novelty**: 别人都用 prototype 计算损失, 我们用作中间层的 attention input → **使用位置 + 角色都不同**

---

## §4 Experiments

### §4.1 Setup
- 数据集: PACS / Office-Caltech10 (Digits 可选)
- 基线: F2DC (vanilla, release 代码), FDSE, FedAvg, FedBN, FedProto, MOON
- F2DC release 代码我们 patch 了 8 个 bug 才能跑 (诚信 footnote)
- Setup: K=10 clients, 30%/20% partition, R=100, E=10
- **Fixed allocation** 严格 3-seed: photo:2/art:3/cartoon:2/sketch:3 (PACS), caltech:3/amazon:2/webcam:2/dslr:3 (Office)

### §4.2 Main Results (3-seed mean ± std)

#### PACS (3-seed)

| Algo | s2 | s15 | s333 | mean | per-domain |
|---|:--:|:--:|:--:|:--:|:--:|
| FedAvg | TBD | ... | ... | ... | ... |
| FedBN | ... | ... | ... | ... | ... |
| FedProto | ... | ... | ... | ... | ... |
| MOON | ... | ... | ... | ... | ... |
| F2DC (复现) | TBD | TBD | TBD | TBD | ... |
| **PG-DFC (我们)** | **TBD** | **TBD** | **TBD** | **TBD** | ... |

#### Office-Caltech10 (3-seed)
(同上)

### §4.3 Per-Domain Analysis (核心 contribution)

- 显示 small domain (art, dslr) 的 PG-DFC 增益最大 (+5-13pp)
- amazon outlier 现象 (合成商品图 vs 真实场景)
- 对应表/图: Fig 1 = per-domain Δ bar chart

### §4.4 Ablation

#### §4.4.1 PG-DFC 各组件
| Variant | Office mean | PACS mean |
|---|:--:|:--:|
| F2DC (no proto) | baseline | baseline |
| + prototype (no attention, mean) | TBD | TBD |
| + cosine attention | TBD | TBD |
| + server EMA β=0.8 | TBD | TBD |
| + Q from r_feat (vs nr_feat) | TBD | TBD |
| Full PG-DFC | TBD | TBD |

#### §4.4.2 超参敏感性
- proto_weight ∈ {0.1, 0.3, 0.5}
- attn_temperature τ ∈ {0.1, 0.3, 0.5, 1.0}
- server_ema_β ∈ {0, 0.5, 0.8, 0.95}

#### §4.4.3 收敛速度对比
- F2DC vs PG-DFC 的 best-round / convergence-round
- Office s15: PG-DFC R87 vs F2DC R98 (-11 round, 11% earlier)
- PACS s2: PG-DFC R64 vs F2DC R76 (-12 round, 16% earlier)

### §4.5 Trajectory Analysis (关键 motivation 证据)

- per-domain test acc trajectory (Fig 2)
- 关键发现: amazon 在 proto_weight 启动后 (R30+) 趋平甚至降, caltech 加速涨
- 这印证 amazon 是 outlier domain 的诊断

---

## §5 Discussion

### §5.1 PG-DFC 的边界
- 增益取决于 cross-client domain 的视觉相似度
- 当存在极端 outlier domain (amazon synthetic vs others realistic), 增益减弱
- Future work: outlier-aware aggregation

### §5.2 跟 FedDSA (我们之前 PFLlib 工作) 的关系
- FedDSA 在 backbone 内做 sem/sty 解耦 + 风格仓库
- PG-DFC 在 F2DC 框架内做 cross-client prototype 引导
- 两者可以组合: PG-DFC 的 prototype + FedDSA 的 disentangled prototype
- 留给 future work

### §5.3 Limitations
- 通信开销: prototype bank 每 round 上传下发 (但 14KB/round 可忽略)
- Outlier domain (amazon) 上增益有限 (Office +0.69pp 边缘)
- 需要 paper Sec 4.1 同样诚实承认这一点 (跟 F2DC 一样诚信)

---

## §6 Conclusion

- 重申 cross-client decoupling 概念
- PG-DFC 作为这个概念的首次实现
- Per-domain insight: small domain 增益最大
- Future: outlier-aware aggregation, multi-prototype hierarchy

---

## 实验数据回填 checklist (跑完后填)

### Wave 1 PACS s=2 (rand allocation, sc5, 已完成)
- [x] R0 sanity (vanilla 等价): max 68.015, R76 best
- [x] R1 full v32 (τ=0.3): max 66.402, R73 best
- [x] R2 τ=0.5: max 67.945, R64 best  ← best 配置

### Wave 2 PACS s=15+333 (fixed allocation, sc5)
- [ ] s=15: R47 max=69.115, 跑中
- [ ] s=333: R46 max=67.472, 跑中
- 完成时间预计: ~19:30

### Wave 1 Office s=2 (rand, sc5, 已完成)
- [x] R0: 59.44
- [x] R1: 65.26
- [x] R2 (τ=0.5 best): **66.87**
- [x] R4 (no server EMA): 62.57

### Wave 2 Office s=15+333 (fixed, sc5, 已完成)
- [x] s=15: 61.61, R87 best per-domain [64.29 75.26 56.90 50.00]
- [x] s=333: 60.89, R98 best per-domain [66.96 76.84 43.10 56.67]

### F2DC v2 fixed baselines (sc3, 4/6 完成)
- [x] Office s=15: AVG Best 60.80, per-domain (last) [61.2 74.2 43.1 40.0]
- [x] Office s=333: AVG Best 60.32, per-domain (last) [61.6 80.0 50.0 43.3]
- [x] Digits s=15: 93.74
- [x] Digits s=333: 93.43
- [ ] PACS s=15: 跑中 (sc3)
- [ ] PACS s=333: 跑中 (sc3)

### 起床后能算的 3-seed mean

#### Office (fixed, 严格)
- F2DC mean (s15+s333): **60.56**
- PG-DFC mean (s15+s333): **61.25**
- **Δ = +0.69pp** (边缘, 在 noise 范围)

#### PACS (待 F2DC s15+s333 fixed 完成)
- F2DC mean (s15+s333): TBD
- PG-DFC mean (s15+s333): seed=15 R47 已 69.12, 估计完成 70+
- **Δ 预测 +5-8pp** (paper grade, 等数据确认)

---

## 论文写作下一步

1. 等 PACS Wave 2 + F2DC v2 PACS 完成 (~3h)
2. 算严格 3-seed mean
3. 选定 paper 主推数字: PACS Δ + Office Δ + per-domain insight
4. 按本大纲写正文 (10-15 页)
