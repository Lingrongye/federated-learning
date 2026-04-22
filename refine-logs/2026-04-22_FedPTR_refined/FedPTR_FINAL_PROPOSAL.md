# FedPTR — Final Refined Proposal (V1)

**日期**: 2026-04-22
**目标**: top-venue (CVPR 2026 / ICCV 2026) 可投的 FL 跨域泛化方案
**数据**: PACS / Office-Caltech10 / DomainNet
**目标**: 超 FDSE (CVPR 2025) +2pp on 3 datasets

---

## 0. 一句话方案

> **FedPTR** = FedBN + Class-level Prototype Trajectory 预测 + Class-Conditional Style Bank (CC-Bank) + Per-Client Learnable α
> 
> 首次把 prototype 当**时间曲线**（速度+曲率）建模; 用 **DG 理论 (Ben-David) 背书的 class-conditional** 风格对齐; per-client **difficulty-aware** 自适应增强强度.

---

## 1. 问题锚定

### 1.1 Cross-domain FL 的三个痛点

| 痛点 | 证据 | 我们怎么打 |
|:---:|:---:|:---:|
| **Prototype 按 round 独立聚合, 时间动力学被忽略** | FedProto/FPL/FedCA 都按 round 快照, 无时间建模 | **FedPTR**: 速度 + 曲率 + 预测 |
| **Style augmentation 全类 pool, 引入 class/style 混淆** | FedFA/FedCA 都 domain-level, class-conditional 为空 | **CC-Bank**: (class, domain) 二维 |
| **所有 client 同等增强强度, 难域 (Caltech 73%) 和易域 (DSLR 100%) 一刀切** | FedFA Σ 硬编码, FedCA AdaIN 直接替换 | **Learnable α**: per-client 自适应 |

### 1.2 与竞品的正交性（关键, novelty 防御）

| 竞品 | 他们做什么 | 我们不撞车的点 |
|:---:|:---:|:---:|
| **FDSE** (CVPR 2025) | 层分解 DFE/DSE, 擦除风格 | 我们**保留**风格做资产, 且做时间动力学 |
| **FedFA** (ICLR 2023) | Gaussian reparam on BN stats, **domain-level**, 硬编码 variance | 我们 **class-conditional** + **per-client α**, 不硬编码 |
| **FedCA** (ESWA 2025) | Shallow conv style bank + ASA, **domain-level** | 我们 **class-conditional bank** + **prototype trajectory** |
| **FedCPD** (IJCAI 2025) | Memory distillation, 按 round 独立 | 我们做 **一阶速度 + 二阶曲率 + 预测** 的时间动力学 |
| **FedDr+** (2024) | Dot-regression classifier alignment | 我们不用 classifier weight 作 prototype, 而是**特征均值 prototype + trajectory** |

---

## 2. 方法详述

### 2.1 架构 (纯 FedBN, 零新增模块)

```
输入 x
  ↓
AlexNet encoder (BN 层完整本地化, 不聚合 γ/β/running_stats)
  ↓ feature h ∈ R^{1024}
  ↓
[训练时: CC-Bank 采样 → feature-level AdaIN → α 混合]
  ↓
Linear classifier W ∈ R^{10 × 1024}
  ↓
logits → CE loss
```

**关键**: 
- ✅ 完整 FedBN (不是半 FedBN, γ/β/running 都本地)
- ✅ 不加 projection head (classifier 直接接 1024d feature)
- ✅ 不加双头 (no style head)

### 2.2 三个组件的数学严格化

#### 组件 1: Class-level Prototype Trajectory (FedPTR core)

**定义全局类原型**:
```
p_c^t = Σ_k (n_{c,k} / N_c) · p_{c,k}^t       # n_{c,k}: client k 中类 c 样本数, N_c: 总
p_{c,k}^t = (1/n_{c,k}) Σ_{x:y=c} h_k(x)       # client k 的类 c 本地 prototype
```

**时间动力学**:
```
v_c^t = p_c^t - p_c^{t-1}                      # 一阶速度
κ_c^t = ||v_c^t - v_c^{t-1}||₂                 # 二阶曲率 (漂移速度变化)
p̂_c^{t+1} = p_c^t + η · v_c^t                  # 一阶预测 (η=0.5 默认)
```

**Client-side alignment loss (用预测位置)**:
```
L_align = Σ_c Σ_{x: y=c} λ_c^t · (1 - cos(h_k(x), sg(p̂_c^{t+1})))
```
- `sg(·)` = stop-gradient, 防止 prototype 跟 encoder 互相"作弊"
- `λ_c^t = 1 / (1 + β·κ_c^t)` = 曲率大的类 alignment 权重降 (β=1.0 默认)

**为什么 class-level trajectory 不怕 client 少**:
- Trajectory 跟踪的是**每类的全局 prototype 轨迹**
- 7-10 类 × R=200 round = 1400-2000 数据点, 样本量充足
- 跟 client 数 (4-6) **无关**

#### 组件 2: CC-Bank (Class-Conditional Style Bank)

**Client upload** (每 round):
```
对每个 class c, client k 上传 class-conditioned pooled stats:
  μ_{c,k}^t = (1/n_{c,k}) Σ_{x: y=c} h_k(x)    # 类 c 特征均值 (= p_{c,k}^t)
  σ_{c,k}^t = sqrt((1/n_{c,k}) Σ_{x: y=c} (h_k(x) - μ_{c,k}^t)²)    # 类 c 特征 std
```

**Server-side bank** (2D index):
```
Ψ^t = {(μ_{c,k}^t, σ_{c,k}^t) : c ∈ classes, k ∈ clients}
Ψ 结构: dict[class_c][client_k] = (μ, σ)
```

**Augmentation sampling** (client-side, 训练时):
```
for (x, y=c) in batch:
  # 从其他 client 的同类 bank 采样
  other_clients = {k' : k' ≠ self.id}
  k* ~ Uniform(other_clients)             # 随机选另一个 client
  (μ', σ') = Ψ[c][k*]                     # 同类不同 client 的 style
  
  # Feature-level AdaIN (注意用本地 batch stats, 不是 running)
  μ_self, σ_self = batch_stats(h_k(x))
  h_aug = σ' · (h_k(x) - μ_self) / σ_self + μ'
```

**关键差异 (vs FedFA)**:
- FedFA: `Ψ = {(μ_k, σ_k) : k ∈ clients}` — 1D, domain-level
- 我们: `Ψ = {(μ_{c,k}, σ_{c,k}) : c × k}` — 2D, **class-conditional**

#### 组件 3: Per-Client Learnable α

**Per-client scalar**:
```
α_k = sigmoid(φ_k) ∈ [0, 1]               # φ_k 是可学标量, 每 client 一个
```

**Final feature mix**:
```
h_final = α_k · h_aug + (1 - α_k) · h_k(x)
```

**Training signal**:
- α_k 通过 L_CE(classifier(h_final), y) 的梯度自然更新
- **不聚合**: α_k 在 private_keys 里, 每 client 独立

**自适应的直觉**:
- Caltech loss 高 → encoder 更依赖 h_aug 救自己 → gradient push α 变大
- DSLR loss 低 → 不需要 h_aug → α 自然小

### 2.3 总损失

```
L_total = L_CE(W·h, y)                                          # 1. 原始 CE
        + λ_CE_aug · L_CE(W·h_final, y)                         # 2. 增强后 CE
        + λ_align  · Σ_c λ_c · (1 - cos(h, sg(p̂_c^{t+1})))      # 3. trajectory alignment
```

**超参默认**:
- `λ_CE_aug = 1.0`, `λ_align = 0.5`, `β = 1.0` (曲率调节), `η = 0.5` (预测步长)

### 2.4 算法 Pseudocode

```python
# =============== SERVER side (every round t) ===============
def server_round(t, received_clients):
    # 1. Aggregate model (FedBN: skip all BN params)
    global_model = FedBN_aggregate(received_clients)
    
    # 2. Update class prototypes p_c^t
    for c in classes:
        p_c[t] = weighted_avg({k.p_c_local for k in received_clients}, 
                               weights={k.n_c for k in received_clients})
    
    # 3. Compute velocity, curvature, prediction
    for c in classes:
        v_c[t] = p_c[t] - p_c[t-1]
        κ_c[t] = norm(v_c[t] - v_c[t-1])
        p_hat_c[t+1] = p_c[t] + η · v_c[t]
        λ_c[t] = 1 / (1 + β · κ_c[t])
    
    # 4. Update CC-Bank (append new class-conditional stats)
    for k in received_clients:
        for c in classes:
            Ψ[c][k.id] = (k.μ_c, k.σ_c)
    
    # 5. Broadcast: model, Ψ, p_hat, {λ_c}
    return {'model': global_model, 'bank': Ψ, 'pred_protos': p_hat, 
            'class_weights': λ_c}

# =============== CLIENT k side (local training, round t) ===============
def client_train(package, local_data, local_model, φ_k):
    load_shared_keys(local_model, package['model'])  # 不覆盖 BN/style params
    Ψ = package['bank']
    p_hat = package['pred_protos']  # dict: class c -> predicted prototype
    λ_c_dict = package['class_weights']
    
    α_k = sigmoid(φ_k)  # current scalar
    
    for epoch in range(E):
        for x, y in local_data:
            h = encoder(x)                           # BN 本地
            
            # Augmentation via CC-Bank
            for i, c in enumerate(y):
                k_other = random.choice(other_client_ids)
                μ_prime, σ_prime = Ψ[c][k_other]
                μ_self = h[i].mean(dim=-1)
                σ_self = h[i].std(dim=-1) + 1e-5
                h_aug[i] = σ_prime * (h[i] - μ_self) / σ_self + μ_prime
            
            # Final feature
            h_final = α_k * h_aug + (1 - α_k) * h
            
            # Losses
            L_ce       = CE(classifier(h), y)
            L_ce_aug   = CE(classifier(h_final), y)
            L_align    = sum(λ_c_dict[c] * (1 - cosine(h[i], p_hat[c].detach())) 
                             for i, c in enumerate(y)) / len(y)
            
            L = L_ce + λ_CE_aug * L_ce_aug + λ_align * L_align
            
            L.backward()
            optimizer.step()   # updates encoder, classifier, φ_k
    
    # Compute upload stats
    μ_c_k, σ_c_k = compute_class_conditional_stats(local_data, encoder)
    p_c_k = {c: μ_c_k[c] for c in classes}
    
    return {'model': local_model, 'μ_c': μ_c_k, 'σ_c': σ_c_k, 'p_c': p_c_k}
```

---

## 3. Pilot 实验 (证明 CC-Bank 必要性)

### 3.1 Pilot 问题

**Claim**: PACS 上类间 style drift 异质性 > 20% → CC-Bank 显著优于 domain-level pool

### 3.2 量化指标

**Domain-level drift** (baseline, FedFA 用的):
```
d_domain(d1, d2) = ||avg_all_c(μ_{c,d1}) - avg_all_c(μ_{c,d2})||₂
```

**Class-conditional drift** (我们用的):
```
d_class_c(d1, d2) = ||μ_{c,d1} - μ_{c,d2}||₂
```

**异质性 metric**:
```
heterogeneity = std_c(d_class_c(d1, d2)) / mean_c(d_class_c(d1, d2))
```

如果 heterogeneity > 20% → 不同类的 cross-domain drift 差异显著 → **class-conditional 必要**

### 3.3 Pilot 实验设计

1. 训一个 FedBN baseline (不加任何 augmentation)
2. R=50 收敛后, 每个 client 提取每类的 `(μ_c, σ_c)`
3. 对每对 domain (d1, d2), 算所有类的 `d_class_c(d1, d2)` 分布
4. 看 heterogeneity 是否 > 20%

**预期数据** (基于 PACS 特性):
- PACS 4 domain × 7 class, 计算 6 × 7 = 42 个 `d_class_c`
- Art vs Sketch: dog 的 drift 大 (有机纹理被笔触重塑), giraffe 轮廓简单 drift 小
- 如果 heterogeneity > 20% → paper 里直接引用作 motivation

**成本**: 半天 GPU (已有 FedBN baseline 可复用)

---

## 4. 消融矩阵 (每层独立可证)

| # | Config | Args | PACS 预期 | Office 预期 | DN 预期 | 验证什么 |
|:-:|---|---|:---:|:---:|:---:|:---:|
| 0 | FedBN | baseline | 79.01 | 88.68 | 72.08 | — |
| 1 | +Trajectory alignment 无曲率 | η=0.5 β=0 | 79.5 | 89.2 | 72.5 | 时间预测帮? |
| 2 | +Curvature reweight | η=0.5 β=1 | 79.8 | 89.5 | 72.8 | 类漂移抑制帮? |
| 3 | +Bank (domain-level, FedFA-like) | + FedFA bank | 80.3 | 90.2 | 73.3 | 风格 bank 基础涨 |
| 4 | +Bank (**class-conditional**) | + CC-Bank | 81.2 | 91.0 | 73.8 | **class vs domain diff** |
| 5 | +Learnable α | + adaptive intensity | **82.0** | **92.5** | **74.3** | 难度自适应 |

**vs FDSE 81.57/90.58/72.21**:
- PACS: +0.4 ~ +0.6
- Office: **+1.9 ~ +2.0** ← 主攻
- DomainNet: +2.0 ~ +2.1

每层独立 +0.5pp 左右, 全开 +3pp over FedBN, +2pp over FDSE.

### 诚实风险预估
- 预期数字基于 "组件乘法效应假设", 实际可能**次线性累加** (+1.5pp 而非 +3pp)
- Worst case: PACS +0 (FedPTR 在 PACS 已经接近饱和), Office +1pp (关键战场)
- Best case: 全 dataset +2~3pp, paper 完美

---

## 5. Reviewer Attack 预案 (Top-5 最危险)

### Q1: "FedPTR 跟 FedCPD (IJCAI 2025) / prototype momentum 工作有啥区别?"
- **Answer**:
  - FedCPD 做 **memory distillation**, 每 round 独立 contrast
  - Prototype momentum 只做**一阶 EMA**, 没 velocity/curvature/prediction
  - 我们是**二阶动力学** (曲率) + **预测** (p_hat), 这组合是新的
  - 消融: 关掉预测 / 关掉曲率 → 分别下降, 证明两者独立贡献

### Q2: "CC-Bank 就是 FedCA 加 class 维度, novelty 不够?"
- **Answer**:
  - FedCA 没有做 class-conditional, 也没给 **DG 理论背书**
  - Ben-David 2010 bound 明确说条件分布对齐 (给定 y) > 边缘分布对齐
  - 我们是**首次在 FL 引入 Ben-David 条件对齐 + style bank**
  - Pilot 实验 (类间 drift 异质性 > 20%) 提供 empirical 动机

### Q3: "Learnable α 就是可学权重, 太 trivial?"
- **Answer**:
  - 不是普通 attention 权重, 是 **per-client injection intensity**
  - 配合 class-conditional bank 才有意义 (选什么 style × 用多少)
  - 消融 α 固定 vs 可学 → 证明自适应的价值
  - PACS vs Office 不同难度下 α 的学出值不同 → 机制可解释性

### Q4: "3 个组件 novelty 分别都不够高, 整体贡献是 incremental?"
- **Answer**:
  - 每个组件填补不同 gap (时间 / 类别 / 样本量)
  - 消融矩阵证明累积增益 +3pp 不是随机
  - 跟 FDSE/FedFA/FedCA **三个方向正交**
  - Paper claim 是"systematic exploration of 3D design space", 不是单点创新

### Q5: "如果把 FedFA + FedCA + FedCPD combine 起来是不是就是你们?"
- **Answer**:
  - FedFA + FedCA 都没 class-conditional (我们的 CC-Bank 是新)
  - FedFA + FedCA + FedCPD 都没 trajectory prediction (我们 FedPTR 是新)
  - 任何 3-way combination 都**缺一个或多个核心组件**, 总的 delta 是我们的 contribution
  - 实验验证: 我们比 `FedFA+CA` 高 +X pp, 比 `+CPD` 高 +Y pp

---

## 6. 实现计划 (3 Stages)

### Stage 1: Pilot 验证 (0.5 天)
- 复用 seetacloud2 上已跑完的 FedBN baseline ckpt
- 写 Python 脚本算类间 style drift 异质性
- **Go/No-Go**: heterogeneity > 20% → 继续; 否则 revise motivation

### Stage 2: Minimal Viable Implementation (1.5 天)

新建文件: `FDSE_CVPR25/algorithm/fedptr.py` (~230 行)

代码分块:
```python
# Section 1: FedPTR Model (纯 FedBN, 无改动) -- 30 行
# Section 2: Server class -- 120 行
#   - init_agg_keys: FedBN 完整本地 (所有 BN)
#   - aggregate_prototypes: 加权平均 class prototypes
#   - update_trajectory: velocity + curvature + prediction
#   - update_cc_bank: 2D dict 管理
#   - pack: 下发 {model, Ψ, p_hat, λ_c}
# Section 3: Client class -- 80 行
#   - unpack: 加载 shared_keys (保留 BN + φ_k)
#   - augment_with_ccbank: 按 class 采样其他 client 的 style
#   - compute_losses: CE + CE_aug + L_align (curvature-weighted)
#   - update_alpha: sigmoid scalar per-client
# Section 4: init_global_module hook -- 10 行
```

新 config: `FDSE_CVPR25/config/{pacs,office,domainnet}/fedptr_*_r200.yml`

单元测试 (15+ tests):
- test_trajectory_velocity_curvature
- test_cc_bank_2d_structure
- test_ccbank_sampling
- test_alpha_gradient_flow
- test_fedbn_aggregation_complete
- test_trajectory_prediction_bounded
- test_class_curvature_weight
- ...

### Stage 3: 全量部署 (3-5 天)

1. Smoke test: Office seed=2 R=50, 看 Best > 89.0 (FedBN + 轻微)
2. 3-seed 消融 × 3 datasets = 45 runs, GPU 约 60h
3. 回填 NOTE.md + 对比 FDSE/FedFA/FedCA

---

## 7. 目标 venue + 时间线

| 阶段 | 时间 | venue 目标 |
|:---:|:---:|:---:|
| Pilot + Smoke | 2 天 | 内部决策 |
| 全量消融 | 5 天 | Paper draft 起点 |
| Paper writing | 10 天 | **CVPR 2026 (Nov 2026)** / ICCV 2026 (March 2027) |
| Rebuttal 预案 | — | 提前准备 Q1-Q5 答案 |

如果全量数据不达 +2pp: **WACV 2026 / BMVC 2026 / AAAI 2026**

---

## 8. 命名最终决定

- **主名**: **FedPTR** (Federated Prototype Trajectory Regularization)
- **副标题**: "with Class-Conditional Style Bank and Adaptive Intensity"
- **全称**: FedPTR-CCA

---

## 9. Risk Register

| Risk | 概率 | 缓解 |
|:---:|:---:|:---:|
| Trajectory 预测在小 client 数下不稳 | 中 | Stage 1 pilot 预先验证 + β 保守 (=1) |
| CC-Bank 在小类样本下 μ/σ 估计噪声大 | 中 | `n_{c,k} ≥ 5` 才上传, 不够的跳过该 bank entry |
| Learnable α 收敛到极端 (0 或 1) | 低 | L1 正则 push α 向 0.5, 禁止退化 |
| Office Caltech 本身瓶颈在 AlexNet capacity | 中 | paper 在 ablation 注明, 不强 claim Caltech 必超 FDSE |
| 历史 AdaIN 失败教训 (EXP-040/059/061) | 中 | 用 class-conditional 避免 label-noise, 配合 α 自适应 |

---

## 10. 关键时间线建议

**今晚 → 明早** (约 8h):
1. Stage 1 pilot (类间 drift 异质性) → 1h
2. 写代码 (fedptr.py) → 4h
3. 单元测试 (15+ tests) → 2h
4. Smoke test 启动 → 1h

**明天早** (收据):
- Smoke test 结果 → Go/No-Go 决策
- 如果 Office seed=2 R50 > 89.0 → 全量 3-seed 部署
- 否则 debug 或切方案 W/V

**下周** (全量):
- 45 runs 消融完成
- 回填 paper draft 数据

---

## 11. 下一步

1. 用户确认这个 refined proposal
2. 进入 **research-review** (GPT-5.4 深度 review) 或跳过直接实现
3. 启动 Stage 1 pilot
4. 代码实现 (230 行, 1.5 天)
5. Smoke test → 全量

**等你 Gate**: 方案定稿 → 开始 research-review 还是直接写代码?
