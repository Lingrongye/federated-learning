# 论文精读: FedOMG (ICLR 2025)

**创建日期**: 2026-04-23
**阅读目的**: 判断 FedOMG 的 server-side gradient matching 能否叠加到 FedDSA-SGPA 攻 Office -2.49pp 的缺口

---

## 1. 基本信息

| 字段 | 值 |
|------|---|
| 标题 | Federated Domain Generalization with Data-Free On-Server Matching Gradient |
| 会议 | ICLR 2025 |
| 作者 | Trong-Binh Nguyen, Minh-Duong Nguyen, Jinsun Park, Quoc-Viet Pham, Won Joo Hwang |
| 单位 | Pusan National University (韩国) + Trinity College Dublin (爱尔兰) |
| arxiv | 2501.14653v2 (2025-05-26) |
| 代码 | https://github.com/skydvn/fedomg |

**TL;DR**:
只在 server 端做 gradient 凸组合优化,找一个 "invariant gradient direction" 替代 FedAvg 的简单平均。不改 client,不加通信开销,号称能和 FedAvg/FedRod/FedBabu/FedSR/FedIIR/FedSAM 做**正交叠加**。PACS ResNet-18 pretrained leave-one-domain-out avg **88.4**, FedSAM+OMG 到 **88.8**。

---

## 2. 核心问题

**问:** server 端拿到每个 client 的 `g_u = θ_u^(r,E) − θ^(r)`, 能不能不靠额外的 client 数据就找出一个"域不变"的方向?

**关键洞察 (Shi et al. Fish ICLR 2022 的 Gradient Inner Product 思路的 FL 版本)**:
- 如果模型 θ 是 domain-invariant,那么不同域的 gradient `g(θ; D_u)` 应该方向一致(内积 > 0)
- 反过来,**把各 client gradient 之间内积最大化 = 在强迫 θ 朝 "各域都同意" 的方向走**
- 原 DG 版 Fish 要算二阶导(对 GIP 本身求导),FL 版做不起 → FedOMG 的贡献就是**绕过二阶导**

**两个分解:**
- `g_u^(r)` 里既有 "各域共性的下降方向", 也有 "只对域 u 有利的特有方向"
- 共性方向 → 要保留; 特有方向 (domain-bias) → 要丢

---

## 3. 方法精读 (核心)

### 3.1 问题重写为 Meta-Learning

原 DG 目标 (需要所有客户端 raw data):
$$\theta^* = \arg\min_\theta E[\theta; D_S] + \lambda \sum_{u,v \in U_S} d_M(P(\theta; D_u), P(\theta; D_v))$$

重写为两阶段 meta-learning:
- **Local update** (client): `φ_u^(r) = θ_g^(r-1) − η_l · g(θ_g^(r-1); D_u)` (标准 FedAvg 本地 SGD)
- **Meta update** (server): `θ_g^(r) = θ_g^(r-1) − η_g · g_IGD^(r)`

### 3.2 关键简化: IGD = Γ · g (凸组合)

**不**在 M 维参数空间里搜 invariant direction (M = 11.69M for ResNet-18), 而是在 `U_S` 维 (客户端数 = 4 for PACS) 系数空间里搜:

$$g_{IGD}^{(r)} = \Gamma \cdot g^{(r)} = \sum_{u \in U_S} \gamma_u \cdot g_u^{(r)}$$

其中 `Γ = {γ_1, ..., γ_{U_S}}` 是**可学习的客户端权重向量**, `g^(r) = {g_1^(r), ..., g_{U_S}^(r)}`.

- **Γ 是什么**: 一个长度为 `U_S` 的 scalar 向量 (PACS 4-client 只有 4 个数), 约束 `Σ γ_u = 1`
- **维度从 M 降到 U_S**: 11.69M → 4, 这就是为什么能在 server 上很便宜地跑

### 3.3 Server-side Objective (Eq. 11 / Eq. 15)

$$\Gamma^* = \arg\min_\Gamma \; \Gamma g^{(r)} \cdot g_{FL}^{(r)} + \kappa \|g_{FL}^{(r)}\| \cdot \|\Gamma g^{(r)}\|$$

注意这里 **argmin** (paper Eq. 15 写的是 argmin, 与 Eq. 11 的 argmax 等价——是对 cosine 做 -cos 方向最小化). 组合含义:
- **第一项 `Γg · g_FL`**: gradient matching — 找让 `Γg` 与 `g_FL` 方向一致的 Γ (g_FL 就是标准 FedAvg 的平均梯度)
- **第二项 `κ · ‖g_FL‖ · ‖Γg‖`**: searching space limitation — 限制 `Γg` 的范数不偏离 `g_FL` 太远 (相当于一个 M-ball 软约束)

### 3.4 最终更新公式 (Theorem 1, Eq. 16-17)

$$g_{IGD}^{(r)} = g_{FL}^{(r)} + \frac{\kappa \|g_{FL}^{(r)}\|}{\|\Gamma^* g^{(r)}\|} \cdot \Gamma^* g^{(r)}$$

$$\theta_g^{(r)} = \theta_g^{(r-1)} - \eta_g \cdot g_{IGD}^{(r)}$$

**解读**:
- `g_IGD = g_FL + κ · 单位化(Γ*g)` — 就是**在 FedAvg 梯度之上,朝 "各 client 都同意的方向" 推一小步**
- κ 控制推多少. κ → 0 退化为 FedAvg (Corollary 1). paper 推荐 κ = 0.5
- Γ* 的求解:paper 用 21 次迭代 + 动量 0.5 + 训练 lr=25 (??? 看起来是 server-side Γ 的更新 lr,对系数向量做 SGD,不是模型 lr)

### 3.5 "Orthogonality" claim (能 plug into 别的方法)

**机制**: `g_FL` 是参考 FL 算法的聚合梯度 (可以是 FedAvg, FedRod, FedBabu, FedSAM, FedSR, FedIIR 任意一种). `g_u` 是那个算法下 client 的 local delta.

→ FedOMG 只负责**对聚合后的方向做二次修正**, client-side 照常用 FedRod / FedBabu / FedSAM 的本地逻辑. 所以声称"正交".

**但 paper 自己也承认有冲突**: FedIIR / FedSR 本身就在 client 侧做 gradient-alignment regularization, 与 FedOMG 的 server-side alignment **功能重叠**, 所以 FedIIR+OMG / FedSR+OMG 的增益不如 FedSAM+OMG 明显.

---

## 4. 算法流程 (Algorithm 1)

```
每轮 r:
  # Client side (跟 FedAvg 一模一样)
  for u in U_S:
    θ_u^(r) ← θ_g^(r)
    for e in E:
      mini-batch ζ, gradient step: θ_u ← θ_u − η·∇E(θ_u, ζ)
    upload θ_u^(r,E) to server

  # Server side (FedOMG 新加的)
  for u in U_S:
    g_u^(r) = θ_u^(r,E) − θ_g^(r)        # local delta
  g_FL^(r) = mean(g_u^(r))                # FedAvg baseline gradient

  # Solve Γ* by 21-iteration SGD on coefficient vector
  Γ* = argmin_Γ [ Γg·g_FL + κ·‖g_FL‖·‖Γg‖ ]

  # Apply correction
  g_IGD = g_FL + κ·‖g_FL‖ / ‖Γ*g‖ · Γ*g
  θ_g^(r+1) = θ_g^(r) − η_g · g_IGD
```

---

## 5. 实验 setup

| 维度 | 值 |
|------|---|
| Backbone (FDG) | **ResNet-18 PyTorch pretrained** (不是 scratch) |
| Backbone (FL) | CNN 2-conv |
| PACS 客户端数 | 4 (leave-one-domain-out,match source domains) |
| Join rate | 1.0 (4/4 全参与) |
| Rounds | **R=100** global |
| Local epochs | E=5 |
| Local lr | 0.001 (FDG) / 0.005 (FL) |
| Batch size | 16 |
| Optimizer | SGD |
| Global lr η_g | 0.05 |
| Searching radius κ | 0.5 |
| Γ 迭代次数 | 21 |
| Γ 训练 lr | 25 (server-side) |
| Γ momentum | 0.5 |
| 评估 | leave-one-domain-out,3-run avg |

---

## 6. PACS 数字 (leave-one-domain-out, ResNet-18 pretrained)

### Main Table (Tab. 2 + Tab. 7, per-domain test acc)

| 方法 | P | A | C | S | **Avg** |
|------|---|---|---|---|---------|
| FedAvg | 92.7 | 77.2 | 77.9 | 81.0 | 82.7 |
| FedGA | 93.9 | 81.2 | 76.7 | 82.5 | 83.5 |
| FedSAM | 91.2 | 74.4 | 77.7 | 83.3 | 81.6 |
| FedIIR | 94.2 | 82.9 | 75.8 | 81.9 | 83.7 |
| FedSR | 94.0 | 82.8 | 75.2 | 81.7 | 83.4 |
| StableFDG | 94.8 | 83.0 | 79.3 | 79.7 | 84.2 |
| **FedOMG** | **98.0** | **89.7** | 81.4 | 84.3 | **88.4** |
| **FedIIR+OMG** | 97.7 | 83.0 | 80.8 | 79.3 | 85.2 |
| **FedSAM+OMG** | **98.3** | 88.9 | **82.7** | **85.5** | **88.8** |
| FedSR+OMG | 97.2 | 83.2 | 79.8 | 79.3 | 84.8 |

**+OMG 平均增益** (PACS Avg):
- FedAvg → FedOMG: **+5.7pp** (82.7 → 88.4)
- FedIIR → FedIIR+OMG: +1.5pp (83.7 → 85.2)
- FedSAM → FedSAM+OMG: **+7.2pp** (81.6 → 88.8)
- FedSR → FedSR+OMG: +1.4pp (83.4 → 84.8)

> **注意 setting 差异**: FedOMG PACS 用 ResNet-18 **pretrained**,而我们 FedDSA-SGPA 走的是从头训 ResNet-18。他们的 FedAvg baseline 82.7 比我们 FDSE 复现 79.91 高得多 (pretrained 起点不同)。不能直接比数字,但**相对增益 +5.7pp** 是主要参考信号。

### Centralized 对照 (不公平,只作参考)

| 方法 | PACS Avg |
|------|---------|
| ERM (centralized) | 86.7 |
| Fish | 86.9 |
| Fishr | 85.5 |
| **FedOMG (federated!)** | **88.4** ← 号称已经超过 centralized |

---

## 7. Ablation

### 7.1 κ (searching radius) — Fig. 4

- κ → 0: 退化为 FedAvg
- κ ≈ 0.5: **最优**
- κ → 1.0+: 性能下降 (搜索空间过大,Γ 优化易陷入次优)

### 7.2 Global lr η_g — Fig. 3

- EMNIST/CIFAR10 (无 domain shift): 大 η_g 好
- **PACS/VLCS/OfficeHome (有 domain shift): 小 η_g 好** (~0.05)

### 7.3 Local epochs E — Fig. 6

- E=5 最优
- E < 5: local gradient 代表性不够
- E > 5: over-fitting to local data,漂移太远

### 7.4 对不同 base method 的 gain

| Base | Base Avg (PACS) | +OMG Avg | Δ |
|------|:---------------:|:--------:|:-:|
| FedAvg | 82.7 | 88.4 (标量 FedOMG) | **+5.7** |
| FedSAM | 81.6 | 88.8 | **+7.2** |
| FedIIR | 83.7 | 85.2 | +1.5 |
| FedSR | 83.4 | 84.8 | +1.4 |

> 规律: **base 方法没做 client-side gradient alignment 的 (FedAvg, FedSAM) 收益最大**; base 方法自己就在 align gradient 的 (FedIIR, FedSR) 会有功能重叠,收益小.
>
> **推论**: 我们的 FedDSA-SGPA 不做 client-side gradient alignment (做的是 semantic prototype alignment,性质不同),按这个规律**应该属于"高收益"组**.

---

## 8. 能借鉴给我们 FedDSA-SGPA

### 8.1 能叠加吗? — **理论上可以,且预期增益高**

**判断理由**:

1. **机制层面不冲突**:
   - FedDSA-SGPA 的个性化在 **feature space** (semantic head + style head + prototype),在 client-side
   - FedOMG 的修正在 **parameter space** (骨干 conv 层的 gradient),在 server-side
   - 两个作用域互不干涉

2. **功能互补**:
   - 我们的 `serverdsa.py` 聚合 = 骨干 + 语义头 + 语义分类器走 FedAvg,风格头不聚合
   - FedOMG 可以**替换那个 FedAvg 步骤**,对骨干 + 语义头的聚合方向做二次修正
   - 风格头依然不聚合 (FedOMG 不影响)

3. **额外通信开销 = 0**:
   - Client 上传的还是 `θ_u^(r,E) − θ_g^(r)` (跟 FedAvg 一样)
   - Server 用手里已有的 `g_u` 算 Γ,没有额外轮次/流量

### 8.2 Drawback 与实现复杂度

1. **Server 计算开销**: 每轮 21 次 Γ 的 SGD 迭代. 对 PACS 4-client,Γ 是 4 维向量,几乎可忽略. OfficeCaltech 4-client 也一样.
2. **额外 hparam**: κ (搜索半径) + η_g (global lr) + 21 iter + Γ-lr=25 + Γ-momentum=0.5. paper 给出 PACS/VLCS/OfficeHome 统一用 κ=0.5, η_g=0.05. 可直接照抄.
3. **风险: double-sided alignment conflict**:
   - FedDSA 有 L_InfoNCE / L_orth / L_HSIC 已经在 align 语义空间
   - FedOMG 又在 param space align
   - 可能出现 "两边推不同方向" 的拉锯
   - **缓解**: 只对**骨干 conv 层**做 FedOMG,语义头/分类器走 FedAvg 就好 (部分应用)

### 8.3 对 Office 缺口 (-2.49pp) 的预期

| 情形 | 预期 Office AVG Best 增益 |
|------|:------------------------:|
| 直接叠加 FedOMG 修正骨干聚合 | +1.5 ~ +3pp (照 paper FedAvg→FedOMG +5.7pp on PACS,我们不 pretrained 打 5 折) |
| 对 Art 是否特别有帮助? | **是**. paper Fig. 5 显示 FedOMG 把 PACS S 域从 outlier 拉回均值,Office Caltech 作为 outlier 域同样适用 |

**对 PACS (目前 +0.73 领先)**: 预期略涨 (因为我们已经赢了,头部效应衰减). **主要想攻 Office**.

### 8.4 给 serverdsa.py 的 30 行 pseudo-code

```python
# serverdsa.py 伪代码 (加在现有 aggregation 之前)

def aggregate_with_omg(self, client_deltas, client_weights, kappa=0.5,
                       n_iter=21, gamma_lr=25.0, gamma_momentum=0.5):
    """
    client_deltas: list of dict {param_name: Δθ_u} for u=1..U
    只对 'backbone.*' 和 'semantic_head.*', 'sem_classifier.*' 层应用 FedOMG
    style_head.* 和 BN 层不走 FedOMG (保持原本地化)
    """
    # 1) 把每个 client 的 Δθ flatten 成一个向量
    g_list = [self._flatten_shared_layers(d) for d in client_deltas]  # [U, M]
    g_stack = torch.stack(g_list)  # [U, M]

    # 2) 标准 FedAvg 梯度 (按样本数加权)
    w = torch.tensor(client_weights) / sum(client_weights)  # [U]
    g_FL = (w.view(-1, 1) * g_stack).sum(dim=0)  # [M]
    g_FL_norm = g_FL.norm()

    # 3) 初始化 Γ (长度 U 的系数向量)
    Gamma = torch.ones(len(g_list), requires_grad=True) / len(g_list)
    opt = torch.optim.SGD([Gamma], lr=gamma_lr, momentum=gamma_momentum)

    # 4) 21 iter 优化 Eq. 15: min_Γ Γg·g_FL + κ·‖g_FL‖·‖Γg‖
    for _ in range(n_iter):
        opt.zero_grad()
        Gg = (Gamma.view(-1, 1) * g_stack).sum(dim=0)  # [M]
        loss = -(Gg * g_FL).sum() + kappa * g_FL_norm * Gg.norm()
        #     ^ 负号因为 matching 是 argmax ⟨·,·⟩,等价 argmin -⟨·,·⟩
        loss.backward()
        opt.step()
        # 约束 Σ γ_u = 1 可选 project 或 softmax 参数化

    # 5) 计算 g_IGD (Eq. 16)
    with torch.no_grad():
        Gg_star = (Gamma.view(-1, 1) * g_stack).sum(dim=0)
        g_IGD = g_FL + kappa * g_FL_norm / Gg_star.norm() * Gg_star

    # 6) reshape 回 param dict 并应用到 global_model
    g_IGD_dict = self._unflatten_shared_layers(g_IGD)
    for name, delta in g_IGD_dict.items():
        self.global_model.state_dict()[name].add_(delta)
    # 其余不共享的层 (style_head, BN) 保持原逻辑
```

**关键实现点**:
- `_flatten_shared_layers`: 只拼接需要 FedOMG 修正的层 (建议只对 `backbone.conv*` 的 weight,避开 BN running stats)
- `Gamma` 可以用 softmax 参数化天然满足 `Σ γ_u = 1`: `gamma_norm = softmax(logits)`, 然后优化 logits
- 如果内存紧张 (ResNet-18 11.69M × 4 client × float32 = 187MB),可以**分层** FedOMG: 每个 conv block 独立求 Γ

### 8.5 实验建议 (如果要验证)

**最小可行实验 (Office-Caltech, 3-seed)**:
1. Baseline A: 现有 FedDSA-SGPA orth_only (3-seed mean 89.09)
2. Treatment B: FedDSA-SGPA + FedOMG(κ=0.5, η_g=0.05) 只对骨干 conv
3. 如果 B > A by ≥ +1.5pp → 加入 main result
4. 如果 B - A < +1pp → 考虑对 semantic head 也上 OMG

**PACS 作为 sanity check** (已经领先 FDSE +0.73, OMG 不应该让它崩):
- 跑一个 single seed 验证 FedDSA + OMG 在 PACS 不退 → 过关
- 如果退了,说明 alignment 拉锯成立,只对 Office 用 OMG

---

## 9. 一句话总结

**FedOMG = 在 server 端对 FedAvg 的平均梯度做一个 U_S 维凸组合方向修正,把"所有 client 都同意的方向"放大一点,在 PACS ResNet-18 pretrained leave-one-out 从 FedAvg 82.7 推到 88.4 (+5.7pp), 零通信开销, 且理论上和我们 FedDSA-SGPA (做 feature-space 解耦而非 param-space alignment) 正交,预期能为 Office 补上 +1.5~3pp 缺口。**
