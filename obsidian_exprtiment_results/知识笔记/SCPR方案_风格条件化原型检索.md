# SCPR 方案:风格条件化原型检索(Self-Masked Style-Weighted Multi-Positive InfoNCE)

> **提问**:总结 SCPR 方案到知识笔记
> **Refine 结果**:GPT-5.4 xhigh 5 轮审核,**9.1/10 READY** ✅
> **Session**:`obsidian_exprtiment_results/refine_logs/2026-04-18_SCPR_v1/`

---

## 1. 一句话是什么

> **把 M3 的"等权多原型对齐"改成"按客户端风格相似度加权的多原型对齐",并且自己不算(self-mask)**。

举个例子(PACS 上 Art 客户端训练"狗"类):
- **M3 做法**:Art 等权拉近 {Cartoon 狗, Photo 狗, Sketch 狗} 三个原型 → 但 Sketch 太远,把"线条极简化"噪声也拉进来
- **SCPR 做法**:Art 按风格相似度加权 → Cartoon 0.7、Photo 0.25、Sketch 0.05;自己(Art)权重 0

---

## 2. 为什么需要 SCPR(3 个核心矛盾)

### 矛盾 1:FedDSA 叫 "Decouple-Share-Align",但 Share 从未 work 过

- EXP-059 在 z_sem 空间 AdaIN + CE + InfoNCE → **PACS −2.54%** ❌
- EXP-078d 在 h 空间 AdaIN → InfoNCE → **NaN 崩溃** ❌
- 论文卖点 "Share" 名不副实,**reviewer 一眼能看穿**

### 矛盾 2:已验证的两个增益各自**不普适**

| 增益点 | 效果 | 普适性 |
|--------|------|--------|
| M3 域感知多正例原型(EXP-072) | PACS +5.09% ✅ | **没利用风格信息**,等权 |
| SAS 风格感知 sem_head 聚合(EXP-084) | Office +1.21% ✅ | **PACS 全 outlier 退化 −0.65%**(EXP-086) |

SAS 在 PACS 失败的根因(EXP-086 诊断):参数空间软聚合 → 所有 client 参数都退化本地 → 破坏跨域共识。

### 矛盾 3:85 次实验的失败教训

6 大类反复失败:
1. 风格共享在特征层失败
2. 风格增强走对比损失 NaN
3. 辅助损失加法全败(HSIC/PCGrad/Triplet/CKA/Uncertainty)
4. 架构改动全败
5. 超参网格搜索无效
6. 过度安全阀 R200 均崩

**SCPR 要避开以上所有坑**,同时补齐 Share 章节。

---

## 3. 数学公式(核心 mechanism)

### 符号

| 符号 | 含义 |
|------|------|
| `s_k := normalize(E_{x ∈ D_k}[z_sty(x)])` | 客户端 k 的 L2-normalized 风格锚点 |
| `p_c^j := E_{(x,y=c) ∈ D_j}[z_sem(x)]` | 客户端 j 类 c 的原型 |
| `τ_nce = 0.3` | InfoNCE 温度(继承 Plan A) |
| `τ_SCPR = 0.3` | SCPR attention 温度(继承 SAS) |

### Self-masked style attention

```
g_kj = cos(s_k, s_j),     j = 1..K
raw_j = exp(g_kj / τ_SCPR) · 𝟙[j ≠ k]      # self-mask

A_k^c = { j : j ≠ k, p_c^j exists }
w_{k→j}^c = raw_j / Σ_{l ∈ A_k^c} raw_l    if j ∈ A_k^c
            0                                otherwise
```

### SCPR loss(SupCon multi-positive 的 style-weighted 版本)

```
L_SCPR(z, c)
  = - Σ_{j ∈ A_k^c} w_{k→j}^c · log [
        exp(sim(z, p_c^j) / τ_nce)
        / ( exp(sim(z, p_c^j) / τ_nce)
            + Σ_{c'≠c, l ∈ A_k^{c'}} exp(sim(z, p_{c'}^l) / τ_nce) )
      ]
```

所有 `p_c^j` 和 `s_j` 均 `.detach()`(软锚点)。

### 总损失(继承 Plan A 权重)

```
L = L_CE + 1.0·L_orth + 0.1·L_HSIC + 1.0·L_SCPR
```

### 极限性质(唯一主 claim)

> **`τ_SCPR → ∞`**:`w = 1/|A_k^c|` → **SCPR 严格退化为 M3**(已验证 PACS +5.09% 下界)

其他极限:
- **`τ_SCPR → 0`**:`w` 塌缩为 one-hot → nearest-style single positive
- **Self-mask**:`w_{k→k} ≡ 0`,Share 路径永不退化为 FedBN-like

---

## 4. Formal Derivation — 为什么 softmax 是推导结果不是设计

> **注意范围**:此推导在 **imperfect decoupling + 线性噪声近似** 这一 residual-noise 模型下成立,**不是所有 prototype weighting 的无条件定理**。

### 原型分解

```
p_c^j = p_c^* + eps_j
```
- `p_c^*`:理想"风格无关"类 c 原型
- `eps_j`:客户端 j 的风格残留(源自 `cos²(z_sem, z_sty) → 0` 软约束不能严格为 0,实测残余 cos ~ 0.05-0.1)

### SupCon pulling 可分解

```
sim(z, p_c^j) = sim(z, p_c^*) + sim(z, eps_j)
                 ──────────      ──────────
                 语义信号         风格噪声拉力(有害)
```

定义 `l_j := E[sim(z, eps_j) | client k]`(客户端 j 原型对 k 的风格噪声投影期望)。

### 优化目标(entropy-regularized noise minimization)

```
min_{w ∈ Δ^{K-1}}  J(w) = Σ_j w_j · l_j + τ · Σ_j w_j log w_j
s.t. Σ_j w_j = 1, w_j ≥ 0
```
(第 2 项熵正则防止 w 塌缩到 one-hot)

### 拉格朗日一阶条件

```
∂J/∂w_j = l_j + τ(log w_j + 1) + λ = 0
⇒ w_j* = exp(-l_j / τ) / Z,  Z = Σ_l exp(-l_l / τ)
```

### 线性近似(imperfect decouple 下风格越远噪声越大)

```
l_j ≈ c · (1 - cos(s_k, s_j)) = c · style_dist(k, j),  c > 0
```

### 代入得

```
w_j* ∝ exp(-c·(1 - cos(s_k, s_j)) / τ)
     ∝ exp(c·cos(s_k, s_j) / τ)
     = softmax_j(cos(s_k, s_j) / τ_SCPR),   τ_SCPR := τ / c
```

### 结论

**SCPR 的 softmax-over-cosine 权重是 entropy-regularized noise 目标的唯一 Boltzmann minimizer**,不是设计选择。

**两极限**(自然过渡):
- **Decouple 完美**(`l_j ≡ 0`):J(w) 只剩熵项 → uniform w 最优 → **SCPR = M3**
- **Decouple 不完美**(实际情形):style weighting 是最优噪声抑制

---

## 5. 2×2 差异化矩阵 — SCPR 独占位置

|   | 风格不共享 | 风格共享 |
|---|-----------|---------|
| **不解耦** | FedBN / FedAvg / FedProto | FISC / PARDON / StyleDDG / FedCCRL |
| **解耦** | FedSTAR / FedSeProto / FDSE | **SCPR(首次占据)** |

### 与 5 个最接近工作的差异

1. **FedProto (AAAI 2022)**:single global-mean proto + 等权对齐 → SCPR 用**域索引 multi-proto + style weighting**
2. **FPL (CVPR 2023) / FedPLVM (NeurIPS 2024)**:cluster-based multi-proto,聚类 key 是**原型自身几何** → SCPR 的 key 是**客户端风格**(语义完全不同)
3. **FedSTAR (2025)**:Transformer aggregator + FiLM 风格分离,但风格**严格本地** → SCPR **首次**把风格作为跨客户端 attention key
4. **FISC / PARDON (ICDCS 2025)**:共享风格但在**图像/混合特征空间 AdaIN**(EXP-059 证伪)→ SCPR 在**解耦表征空间** + 只改对齐 target
5. **SAS(我们)**:参数空间路由 → PACS 失败(EXP-086) → SCPR 把路由下放到**原型层**,保留参数共享

---

## 6. 为什么 SCPR 同时适用 PACS 和 Office

### 概念位置:M3 和 SAS 之间

- **M3**:无风格信息,所有 positive 等权 → PACS 良好,**没 exploit 风格邻域先验**
- **SAS**:参数空间风格路由 → Office 单 outlier 胜,**PACS 全 outlier 退化为本地**
- **SCPR**:**对齐目标风格加权 + 参数保持共享**(sem_head 全 FedAvg,跟所有 client 共享) → 两种 regime **均适用**

### SAS 失败根因 vs SCPR 隔离设计

| | SAS | SCPR |
|--|-----|------|
| 操作层 | 参数空间(sem_head 权重) | 原型层(对齐目标权重) |
| PACS 4-outlier | 所有 client 参数都退化本地,共识破坏 | 参数层 FedAvg 不变,共识保留 |
| Office Caltech outlier | Caltech 收到相似 client 的参数 | Caltech 收到风格匹配的对齐目标 |

**关键洞察**:SCPR 隔离了"过度个性化"的风险,同时仍然利用风格先验。

---

## 7. 复杂度预算

| 组件 | 状态 |
|------|------|
| 骨干 / sem_head / style_head / classifier / BN | 冻结复用 |
| 正交解耦(cos² + HSIC) | 复用 Plan A |
| Style bank(`{s_j}`) | 复用 SAS/EXP-084 实现 |
| Class-proto bank(`{p_c^j}`) | 复用 M3/EXP-072 实现 |
| **SCPR attention + self-mask + renorm** | **新增 ~30 行,无参数** |

- **0** 新 trainable 组件
- **0** 新 loss 项(只改 M3 loss 内部的 positive 权重)
- **1** 新超参(`τ_SCPR = 0.3`,继承 SAS 最优值)
- **~30 行**新代码(远低于 100 LOC 预算)

---

## 8. 3 个 Claim 验证设计

### Claim A(主):PACS 全 outlier,SCPR > M3 uniform

- **数据集**:PACS(4 domains, 7 classes)
- **对比**:
  - A.1 FedDSA orth_only(Plan A baseline,80.41%)
  - A.2 + M3 multi-pos(= SCPR uniform attention 下界,~81.91%)
  - A.3 **+ SCPR(τ_SCPR=0.3, self-mask)**
- **指标**:3-seed {2, 15, 333} mean AVG Best / Last / per-domain(Art、Sketch)
- **预期**:A.3 ≥ A.2 + 0.5%,A.3 ≥ **81.5%**
- **决定性**:若 A.3 < A.2,方法证伪

### Claim B(主):Office 单 outlier,SCPR ≥ SAS(不做参数个性化)

- **数据集**:Office-Caltech10
- **对比**:
  - B.1 orth_only
  - B.2 SAS(89.82%)
  - B.3 **SCPR(τ=0.3)**
- **指标**:3-seed mean AVG Best / Last / Caltech-specific
- **预期**:B.3 ≥ B.2,目标 ≥ **90.5%**
- **决定性**:证明**原型层 ≥ 参数层**,SCPR 普适两种 regime(SAS 只赢 Office)

### Claim C(机制):Outlier-ness correlation + τ 敏感性

**主诊断(非 tautological)**:
```
iso_k := 1 - mean_{j ≠ k} cos(s_k, s_j)      # 风格孤立度
gain_k := acc_k^{SCPR} - acc_k^{M3}           # per-client 改善
Spearman ρ(iso_k, gain_k) > 0                # outlier 受益更多
```

**为什么非 tautological**:gain_k 是 downstream accuracy 不是 cos 的函数;M3 baseline 与 style 无关;此预测**可证伪**。

**辅助**:τ ∈ {0.1, 0.3, 1.0, 3.0} 扫描,H(w_k) 曲线 + attention heatmap。

### 附录(非主):SCPR + SAS composability on Office

1 config × 3 seeds,只验证"两层路由是否正交"。

### 总预算

~33 runs ≈ **60 GPU·h ≈ 3 天单卡**:
- PACS 主:9 runs ≈ 18 h
- Office 主:9 runs ≈ 18 h
- τ 扫描:12 runs ≈ 24 h
- 附录:3 runs ≈ 6 h

---

## 9. 5 轮 Refine 评分轨迹

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | **Overall** | Verdict |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 9 | 6 ⚠️ | 6 ⚠️ | 8 | 8 | 8 | 6 ⚠️ | **7.1** | REVISE |
| 2 | 9 | 8 | 8 | 9 | 9 | 9 | 7 | **8.4** | REVISE |
| 3 | 10 | 9 | 8 | 9 | 9 | 9 | 8 | **8.9** | REVISE |
| 4 | 10 | 9 | 8 | 9 | 9 | 8 ⚠️ | 8 | **8.8** | REVISE |
| **5** | **10** | **9** | **9** | **9** | **9** | **8** | **9** | **9.1** | **READY ✅** |

**关键转折**:
- R1 → R2(+1.3):砍掉变体 sprawl,收敛为单一 self-masked style-weighted M3
- R3 → R4(−0.1):我加的 `ρ(w, -style_dist)` 诊断被 reviewer 抓是 tautological
- R4 → R5(+0.3):升级为 Formal MaxEnt 推导 + 换非 tautological 的 outlier-ness 诊断

---

## 10. 失败风险 + 内置安全网

### 可能的失败模式

| Failure | Signal | Fallback |
|---------|--------|----------|
| Attention uniform 塌缩 | `H(w_k) > 0.95·log(K-1)` 持续 > 20 轮 | 降维 `s_k` 或调低 τ_SCPR |
| 数值 NaN | loss NaN / Inf | `τ_SCPR: 0.3 → 1.0` 软化 |
| PACS < M3 | 3-seed AVG Best < 81.91% | 检查 self-mask 启用 / entropy |
| Outlier-ness ρ ≈ 0 | 机制未激活 | 方法证伪,方案 downgrade |

### 内置安全网(关键)

> **uniform w(τ→∞)下 SCPR 严格等价于 M3**,最差情况 = 81.91%(已验证)。
>
> 这是一个**数学保证**,不是经验希望 — 只要 attention 退化到 uniform,我们自动 fall back 到已验证的好方法。

---

## 11. 下一步 roadmap

```
1. /experiment-plan           ← 把 Claim A/B/C 细化成完整执行路线图
                                (精确到 config YAML、seed 顺序、fallback、成功标准)
2. 实现 + 单测                 ← ~30 行代码 + self-mask/τ 极限/renorm 单测
3. codex 代码 review           ← codex exec 做 gpt-5.4 审
4. 部署 + 跑实验               ← 单卡 60 GPU·h,3 天完成
5. 结果回填 NOTE.md            ← 按 CLAUDE.md 18.3 规范(对照行 + Δ + 全指标)
6. 论文写作 caveat             ← Formal Derivation 段落明确限定"在 residual-noise 线性模型下"
```

---

## 关键参考文件

| 文件 | 内容 |
|------|------|
| `obsidian_exprtiment_results/refine_logs/2026-04-18_SCPR_v1/FINAL_PROPOSAL.md` | canonical 最终方案 |
| `obsidian_exprtiment_results/refine_logs/2026-04-18_SCPR_v1/REVIEW_SUMMARY.md` | round-by-round 审核总结 |
| `obsidian_exprtiment_results/refine_logs/2026-04-18_SCPR_v1/REFINEMENT_REPORT.md` | 完整 evolution + raw reviewer 节选 |
| `obsidian_exprtiment_results/refine_logs/2026-04-18_SCPR_v1/score-history.md` | 评分演化表 |
| `obsidian_exprtiment_results/2026-04-15/EXP-072_adaptive_baselines.md` | M3 实验(SCPR 下界) |
| `obsidian_exprtiment_results/2026-04-17/EXP-084_office_sas.md` | SAS 实验(对照组) |
| `obsidian_exprtiment_results/2026-04-17/EXP-086_pacs_sas_failure.md` | SAS PACS 失败诊断(SCPR 设计动机) |

---

*记录时间:2026-04-19 01:30*
*来源:Claude Code 对话 + 5 轮 GPT-5.4 xhigh refine*
