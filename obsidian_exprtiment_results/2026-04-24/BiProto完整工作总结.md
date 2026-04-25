# FedDSA-BiProto 完整工作总结 (2026-04-24 ~ 2026-04-25)

> 本文档整合从 research-refine 设计 → 代码实现 → 实测 → 诊断 → 修复方案探索的全部内容.
> 关联文档:
> - 学术版方案: `refine_logs/2026-04-24_feddsa-biproto-asym_v1/FINAL_PROPOSAL.md`
> - 大白话版方案: `知识笔记/大白话_FedDSA-BiProto方案.md`
> - 实验目录: `experiments/ablation/EXP-126_biproto_S0_office_gate/`, `EXP-127_biproto_full_r200/`

---

## 第一阶段: 方案设计 (Research Refine 4 轮)

**位置**: `refine_logs/2026-04-24_feddsa-biproto-asym_v1/`

| 轮次 | Score | Verdict | 关键变化 |
|:-:|:-:|:-:|---|
| R0 | — | 初稿 | 双原型 (Pc + Pd) + L_sem_proto + L_sty_proto + L_proto_excl + 非对称 encoder + 5 套可视化 |
| R1 | 6.5 | REVISE | 删 L_sem_proto；Pd 改 domain-indexed；C0 改 matched intervention；Visual 5→3；预算 stage-gated |
| R2 | 7.8 | REVISE | C0 fix（freeze encoder_sem only）；hybrid ST axis（forward=Pd, grad via batch centroid）；Pc 降为 monitor only |
| R3 | 8.25 | REVISE | ST normalize 明确；present_classes only；"to our knowledge" + Limitations §；−Pd MANDATORY |
| **R4** | **8.75** | **REVISE (near-READY)** | Pd 初始化、claim downgrade pre-register、−encoder_sty 单 factor、ST estimator transparency |

**最终方案 BiProto v4 核心机制**:
- 非对称 encoder：encoder_sem (AlexNet 14M) + encoder_sty (1M MLP on (μ,σ) taps from conv1-3)
- 双原型：Pc (class, monitor only) + Pd (domain, federated EMA, 训练 anchor)
- 5 个 loss：L_CE + L_CE_aug + λ₁·L_orth + λ₂·L_sty_proto + λ₃·L_proto_excl
- L_sty_proto = InfoNCE(z_sty → Pd[k]) + 0.5·MSE
- L_proto_excl = cos²(class_axis, domain_axis)，domain_axis 用 hybrid ST trick

**产出文档**: FINAL_PROPOSAL.md / REVIEW_SUMMARY.md / REFINEMENT_REPORT.md / 大白话_FedDSA-BiProto方案.md / 4 轮 review-raw + refinement md

---

## 第二阶段: 代码实现 + 集成 (~30 GPU-min 调试)

**位置**: `FDSE_CVPR25/algorithm/feddsa_biproto.py` (~600 行)

| 模块 | 实现 |
|---|---|
| `FedDSABiProtoModel` | 含 encoder_sem (inherit AlexNet) + StatisticEncoder (1M MLP) + Pc/Pd register_buffer |
| `Server` | 独立 init_algo_para 22 项；自定义 _init_agg_keys (Pd/Pc 不参与 FedAvg)；override iterate (Pc/Pd EMA) |
| `Client` | _maybe_freeze_encoder_sem + 5-loss train_step + Bell schedule + present_classes only + ST hybrid axis + sparse-batch fallback |
| `init_global_module` | 自动从 task name 推断 (PACS=7类/4client, Office=10类/4client) |
| `init_ckpt` | 加载 orth_only ckpt 用于 C0 gate (strict=False) |

**修复的 6 个 bug** (在调试和 codex review 中发现):

1. `algo_para` 索引错位 (父类已吃前 18 项)
2. optimizer 重建用 `type()` 不兼容 Adam → 改 SGD + filtered params
3. `w_sty_proto=0` 时 MSE 被零乘失效 → InfoNCE 和 MSE 独立权重
4. `bc_d = z_sty[(y==y)].mean()` 写法误导 → `z_sty.mean(dim=0)`
5. Client.num_rounds 缺失 → server pass num_rounds
6. Client.train 末尾缺 _local_protos 等占位 → 加 placeholder 兼容父类 pack()

**配置**: 6 个 yml (Office/PACS × {S0 gate A, S0 gate B baseline, R200 full})

---

## 第三阶段: EXP-126 S0 Matched-Intervention Gate (~6 GPU-h, 4 runs)

**位置**: `experiments/ablation/EXP-126_biproto_S0_office_gate/`

| 设置 | 内容 |
|---|---|
| 加载 ckpt | EXP-105 orth_only Office R200 s=2 (best=89.38) / orth_only PACS R200 s=2 (best=78.26) |
| 冻结 | encoder_sem only，head + encoder_sty 可训 |
| Round | R30 (warmup ws=5/we=10/rd=25 压缩 schedule) |
| 4 runs | Office_A (BiProto-lite) / Office_B (head-only baseline lp=0/le=0) / PACS_A / PACS_B |

**结果**:

| 数据集 | A best | B best | Δ best | A last10 | B last10 | Δ last10 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Office | 88.26 | 88.26 | **+0.00** | 87.99 | 88.08 | **−0.09** |
| PACS | 75.82 | 75.82 | **+0.00** | 75.69 | 75.59 | **+0.10** |

**判决**: Δ ≪ +0.3pp 阈值 → 按 NOTE.md 设计 "Negative S0 sufficient to kill" 应该 KILL BiProto。但 R4 reviewer 警告过 "C0 是 pruning test, not causal proof"，用户决定**绕过 S0 跑完整 R200 验证**。

---

## 第四阶段: EXP-127 完整 R200 Pipeline (~30 GPU-h, 6 runs)

**位置**: `experiments/ablation/EXP-127_biproto_full_r200/`

**6 runs**: BiProto × {Office, PACS} × 3 seeds (2/15/333)，标准 Bell schedule (warmup=50 / peak=80-150 / ramp_down=200)，from-scratch fz=0。lab-lry GPU 1 并发，wall ~5h。

### Accuracy 真实结果

**Office**:
| seed | BiProto | orth_only | FDSE | Δ vs orth | Δ vs FDSE |
|:-:|:-:|:-:|:-:|:-:|:-:|
| s=2 | 88.00 | **90.74** | **92.39** | −2.74 | −4.39 |
| s=15 | 86.65 | **90.11** | **91.24** | −3.46 | −4.59 |
| s=333 | **92.46** | 91.52 | 88.11 | **+0.94** | **+4.35** |
| **mean** | **89.04** | **90.79** | **90.58** | **−1.75 ❌** | **−1.54 ❌** |

**PACS**:
| seed | BiProto | orth_only | FDSE | Δ vs orth | Δ vs FDSE |
|:-:|:-:|:-:|:-:|:-:|:-:|
| s=2 | 80.70 | **82.02** | **82.16** | −1.32 | −1.46 |
| s=15 | **80.76** | 80.86 | 80.14 | −0.10 | +0.62 |
| s=333 | 81.67 | **83.65** | **82.33** | −1.98 | −0.66 |
| **mean** | **81.05** | **82.18** | **81.54** | **−1.13 ❌** | **−0.49 ❌** |

**判决**: BiProto 在两个数据集上**均输给 orth_only 和 FDSE**。

---

## 第五阶段: 诊断分析 (4 套诊断, ~3 GPU-h)

### 诊断 1: t-SNE 双面板可视化 (Vis-A)

**位置**: `experiments/ablation/EXP-127_biproto_full_r200/figs/`

- BiProto Office s=333 R200 `tsne_biproto_office_s333_R200.png`
- orth_only Office s=333 R200 `tsne_orth_only_office_s333_R200.png`
- BiProto Office S0 (R30 frozen, 之前的) `EXP-126_*_/figs/tsne_biproto_s0_office_A_s2.png`

**视觉结论**:
- z_sem by class: BiProto 跟 orth_only 都有清晰 cluster
- z_sty by domain: **BiProto 4 团比 orth_only 更紧凑** (看似 disentanglement 成功)
- z_sty by class: BiProto 看起来更乱 (看似 class 信号被推走)

⚠️ 视觉好看 ≠ 真好，需要 SVD 验证。

### 诊断 2: Probe Ladder (Vis-B, 真 test_data)

| 方向 × 容量 | BiProto | orth_only |
|:-:|:-:|:-:|
| z_sem→class linear/MLP-64/256 | 0.941/0.922/0.922 | 0.902/0.843/0.863 |
| z_sem→domain | 0.529/0.333/0.529 | 0.784/0.510/0.608 |
| **z_sty→class** | **0.392/0.137/0.059** | 0.863/0.647/0.784 |
| z_sty→domain | 1.000/1.000/1.000 | 0.902/0.451/0.745 |

**probe 数据看起来 BiProto 大胜**: z_sty→class MLP-256 从 0.78 压到 0.06（−72pp），z_sem→domain 也大幅下降。

### 诊断 3: per-domain accuracy (真 test_data)

| Domain | n | BiProto s=333 | orth_only s=333 | Δ |
|---|:-:|:-:|:-:|:-:|
| Amazon | 112 | 93.75 | 91.96 | +1.79 |
| **Caltech** | 95 | **98.95** | **63.16** | **+35.79** |
| DSLR | 15 | 100 | 100 | 0 |
| Webcam | 29 | 100 | 100 | 0 |

⚠️ **数据有 4 pp 不匹配** record 92.46（client local model 含 BN 跟 server FedAvg model 评估口径差异），BiProto 的"Caltech 大胜"可能是 personalized 评估假象。

### 诊断 4: F2DC 风格 SVD Spectrum + Effective Rank ⭐ **最关键**

**位置**: `experiments/ablation/EXP-127_biproto_full_r200/figs/svd_spectrum_biproto_vs_orth_office.png`

**Effective Rank 表 (128 维 z_sem / z_sty 实际占了多少维)**:

| Model | Seed | z_sem ER (/128) | **z_sty ER (/128)** | Office Best |
|---|:-:|:-:|:-:|:-:|
| BiProto | s=2 | 8.61 | **16.07** | 88.00 |
| BiProto | s=15 | 8.70 | **17.06** | 86.65 |
| BiProto | s=333 | 8.96 | **2.73 ⚠️ 严重** | **92.46** |
| orth_only | s=2 | 8.66 | 28.14 | 90.74 |
| orth_only | s=15 | 8.97 | 29.31 | 90.11 |
| orth_only | s=333 | 8.14 | 29.69 | 91.52 |

**铁证**:
- z_sem ER 两边几乎一样 (8.6-9.0) → BiProto 没让 z_sem 退化
- **z_sty ER 大幅塌缩**: 28-30 → 2.7-17 = mode collapse 铁证
- s=333 z_sty 只占 2.73 维（实际坍塌到 D-1=3 个 domain 自由度）

---

## 核心机制发现

### 🐛 Bug 1: Pd 自指死循环导致 z_sty mode collapse

```
z_sty (encoder_sty 输出)
  ↓ ① client 上传 mean
Pd[k] = client k 自己 z_sty 的 EMA mean
  ↓ ② broadcast 回 client
L_sty_proto = MSE(z_sty → Pd[k]) + InfoNCE(target 全是 k)
  ↓ ③ z_sty 拉向 Pd[k]
z_sty 拉向"自己的过去均值" (照镜子整容比喻)
  → 200 round 累积自指 → z_sty 收敛到一个常量点
  → 4 client × 1 点 = z_sty rank-3 collapse
```

### 🐛 Bug 2: collapse 后 L_proto_excl 退化为噪声梯度

```
z_sty 塌 → bc_d 是常量 → Pd[k] 是常量
→ domain_axis = F.normalize(Pd + bc - bc.detach()) ≈ 固定常量向量
→ L_proto_excl = cos²(class_axis, ~常量)
→ encoder_sem 收到方向固定的"噪声梯度"
→ 训练被污染 → accuracy 退步
```

### 🐛 反 paradox: BiProto 越失败越接近 orth_only 越好

| seed | z_sty 塌缩程度 ER | accuracy |
|:-:|:-:|:-:|
| s=333 | 2.73 (最严重) | 92.46 (最高) |
| s=2 | 16.07 | 88.00 |
| s=15 | 17.06 | 86.65 |

**塌缩越彻底，L_proto_excl 越无效，越接近"纯 orth_only"，accuracy 反而最好** —— BiProto 越失败越无害。

### 🔬 评估的 7 个机制级假设

| # | 假设 | 验证结论 |
|:-:|---|---|
| A | head 路径瓶颈 | 未直接验证 |
| B | Caltech outlier | per-domain 看 BiProto Caltech 反升 +35pp（但有 BN 评估差异嫌疑） |
| C | encoder_sem capacity 饱和 | SVD ER 8.7/128 = 6.8% 已确认饱和 |
| **D** | **z_sem 失去 domain 信息伤分类** | **❌ 证伪**（z_sem ER 两边一样） |
| **E** | **Pd 自指无效 / loss 是噪声** | **✅ 证实**（SVD + paradox 双重证据） |
| F | FedAvg head 在 z_sem 越纯下被 outlier 拉 | 未直接验证 |
| G | 解耦在 D=K=4 不是真瓶颈 | 部分支持（与 EXP-110/120 evidence 一致） |

---

## 修复方向探索 (5 个候选 + 严格反事实检验)

### 5 条硬约束

| # | 硬约束 | 解释 |
|:-:|---|---|
| C1 | 不许自指 | z_sty 训练目标不能依赖 z_sty 自己的过去/均值 |
| C2 | 必须有真 negative pair | InfoNCE/contrastive 要有真"反例", 不能 batch 全同 target |
| C3 | collapse-resistant | 即便部分组件受扰, loss 必须仍能产生有意义梯度 |
| C4 | 保持 federated 语义 | Pd 必须真的是"跨 client 共享对象", 不能只是 server 缓存 |
| C5 | 不重蹈 EXP-108 CDANN 坑 | 不要用 GRL adversarial 让 z_sem 去 domain |

### 候选方案对比

| # | 方案 | C1 | C2 | C3 | C4 | C5 | 推荐 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | Dom Classifier 替代 Pd | ✅ | ✅ | ✅ | 🟡 | ✅ | ⭐⭐⭐⭐ |
| 2 | Pd 用 others' mean 更新 | ✅ | ✅ | 🟡 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| **2+** | **Fixed ETF Pd (升级版 2)** | ✅✅✅ | ✅✅ | ✅✅✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| 3 | Cross-client raw negative | ✅ | ✅✅ | ✅ | ✅ | ✅ | ⭐⭐⭐ (通信高) |
| 4 | sty_norm_reg 补丁 | ❌ | ❌ | 🟡 | ✅ | ✅ | ❌ |
| 5 | 删 Pd | ✅ | ✅ | ✅ | ❌ | ✅ | ⭐⭐ (paper 卖点死) |

### 候选 2+ Fixed ETF 严格审视

候选 2+ 设计:
- Pd 初始化为 K-vertex Fixed ETF (K=4 时正四面体顶点), **完全 freeze 不更新**
- L_sty_proto = `CrossEntropy(cos(z_sty, Pd) / τ, target=client_id)` — z_sty 拉向自己 ETF 顶点
- 自动消除 mode collapse (Pd 是常量, 无自指 loop)

**4 层严格审视后的结论**:
1. ✅ 数学上能做到 (z_sem 推到 V⊥ 子空间)
2. ❌ **几何垂直假设可能不成立** — class 跟 domain 在 visual feature 上 entangled (如 Caltech 黑白照片既是 domain 标志也是 calculator class 信号)
3. ❌ **F2DC 论文实证打脸**: domain bias 跟 class info 本质 entangled, 应该 calibrate 不是 separate
4. ❌ **已有 evidence 支持瓶颈在 capacity 不在解耦**: z_sem ER 8.7/128 = 6.8% 表示空间利用率已饱和, 推 z_sem 跟 domain 互斥不解决这个

**预期实测**: Office 大概率 89.5-90.5（接近 orth_only 90.79，不超 FDSE 90.58），赢 FDSE 概率 < 20%。

---

## 投入产出统计

| 项目 | 投入 | 产出 |
|---|:-:|---|
| Research refine 4 轮 | ~2h GPT-5.4 + 1h 写作 | 8.75/10 near-READY 方案 + 完整文档 |
| 代码实现 + 调试 | ~6h 人时 | 600 行 feddsa_biproto.py + 6 configs + 兼容现有诊断脚本 |
| EXP-126 S0 gate | 6 GPU-h | 第一次拿到 BiProto Δ=0 信号 (但被 override) |
| EXP-127 完整 R200 | 30 GPU-h | 6 runs accuracy 真实数据 + ckpt |
| 诊断 (t-SNE/probe/SVD) | 3 GPU-h | 完整诊断套件 + 定位 mode collapse root cause |
| **Total** | **~40 GPU-h + ~10h 人时** | **设计 + 代码 + 全部 evidence + 诊断 + 修复方案探索** |

---

## 当前位置 (决策点)

| 选项 | 简介 | 预期 |
|---|---|---|
| **A. 试候选 2+ Fixed ETF** | 50 行实现 + 1.5 GPU-h smoke + 严格 cutoff | 大概率持平 orth_only，赢 FDSE < 20% |
| **B. 转 F2DC / Calibrator 兜底** | calibrate not separate，50 行 corrector MLP | F2DC paper 已 SOTA，可能直接救 Office |
| **C. 攻 z_sem 维度坍缩根因** | uniformity loss / orthogonal weight reg | 理论上根因 (z_sem ER 6.8%)，但难度大 |
| **D. 写 BiProto negative result paper** | 把现有 evidence 包装成 short paper | mode collapse + SVD 诊断有独立 paper 价值 |

### 已完成产物 ✅

- 完整设计文档（refine 4 轮）
- 600 行实现 + 单元 smoke 验证
- 6 个 yml config (S0 + R200 × Office + PACS)
- 39 GPU-h Stage 0+1+2 真实数据
- 4 套诊断（t-SNE / probe / per-domain / SVD spectrum）
- mode collapse root cause + 修复方案 5 选 1 已对比

### 还差的 ⏸️

- Fixed ETF 候选 2+ 的实测验证（如果决定试）
- 关键发现备忘 N+3 / EXP-127 NOTE.md 最终回填（pending）
- Calibrator 兜底方案的实施（pending）

---

## 重要 Insights (供后续 paper 写作 / 类似工作借鉴)

### Insight 1: t-SNE 和 probe 都会被 mode collapse 骗

**视觉/数值 disentanglement 完美**:
- t-SNE z_sty by domain → 4 团 (因为 z_sty 塌成 4 点, t-SNE 当然画 4 团)
- probe z_sty→class = 0.06 (因为 z_sty 是 domain id 常量, 没 class 信号)
- probe z_sty→domain = 1.00 (z_sty = domain id 本身)

**只有 SVD effective rank 能区分** "真 disentangle" vs "塌缩成 4 点":
- 真 disentangle: ER ~ 28-30 (健康谱)
- Mode collapse: ER ~ 2.73 (实际 D-1 维)

**结论**: 任何 disentanglement 方案的 evaluation 都必须配 SVD spectrum 诊断, 不能只靠 t-SNE + probe.

### Insight 2: 联邦学习里"自指 EMA prototype"是高风险设计

任何 prototype 学习方案如果满足:
1. Prototype P[k] 用 client k 自己上传的特征 mean 做 EMA
2. Loss 把特征拉向 P[k]
3. K 个 client 各自只覆盖一个 cluster (D=K 设置)

→ **必然 mode collapse** (实测 + 理论可证).

修复必须打破至少一条:
- 改 P[k] 来源 (用 others' mean / fixed ETF / cross-client mixing)
- 改 Loss 方向 (推开而非拉向)
- 改 batch composition (每 client 多 cluster)

### Insight 3: "几何垂直 disentanglement" 在 D=K 小场景未必有用

我们假设 z_sem ⊥ z_sty 几何垂直 = 解耦. 但 evidence 表明:

1. z_sem ER 两边相同 (BiProto 没让 z_sem 退化, 也没占更多空间)
2. F2DC 论文证明 domain bias 跟 class info entangled, 几何分离会丢 class 信号
3. EXP-110/120 已证: Office 上各种解耦变体差 < 1pp

→ **解耦不是 Office 真瓶颈, 真瓶颈在 z_sem 表征空间利用率 (ER 6.8% 已饱和) + Caltech outlier**.

### Insight 4: research-refine 流程能识别 design 漏洞但识别不了实证漏洞

Refine 4 轮把方案打到 8.75/10 near-READY, R4 reviewer 明确警告 "remaining gap is empirical not design", 实测 -1.5 pp 输 FDSE 印证 reviewer 的话.

教训: design 完美 ≠ 实证有效. **C0 matched-intervention gate 设计的初衷就是 cost-effective falsification, 实测 Δ=0 应该 kill 而不是绕过**. 但 R4 reviewer 也警告过 "Negative C0 sufficient to kill", 我们没听 (用户的 B 选项决定也是合理验证).

---

## 关联文件 (索引)

### 设计文档
- `refine_logs/2026-04-24_feddsa-biproto-asym_v1/FINAL_PROPOSAL.md`
- `refine_logs/2026-04-24_feddsa-biproto-asym_v1/REVIEW_SUMMARY.md`
- `refine_logs/2026-04-24_feddsa-biproto-asym_v1/REFINEMENT_REPORT.md`
- `refine_logs/2026-04-24_feddsa-biproto-asym_v1/round-{0-3}-{proposal,review,refinement}.md`
- `refine_logs/2026-04-24_feddsa-biproto-asym_v1/round-{1-4}-review-raw.txt`

### 大白话解释 (含流程图)
- `知识笔记/大白话_FedDSA-BiProto方案.md`

### 代码
- `FDSE_CVPR25/algorithm/feddsa_biproto.py` (~600 行)
- `FDSE_CVPR25/config/{office,pacs}/feddsa_biproto_*.yml` (6 个)
- `FDSE_CVPR25/scripts/visualize_tsne.py` (改动支持 BiProto)

### 实验目录
- `experiments/ablation/EXP-126_biproto_S0_office_gate/`
- `experiments/ablation/EXP-127_biproto_full_r200/`
  - `figs/tsne_biproto_office_s333_R200.png`
  - `figs/tsne_orth_only_office_s333_R200.png`
  - `figs/svd_spectrum_biproto_vs_orth_office.png`
  - `figs/svd_spectrum_results.json`
  - `probe_results.json`

### 关键 ckpt (lab-lry)
- BiProto Office best: `~/fl_checkpoints/feddsa_s333_R200_best137_1777085380` (best=92.46)
- orth_only Office best: `~/fl_checkpoints/feddsa_s333_R200_best155_1776428179` (best=91.52)
- 其他 4 个 BiProto Office/PACS s={2,15,333} ckpt 也都已保存

---

## 一句话总结

**BiProto 方案设计层面 8.75/10 near-READY, 代码完整实施且 6 bug 修完, 但 EXP-127 实测在 Office 输 -1.75pp / PACS 输 -1.13pp; SVD 诊断揭示 root cause 是 Pd EMA 自指导致 z_sty rank-3 mode collapse 让 L_proto_excl 退化为噪声梯度, 反 paradox 现象 (越塌缩越好) 完美佐证; 
