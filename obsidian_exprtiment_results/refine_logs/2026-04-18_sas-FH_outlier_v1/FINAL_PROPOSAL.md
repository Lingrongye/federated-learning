# 研究方案：面向联邦域泛化 outlier 域恢复的风格条件化分类器路由（Style-Conditioned Classifier Routing）

## 问题锚点 (Problem Anchor)

- **核心问题**：FedDSA 方案A（sas 仅个性化 `semantic_head`，`classifier head` 与 backbone 通过 FedAvg 聚合）在 outlier 域严重失败。3-seed 证据：PACS art_painting Client 0 AVG 仅 65.69%（其他 3 域 83-89%），Office Caltech 77.68%（其他 3 域 89-100%）。70 个 art 错分样本全部 conf ≥ 0.85（自信地错），错分集中在有机体类互串（person↔dog↔horse↔guitar）。
- **必须解决的瓶颈**：当同一模型在多域共用分类器时，outlier 域的分类决策边界被其他域的语义偏好稀释。EXP-086 PACS 方案A 在 4 个域上都是 −0.65/−1.03 AVG（PACS 4 域都是彼此 outlier，共享分类器双向拖累）；EXP-084 Office 方案A Caltech +2.4/+3.6 但距 FDSE 仍 −3.9%（sem_head 已捕获部分红利，未捕获完）。
- **非目标**：
  - 不换骨干（保持 AlexNet 对齐 FDSE 基线）
  - 不引入新可训练模块或新损失项
  - 不改推理方式（TTA / temperature scaling 已被明确拒绝）
  - 不做 per-sample sas（保持 client-level sas 的通信效率）
- **约束条件**：
  - 3-seed {2, 15, 333} × R200 × E=1 对齐 FDSE 实验协议
  - SC2 单卡串行，单 run 2-4h
  - 代码改动 ≤ 50 行
  - 与 EXP-083 / EXP-084 / EXP-086 完全可比（同 config，仅 sas 聚合范围不同）
- **成功标准**：
  1. Caltech Client 0 AVG Best **≥ 77.0%**（EXP-084 为 75.0，+2%；逼近 FDSE 78.9%）
  2. 其他域 drop < 1%
  3. Office AVG Best **≥ FDSE 的 90.58**
  4. PACS 仅作为附录中的边界条件证据，不进入主路径成功判据

## 技术缺口 (Technical Gap)

方案A 的 sas 只覆盖投影头 `semantic_head`，最后的线性分类器 `head` 仍 FedAvg。`head (1024 → num_classes FC)` 是**唯一决定类间决策边界**的层。

Caltech 作为风格 outlier，收到的个性化 `sem_head` 让特征朝自己的风格方向走，**但所有 client 共享同一条分类平面** → 平面向 photo/sketch/amazon 多数派偏好倾斜。结果：**特征对了，边界错了（feature right, boundary wrong）**。

**缺失的机制**：一个决定"**哪些 client 应该共享分类边界**"的信号。FedROD (ICLR 23) 给每个 client 一个独立分类器（per-client 个性化，**均匀，无条件化**）；FedAlign (CVPR Workshop 25) 把分类器个性化与跨 client 特征对齐绑定在一起（不用风格信号）。

**核心 claim**：*风格相似度是分类器边界共享的自然路由信号；在风格驱动的 outlier 域场景下，分类器级别的个性化是最小充分的个性化粒度（minimal sufficient personalization granularity）。*

**最小充分干预**：把 `classifier.*` keys 移入 sas_keys，零新参数，约 30 行代码。

## 方法主论点 (Method Thesis)

- **一句话主论点**：**在风格异质性条件下，联邦学习在分类器层面的关键问题不是"要不要"个性化，而是"如何路由"共享；而风格相似度就是天然的路由信号。**
- **具体实现**：把 FedDSA 方案A 的风格感知聚合 (sas) 从 `semantic_head` 扩到整个 head 链（投影头 + 分类器），零新可训练模块。
- **时代契合**：FedROD / FedAlign / DualFed 已在 pFL 中确立 "per-client 分类器" 范式。**我们首次把这个个性化条件化到风格相似度上**。

## 贡献聚焦 (Contribution Focus)

- **核心贡献**：风格条件化分类器路由 (**sas-FH**)。
- **可选支撑贡献**：**无**。
- **明确声明的非贡献**：
  - 不 claim 分类器个性化本身的 novelty（FedROD 已做）
  - 不加 MixStyle / SupCon / focal loss
  - 不改 BN 策略（保持 FedBN 本地化）
  - 不做 per-sample / multi-centroid 聚合（属不同方向）

## 方法细节 (Proposed Method)

### 复杂度预算 (Complexity Budget)
- **冻结 / 复用**：方案A 全架构（AlexNet + sem_head + style_head + classifier）、style_proto 提取、FedBN BN 本地化、L_orth、sas softmax、所有超参
- **修改项**：server `pack()` 中 `classifier.weight/bias` 的聚合路径 — 从 FedAvg 改为 sas 个性化
- **新增可训练模块**：**0 个**
- **故意不加**：per-client sas_tau、style_head 个性化、骨干 sas、classifier-bias-only sas

### 系统结构图 (System Overview)

```
Client i:                                     Server:
  encoder(x) → h                              维护 client_sem_states[j]
  h → sem_head → z_sem → head → logits       维护 client_head_states[j]  [NEW]
  h → style_head → z_sty → style_proto       维护 client_style_protos[j]
  
  本地 SGD: CE + λ_orth · L_orth              对每个目标 client i 在 pack 时：
                                                w_{ij} = softmax(cos(sp_i, sp_j)/τ)
  上传: encoder, sem_head, style_head,           per_sem_i  = Σ w_{ij} · sem_state[j]
        head, style_proto                        per_head_i = Σ w_{ij} · head_state[j]  [NEW]
                                                encoder, style_head, BN → 标准 FedAvg
                                                **同时保存同轮 global_head 快照 (供 Claim 2 用)**
                                              下发个性化的 (encoder, sem_head_i, head_i) 给 client i
```

### 配置开关 (sas flag 0..4)

```
sas=0: 无个性化 (基线 B0 = EXP-083)
sas=1: 仅 sem_head sas, classifier FedAvg 按样本量加权 (B1 = 方案A = EXP-084)
sas=2: sem_head + classifier, 都风格条件化  ← 我们的方法 (A2)
sas=3: sem_head sas + classifier 均匀平均    ← 关键 counterfactual C2
sas=4: sem_head sas + classifier 完全本地    ← 次级消融 C1
```

### 聚合规则（k ∈ sas_keys, 每轮参与集合 S_t 一致）

- **A2 (sas=2)**：`per_param_i[k] = Σ_{j ∈ S_t} w_{ij} · client_state[j][k]`，其中 `w_{ij} = softmax(cos(sp_i, sp_j) / sas_tau)`
- **C2 (sas=3)**：`per_param_i[k] = (1/|S_t|) · Σ_{j ∈ S_t} client_state[j][k]` — 所有 client 收到**同一个** head
- **C1 (sas=4)**：`per_param_i[k] = client_state[i][k]` — 只取本地，不共享

**Office c4 中 proportion=1.0, 所以 S_t = {0,1,2,3} 每轮恒定。** A2 与 C2 聚合的**参与集合完全相同**，差异只在权重策略。

Client 端训练**保持不变**：CE + λ_orth · L_orth。

### 代码集成 (Integration)
- **文件**：`FDSE_CVPR25/algorithm/feddsa_scheduled.py` `pack()` 方法
- **Config**：`sas` flag 从 {0,1} 扩展到 {0,1,2,3,4}
- **代码量**：约 40 行（含 C1/C2 分支 + 单元测试 + `global_head` 快照保存）
- **Client 端**：不变

### 训练方案 (Training Plan)
- **调度**：R200, E=1, B=50, LR=0.05, decay=0.9998（对齐 FDSE）
- **Seeds**：{2, 15, 333}
- **数据**：Office-Caltech10 c4（主要 benchmark），PACS c4（仅作附录）
- **Loss**：CE + λ_orth · L_orth
- **无 warmup / curriculum / 伪标签 / 新数据增强**

### 是否使用前沿基元 (Modern Primitive Usage)
**无**。Reviewer 已确认：foundation-model 时代的组件对这个问题不是自然匹配 — 会改变论文的核心，而不是解决锚定的瓶颈。

### 失败模式与诊断 (Failure Modes)
- **失败 1**：Caltech 无提升
  - 检测：R100 时 Caltech AVG < 75.5
  - 回退：用 Claim 2 的 swap 诊断定位；若 swap 也无效应 → 分类器不是瓶颈 → 停止
- **失败 2**：训练不稳定（drop > 3%）
  - 检测：任一 seed 的 AVG Last < AVG Best − 3%
  - 回退：retry `sas_tau=0.5`

### 创新性与优雅性论证 (Novelty & Elegance)

**最接近工作**：
- FedROD (ICLR 23)：per-client 分类器 + **均匀**聚合
- FedAlign (CVPR Workshop 25)：分类器个性化 + 跨 client 特征对齐
- FedDSA 方案A (EXP-084)：sas 仅作用于 sem_head

**精确差异**：
- vs FedROD：风格相似度决定分类器共享方向，**非均匀**
- vs FedAlign：**不加** 特征对齐，纯聚合层面的改动
- vs 方案A：sas 覆盖整个 head 链，且**升级为路由问题的框架**

**论文故事（一句话）**：*"联邦域泛化中的分类器聚合是一个路由问题，不是一个尺度问题；风格相似度提供了路由。"*

## Claim 驱动的验证方案 (Claim-Driven Validation)

### Claim 1 (主 claim)：A2 > C2 证明风格路由因果性

**主表格（最终论文用）**：
| 编号 | sem_head | classifier | Seeds | 来源 |
|------|----------|-----------|-------|------|
| B0 | FedAvg | FedAvg | 3 | EXP-083 已有 |
| B1 | sas | FedAvg 按样本加权 | 3 | EXP-084 已有 |
| **A2 (我们的)** | **sas** | **sas 风格条件化** | **3 (新)** | 主目标 |
| **C2** | **sas** | **均匀平均** | **3 (新)** | **关键 counterfactual** |
| C1 (次级) | sas | 本地只 | 1 | 消融 |

**协议**：
1. **Triage（单 seed s=2，6h）**：pilot 跑 A2 + C1 + C2
2. **决策规则（无 margin 豁免）**：
   - 若 `AVG_Best(A2) − max(AVG_Best(B1), AVG_Best(C2)) ≥ 0.5`：**A2 和 C2 都升级到 3-seed {2, 15, 333}**
   - 若 `< 0.5`：停止（主论点被伪证）

**决定性对比**：
- A2 > B1：整 head sas 优于半 head sas（主效应）
- A2 > C2：**风格条件化 > 均匀条件化（核心创新点）**
- A2 > C1：分类器应该共享（次级）

**统计报告**：
- **主要**：3-seed 均值 ± 标准差 + per-seed Δ(A2 − C2) + 效应量 (effect size)
- **次要（补充）**：paired t-test，附 n=3 caveat — **不作为主显著性声明**
- **一致性要求**：A2 > C2 在**所有 3 个 seed 上**都成立，才能声明"confirmed"

**指标**：AVG Best / AVG Last / per-domain Caltech Best/Last

**预期**：A2 Caltech ≥ 77；A2 AVG ≥ 90.5（超过 FDSE 90.58）；A2 − C2 ≥ 1%。

### Claim 2 (机制诊断，同轮 swap，零训练)

加载 A2 R149 best-round checkpoint。Checkpoint 保存：
- (i) 每个 client 的个性化 `(encoder_i, sem_head_i, head_i)`
- (ii) **同轮 pack 时的 FedAvg 快照 `global_head`** — 在 R149 聚合 `head.*` 时作为标准中间量计算得出

**Per-client swap 实验**：
- 条件 (a)：`(encoder_i, sem_head_i, head_i)` 在 client i 的 test set 上
- 条件 (b)：`(encoder_i, sem_head_i, global_head)` 在同一 test set 上

**同轮保证**：两个 head 都来自 R149 — 避免训练动态漂移造成 confound。

**预测特征**：
- Caltech：(a) > (b) 差距 ≥ 2%
- Amazon / DSLR / Webcam：(a) ≈ (b)，Δ < 0.5%

**解读**：直接证明"分类器共享专门伤害 outlier 域，而且问题定位在 head 层"。

### 机制可视化（Claim 2 的补充证据）：风格相似度矩阵 + 路由权重

从 A2 R149 提取并可视化：
- 4×4 矩阵 `S_ij = cos(sp_i, sp_j)`
- 4×4 矩阵 `W_ij = softmax(S / sas_tau)`

Heatmap。

**定性预期**：Caltech 行的自我集中度高于其他行；其他行更分散。具体数值按实际观察报告，不预承诺阈值。

### Claim 3 (附录，可选)：PACS 作为边界条件

仅在 Office A2 成功时才跑 PACS A2 × 3-seed。预期 AVG ≤ EXP-086 方案A 的 79.76。

解读：PACS 4 域互为 outlier → 无 majority 参考 → routing 无意义。**仅附录**，不作协同 claim。

## 实验交接输入 (Experiment Handoff)

- **必须验证**：Claim 1 A2 > C2 ≥ 1% 在 3 seed 上 + per-seed 一致性；Claim 2 swap 特征成立。
- **必须隔离**：A2 > C2（风格条件化的因果性）。
- **关键数据集 / 指标**：Office-Caltech10 c4 的 AVG Best 与 Caltech Best/Last。
- **最高风险假设**：
  1. style_proto 足以区分 Caltech（✅ EXP-084 已验证）
  2. classifier 层是瓶颈（Claim 2 直接测试）
  3. 风格条件化 > 均匀条件化（Claim 1 A2 vs C2 直接测试）

## 计算预算与时间线

| 实验 | 运行数 | 单 run | 串行总时 |
|------|-----|--------|---------|
| Triage：Office s=2 × {A2, C1, C2} | 3 | 2h | 6h |
| **升级（若 triage 通过）：A2 + C2 × {s=15, s=333}** | 4 | 2h | 8h |
| Claim 2 swap + 机制矩阵可视化 | — | 0.5h | 0.5h |
| Claim 3 PACS A2 × 3-seed（附录可选） | 3 | 3.5h | 10.5h |
| **必跑（若 triage 通过）** | 7 | — | **~14.5h** |
| **完整（含 PACS）** | 10 | — | ~25h |

- **GPU-hours（必跑）**：~14.5
- **代码实现**：1h（含单元测试 + `global_head` 快照保存）
- **关键路径**：Triage 6h → 决策闸门 → 升级 8h → 论文撰写
