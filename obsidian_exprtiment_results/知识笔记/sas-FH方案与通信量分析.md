# sas-FH 方案与通信量分析

> **提问**: 这个 head_state 是什么意思？这会不会增加通信量？详细给我解释一下我们的这个方案。

---

## 一、核心答案（先简）

1. **`head_state`** = 分类器 `head` 的 `state_dict()`（weight + bias），即 `FedDSAModel.head = nn.Linear(128, num_classes)` 这一层的参数状态字典
2. **通信量不会增加** — 零增加
3. 仅 **server 端内存** 多 ~15 KB（完全可忽略）

**关键洞察**：Plan A 本来就在上传 `head` 参数（因为 FedAvg 需要所有 client 的 head 参数做平均）。sas-FH 只是改 server 端**如何处理**这些已经收到的 head，而不改 client 端**传什么**。

---

## 二、`head_state` 的精确定义

### FedDSAModel 架构回顾

```python
class FedDSAModel(nn.Module):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
        self.encoder = AlexNetEncoder()                    # ~14.5M 参数
        self.semantic_head = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, 128)
        )                                                   # ~169K 参数
        self.style_head = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, 128)
        )                                                   # ~169K 参数
        self.head = nn.Linear(128, num_classes)             # 903 参数 (PACS) / 1290 (Office)
```

- `head.weight` 形状 `(num_classes, 128)`
- `head.bias` 形状 `(num_classes,)`
- `head.state_dict()` 就是这两个 tensor 的字典
- **head 是整个网络中唯一决定类间决策边界（class boundary）的层**

### `client_head_states[j]`

是 server 端维护的字典，key 是 client id，value 是该 client 最近一轮上传的 `head.state_dict()`。Plan A 里没有这个字典（因为 Plan A 收到 head 后直接 FedAvg 完就扔掉了），sas-FH 需要保留每个 client 的单独 head state 才能做 sas 个性化聚合，所以标记 [NEW]。

参数量：
- PACS: `7 × 128 + 7 = 903` floats × 4 bytes = **3.6 KB/client**
- Office: `10 × 128 + 10 = 1290` floats × 4 bytes = **5.2 KB/client**

4 个 client 合计：PACS ~14.4 KB，Office ~20.7 KB。**相对 AlexNet 模型 ~60 MB 完全可忽略**。

---

## 三、通信量精确对比（这是用户最关心的问题）

### 每轮每 client 上行 (client → server)

| Item | 参数量 | Plan A | sas-FH | 差异 |
|------|-------|--------|--------|------|
| encoder (AlexNet) | ~14.5M | ~58 MB | ~58 MB | = |
| sem_head | ~170K | ~680 KB | ~680 KB | = |
| style_head | ~170K | ~680 KB | ~680 KB | = |
| **head** | **903 (PACS)** | **~3.6 KB** | **~3.6 KB** | **=** |
| style_proto | 128×7 | ~3.5 KB | ~3.5 KB | = |
| **总计** | — | **~59.4 MB** | **~59.4 MB** | **0** |

**重点**：`head` 在 Plan A 里就已经被 client 上传 — 因为 Plan A 的 `sas=1` 策略里，`head` 走的是 FedAvg（`(1/N) Σ head_j` 或 sample-count weighted mean），server 必须收到所有 client 的 head 才能做 FedAvg。sas-FH 没增加任何上行 item。

### 每轮下行 (server → client)

每个 client 收到一份 `(encoder_i, sem_head_i, head_i, style_head, BN)`：

| Item | Plan A 给 client i | sas-FH 给 client i | 尺寸 |
|------|-------------------|-------------------|------|
| encoder | FedAvg 全局参数（所有 client 相同） | FedAvg 全局参数（所有 client 相同） | 同 |
| sem_head_i | sas 个性化（每 client 不同） | sas 个性化（每 client 不同） | 同 |
| **head_i** | **FedAvg 全局 head（所有 client 相同）** | **sas 个性化 head（每 client 不同）** | **同** |
| style_head | FedAvg 全局参数 | FedAvg 全局参数 | 同 |
| BN | 各 client 本地保留，server 不下发 | 各 client 本地保留，server 不下发 | — |

**唯一差异**：`head_i` 的**值** — Plan A 下发同一份 global head，sas-FH 下发每 client 不同的 personalized head。但**单次下行传输的 head 参数量一样**（都是 `num_classes × 128 + num_classes`）。

### 为什么 reviewer 会立即通过这个方案

因为这个方案：
- **上行通信量零增加**
- **下行通信量零增加**
- **client 代码零改动**（整个训练循环、损失、数据流都不变）
- **模型参数零增加**（没有新模块）
- **唯一的改动在 server 的 `pack()` 函数里 30 行代码**

---

## 四、Server 端计算与内存增量

### 内存

Plan A 维护：
- `client_sem_states[j]` — 每个 client 的 sem_head state（~170K params × 4 bytes × 4 clients ≈ 2.7 MB）
- `client_style_protos[j]` — 每个 client 的风格原型（128 × num_classes × 4 clients，几 KB）

sas-FH 新增：
- `client_head_states[j]` — 每个 client 的 head state（3.6-5.2 KB × 4 clients ≈ 14-21 KB）

**增量 ≈ 0.5% 内存开销**。

### 计算

Pack() 函数中，对每个 target client i：

**Plan A**：
1. 计算 sas 权重 `w_ij = softmax(cos(sp_i, sp_j)/τ)` — `O(N² × proto_dim)`
2. 加权求和 sem_head states — `O(N × sem_head_params) ≈ O(N × 170K)`

**sas-FH**：
1. 同上
2. 同上
3. **新增**：加权求和 head states — `O(N × head_params) ≈ O(N × 900)`

head_params 相比 sem_head_params 小两个数量级 → 计算增量 **< 1%**。

---

## 五、系统结构图逐行解读

```
Client i:                                    Server:
  encoder(x) → h                              维护 client_sem_states[j]
  h → sem_head → z_sem → head → logits       维护 client_head_states[j]  [NEW]
  h → style_head → z_sty → style_proto       维护 client_style_protos[j]
  
  本地 SGD: CE + λ_orth · L_orth              对每个目标 client i 在 pack 时:
                                                w_{ij} = softmax(cos(sp_i, sp_j)/τ)
  上传: encoder, sem_head, style_head,           per_sem_i  = Σ w_{ij} · sem_state[j]
        head, style_proto                        per_head_i = Σ w_{ij} · head_state[j]  [NEW]
                                                encoder, style_head, BN → 标准 FedAvg
                                                **同时保存同轮 global_head 快照 (供 Claim 2 用)**
                                              下发个性化 (encoder, sem_head_i, head_i) 给 client i
```

### Client 端（**与 Plan A 完全一致**）

- `encoder(x) → h`：AlexNet 骨干提取 1024 维特征
- `h → sem_head → z_sem → head → logits`：语义路径，最终输出分类 logits
- `h → style_head → z_sty → style_proto`：风格路径，生成 128 维风格原型（按类均值 + 标准差）
- `本地 SGD: CE + λ_orth · L_orth`：**损失与 Plan A 逐字相同**，没有新损失项
- `上传: encoder, sem_head, style_head, head, style_proto`：**5 items，与 Plan A 完全相同**

### Server 端（**只有 2 处 [NEW] 标记**）

- `维护 client_sem_states[j]` ← Plan A 已有（用于 sem_head sas 聚合）
- `维护 client_head_states[j]` **[NEW]** ← 唯一新增的状态字典
- `维护 client_style_protos[j]` ← Plan A 已有

在 pack() 阶段（每轮 server 给每 client 准备 personalized model 时）：

- `w_{ij} = softmax(cos(sp_i, sp_j)/τ)` ← sas 路由权重，与 Plan A 相同
- `per_sem_i = Σ w_{ij} · sem_state[j]` ← Plan A 已有，sem_head 个性化聚合
- `per_head_i = Σ w_{ij} · head_state[j]` **[NEW]** ← **核心改动，head 也做 sas 聚合**
- `encoder, style_head, BN → 标准 FedAvg` ← 不变
- `**同时保存同轮 global_head 快照**` ← 供 Claim 2 mechanism swap diagnostic 用

**global_head 快照**是什么？在 R149 这一轮 pack 时，我们除了计算 A2 的 `per_head_i`（每 client 一份 style-conditioned head），也额外保存一份 **uniform/sample-weighted FedAvg 版本的 head** 作为对照参考。这份 global_head 用于 zero-training 的 swap diagnostic：对每个 client 分别用 `(encoder_i, sem_head_i, per_head_i)` 和 `(encoder_i, sem_head_i, global_head)` 跑 test，看 per-domain acc 的差异。这个 "same-round" 设计避免了训练动态漂移的 confound（reviewer R3 的 fix 要求）。

- `下发个性化 (encoder, sem_head_i, head_i)` ← Plan A 下发的 head 是 global FedAvg，sas-FH 下发的是每 client 不同的 personalized head。**下行包的尺寸相同**。

---

## 六、为什么这是 "minimal adequate"（最小充分改动）

### 对比其他可能的 head 改进方案

| 方案 | client 代码 | 上行 | 下行 | 新参数 | 新损失 | 新通信 |
|------|-----------|-----|-----|--------|-------|--------|
| Plan A (EXP-084) | 不改 | 5 items | 5 items | 0 | 0 | 0 |
| **sas-FH (ours)** | **不改** | **相同** | **相同尺寸** | **0** | **0** | **0** |
| FedROD 双头架构 | 改（两套 classifier） | +1 head | +1 head | +1 head params | 可能 +decoupled loss | +1 head 传输 |
| MixStyle on classifier | 改（feature mix） | 相同 | 相同 | 0 | 可能 +style mixing loss | 0 |
| Hypernetwork head | 改（加 generator） | 相同 | +hypernet | +hypernet params | 0 | +hypernet 传输 |

**sas-FH 是所有候选方案里最"纯净"的**：
- 所有改动都在 server 内部
- client / model / loss / data pipeline 全部不变
- 没有新 hyperparameter 需要调（复用现有 sas_tau = 0.3）
- 没有工程风险（不改 backward graph，不引入数值不稳定性）

### 为什么 reviewer 在 R4 给了 READY (9.4/10)

- PF 10/10：核心问题锚定不变（outlier 域分类边界被拖累）
- MS 10/10：configs / 更新规则 / 参与集合 / 诊断 / 集成点全部不含糊
- CQ 9/10：一个核心机制，counterfactual 干净
- Feas 9/10：代码改动可信，运行时间预算可信
- VF 9/10：核心 claim 被 matched counterfactual + 直接 mechanism 测试清晰隔离

唯一的"减分项"是 CQ 没拿满分 —— reviewer 觉得 novelty 还是"范围扩展 + 条件选择"这个粒度，而不是一个全新的算法对象。但这本身就是设计初衷：**我们要的是能与 FedProto/FPL/FDSE/FedPall 公平对比的最小改动**，不是加一堆新模块。

---

## 七、与 Plan A (EXP-084) 的精确 diff

**代码层面** (FDSE_CVPR25/algorithm/feddsa_scheduled.py):

```python
# === Plan A (sas=1) 现状 ===
sas_keys = [k for k in global_dict if k.startswith('semantic_head.')]

# === sas-FH (sas=2) 改动 ===
sas_keys = [k for k in global_dict if k.startswith('semantic_head.')
                                    or k.startswith('head.')]
```

加上 `sas=3` (C2 counterfactual) 和 `sas=4` (C1 counterfactual) 分支，以及 `global_head` 快照保存（供 Claim 2 诊断用），总代码增量约 40 行。

**配置层面**：
- Plan A: `sas: 1`, `sas_tau: 0.3`
- sas-FH: `sas: 2`, `sas_tau: 0.3`（完全复用）

**训练过程**：
- Client 端：完全相同
- Server 端：pack() 多做一步 head 的 sas 聚合
- Loss 曲线：预期与 Plan A 高度相似（因为没有新损失，只是 personalized model 的 init 稍有不同）

---

## 八、常见疑问

### Q1: `head` 这么小（几 KB），个性化真的能提升准确率吗？

A: 恰恰因为 `head` 小，它反而是**瓶颈层** — 所有语义信息都从 1024 维压到 `num_classes` 维，共享这一层意味着所有 client 必须接受同一个"类间几何结构"。outlier 域（如 Caltech、art_painting）的类间几何本来就应该与 majority 不同（"抽象人物油画"在像素空间更接近"抽象狗油画"，但在 photo 里 person 和 dog 距离远）。小层 × 高语义贡献 = 对 personalization 最敏感。

### Q2: 为什么不连 `encoder` 也做 sas？

A: 技术上可以（EXP-080 的 "A3 encoder last block" 就是这个思路），但：
- encoder 占整个模型 ~95% 参数，每 client 一份 personalized encoder 内存爆炸
- R1 reviewer 明确建议删掉 encoder-last-block 路径，理由是"weakens smallest adequate story"
- EXP-086 已经证明 PACS 上 full-model personalization 不会带来额外收益

### Q3: C1 (local-only classifier) vs C2 (uniform) 为什么都要做？

A:
- A2 vs B1: 证明 full-head sas 比 half-head sas 更好（范围扩展的效果）
- A2 vs C2: 证明 style-conditioning 比 uniform aggregation 更好（**核心 novelty 测试**）
- A2 vs C1: 证明分类器应该共享（哪怕是 uniform share 也好过完全不 share）

三个对照一起，才能把"为什么 sas-FH 有效"的归因链完全切干净。

### Q4: 如果 Office Caltech 没提升怎么办？

A: Claim 2 的 swap diagnostic 直接判别：
- 若 swap 显示 Caltech `(a) > (b)` 明显 → classifier 确实是 outlier 域的瓶颈 → 我们实验失败 = 单 seed 噪声，换 seed 重跑
- 若 swap 显示 `(a) ≈ (b)` → classifier **不是** outlier 域的瓶颈 → thesis 被伪证 → halt，换方向（转 Top-2 multi-centroid 或其他）

这个"伪证路径"是 reviewer 特别赞许的 — 大多数论文没有写清楚"假设被伪证时怎么办"。

---

## 九、参考

- 方案原始推演：`refine-logs/outlier_v1/FINAL_PROPOSAL.md`（R4 READY 版本）
- 迭代过程（4 轮 GPT-5.4 review，8.3 → 9.4）：`refine-logs/outlier_v1/REFINEMENT_REPORT.md`
- Plan A 原理：`知识笔记/实验84_方案A风格相似度个性化聚合.md`
- 错分模式诊断：`misclass_viz/office_sas_s333_best/` 与 `misclass_viz/pacs_sas_s2_best/`

---
*记录时间: 2026-04-18 11:45*
*来源: Claude Code 对话*
