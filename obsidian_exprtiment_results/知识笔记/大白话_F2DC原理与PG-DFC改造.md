# 大白话: F2DC 每个组件干嘛 + 我们改了啥 (PG-DFC v3.2)

## 一句话总结

> **F2DC**: 把图像特征**逐像素**切成"该类的部分 (robust)"跟"无关的部分 (non-robust)", 用 conv 把无关那块**重建**回去, 让模型学到"什么是真信号什么是噪声".
> **我们的 PG-DFC**: 在 F2DC 重建那一步**加了 prototype 提示** — server 收所有 client 的"类原型"做平均下发回 client, 让 client 重建时能参考"全班对这个类的共识答案".

---

## 第一部分: F2DC 在干啥 (大白话)

### 一句话比喻

把一张狗的照片当成 100 个像素拼图. F2DC 干的事:

1. **DFD (切分员)**: 用一个小神经网络看每个拼块, 给每块判一个 0-1 mask: "这块跟'狗'真有关吗?". 有关 = robust 块, 无关 = non-robust 块.
2. **DFC (修补员)**: 把 non-robust 那部分用 conv **重新画一遍**, 让重建出来的也带"狗"的特征 (本来 non-robust 那部分丢了, 现在让它也能贡献分类信息).
3. **3 个分类器同时打分**: 用 robust 块预测对 ✓, 用 non-robust 块**预测错** ✓ (这是反向信号, 让 mask 学准), 用重建块预测对 ✓.
4. **角度损失**: 让 robust 跟 non-robust 在 feature 空间分得开 (cosine 距离大).

→ 训练 100 round, 模型学会 "看狗的本质特征" + "把 non-robust 部分救回来".

---

## 第二部分: F2DC 整体流程 (一图看懂)

```
                  输入图片 (狗.jpg, 224×224)
                         ↓
                   [ResNet conv1-4]
                         ↓
                  feature map out (512, 28, 28)
                         ↓
              ┌─── DFD 切分员 ────┐
              │  小 conv → mask   │
              │  mask ∈ [0,1]     │  (Gumbel-sigmoid 把硬分类弄成可微)
              └───┬─────────┬─────┘
                  ↓         ↓
         r_feat = out·mask    nr_feat = out·(1-mask)
         (狗的本质部分)         (无关/背景部分)
              │                     │
              │                     ↓
              │              [DFC 修补员]
              │              小 conv 重建 → rec_units
              │              rec_feat = nr_feat + rec_units·(1-mask)
              │                     │
              │                     ↓ (修过的非robust部分也带类信息)
              │             ┌──────┴───────┐
              │             ↓              ↓
              │       aux分类(rec)      ─── 用于 DFC_loss
              │                       
              ├──────────────┬─────────┐
              │              │         │
        aux分类(r_feat) aux分类(nr_feat) cosine(r_feat, nr_feat)
              │              │         │
              ↓              ↓         ↓
         DFD_dis1_loss  DFD_dis2_loss  DFD_sep_loss
         (用 robust       (用无关部分     (让两边
         分类正确)         应该分类错)     在向量空间分开)

最终特征 = r_feat + rec_feat
        ↓
    AvgPool → linear → out (logits)
        ↓
    主分类 CE_loss
```

→ **5 个 loss 同时训练**: CE + DFD_dis1 + DFD_dis2 + DFD_sep + DFC

---

## 第三部分: F2DC 4 大组件细节

### 🅰 DFD (Domain Feature Decoupler) — 切分员

**代码**: `backbone/ResNet_DC.py:7-29` (class DFD)

**做什么**:
1. ResNet 跑完 conv 后得到 `out` (B, 512, H, W) — 每个像素 512 维
为什么
2. DFD 是个 4 层小 conv (`conv → bn → relu → conv`), 跑完后输出一个 logit map
3. 过 sigmoid + Gumbel-sigmoid → 得到 mask, **每个像素一个 [0,1] 值**, 表示"这个像素跟该类相关吗"
4. 输出:
   - `r_feat = out × mask` (该类相关那部分)
   - `nr_feat = out × (1-mask)` (无关那部分)

**为什么用 Gumbel-sigmoid 而不是普通 sigmoid**:
- 我们希望 mask 接近 **离散** 0 或 1 (要么是 robust 要么不是, 不要灰色), 但又要能反向传梯度
- Gumbel-sigmoid 加随机噪声让训练时 mask 接近离散, 测试时 (`is_eval=True`) 直接取 0.5 噪声 → 平滑过渡
- 实现: `gumbel_sigmoid.py:5-36`

**输出**: `r_feat`, `nr_feat`, `mask` 三个 (B, 512, H, W)

---

### 🅱 DFC (Domain Feature Compensator) — 修补员

**代码**: `backbone/ResNet_DC.py:32-47` (class DFC)

**做什么**:
1. 拿 `nr_feat` (无关那部分) 喂给一个 4 层小 conv (跟 DFD 同结构)
2. 输出 `rec_units` — "重建出来的非robust部分该长啥样"
3. 最后:
   ```python
   rec_feat = nr_feat + rec_units × (1 - mask)
   ```
   注意: `rec_units` 只在 `mask=0` 的位置加进去 (即 non-robust 区), robust 区原样保留

**直觉理解**:
- 一张狗照片中, 背景 (non-robust 部分) 本来就不该贡献类信息
- 但 DFC 强行让 conv 学习"如果这部分也要预测狗, 该怎么画?"
- → 强迫模型学到 **"狗"的特征如何分布在整张图**, 不只是脸部
- → 数据增强 + 正则化效果

**关键约束**: `rec_units * (1-mask)` 限制只在 nr 区 修. 这样不破坏 robust 区, 只补 nr 区.

---

### 🅲 双分类器 + 5 个 loss

**代码**: `f2dc.py:85-114`

**两个分类器**:
- `self.aux = Linear(512 → num_classes)` — auxiliary 分类器, 给 r_feat / nr_feat / rec_feat 分别打分
- `self.linear` — 主分类器, 给 r_feat + rec_feat 的最终融合特征打分

**5 个 loss 干啥**:

| Loss | 公式 | 目的 |
|---|---|---|
| **CE_loss** | `criterion(out, labels)` | 主分类要对 |
| **DFD_dis1** | `criterion(aux(r_flat), labels)` | r_feat 应能正确分类 (验证 mask 切对了) |
| **DFD_dis2** | `criterion(aux(nr_flat), wrong_labels)` ⚠️ | nr_feat **应该预测成第二高的错误标签** (反向监督, 让 mask 把无关信号过滤干净) |
| **DFD_sep** | `cosine(r_flat, nr_flat) / tem` | 让 r_feat 跟 nr_feat 在 feature 空间**距离大** (角度分得开) |
| **DFC_loss** | `criterion(aux(rec_flat), labels)` | 重建后的 rec_feat **应能正确分类** (验证 DFC 修对了) |

**总损失**:
```
loss = CE + λ1 × (DFD_dis1 + DFD_dis2 + DFD_sep) + λ2 × DFC
```
默认 λ1 = λ2 = 1.0.

**`wrong_high_label` 是啥** (`f2dc.py:16-20`):
```python
def get_pred(out, labels):
    pred = out.argmax  # 预测 top-1
    second_pred = out.argsort top-2
    wrong_high_label = pred 跟真实 label 一样 ? second_pred : pred
```
即: 如果模型预测对了, 就用 top-2 当"错误标签"; 如果预测错了, 就用 top-1 (反正错的). **DFD_dis2 让 nr_feat 学到 "如果只看无关部分, 应该预测错"** — 这强迫 mask 把类信号都路由到 r_feat 里.

---

### 🅳 训练时序 (一个 round 的流程)

**代码**: `f2dc.py:38-53` (loc_update) + `models/utils/federated_model.py` (aggregate_nets)

```
[Round 开始]
  ↓
server: 选 N 个 online_clients
  ↓
每个 client (本地训练):
  ├─ 10 个 local_epoch:
  │  └─ 每个 batch:
  │     ├─ forward: out, r_flat, nr_flat, rec_outputs ← ResNet+DFD+DFC
  │     ├─ loss = CE + DFD + DFC (5 个加权和)
  │     ├─ backward + optimizer.step
  │     └─ ResNet/DFD/DFC weights 都被更新
  │
  └─ 训练完, weights 上传 server
  ↓
server: aggregate_nets()
  ├─ 按 sample_share 加权平均所有 client 的 conv/DFD/DFC 参数
  └─ (BN running stats 也聚合 — 跟 FedBN 不同)
  ↓
下发回每个 client
  ↓
[下个 round]
```

→ **这是 vanilla F2DC**, 没用 prototype, 没区分 BN.

---

## 第四部分: 我们的改造 (PG-DFC v3.2)

### 一句话总结改动

> **F2DC 的 DFC 修补员只用 `nr_feat` 自己的信息重建 (闭门造车). 我们让 DFC 修补员同时参考 server 下发的"班级标准答案 (class prototype)" — 通过 cross-attention 注入.**

---

### 改动 1: DFC → DFC_PG (加 cross-attention)

**代码**: `backbone/ResNet_DC_PG.py:26-153` (class DFC_PG)

**新增**:
```python
self.q_proj = nn.Linear(C, C)    # query 投影
self.k_proj = nn.Linear(C, C)    # key 投影
self.v_proj = nn.Linear(C, C)    # value 投影
self.register_buffer('class_proto', torch.zeros(num_classes, C))
                                  # ← server 下发的全局原型, 每类一个 512 维
```

**新 forward (line 84-153)**:

```
nr_feat (B, 512, H, W)
   ↓
路径 1 (F2DC 原版): 用 conv 重建 → rec_units

路径 2 (PG-DFC 新增): cross-attention
   ├─ q = q_proj(pool(r_feat))         # 用 robust 部分当 query
   ├─ k = k_proj(class_proto)          # 全局原型当 key
   ├─ v = v_proj(class_proto)          # 全局原型当 value
   ├─ attn = softmax(q · k / temp)     # 当前样本跟 7 个类原型谁最像
   └─ proto_clue = attn @ v            # 加权和 → "对当前样本的类提示"
   
最终: rec_feat = nr_feat + (1-mask)·rec_units + (1-mask)·proto_weight·proto_clue
                  ↑F2DC 原版不变↑      ↑新增: prototype 提示↑
```

**直觉**: 
- F2DC 原版只让 conv "凭空想象"该类长啥样
- PG-DFC 多一份"班级标准答案"参考, 帮 conv 知道"狗的标准特征 = 这个 512 维方向"

**关键超参**:
- `proto_weight = 0.3` (注入强度, 不能太大, 否则盖过 conv 重建)
- `attn_temperature = 0.5` (注意力温度, 越小注意力越尖锐)

**warmup**: 前 30 round `proto_weight = 0` (因为 server 还没 prototype), 30-50 round 线性 ramp 到 0.3.

---

### 改动 2: client 上传类原型 (新流程)

**代码**: `models/f2dc_pg.py:195-216`

每个 client 训练时, 在每个 batch 累加 `r_flat` (DFD 切出的 robust 部分):
```python
sum_feat_round.index_add_(0, labels, ro_flatten.detach())
count_round += bincount(labels)
```

训练完一个 round, 算每类的 mean:
```python
local_proto[c] = sum_feat_round[c] / count_round[c]   # (num_classes, 512)
self.local_protos[index] = local_proto.cpu()           # 上传
```

→ 每个 client 训练完一轮, 多上传一个 `(num_classes, 512)` 的原型矩阵 (除了模型参数).

---

### 改动 3: server 聚合原型 (raw 空间 EMA)

**代码**: `models/f2dc_pg.py:221-292` (aggregate_protos_v3)

```
1. 每类分别处理:
   - 收集所有 valid client 的 local_proto[c]
   - L2-normalize 每个 (变 unit 球面方向)
   - 等权平均 (raw 空间)
   - new_aggregated[c] = mean(normed_protos)

2. server EMA:
   global_proto_raw = β·global_proto_raw + (1-β)·new_aggregated
   (β = 0.8, 跨 round 平滑)

3. unit normalize:
   global_proto_unit = global_proto_raw / norm  ← 下发用
```

**为啥 EMA**: 防止某 round 某 client 的 proto 抖动太大影响全局.
**为啥 raw 空间 EMA**: 在 unit 球面 EMA 数学上不一致 (凸组合不在球面上).

---

### 改动 4: server 下发 prototype + DaA 聚合权重

**代码**: `models/f2dc_pg.py:78-101`

```python
for net in self.nets_list:
    if hasattr(net, 'dfc_module') and hasattr(net.dfc_module, 'set_proto_weight'):
        net.dfc_module.set_proto_weight(current_pw)   # warmup ramp
        if self.global_proto_unit is not None:
            net.dfc_module.class_proto.copy_(self.global_proto_unit.to(self.device))
            ↑ 把 prototype 塞进 client model 的 buffer ↑
```

**DaA (Domain-Aware Aggregation, Eq 10/11)**:
- 默认: `freq[i] = sample_share[i]` (FedAvg)
- DaA on: `freq[i] = sigmoid(α·sample_share - β·d_k) / sum`
  - `d_k = sqrt(C/2) × |n_k/N - 1/Q|` 衡量 client 偏离平均的程度
  - 偏离大的 client 适度升权 → 罕见 domain 不被淹没
- 实现: `models/utils/federated_model.py:_compute_daa_freq`

---

## 第五部分: PG-DFC vs F2DC 一图对比

```
F2DC: DFC 修补员闭门造车 (只用 nr_feat 重建)
   nr_feat (无关部分) ──→ [conv 重建] ──→ rec_units ──→ rec_feat

PG-DFC: DFC 修补员开卷考试 (参考全班共识)
   nr_feat ─────────────→ [conv 重建] ──→ rec_units ─┐
                                                      │
   r_feat 池化 ─→ q ─┐                               ├→ rec_feat
                     │                                │
   class_proto ─→ k,v┘                                │
   (server 下发)      cross-attn → proto_clue ─────────┘
```

→ **PG-DFC 改动量极小** (只在 DFC 加 attention 路径), 但训练时序多了 2 步:
1. client 多上传一个 `(C, 512)` proto 矩阵
2. server 多算一次 prototype 聚合 + EMA + 下发

---

## 第六部分: 改动的诊断意义 (回到诊断指标话题)

### 我们怎么知道 PG-DFC 起作用了?

**诊断 1: prototype 收敛吗** (`pg_proto_norm_mean` 监控)
- 如果 `global_proto_unit` 的 norm 一直 0 → prototype 路径没生效, 等价 F2DC
- 如果 norm 在 0.5-1.0 收敛 → 正常工作

**诊断 2: cross-attention 真的在传信号吗** (`proto_signal_ratio` 监控)
- `proto_signal_ratio = proto_clue.abs().mean() / rec_units.abs().mean()`
- 太小 (<0.05) → prototype 信号被 conv 重建淹没, 等价 F2DC
- 太大 (>1.0) → prototype 主导, 可能压制 client 个性
- 我们设 `proto_weight=0.3`, 期望 ratio 在 0.1-0.3

**诊断 3: client 是不是被 prototype 同化** (cos similarity, 之前的争论)
- ⚠️ **重要纠正**: 训练时累加的 `local_proto` cos sim 0.97 是 **DFD 切出 robust 部分共识** + **class supervision** 的副作用, 不代表 client feature 被 prototype 拉过去 (PG-DFC 只在 DFC 修补员里用 prototype, 没在 r_feat 上拉)
- 用 final test set feature 反推的 cos sim 是 0.80, 跟 vanilla F2DC 一样 → **真实 client 多样性没被 PG-DFC 减弱**

**诊断 4: PG-DFC 比 F2DC 强了多少** (主表 acc)
- Office: PG-DFC+DaA 60.65 last vs F2DC+DaA 59.90 last (+0.75pp 微胜)
- PACS s=15: PG-DFC+DaA 71.75 best vs F2DC+DaA 75.01 best (-3.26pp 反输 ⚠️)
- → **PG-DFC 在 office 微赢, 在 PACS 输** — 说明 prototype guidance 跟数据集的"类内方差/domain 异质性"有交互

---

## 一句话记住

| 谁 | 干啥 | 怎么实现 |
|---|---|---|
| **DFD** | 切 robust / non-robust | 4 层 conv + Gumbel-sigmoid mask |
| **DFC** | 重建 non-robust 部分 | 4 层 conv 凭空想象 |
| **DFC_PG (我们改)** | 重建 + 参考全班标准答案 | DFC + cross-attention(class_proto) |
| **prototype 上传** | client 累加每类 r_flat 平均 | sum_feat / count → local_proto |
| **prototype 聚合** | server L2-norm 等权 + EMA | beta=0.8 raw 空间 |
| **DaA** | 罕见 domain 升权 | sigmoid(α·share - β·deviation) |

---

## 后续诊断关注点

1. ✅ Office 已验证: PG-DFC+DaA 微胜 (+0.75pp last)
2. ⏳ PACS 等 R100 完成: 已知 s=15 PG-DFC+DaA 反输 F2DC+DaA, 等 s=333 验证
3. ⏳ 等 cold path 出图后写"为啥 PACS 上 PG-DFC 反输"分析

---

*本笔记生成日期: 2026-04-29*
*参考: 大白话_FedDPG方案.md 风格, F2DC 论文, PG-DFC v3.2 实现*
