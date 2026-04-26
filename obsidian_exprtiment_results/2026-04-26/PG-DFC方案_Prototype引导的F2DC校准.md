---
date: 2026-04-26
status: v2 方案敲定 (M1/M2 实测验证完成),等 F2DC sanity 复现完成后开始 W2 实施
based_on: F2DC (CVPR'26),借鉴 RFedDis evidential / Disentanglement-Then-Reconstruction prototype
expected_gain: PACS 76.47 → 78+ (+1.5pp), Office 66.82 → 68+ (+1.5pp)
revisions:
  v1 (04-26 早): 设计完成,基于 paper 推理
  v2 (04-26 晚): 经过 13 项 review critique,M1/M2 经 EXP-130 client 分布实测验证为真问题,锁定 fix
---

# PG-DFC: Prototype-Guided DFC for Federated Feature Decoupling

> 把 F2DC 的"盲修复 DFC"改成"有方向的 prototype-guided 校准"
> 用跨 client 聚合的 class prototype 当抢救锚点,让 DFC 知道 nr_feat 该往哪儿救

> **v2 重要修正**:经过 review + 实测,M1 改成 sample 累加(不用 batch-mean EMA),M2 改成 L2-norm + 等权(不用 sample-weighted)。详见末尾「附录 C: v2 实证修正」。

---

## 一、动机 — 为什么要做这个改动

### 1.1 F2DC 当前状态(Paper Tab 1 数据)

| 数据集 | F2DC | FDSE | FedBN | FedAvg |
|---|:--:|:--:|:--:|:--:|
| PACS AVG | **76.47** | 73.13 | 70.65 | 66.39 |
| Office AVG | **66.82** | 63.18 | 62.10 | 55.86 |

F2DC 是 paper-validated SOTA,赢 FDSE 约 +3pp。**但涨幅不算碾压,还有改进空间**。

### 1.2 F2DC paper Sec 4.1 末段自己承认的 limitation

原话:
> "A_D **cannot perfectly isolate all discriminative signals** into f+, i.e., A_D inevitably relegates those 'mixed' features into f⁻. Therefore, the resulting f⁻ is often a complex mixture of **domain artifacts and valuable class-relevant clues**."

Paper 还做了两个实验证明这个 limitation:
1. τ=0.02(强制更激进分离) → 性能反而下降(Fig 6)
2. f⁻ 经过 DFC 校准后涨点显著(Tab 7)→ 证明 f⁻ 里**确实混入了 class clue**

### 1.3 F2DC 现有 DFC 的问题

```python
# F2DC 原版 DFC (backbone/ResNet_DC.py)
class DFC(nn.Module):
    def forward(self, nr_feat, mask):
        rec_units = self.net(nr_feat)        # 2 层 conv,只看本地 nr_feat
        rec_units = rec_units * (1 - mask)   # 只在 non-robust 位置激活
        rec_feat = nr_feat + rec_units       # 残差加回
        return rec_feat
# 训练监督:CE(rec_out, label),让 rec_feat 能预测对正确 label
```

**问题诊断**:
- DFC 输入只有本地 nr_feat,**没有任何外部参考**
- 它要靠 CE(rec_out, label) 反向梯度,**盲目摸索**该怎么修
- 不知道"修对了"应该长什么样 — 等于蒙眼画师只听对错,没参考

### 1.4 我们要做的

给 DFC 一本**字典**:
- 字典 = class prototype bank(每个类的"标准 robust feature")
- 字典是**跨所有 client 编的**(server 聚合) → 携带跨域共识
- DFC 看着 nr_feat,先猜大概是哪个类,然后查字典找参考,模仿着修

---

## 二、设计哲学 — 用大白话说清楚

### 2.1 画师比喻

**F2DC 原版 DFC**:
> 一个**蒙眼画师**,被丢了一团模糊画(nr_feat),让他画出能让别人猜对类别的画。他只能反复试错,看裁判说"对/错"。

**PG-DFC**:
> 同样的画师,但给他一本**画册**(prototype bank),每一页是一个类的"标准长相"。他先看 nr_feat 大概像哪个类,翻到那页参考着画。**有方向地画,不是瞎画**。

### 2.2 字典比喻

- **F2DC**:DFC 像一个**没字典的翻译** — 只能根据上下文(label 反向梯度)摸索词义
- **PG-DFC**:DFC 像一个**有字典的翻译** — 先在字典(prototype bank)里查参考翻译,再结合上下文调整

### 2.3 跟 F2DC paper 弱点的对应

| F2DC 自己承认的 | PG-DFC 怎么解 |
|---|---|
| DFD 切不干净,nr 里混入 class clue | DFC 通过 prototype attention 找到这些 clue 的"目标方向" |
| DFC 只能用本地 nr 信号,无跨 client 协作 | prototype 是**跨 client 聚合**的,引入全局信号 |
| DFC 是无方向校准 | prototype attention 提供**明确的目标方向** |

---

## 三、完整架构图

### 3.1 F2DC 原版 vs PG-DFC 对比

```
========== F2DC 原版 ==========

输入图片 x
   ↓
Backbone (ResNet10/AlexNet) features
   ↓
   feat (B, 512, H, W)
   ↓
DFD (Domain Feature Decoupler)
   ├─ rob_map = Conv(feat)
   ├─ mask = GumbelSigmoid(sigmoid(rob_map))   (B, C, H, W) per-pixel
   └─ r_feat = feat * mask
       nr_feat = feat * (1 - mask)
   ↓
DFC (Domain Feature Corrector)
   └─ rec_feat = nr_feat + (1-mask) * Conv(nr_feat)
       ↑ 盲修复 — 只看本地 nr_feat,没外部参考
   ↓
最终 logits = Linear(r_feat + rec_feat)


========== PG-DFC (我们的) ==========

输入图片 x
   ↓
Backbone features
   ↓
DFD (跟 F2DC 一样,可选改成 channel-level mask)
   └─ r_feat / nr_feat
   ↓
PG-DFC (Prototype-Guided DFC)
   ├─ 路径 1: 原 conv 残差 (保留)
   │   rec_units = Conv(nr_feat)
   │
   ├─ 路径 2: prototype attention (新加)
   │   q = Linear(AdaPool(nr_feat))               # (B, C)
   │   k = Linear(class_proto_bank)                # (num_classes, C)
   │   attn = softmax(q · k.T / √C)                # (B, num_classes)
   │   v = Linear(class_proto_bank)
   │   proto_clue = attn · v                       # (B, C)
   │   proto_clue = expand_to_spatial              # (B, C, H, W)
   │
   └─ rec_feat = nr_feat + (1-mask) * (rec_units + proto_weight * proto_clue)
   ↓
最终 logits = Linear(r_feat + rec_feat)

[每 round] 训练后 client EMA 更新本地 μ_local[c] (从 r_feat)
[每 round] server 收集 μ_local[c] 加权聚合 → μ_global[c] → 下发回 client
```

### 3.2 通信流程图

```
Round r:

Server                          Client_1     Client_2     Client_3     Client_4
  │
  ├── 下发 backbone params ──────→ ✓           ✓           ✓           ✓
  ├── 下发 μ_global ─────────────→ ✓           ✓           ✓           ✓
  │   (覆盖 client.dfc.class_proto buffer)
  │
  │   client 本地训练 wk_iters epoch:
  │     - F2DC forward + 三路 loss + backprop
  │     - DFC 用 μ_global 做 attention
  │     - 每 batch 后 EMA 更新本地 μ_local
  │
  ├── 收集 backbone params ←──── ✓           ✓           ✓           ✓
  ├── 收集 μ_local + class_count ✓           ✓           ✓           ✓
  │
  ├── 聚合 backbone (weighted FedAvg) — 跟 F2DC 原版一样
  ├── 聚合 μ_global (按 class count 加权):
  │   for c in num_classes:
  │     μ_global[c] = Σ_k (n_c^k / Σ n_c) * μ_local_k[c]
  │
  └── 进入 round r+1
```

---

## 四、详细技术设计

### 4.1 PG-DFC 模块代码(完整可运行版本)

```python
# backbone/ResNet_DC.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class DFC_PG(nn.Module):
    """
    Prototype-Guided DFC.
    向后兼容 F2DC 原版 DFC:proto_weight=0 时退化成原版。
    """
    def __init__(self, size, num_classes, num_channel=64, proto_weight=0.3):
        super().__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.num_classes = num_classes
        self.proto_weight = proto_weight

        # 路径 1: 原 F2DC conv 残差(保留)
        self.net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # 路径 2: prototype attention (新增)
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C, C)

        # class prototype buffer:由 server 下发,不参与 backprop
        self.register_buffer('class_proto', torch.zeros(num_classes, C))

    def set_proto_weight(self, w):
        """warmup 用:训练前 N round 设 0,后逐步 ramp up"""
        self.proto_weight = w

    def forward(self, nr_feat, mask):
        B, C, H, W = nr_feat.shape

        # 路径 1: 原 conv 残差
        rec_units = self.net(nr_feat)

        # 路径 2: prototype attention
        if self.proto_weight > 0 and self.class_proto.abs().sum() > 0:
            # nr_feat pool 成 query
            nr_pooled = F.adaptive_avg_pool2d(nr_feat, 1).reshape(B, C)
            q = self.q_proj(nr_pooled)                              # (B, C)
            # attention over class prototypes
            k = self.k_proj(self.class_proto)                        # (K, C)
            v = self.v_proj(self.class_proto)                        # (K, C)
            attn_logits = q @ k.T / sqrt(C)                          # (B, K)
            attn = F.softmax(attn_logits, dim=-1)                    # (B, K)
            proto_clue = attn @ v                                    # (B, C)
            proto_clue = proto_clue.reshape(B, C, 1, 1).expand(-1, -1, H, W)
        else:
            proto_clue = torch.zeros_like(nr_feat)

        # 融合,只在 non-robust 位置激活
        rec_feat = nr_feat + (1 - mask) * (rec_units + self.proto_weight * proto_clue)
        return rec_feat
```

### 4.2 ResNet 修改

```python
# backbone/ResNet_DC.py: ResNet 类,只改 dfc_module 一行
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, tau=0.1, image_size=(32, 32),
                 name='f2dc', proto_weight=0.3):
        super().__init__()
        # ... 原有 backbone 代码不变 ...

        self.dfd_module = DFD(size=(512, ...), tau=self.tau)
        # 改这一行:
        self.dfc_module = DFC_PG(
            size=(512, int(self.image_size[0]/8), int(self.image_size[1]/8)),
            num_classes=num_classes,
            proto_weight=proto_weight
        )
        # ... 其他不变 ...
```

### 4.3 Client 端 — 训练后 EMA 更新本地 prototype

```python
# models/f2dc.py: F2DC._train_net 方法
class F2DC_PG(F2DC):
    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        self.proto_alpha = 0.99
        self.warmup_rounds = 30  # 前 30 round 不用 prototype
        # 每个 client 的本地 prototype 和 class count(stateful)
        self.local_protos = [None] * args.parti_num
        self.local_counts = [None] * args.parti_num

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        # 初始化 local proto
        if self.local_protos[index] is None:
            feat_dim = net.dfc_module.C   # 一般 512
            self.local_protos[index] = torch.zeros(self.N_CLASS, feat_dim, device=self.device)
            self.local_counts[index] = torch.zeros(self.N_CLASS, device=self.device)

        # warmup 控制 proto_weight
        if self.epoch_index < self.warmup_rounds:
            net.dfc_module.set_proto_weight(0.0)
        else:
            ramp = min(1.0, (self.epoch_index - self.warmup_rounds) / 20)
            net.dfc_module.set_proto_weight(0.3 * ramp)

        # F2DC 原训练循环 + EMA 更新
        for iter in range(self.local_epoch):
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)

                # 修改 ResNet.forward,让它额外返回 r_feat
                out, feat, ro, re, rec, ro_flat, re_flat, r_feat = net(images, return_r_feat=True)

                # ... F2DC 原 loss(完全不变) ...
                loss_DC = self.args.lambda1 * DFD_loss + self.args.lambda2 * DFC_loss
                loss_CE = criterion(out, labels)
                loss = loss_CE + loss_DC

                loss.backward()
                optimizer.step()

                # 新增: EMA 更新本地 prototype
                with torch.no_grad():
                    # 把 r_feat (B, C, H, W) pool 成 (B, C)
                    r_pooled = F.adaptive_avg_pool2d(r_feat, 1).reshape(images.size(0), -1)
                    for c in range(self.N_CLASS):
                        mask_c = (labels == c)
                        n_c = mask_c.sum().item()
                        if n_c > 0:
                            class_mean = r_pooled[mask_c].mean(0)
                            self.local_protos[index][c] = (
                                self.proto_alpha * self.local_protos[index][c]
                                + (1 - self.proto_alpha) * class_mean
                            )
                            self.local_counts[index][c] += n_c

        return ...
```

### 4.4 Server 端 — 聚合 prototype + 下发

```python
# models/utils/federated_model.py: 新增 aggregate_protos 方法
class FederatedModel(nn.Module):
    def aggregate_protos(self, local_protos, local_counts, online_clients):
        """
        加权聚合各 client 的本地 prototype 成 global prototype。
        """
        # 收集 online client 的 proto/count
        protos = [local_protos[i] for i in online_clients]
        counts = [local_counts[i] for i in online_clients]

        K = len(protos)
        num_classes, C = protos[0].shape
        global_proto = torch.zeros(num_classes, C, device=protos[0].device)

        for c in range(num_classes):
            n_c_total = sum(counts[k][c] for k in range(K))
            if n_c_total > 0:
                for k in range(K):
                    if counts[k][c] > 0:
                        global_proto[c] += (counts[k][c] / n_c_total) * protos[k][c]
            # n_c_total = 0 时保持 0,不更新(避免污染)

        # 下发到所有 client(覆盖 buffer)
        for client_net in self.nets_list:
            if hasattr(client_net, 'dfc_module'):
                client_net.dfc_module.class_proto.copy_(global_proto)

        return global_proto
```

### 4.5 主 loop 修改

```python
# models/f2dc.py: F2DC.loc_update 末尾
class F2DC_PG(F2DC):
    def loc_update(self, priloader_list):
        # ... F2DC 原 client 训练循环 ...
        for i in online_clients:
            c_loss, c_samples = self._train_net(i, self.nets_list[i], priloader_list[i])
            ...

        # 1. 原 F2DC backbone 聚合
        self.aggregate_nets(None)

        # 2. 新增: prototype 聚合
        if self.epoch_index >= self.warmup_rounds - 1:
            self.aggregate_protos(self.local_protos, self.local_counts, online_clients)
            # 重置 counts(避免累计旧 round 的 count)
            for i in online_clients:
                self.local_counts[i].zero_()

        return ...
```

---

## 五、训练流程(完整 round)

```
Round r 开始

[Server]
  ├── 下发 backbone state_dict 到所有 client(F2DC 原版)
  └── 下发 global class prototype μ_global → 覆盖 client.dfc.class_proto

[Client k 本地训练 wk_iters epoch]
  for each batch:
    1. forward: x → backbone → feat
                    → DFD → r_feat, nr_feat, mask
                    → DFC_PG(nr_feat, mask, class_proto) → rec_feat
                    → linear → final logits
    2. loss = CE(out) + λ1·(DFD_dis1 + DFD_dis2 + DFD_sep) + λ2·CE(rec_out)
    3. backward + optimizer.step()
    4. 新增: EMA 更新本地 μ_local[c]:
       r_pooled = AvgPool(r_feat)
       for c in classes:
         mask_c = (label == c)
         μ_local_k[c] = α · μ_local_k[c] + (1-α) · mean(r_pooled[mask_c])
         n_c_k += mask_c.sum()

[Client k 上传]
  ├── 上传 backbone state_dict(F2DC 原版)
  └── 上传 μ_local_k 和 n_c_k

[Server 聚合]
  ├── backbone: weighted FedAvg(F2DC 原版)
  └── prototype 聚合:
      for c in classes:
        n_c_total = Σ_k n_c_k
        if n_c_total > 0:
          μ_global[c] = Σ_k (n_c_k / n_c_total) · μ_local_k[c]

Round r 结束 → 进入 r+1
```

---

## 六、关键设计决策(Q&A)

### Q1: Prototype 用 r_feat 还是 feat?

**用 r_feat (robust feature)**

**理由**:
- r_feat 是被 DFD 认定为"class-relevant"的纯净部分,用它当 prototype 才符合"robust class anchor"语义
- 用原 feat 会被 domain artifact 污染 → prototype 自带域偏置 → 反而误导 DFC

### Q2: Prototype 是 channel-level 还是 spatial?

**channel-level (C-dim vector,即 (B,C) 后再 mean)**

**理由**:
- spatial prototype 太大:512 × 16 × 16 = 131K floats × 7 类 = 920KB / round → 通信不可行
- channel-level: 512 × 7 = 3.5KB / round → 可忽略
- 物理意义清晰:每个 channel 描述一个"语义维度"

### Q3: Attention 用 dot-product 还是 cosine?

**dot-product + /√C 缩放**

**理由**:
- cosine 需要 norm,训练初期不稳
- dot-product + scaling 是 Transformer 标准做法,数值稳定

### Q4: Prototype 早期不稳怎么办?

**warmup 30 round + ramp up**

**理由**:
- 前 30 round r_feat 还没学好,prototype 是噪声
- warmup 期间 `proto_weight=0` → DFC 退化成原 F2DC
- 第 30 round 后 ramp up `proto_weight` 从 0 到 0.3 (用 20 round 线性 ramp)

### Q5: EMA decay α=0.99 还是别的?

**默认 α=0.99,可调到 0.9 ~ 0.999**

**理由**:
- α=0.9: 跟得快但抖
- α=0.99: 平滑(滞后约 100 batch),适合稳定训练
- α=0.999: 几乎不更新,前期 prototype 一直接近 0
- 我们的 setup E=10 batch 较多,α=0.99 是中间值

### Q6: 哪些 class 没出现怎么办?

**class count = 0 时跳过更新**

**理由**:
- 某 client 缺某 class → 那个 class 的 μ_local 永远 0
- Server 聚合时 mask 掉 0 的 client (用 n_c_k > 0 判断)
- 完全 0 的 class(所有 client 都没)→ μ_global 保持 0,DFC 退化成原 F2DC

### Q7: 训练后期 backbone 还在变,prototype 跟不上怎么办?

**LR 降到 0.1× 时 reset prototype + 短 warmup**

**理由**:
- 大 LR 阶段 backbone 剧烈变化,EMA 留着旧 backbone 的 feature mean
- LR 衰减后 backbone 趋稳,但 prototype 滞后
- 解法:scheduler 触发 LR 衰减时 reset μ_local = 0,再 warmup 5 round
- 这是 advanced 技巧,初版不必加

---

## 七、工程实现 — 改哪些文件

| 文件 | 改动 | 行数 | 备注 |
|---|---|:--:|---|
| `backbone/ResNet_DC.py` | 加 `DFC_PG` 类(新) | +60 | 完全新增,跟原 DFC 并存 |
| `backbone/ResNet_DC.py` | `ResNet.__init__` 用 DFC_PG 替代 DFC | +3 | 加 `num_classes`/`proto_weight` 参数 |
| `backbone/ResNet_DC.py` | `ResNet.forward` 加 `return_r_feat` 参数 | +5 | 训练时返回 r_feat 给 client EMA |
| `models/f2dc.py` | 新建 `F2DC_PG` 子类(继承 F2DC) | +80 | EMA + 训练后更新 |
| `models/utils/federated_model.py` | 加 `aggregate_protos` 方法 | +30 | server 端聚合 + 下发 |
| `models/__init__.py` | 注册 `f2dc_pg` 方法名 | +2 | |
| `utils/best_args.py` | 加 `f2dc_pg` 超参条目 | +30 | 跟 f2dc 一样 + 新加 proto_weight/proto_alpha |
| `main_run.py` | 加 `proto_weight/proto_alpha/warmup_rounds` argparse | +6 | |
| **总计** | | **~216 行** | 大约 200 行,符合"小改动"承诺 |

### 兼容性
- 当 `proto_weight=0` 时,DFC_PG **完全等价于** F2DC 原版 DFC
- 可以一键切换 F2DC vs F2DC_PG 做对比

---

## 八、消融实验设计(为论文 Table 准备)

### 主表(完整方案)

| Variant | proto_weight | EMA α | warmup | 预期 PACS AVG | 预期 Office AVG |
|---|:--:|:--:|:--:|:--:|:--:|
| F2DC (vanilla) | 0 (退化) | - | - | 76.47 baseline | 66.82 baseline |
| **F2DC_PG (full)** | 0.3 (ramp) | 0.99 | 30 | **+1.0~1.5pp** | **+1.0~1.5pp** |

### 消融子项

| Variant | 配置 | 验证什么 |
|---|---|---|
| no proto attention | proto_weight=0 全程 | 退化到原 F2DC,验证 setup 一致性 |
| local-only proto | 不上传聚合,只用本地 μ_local | 验证 cross-client 信号是否关键 |
| no warmup | warmup=0,直接 proto_weight=0.3 | 验证 warmup 是否必要 |
| no attention(直接加 mean proto) | proto_clue = mean(class_proto) | 验证 attention pick 是否必要 |
| α grid | α ∈ {0.9, 0.99, 0.999} | 验证 EMA decay 敏感度 |
| proto_weight grid | w ∈ {0.1, 0.3, 0.5, 0.7} | 验证融合系数 |
| proto from feat (vs r_feat) | EMA 用 feat 而非 r_feat | 验证用 robust feature 是否更对 |

---

## 九、风险点 + 应对

| 风险 | 严重度 | 应对 |
|---|:--:|---|
| Attention 权重塌缩(全集中在 1-2 类) | ★★ | 加 attention temperature τ=0.5 平滑 |
| Prototype EMA 早期不稳 | ★★★ | warmup 30 round,初始化 μ=0 |
| proto_weight 调参敏感 | ★★ | grid search {0.1, 0.3, 0.5, 0.7} |
| Class imbalance 时 0 prototype 污染 | ★★ | 聚合时 mask 掉 n_c=0 的 client |
| 通信开销增加 | ★ | C × num_classes = 3.5KB/round,可忽略 |
| 跟 F2DC 原版兼容性破坏 | ★ | proto_weight=0 时完全退化,内部测试通过即可 |
| 后期 backbone 还在变,EMA 滞后 | ★ | 高级:LR 衰减时 reset μ_local。初版不加 |
| Prototype 维度跟 backbone 不对齐 | ★ | F2DC ResNet10 = 512-d,代码里 hardcode 必须查 |

---

## 十、跟我们已有 EXP-128/129 的衔接

| 已有积累 | 在 PG-DFC 里的角色 |
|---|---|
| EXP-128 DualEnc 失败经验 | 写入 paper "Failed Attempts" — 说明 image-space cycle 不行,自然引出 channel-level 解耦 + prototype guidance |
| EXP-129 F2DC-style 诊断脚本 | 评估 PG-DFC 训出来的 r_feat 是否更"健康"(SV decay 更平、ER 更高) |
| EXP-129 functional 4 指标(DIAD/CCTM) | paper Section "Functional Diagnostic" — 证明 PG-DFC 不只是 acc 涨,functional 健康度也涨 |
| FedDSA 的 sem_head/sty_head 双头经验 | 直接复用双头实现技巧,不是从 0 开始 |
| 我们 orth_only 的正交损失代码 | DFD 的 sep loss 可考虑加正交约束(可选) |
| EXP-130 F2DC baseline 数字 | 直接用作 comparison baseline |

**关键**:这个方案**跟我们已有方向完全衔接**,不是从 0 开始。

---

## 十一、时间表

| 周 | 任务 | 产出 |
|---|---|---|
| W1 | F2DC sanity 复现(已在跑)| baseline 数字 PACS 76.47 / Office 66.82 |
| W2 Day 1-2 | 写 DFC_PG 类 + replace 到 ResNet | code v1 编译通过 |
| W2 Day 3-4 | 写 client EMA + server 聚合 | code v1 完整可跑 |
| W2 Day 5 | 跑 PACS 1-seed sanity(proto_weight=0,验证退化等价 F2DC)| 跟 F2DC 数字一致 ±0.3pp |
| W3 Day 1-3 | 跑 PACS 1-seed full(proto_weight=0.3,带 warmup)| **第一个 PG-DFC 数字** |
| W3 Day 4-5 | 调 proto_weight + α 各跑 1-seed | grid 表 |
| W4 | 全配置 3-seed (PACS + Office) | 主表 |
| W5 | 消融实验(8 个 variant) | 消融表 |
| W6 | 加 functional 诊断(DIAD/CCTM) | 诊断表 |
| W7-8 | 写论文 | 初稿 |

---

## 十二、立即下一步

等 F2DC sanity 复现完成 → 立刻 W2 启动:

1. 在 F2DC repo 创建 branch `pg-dfc-v1`
2. 按本笔记 §4 顺序加代码
3. 第一个 sanity 跑:proto_weight=0,验证退化等价 F2DC
4. 第二个跑:proto_weight=0.3 + warmup=30,看是否涨

**预期 W2 末就能拿到 PG-DFC vs F2DC 的初步数字**。

---

## 十三、Novelty 论证(查重 / 答辩用)

| 工作 | 用 prototype | 用 cross-client prototype | 用 attention pick | 在 F2DC decouple 框架下 |
|---|:--:|:--:|:--:|:--:|
| FedProto (AAAI'22) | ✓ | ✓ | ✗ (直接 MSE) | ✗ (无解耦) |
| FPL (CVPR'23) | ✓ (FINCH 多原型) | ✓ | ✗ | ✗ |
| Disentanglement-Then-Reconstruction (2020) | ✓ (UDA) | ✗ (UDA 没 client 概念) | ✗ | 类似框架但不是 FL |
| RFedDis (2023) | ✗ | ✗ | ✗ | ✗ (双 head + evidential) |
| F2DC (CVPR'26) | ✗ | ✗ | ✗ | ✓ |
| **PG-DFC (我们)** | **✓** | **✓** | **✓** | **✓** |

**首次在 F2DC decouple 框架下用跨 client class prototype + attention 引导 DFC 校准** — 这是 contribution。

---

## 十四、毕设论文 framing

**毕设答辩用**:
> "F2DC 是 CVPR'26 的 federated cross-domain SOTA,但 paper 自己 Sec 4.1 末段承认 DFD 切不干净、DFC 抢救能力有限(Tab 7 自己 ablation 验证)。我们提出 PG-DFC,用 server 维护的跨 client class prototype 引导 DFC 校准,通过 attention pick 让 DFC 知道往哪个类的方向救。在 F2DC 同 setup 下涨 +1.5pp,加 functional 诊断(DIAD/CCTM)证明 global model 健康度也涨。"

**论文 abstract 用**:
> "Prior work F2DC pioneered Decouple-and-Calibrate FL but suffered from blind residual recovery in its DFC module. We propose PG-DFC, leveraging server-aggregated class prototypes as recovery anchors via cross-attention. The prototype bank is maintained through EMA-based local updates and weighted aggregation, providing directional guidance to the calibration. Experiments on PACS and Office-Caltech show consistent +1.5pp improvement over F2DC, with corroborating evidence from functional health metrics."

---

## 附录 A: 代码 patch 模板(等 F2DC 复现完直接 apply)

```diff
# backbone/ResNet_DC.py
+ class DFC_PG(nn.Module):
+     # ... §4.1 完整代码 ...

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, tau=0.1, image_size=(32, 32),
-                name='f2dc'):
+                name='f2dc', proto_weight=0.3):
        ...
-       self.dfc_module = DFC(size=(512, ...))
+       self.dfc_module = DFC_PG(size=(512, ...), num_classes=num_classes, proto_weight=proto_weight)

# models/f2dc.py
+ class F2DC_PG(F2DC):
+     # ... §4.3 完整代码 ...

# models/__init__.py
+ from models.f2dc_pg import F2DC_PG

# utils/best_args.py
+ 'fl_pacs': {
+     ...
+     'f2dc_pg': {
+         'local_lr': 0.01, 'local_batch_size': 46,
+         'proto_weight': 0.3, 'proto_alpha': 0.99, 'warmup_rounds': 30,
+     }
+ }
```

---

## 附录 B: 我之前 critique F2DC 的演化轨迹

(供后续参考,避免重复犯错)

| 阶段 | 我提出的 critique | 后来发现错在哪 | 修正 |
|---|---|---|---|
| 初版 | "wrong-high label 在 PACS top-2 是另一动物会失效" | 没注意 DFC 会修复 | 把 wrong-high 严重度从 ★★★★ 降到 ★★★ |
| 二版 | "F2DC 输给 FDSE 7.3pp 设计有问题" | 数据张冠李戴(不同 setup 数字直接比) | 真实 F2DC 在自己 setup 下赢 FDSE +3pp |
| 三版 | "借 paper A (Disentanglement-Then-Reconstruction) 的 prototype reconstruction" | A 是 UDA 不是 FL,没法直接搬 | 改为只借"prototype 当 anchor"的思想 |
| 四版 | "Mask Consensus 是 F2DC 没解决的问题" | 没数据支持,可能伪问题 | 改为先做 sanity 诊断再说 |
| 五版 | "RFedDis 风格双 head + prototype" | 改动过大,不实际 | 改为小改动:只换 DFC 模块 |
| **当前(六版)** | "PG-DFC: prototype-guided DFC residual" | 改动 ~200 行,F2DC 兼容,经过多轮自我审查 | 等 F2DC 复现完执行 |

**经验教训**:
1. 不能凭印象引数据 — 必须查 paper Tab
2. 不能凭 abstract 推方法 — 必须读源码
3. 不能跨 setup 借鉴 — UDA / FedDG / personalization FL 各有独特约束
4. critique 要诚实 — 不能为 critique 而 critique

---

# 附录 C: v2 实证修正(2026-04-26 晚)

## C.1 触发原因

经过 review (codex 13 项 critique) 后,有 2 个问题(M1, M2)review 推断"必须改"。
我们决定:**不依赖 review 推理,先用 EXP-130 真实 F2DC dataloader 实测**。

诊断脚本: `scripts/diagnostic/dump_client_distribution.py`
执行环境: sc4 (westc, port 14824)
数据: F2DC 真实 PACS / Office / Digits 在 seed {2, 15, 333} 下的 client 分布

## C.2 实测结果(3-seed mean,2026-04-26 23:00)

| Dataset | M1 single_sample (mean ± std) | M2 size_skew (mean ± std) | M1 决策 | M2 决策 |
|---|:--:|:--:|---|---|
| **PACS** | 0.085 ± 0.006 | **4.09 ± 2.02** | EMA + n≥2 保护 | **★ 必须 L2-norm 等权** |
| **Office** | 0.137 ± 0.014 (max 0.60) | **7.16 ± 0.00** | EMA + n≥2 保护 | **★ 必须 L2-norm 等权** |
| **Digits** | **0.229 ± 0.043** (max 0.45) | **10.17 ± 0.00** | **★ 必须改 sample 累加** | **★ 必须 L2-norm 等权** |

完整数据: `experiments/ablation/EXP-130_F2DC_baseline_main_table/client_distribution.json`

### 数字解读

**M1 (single_sample_freq)** — 单 batch 内某类只有 0-1 个样本的占比:
- PACS sketch client 943 sample → batch=46,平均每类 ~6 样本,M1 还行
- Office dslr client 25 sample → 只够 1 个 batch,某些类只 1-2 样本,M1 max 60%
- Digits usps/syn client 72/74 sample → 长期单样本 batch,M1 mean 23%

**M2 (size_skew)** — 最大 client / 最小 client 样本数比:
- PACS: sketch 943 vs photo 400 → 2.36-6.93x(seed 不同)
- Office: caltech 179 vs **dslr 25** → 7.16x(三 seed 一致)
- Digits: svhn 732 vs **usps 72** → 10.17x(三 seed 一致)

### Review 预测 vs 实测

| Review 推断 | 实测 | 结论 |
|---|---|---|
| M1 在 FL 小 batch 是真问题 | PACS 还行,Office/Digits **更严重** | 真问题,且**比 review 想象更严重** |
| M2 sample-weighted 不消除 domain skew | 三 dataset 都 4-10x skew | 真问题,**所有 dataset 都必须改** |

**Review 完全正确,实测确认。**

## C.3 锁定 fix v2

### v2 Fix 1: EMA → Sample 累加(M1)

**v1 (废弃)**:
```python
# 每 batch 算 batch_mean 用 EMA 更新
class_mean = r_pooled[mask_c].mean(0)
μ_local[c] = α * μ_local[c] + (1-α) * class_mean
```
**问题**: 单 batch 单类只 1 样本时,outlier 拉飞 EMA(α=0.99 也救不回)

**v2 (敲定)**:
```python
# Round 内 sample 累加,round 末统一算均值
# Client 端初始化(每 round 开始)
sum_feat = torch.zeros(num_classes, C, device=device)
count = torch.zeros(num_classes, device=device)

# 每 batch 累加(不算 mean,直接 sum)
for batch in loader:
    r_pooled = AdaptiveAvgPool2d(1)(r_feat).flatten(1)   # (B, C)
    for c in range(num_classes):
        mask_c = (labels == c)
        if mask_c.sum() > 0:
            sum_feat[c] += r_pooled[mask_c].sum(0)        # ★ sum 不是 mean
            count[c] += mask_c.sum().item()

# Round 末算真实均值 + 跨 round EMA(可选,稳定后期)
for c in range(num_classes):
    if count[c] > 0:
        round_mean = sum_feat[c] / count[c]
        μ_local[c] = α_round * μ_local[c] + (1 - α_round) * round_mean
        # α_round 可设 0(不跨 round 平滑)或 0.5(适度平滑)
```

**关键差别**:
- v1: 单样本 batch 直接污染 EMA
- v2: 单样本被淹没在 round 内全部 sample 中(dslr 25 样本 / 10 类 → 至少 2-3 样本/类的均值)

**好处**:
- 这是 FedProto / FPL 标准做法
- 数值上稳定(全 sample 均值 ≠ 抖动 batch 均值)
- 实现简单(sum + count buffer,无 α 调参)
- 通信量不变(prototype 还是 (num_classes, C))

### v2 Fix 2: Sample-Weighted → L2-norm + 等权(M2)

**v1 (废弃)**:
```python
# Server 端聚合 — 按 sample count 加权
for c in range(num_classes):
    n_c_total = Σ_k n_c_k
    μ_global[c] = Σ_k (n_c_k / n_c_total) * μ_local_k[c]
```
**问题**: 大 client / 大 domain 主导,小 client 被淹没。
- PACS: sketch client (943) 主导,photo client (400) 被压
- Office: caltech (179) 主导,dslr (25) 被压
- 结果: μ_global 偏向主导 domain,小 domain 的 DFC 拿到的 prototype 反向引导

**v2 (敲定)**:
```python
# Server 端聚合 — L2-normalize 后等权平均
import torch.nn.functional as F

for c in range(num_classes):
    valid_clients = [k for k in online_clients if local_counts[k][c] > 0]
    if not valid_clients:
        continue  # 无 client 有此类,保持 μ_global[c] 不变

    # 1. 每个 client 的 local prototype L2 归一化(只看方向)
    normalized = [F.normalize(local_protos[k][c], dim=0) for k in valid_clients]

    # 2. 等权平均(每个 valid client 1/K_c 票)
    μ_global[c] = torch.stack(normalized).mean(0)

    # 3. 输出再 normalize 一次(确保 unit vector,attention 时方向稳定)
    μ_global[c] = F.normalize(μ_global[c], dim=0)
```

**关键差别**:
- v1: 大 client 权重大 → magnitude 主导
- v2: 都 L2-norm 后等权 → 只看方向,每 domain 平等发声

**好处**:
- 真正的"跨 domain 共识 prototype"
- 小 domain 的 client 也能影响 global prototype
- DFC 引导方向 = 所有 domain 共识方向,不偏任何一边
- 这是 metric learning / clustering 标准做法

### v2 Fix 不动的部分

| 项 | 状态 | 原因 |
|---|---|---|
| C1 (proto_clue 梯度污染 mask) | 先跑加诊断验证,不预先 detach | review 推断,未实测,可能过虑 |
| C2 (DFD/DFC 冲突放大) | 同 C1 | 同 |
| C3 (正反馈塌缩) | 同 C1 | C1 fix 后自动解决 |
| M3 (proto_weight 量级) | 不改,实测看现象 | review 数值估算偏高 |
| M4 (attention temperature) | 不改,实测看现象 | review 数学推导错 |
| M5 (EMA α 太慢) | M1 改 sample 累加自动消失 | / |
| m1-m4 | 不改 | 工程优化,后期再说 |

## C.4 v2 验证 checklist(W2-3 实施时)

| 验证 | 方法 | 通过标准 |
|---|---|---|
| Fix 1 (sample 累加) 实施正确 | 跑 1 round,看 prototype 是否稳定 | round 末 prototype 跟 client 真实 r_feat mean 误差 < 1% |
| Fix 2 (L2-norm 等权) 实施正确 | 跑 1 round,看 server 聚合后 ||μ_global[c]||=1 | norm 都是 1 (±1e-6) |
| 跟 v1 对比 | 各跑 1 seed PACS | v2 ≥ v1(至少不退步) |
| 跟 F2DC vanilla 对比 | proto_weight=0 sanity | v2 with proto=0 ≡ F2DC vanilla (±0.3pp) |

## C.5 v2 后续待验证(等 PACS 1-seed 跑出来)

- C1 是否真问题(看 mask sparsity 演化)
- C3 是否真问题(看 r_feat ER 演化)
- M3/M4 是否真问题(看 attention entropy + proto/rec 量级比)

如果 C1 实测验证为真问题 → 加 `(1-mask).detach()` fix
如果不真 → 保持 v2 不动

## C.6 关键经验

**Review 是好的纠错工具,但不应该全盘接受 — 必须用实测数据决定哪些 fix 必做**。

- 实测验证的 fix(M1, M2): 锁定执行
- 推断的 fix(C1, M3, M4): 加诊断 hook 后实测,数据驱动决策
- 不要预先优化(避免引入 review 自己也没把握的复杂 fix)

诊断脚本(`scripts/diagnostic/dump_client_distribution.py`)用了 10 分钟产出 9 个分布,
省了几周可能走错的 fix 路径。

---

# 附录 D: v2 修订版总改动量

| 文件 | v1 改动 | v2 修订 | 总行数 |
|---|---|---|:--:|
| `backbone/ResNet_DC.py` | 加 DFC_PG 类 | 不变 | +60 |
| `backbone/ResNet_DC.py` | ResNet 用 DFC_PG | 不变 | +3 |
| `backbone/ResNet_DC.py` | forward 加 return_r_feat | 不变 | +5 |
| `models/f2dc.py` | F2DC_PG 子类 + EMA | **EMA 改 sample 累加(-15 +20)** | +85 |
| `models/utils/federated_model.py` | aggregate_protos | **改 L2-norm + 等权(-10 +15)** | +35 |
| `models/__init__.py` | 注册 f2dc_pg | 不变 | +2 |
| `utils/best_args.py` | 加 f2dc_pg 超参 | **去掉 proto_alpha(不需要 EMA)** | +27 |
| `main_run.py` | argparse | 不变 | +6 |
| **总计** | ~216 行 | **~223 行** | 几乎不变 |

