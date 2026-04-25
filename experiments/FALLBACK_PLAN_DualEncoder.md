# FedDSA-DualEncoder —— 兜底方案(独立双通道 encoder)

**创建日期**: 2026-04-24
**方案性质**: **真·物理解耦**,代价大但**架构层面保证 z_sem 和 z_sty 不共享任何 trunk**
**和 FedDSA-Calibrator 方案的区别**: Calibrator **接受并利用** class mirror 泄漏;DualEncoder **从源头杜绝**泄漏
**实施成本**: ~100-150 行代码(比 Calibrator 重),参数量约 2× orth_only

---

## 一、动机 —— 直击 shared trunk 诅咒的根源

### 1.1 Decouple Probe 的核心发现(2026-04-24)

可视化证明:
- `L_orth` 让权重层面几乎完美正交(corr 0.025)
- 两路对 1024 个 trunk channel 的使用独立(pearson ≈ 0)
- **但 MLP-256 probe 仍能从 z_sty 读出 81% class 信号**

**唯一解释**:L_CE 让 trunk(AlexNet encoder)的**每个 channel 都弥散了 class 信息**。无论 z_sty 挑哪批 channel,每个 channel 里都有 class 残余 —— 在 head 层面怎么约束都治不好。

### 1.2 本方案的哲学

**既然 shared trunk 是诅咒源头,就不要 shared trunk**。

- 给 semantic 路径一个**独立的 encoder_sem**(不经 L_CE 之外的 loss)
- 给 style 路径一个**独立的 encoder_sty**(只被 L_orth / L_style_aux 约束)
- 两个 encoder **权重完全独立,前向完全不共享任何 tensor**

这样:
- encoder_sem 的 1024 channel 里全是 class 信号(L_CE 训出来的)—— 符合目的
- encoder_sty 的 1024 channel 里**没有任何 L_CE 梯度影响过**,class 信号只能从 L_orth 推开(弱)或 L_style_aux 显式指定(强)—— 可控

**物理上不共享 = 诅咒不存在**。

---

## 二、架构对比

### 2.1 现状(shared trunk,诅咒暴露)

```
x
 ↓
AlexNet encoder (1×60M params, 共享 trunk)
 ↓ pooled [1024]
 ├── semantic_head → z_sem → classifier → L_CE
 └── style_head   → z_sty
             ↑
        L_orth(z_sem, z_sty)
```

**问题**:L_CE 梯度通过 pooled 反传到整个 encoder,每个 channel 都学了 class → z_sty 读哪个 channel 都有 class。

### 2.2 DualEncoder(独立 trunk)

```
x
 ├──────────────┬──────────────┐
 ↓              ↓              
encoder_sem    encoder_sty      ← 两个独立 AlexNet,权重完全分开
 ↓              ↓
pooled_sem    pooled_sty
 ↓              ↓
semantic_head  style_head
 ↓              ↓
z_sem          z_sty
 ↓              ↑
classifier ─── L_orth ──→ (z_sty 无主任务约束)
 ↓
L_CE (只训 encoder_sem + classifier)
```

**关键点**:
- `encoder_sem` 只被 **L_CE 梯度**训 → 所有 channel 学 class 信号(预期)
- `encoder_sty` 只被 **L_orth 梯度**训(+ 可选 L_style_aux)→ trunk 不含 class 梯度
- 两路梯度 pipeline **完全隔离**,诅咒机制消失

---

## 三、三个变体(按参数量递增)

### 变体 A:完全独立双 encoder(最干净,参数 2×)

```python
self.encoder_sem = AlexNetEncoder()     # ~60M
self.encoder_sty = AlexNetEncoder()     # ~60M (独立权重)
self.semantic_head = MLP(1024, 128)     # ~130K
self.style_head    = MLP(1024, 128)     # ~130K
self.classifier    = Linear(128, num_classes)

def forward(self, x):
    pooled_sem = self.encoder_sem(x)    # [B, 1024]
    pooled_sty = self.encoder_sty(x)    # [B, 1024]
    z_sem = self.semantic_head(pooled_sem)
    z_sty = self.style_head(pooled_sty)
    logits = self.classifier(z_sem)
    return {'logits': logits, 'z_sem': z_sem, 'z_sty': z_sty}
```

**参数量**: ~120M (vs orth_only 60M,**2×**)
**显存**: 增加约 1.5×(两份 activation 但共享 batch)
**训练时间**: +50~80%(双前向 + 双反向)
**通信**: FL aggregate 时两个 encoder 都要传 → **通信量 2×** ⚠️

### 变体 B:Early share + late split(参数 1.3×)

```python
# 共享前 2 层(conv1-2,低级 edge/texture 特征,不区分 semantic vs style)
self.shared_early = nn.Sequential(conv1, bn1, relu1, maxpool1,
                                   conv2, bn2, relu2, maxpool2)

# 后 3 层分叉(conv3-5,高级特征,这里才区分 semantic vs style)
self.encoder_sem_late = nn.Sequential(conv3, ..., conv5, maxpool5)
self.encoder_sty_late = nn.Sequential(conv3, ..., conv5, maxpool5)  # 独立权重

# 两条路径的 avgpool + fc 也独立
```

**逻辑**: 低级特征(纹理、边缘)两路都要用,共享 OK;高级特征(class 判别 vs style 表达)才需要分离。
**参数量**: ~80M (1.3×)
**显存**: 增加约 1.2×
**通信**: shared_early 只聚合一次,late 聚合两次 → 通信 ~1.5×
**风险**: 如果 L_CE 梯度通过 late_sem 反传到 shared_early,**shared_early 的 channel 仍会学 class** → 诅咒部分回来(虽然弱于变体 A 的 fully shared)

### 变体 C:主 encoder + adapter 分支(参数 ~1.1×)

```python
self.encoder_main = AlexNetEncoder()                # 主 trunk,~60M
self.sem_adapter = nn.Sequential(Linear(1024, 128))  # 小 adapter
self.sty_adapter = nn.Sequential(Linear(1024, 128),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  Linear(128, 1024),
                                  nn.ReLU())         # 更大 adapter, 尝试"重塑" feature
```

**参数量**: ~61M (1.02×)
**风险**: 本质还是 shared trunk 架构,**诅咒依然存在**,只是换了个 adapter 名字。**不推荐**,和 Calibrator 思路重合。

### 变体选择建议

| 变体 | 参数 | 物理解耦 | 推荐度 |
|:-:|:-:|:-:|:-:|
| **A 完全独立** | 2× | ✅ 彻底 | **⭐⭐⭐** |
| B Early share | 1.3× | 🟡 部分 | ⭐⭐ |
| C Adapter | 1.02× | ❌ 伪解耦 | ⭐ |

**首选变体 A**。兜底方案要的就是"物理保证",半吊子不如不做。

---

## 四、Loss 组成

$$
\mathcal{L}_{\text{total}} = \underbrace{\text{CE}(\text{logits}, y)}_{\mathcal{L}_{CE}}
+ \underbrace{\lambda_{\text{orth}} \cdot \cos^2(z_{\text{sem}}, z_{\text{sty}})}_{\mathcal{L}_{\text{orth}}}
+ \underbrace{\lambda_{\text{sty\_aux}} \cdot \mathcal{L}_{\text{style\_aux}}}_{\text{可选: 防 z\_sty 坍缩}}
$$

### 4.1 L_CE(必要)

- 只对 `encoder_sem + semantic_head + classifier` 反传
- encoder_sty 完全不接收 L_CE 梯度

### 4.2 L_orth(必要)

- `cos²(z_sem, z_sty)` 推开两路
- 梯度对 encoder_sem 和 encoder_sty 都传(两路都调整以满足正交)

### 4.3 L_style_aux(可选,防坍缩)

**问题**:如果 encoder_sty 只被 L_orth 约束,可能学成**简单解**(比如全部输出 0 向量,trivially 正交)。

**预防**:加辅助目标给 encoder_sty 一个"要学点啥"的 pressure。候选:

| 选项 | 公式 | 要 label | 备注 |
|---|---|:-:|---|
| **(a) Domain prediction** | `CE(dom_head(z_sty), domain_id)` | 只要 client_id | 逼 z_sty 含 domain 信号(CDANN 变体但**无 GRL**) |
| **(b) Reconstruction** | `MSE(decoder(z_sty), x)` | 无需 label | 让 z_sty 保留重建所需信息(含 style) |
| **(c) Nothing** | 无 L_style_aux | — | 完全放任,风险 z_sty 坍缩成 0 |

**推荐 (a)**:便宜(+9K params dom_head),domain label 天然有(client_id),**不走 CDANN 老路**的关键是**不加 GRL** —— 因为我们这是独立 encoder,z_sem 路径不受这个 loss 影响。

**Loss schedule**:
- `λ_orth = 1.0`(同 orth_only baseline)
- `λ_sty_aux = 0.3`(较弱,只防坍缩,不主导)

---

## 五、FL 聚合策略(关键考虑)

### 5.1 两个 encoder 怎么聚合?

**选项 X:都聚合(FedAvg 式)**
- 每 round 两个 encoder 都上传 server 聚合
- 通信量 2×
- 简单但费带宽

**选项 Y:只聚合 encoder_sem,encoder_sty 本地私有**
- encoder_sty 不参与聚合,每 client 自己保留
- 类似 FedBN 把 BN 保留本地的哲学推广到整个 style encoder
- 通信量回到 1×(甚至更小)
- 对应"style 是本地特征"的先验

**选项 Z:encoder_sem FedAvg + encoder_sty FedBN(只聚合非 BN 参数)**
- 折中方案
- encoder_sty 的卷积权重聚合,BN 保留本地
- style 统计在 client 特化,style 特征结构跨 client 共享

**推荐 Y 或 Z**:既减少通信压力,又符合"style 本地化"的自然假设。

### 5.2 和已有 style bank 机制的关系

FedDSA 原方案有 **style bank**(跨 client 共享 (μ, σ) 做 AdaIN 增强)。DualEncoder 下:
- encoder_sty 本地化(选项 Y/Z)
- 但 style bank 还可以跨 client 共享(把本地 encoder_sty 的 (μ, σ) 上传,用于别的 client 增强)
- **style bank 的 novelty 保留**,不冲突

---

## 六、实施代码骨架(~150 行)

### 6.1 新文件:`FDSE_CVPR25/algorithm/feddsa_dual_encoder.py`

```python
"""FedDSA-DualEncoder: 独立双 encoder 方案(兜底)
两个完整 AlexNet encoder,物理上保证 z_sem 和 z_sty 不共享 trunk.
"""
from algorithm.feddsa_scheduled import (
    AlexNetEncoder, FedDSAServer as _BaseServer,
    FedDSAClient as _BaseClient, FedDSAModel as _BaseModel,
)

class FedDSADualEncoderModel(nn.Module):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128, num_clients=4):
        super().__init__()
        self.encoder_sem = AlexNetEncoder()     # 独立!
        self.encoder_sty = AlexNetEncoder()     # 独立!
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)
        # 可选: 防 z_sty 坍缩
        self.dom_head = nn.Linear(proj_dim, num_clients)

        self.num_classes = num_classes
        self.num_clients = num_clients

    def forward(self, x):
        pooled_sem = self.encoder_sem(x)
        pooled_sty = self.encoder_sty(x)
        z_sem = self.semantic_head(pooled_sem)
        z_sty = self.style_head(pooled_sty)
        logits = self.classifier(z_sem)
        dom_logits = self.dom_head(z_sty)  # 辅助
        return {
            'logits': logits, 'dom_logits': dom_logits,
            'z_sem': z_sem, 'z_sty': z_sty,
        }

class Client(_BaseClient):
    def train_step(self, batch):
        x, y = batch
        out = self.model(x)
        # 主损失
        L_CE = F.cross_entropy(out['logits'], y)
        # 正交
        cos = F.cosine_similarity(out['z_sem'], out['z_sty'], dim=-1)
        L_orth = cos.pow(2).mean()
        # 防坍缩
        dom_id = torch.full((x.size(0),), self.id, device=x.device, dtype=torch.long)
        L_sty_aux = F.cross_entropy(out['dom_logits'], dom_id)
        # 合
        L = L_CE + self.lambda_orth * L_orth + self.lambda_sty_aux * L_sty_aux
        L.backward()
        ...

class Server(_BaseServer):
    def aggregate(self):
        # 选项 Y: 只聚合 encoder_sem, encoder_sty 本地私有
        sem_state = self.collect_state(keys=['encoder_sem.*', 'semantic_head.*', 'classifier.*'])
        aggregated = self.fedavg(sem_state)
        self.broadcast(aggregated)
        # encoder_sty 和 style_head 每 client 保留自己的
```

### 6.2 Config

```yaml
# FDSE_CVPR25/config/office/feddsa_dual_encoder_r200.yml
algorithm: feddsa_dual_encoder
num_rounds: 200
learning_rate: 0.01
batch_size: 50
num_epochs: 5
algo_para:
  lambda_orth: 1.0
  lambda_sty_aux: 0.3
```

---

## 七、预期定量 + 判决

### 7.1 最好情况(target)

| 数据集 | orth_only baseline | DualEncoder 预期 | Δ |
|---|:-:|:-:|:-:|
| Office AVG Best | 89.09 | **91.0~92.0** | **+1.9~2.9 pp** |
| PACS AVG Best | 80.64 | **81.5~82.5** | **+0.9~1.9 pp** |

**逻辑**:
- encoder_sty 不再被 L_CE 梯度影响 → encoder_sem 可以**专注学 class**,不用再和"不要泄漏"拔河
- 专心学 class 的 encoder 表征力更强 → 边界更干净 → accuracy 涨

### 7.2 保守情况

| 数据集 | orth_only | DualEncoder | Δ |
|---|:-:|:-:|:-:|
| Office | 89.09 | **89.0~90.0** | ±0 |
| PACS | 80.64 | **80.0~81.0** | ±0 |

**这个情况也可接受**:至少证明**物理解耦做到了**,probe_sty_class 应该会明显下降(从 0.81 → 0.30 以下),有 representation-level 证据即使 accuracy 持平。

### 7.3 失败情况(需接受的可能性)

- **encoder_sty 坍缩成 trivial**:even with L_sty_aux,如果 aux 信号不够强,z_sty 可能变成无意义表征 → L_orth 几乎无约束 → 等价于只训 encoder_sem → 成本白费
- **参数量膨胀导致过拟合**:小数据集(Office ~2000 图)上 2× 参数可能过拟合 → accuracy 反而退

### 7.4 判决标准

- s=2 在 R40 时 AVG > 87%(收敛正常)
- s=2 R200 AVG Best > 89.5(至少不退 Office)
- 3-seed mean AVG Best > 89.5 → 继续 PACS
- 3-seed mean < 89.0 → kill

---

## 八、Probe 层面的预期差异(关键区分点)

这是 DualEncoder **vs Calibrator 的决定性差异**:

| 指标 | orth_only | Calibrator(兜底 A) | **DualEncoder(兜底 B)** |
|---|:-:|:-:|:-:|
| probe_sty_class linear | 0.24 | **0.24**(不变) | **< 0.15**(近 random)⭐ |
| probe_sty_class MLP-256 | 0.81 | **0.81**(不变,但 corrector 显式提取) | **< 0.30**(显著下降)⭐ |
| 参数量 | 1× | 1.01× | **2×** |
| 通信量 | 1× | 1× | 1× ~ 2×(看聚合策略) |

**DualEncoder 的杀手锏**:即使 accuracy 和 orth_only 持平,**probe_sty_class 暴跌**就是 representation-level 证据 —— 证明"真·解耦"做到了,这在 paper 里有独立价值。

---

## 九、风险清单 + 预防

| 风险 | 机制 | 预防 |
|---|---|---|
| z_sty 坍缩 | L_orth 天然解是 z_sty=0 向量,trivial | **L_sty_aux 强制 z_sty 含 domain 信号** |
| 参数过拟合 | 2× 参数 + 小数据集 | 加 weight decay,监控 train/test acc gap |
| 通信压力 | 两个 encoder 都聚合 | **选项 Y:encoder_sty 本地私有**(不聚合) |
| 显存 OOM | 双前向 × 双反向 | batch_size 减半或 grad accumulation |
| 训练时间 × 2 | 双 pipeline | 接受,或 freeze encoder_sty 跑一段再解冻 |
| 超参调优爆炸 | 新增 λ_sty_aux | 固定 λ_sty_aux=0.3,只扫 λ_orth |

---

## 十、和 Calibrator 方案的对比

| 维度 | FedDSA-Calibrator | **FedDSA-DualEncoder** |
|---|---|---|
| 哲学 | 接受泄漏,**利用**它 | 从源头**杜绝**泄漏 |
| 架构改动 | head 旁加 MLP | **双 encoder** |
| 代码量 | ~50 行 | ~150 行 |
| 参数量 | 1.01× | **2×** |
| 显存 | 1× | 1.5× |
| 训练时间 | 1.05× | **1.5~1.8×** |
| 通信量 | 1× | 1~2× |
| Novelty | 中(calibrate 思想已有)| **高**(FL 里独立双 trunk 少见) |
| probe 下降 | 不下降 | **显著下降**(核心证据) |
| 预期 accuracy 涨幅 | +0.4~2.4 pp | **+0.9~2.9 pp** |

**DualEncoder 的优势**:
- Novelty 更高:独立双 trunk 在 FL 里几乎没人做(原因就是参数量 2×,大家不愿意)
- Probe 层面有**独立叙事**:"我们做到了真·解耦"
- Accuracy 上限更高(理论上)

**Calibrator 的优势**:
- 实施快,代价小
- FL 通信压力无变化
- 对已有 FedDSA 代码改动小

---

## 十一、实施步骤

```markdown
Step 1: 实现 feddsa_dual_encoder.py (~150 行)
  - 继承 AlexNetEncoder,new 两份
  - forward 双 pipeline
  - Client.train_step: L_CE + L_orth + L_sty_aux
  - Server.aggregate: 选项 Y (只聚合 sem)

Step 2: Config
  - office/feddsa_dual_encoder_r200.yml
  - pacs/feddsa_dual_encoder_r200.yml

Step 3: 验证 (CLAUDE.md 强制)
  - ast.parse 语法检查
  - 单元测试: 确认 encoder_sem 和 encoder_sty 的参数梯度 **互相独立**
    - L_CE.backward() 后, encoder_sty.conv1.weight.grad 应为 None(或 0)
    - L_orth.backward() 后两者都有梯度
  - codex review

Step 4: 启动 Office 3-seed pilot (EXP-126_dual_encoder_office)
  - seeds: 2, 15, 333
  - 预计 4h × 3 seed (因为 2× 训练时间),并发 ~5h wall

Step 5: 判决
  - Office 3-seed mean AVG Best > 90.58 → ship,进 PACS
  - probe_sty_class MLP-256 < 0.3 → representation evidence 成立(即使 acc 没赢也有 paper 价值)
  - 两者都 fail → kill,不继续调
```

---

## 十二、决策定位

### 和其他方案的优先级

| 优先级 | 方案 | 理由 |
|:-:|---|---|
| 1 | **跨域 class prototype anchor** | 最轻,novelty 强(首选主攻) |
| 2 | F2DC 选项 C spatial DFD | novelty 最高,但代价大 |
| 3 | **FedDSA-DualEncoder**(本方案) | 兜底 B:真·解耦,参数 2× 换物理保证 |
| 4 | FedDSA-Calibrator | 兜底 A:最轻,但不解决根本问题 |

### 何时启动?

- 1 + 2 都失败时启动兜底
- **本方案(DualEncoder)在兜底里推荐优先于 Calibrator**,因为:
  - Novelty 更高(DualEncoder 在 FL 里少见)
  - Probe 证据叙事独立(Calibrator 没有)
  - 预期 accuracy 涨幅也更高

---

## 十三、一句话总结

**彻底物理隔离**:semantic 和 style 各有一个完整 AlexNet encoder,权重梯度完全不交叉,从架构层面杜绝 `L_CE` 梯度弥散到 style trunk。代价是参数 2×、训练时间 1.5×,但**probe_sty_class 会暴跌**(真正的解耦证据),accuracy 预期 +0.9~2.9 pp。和 Calibrator 方案对比:**DualEncoder 从源头治病,Calibrator 治标**。推荐在主攻失败后**优先启动 DualEncoder** 作为兜底。
