# ES-IBND 方案:前端层级解耦(Early-Split IN-BN Decorrelation for FedBN)

> **提问**:用 research-refine 打磨一下层级结构化解耦改进方向,完成后同步到知识笔记
> **Refine 结果**:GPT-5.4 xhigh 4 轮审核,**9.0/10 READY** ✅
> **Session**:`obsidian_exprtiment_results/refine_logs/2026-04-19_hierarchical-decouple_v1/`

---

## 1. 一句话是什么

> **在 AlexNet 前 2 层 BatchNorm 的位置,把一半通道改用 InstanceNorm(IN),另一半保持 BatchNorm(BN),然后强制两半的空间激活模式互相不相关**(pairwise spatial cosine decorrelation)。

不改 backbone 结构,不加新可训练参数,~80 行代码,作为 Plan A 末梢正交约束的**前端补充正则**。

---

## 2. 为什么 IN+BN 混合能分离 style/content?

### Normalization 的本质区别

| 操作 | 对什么统计 | 效果 |
|------|-----------|------|
| **BN**(Batch Norm)| 对整个 batch 每个 channel 算 μ/σ | 保留 batch 级别的信号(**类别/content**)|
| **IN**(Instance Norm)| 对**每个样本**每个 channel 算 μ/σ | 去除 per-image 的亮度/对比度/色调(**去 style**)|

### IBN-Net (Pan et al. ECCV 2018) 的发现

把一半通道走 IN,一半走 BN **并行**,在 domain generalization(DG)任务上有 **+2-3% 提升**(单机设置)。

**直觉**:让模型同时看到:
- IN 通道:**去风格** feature(style-invariant semantic)
- BN 通道:**保留** batch-level content

### 我们的创新

IBN-Net 只是"用 IBN 替换 BN"就完事,**没有显式正交约束**,也**没在 FL 设置测试过**。

我们:
1. 把 IBN 移植到 **FedBN 框架**(IN 通道 parameter-free 天然无 FL 同步问题)
2. 加 **spatial cos² 正交约束**,让 IN 通道和 BN 通道的空间激活**显式不相关**
3. **和 Plan A 末梢 cos² 并存**(末梢 Plan A 不变,只加前端)

---

## 3. 为什么叫 "Early-Split" 不叫 "Hierarchical"?

R2 reviewer 抓包:只替换前 2 个 BN **不算**深度 hierarchical(hierarchy 要跨多个 depth stages)。

**诚实措辞**:
- ❌ 原想叫 "Hierarchical IBN-Orthogonal Decoupling"(过度包装)
- ✅ 改叫 **"Early-Split IN-BN Decorrelation (ES-IBND)"**
  - Early-Split:只在 backbone 前 2 层做 channel 劈分
  - IN-BN:两种 normalization 并行
  - Decorrelation:施加空间 cos² 正交

这样 reviewer 不能抓"过度声称深度 hierarchy"。

---

## 4. 具体实现(~80 行)

### 第 1 步:IBN 模块(~30 行)

```python
class IBN(nn.Module):
    def __init__(self, channels, in_ratio=0.5):
        super().__init__()
        self.in_ch = int(channels * in_ratio)   # 前 50% 通道走 IN
        self.bn_ch = channels - self.in_ch      # 后 50% 通道走 BN
        self.instance_norm = nn.InstanceNorm2d(self.in_ch, affine=False)  # 0 参数
        self.batch_norm = nn.BatchNorm2d(self.bn_ch)                      # FedBN 本地
        self._cache_in = None
        self._cache_bn = None

    def forward(self, x):
        xin, xbn = torch.split(x, [self.in_ch, self.bn_ch], dim=1)
        out_in = self.instance_norm(xin)
        out_bn = self.batch_norm(xbn)
        self._cache_in, self._cache_bn = out_in, out_bn  # 存 cache 给 loss 用
        return torch.cat([out_in, out_bn], dim=1)
```

### 第 2 步:替换前 2 个 BN(~10 行)

```python
# AlexNetEncoder 里:
# 原:
self.bn1 = nn.BatchNorm2d(64)     # conv1 后
self.bn2 = nn.BatchNorm2d(192)    # conv2 后
# 新:
self.bn1 = IBN(64)                # 32 通道 IN, 32 通道 BN
self.bn2 = IBN(192)               # 96 通道 IN, 96 通道 BN
```

### 第 3 步:加 decorrelation loss(~20 行)

```python
def layer_decorr_loss(ibn):
    xin = F.normalize(ibn._cache_in.flatten(2), dim=2)   # [B, Cin, HW]
    xbn = F.normalize(ibn._cache_bn.flatten(2), dim=2)   # [B, Cbn, HW]
    corr = torch.bmm(xin, xbn.transpose(1, 2))           # [B, Cin, Cbn]
    return corr.square().mean()                          # 推向 0

# 在 Client.train() forward 后:
L_orth_layer = sum(layer_decorr_loss(ibn) for ibn in self.ibn_layers) / N_IBN
```

### 第 4 步:单测(~20 行)

- IBN 前向输出形状正确
- cache 正确保存
- loss 是正数且梯度能回传
- IN 通道的 per-channel spatial mean ≈ 0(验证 IN 确实做了归一化)
- `num_ibn=0` 时等价 Plan A(backward compat)

**总代码增量:约 80 行,0 新可训练参数。**

---

## 5. 核心 loss 公式(R2 locked,R3 confirmed)

```python
xin = F.normalize(ibn._cache_in.flatten(2), dim=2)   # [B, Cin, HW] 单位化
xbn = F.normalize(ibn._cache_bn.flatten(2), dim=2)   # [B, Cbn, HW] 单位化
corr = torch.bmm(xin, xbn.transpose(1, 2))           # [B, Cin, Cbn] cos 相似度
L_orth_layer = corr.square().mean()                  # 推向 0(正交)
```

**关键点**:
- `F.normalize` 已经单位化 → 不再除 `/HW`(R2 reviewer 指出这是 R1 bug)
- loss 值在 [0, 1] 可直接阈值解读(例如训练末应从 0.3 降到 < 0.1)
- 不依赖 spatial mean(R1 的 GAP bug 已修)

---

## 6. 总 Loss(Plan A 不变,只加前端)

```
L = L_CE
  + 1.0 × L_orth_end     (Plan A 末梢 cos²,不变)
  + 0.0 × L_HSIC         (继承 EXP-017 最优,HSIC 不利)
  + 0.5 × L_orth_layer   (NEW,前端)
```

**没有改 Plan A 的任何权重**,只新增一项前端正则。

---

## 7. 3-Way Ablation(铁证 isolation)

| 配置 | 作用 | 对比 |
|------|------|------|
| **A.1 Plan A** | orth_only(末梢 cos²)| baseline(82.31 Best / 81.17 Last)|
| **A.2 IBN-only** | IBN 替换,**不加** L_orth_layer | 看 IBN 结构本身贡献 |
| **A.3 ES-IBND** | IBN + L_orth_layer(OURS)| 主方法 |

**决断逻辑**:
- A.3 > A.2:**证明 decorrelation loss 有独立贡献**(不是只靠 IBN 结构)
- A.2 > A.1:**IBN 结构本身**的 DG 效应(即使不加新 loss)
- A.3 > A.1 但 A.2 ≈ A.1:**主功劳是新 loss**,IBN 只是 enabler

---

## 8. 主决策指标:AVG Last(不是 Best)

### 为什么不用 Best?

R3 reviewer 批评:
> "AVG Best cannot carry the paper by itself. If the gain is only in Best and not Last, reviewers will discount it."

- **AVG Best**:训练过程某轮 peak(可能只是运气)
- **AVG Last**:R=200 最后一轮(反映**收敛质量**)

### 主表报告标准(mean ± std over 3 seeds)

| Config | AVG Best | **AVG Last**(主) | ALL Best | ALL Last | Art | Cart | Photo | Sketch |
|--------|----------|-------------------|----------|----------|-----|------|-------|--------|
| Plan A | 82.31 ± ? | **81.17 ± ?** | ... | ... | ... | ... | ... | ... |
| IBN-only | ? ± ? | **?** | ... | ... | ... | ... | ... | ... |
| ES-IBND | ? ± ? | **?** | ... | ... | ... | ... | ... | ... |

**Worst-domain check**:检查 Sketch / Art 等 outlier domain 不能下跌 >1%(防止"平均赢但某 domain 崩")

---

## 9. vs FDSE(主 baseline CVPR 2025)

| | **FDSE** | **ES-IBND(我们)** |
|---|---|---|
| 层级方式 | 每层 DFE+DSE 结构分解 | 前 2 层换 IBN |
| 新参数 | DSE ≈ DFE/94(~10K 每层,累计 50K+)| **0** |
| 实现 | 改每个 conv 结构 + 新 grouped conv | 只改 normalization scheme |
| Decouple 机制 | KL 对齐 BN running stats | 空间 cos² decorrelation |
| Claim | "domain shift erasor"(擦除派)| "early-split regularizer"(正交派)|

**核心卖点**:**零新参数 + 极小代码**,比 FDSE 更轻量。

---

## 10. vs IBN-Net(Pan et al. ECCV 2018)

| | **IBN-Net** | **ES-IBND(我们)** |
|---|---|---|
| 场景 | 单机 DG | FL 跨域(FedBN)|
| IBN 位置 | 全骨干多处 | 前 2 层,轻量 |
| 损失 | 无显式正交 | **spatial cos² decorrelation**(新增)|
| FL 兼容性 | 未验证 | IN parameter-free,天然兼容 |

**创新点**:首次在 FL 设置验证 IBN + 加显式正交 loss。

---

## 11. vs Plan A(当前 baseline)

| | **Plan A** | **ES-IBND** |
|---|---|---|
| Decouple 位置 | 末梢 128d projection 一次 | 前 2 层 + 末梢 |
| 正交约束 | cos²(z_sem, z_sty) | L_orth_end + L_orth_layer |
| backbone 梯度 | 只间接(通过末梢回传)| 前端直接 |
| 新参数 | 0 | 0(ES-IBND 也 0)|

**关键 hypothesis**:Plan A 卡 ceiling 是因为 backbone 前端没有直接解耦梯度。ES-IBND 在前 2 层直接加梯度,理论上能突破。

---

## 12. 4 轮 refine 评分轨迹

| Round | Overall | Verdict | 本轮关键改动 |
|-------|---------|---------|-------------|
| 1 | 6.7 | REVISE | initial proposal,有致命 math bug |
| 2 | 8.0 | REVISE | 修 math bug(spatial cross-correlation)+ 收窄 claim + 加 IBN-only |
| 3 | 8.7 | REVISE | 锁 loss 公式(去 /HW)+ 改名 ES-IBND + 4 指标 |
| **4** | **9.0** | **READY ✅** | **paper hygiene 吃下**(AVG Last 主 + mean±std + AlexNet scope) |

轨迹:6.7 → 8.0 (+1.3) → 8.7 (+0.7) → **9.0 (+0.3) ✅**

---

## 13. 最大 bug(R1 被 reviewer 抓到)

### Bug:L_orth_layer 在 IN+GAP 下**数学退化**!

**R1 原公式**:
```python
in_feat_l = global_avg_pool(out_in)   # [B, C_in]  ← 这个永远是 0!
bn_feat_l = global_avg_pool(out_bn)
cos_sim = (normalize(in_feat) * normalize(bn_feat)).sum(dim=1)
L = (cos_sim ** 2).mean()
```

**Reviewer 指出**:
> `InstanceNorm2d(affine=False)` 使得每 channel 的 spatial mean **定义上就是 0**(这是 IN 的本质)。
> → `global_avg_pool(out_in) = 0` 恒成立
> → cos_sim = 0 / 0 → loss 无意义

**修复(R2 锁定)**:改用 **spatial cross-correlation**(保留 H×W 空间结构,不做 GAP):
```python
xin = F.normalize(out_in.flatten(2), dim=2)   # [B, Cin, HW] 沿 HW 维度单位化
xbn = F.normalize(out_bn.flatten(2), dim=2)   # [B, Cbn, HW]
corr = torch.bmm(xin, xbn.transpose(1, 2))     # [B, Cin, Cbn]
L = corr.square().mean()
```

**教训**:**任何正交/相关性 loss 写之前都要想"在 affine=False 的 normalization 下是否退化"**。以后写 loss 必须先算一遍极限值。

---

## 14. 验证节奏

| 阶段 | 时间 | 目的 |
|------|------|------|
| **M0 Sanity R=20** | **10 min** | 代码不崩 + L_orth_layer 下降 + 前 20 轮 AVG 不跌 |
| **M1 Full R=200**(PACS) | **24h 并行 3** | 3 configs × 3 seeds = 9 runs 主实验 |
| **M2 Office 附录**(ResNet-18)| 12h | 验证 ResNet backbone 也成立 |
| **总** | **36 GPU·h ≈ 1.5 天** | 快速可证伪 |

R=20 sanity 是关键 —— **10 分钟就能看趋势**,代码实现 + 单测通过后,sanity 如果崩就立刻 fallback,不浪费 24h。

---

## 15. 成功条件(铁证)

**主 claim**:
- ✅ **A.3 AVG Last ≥ 81.67**(+0.5 over Plan A 81.17)
- ✅ **A.3 AVG Last std ≤ 1.5**(稳定性)
- ✅ **A.3 > A.2 on AVG Last**(decorrelation loss 独立贡献)
- ✅ **worst-domain(Sketch/Art)不跌 >1%**(无 hidden crash)

**机制诊断**(零 GPU 成本):
- ES-IBND 末梢 cos²(z_sem, z_sty)**比 Plan A 低**(证明前端 decorrelation 传播)

---

## 16. 诚实 fail-stop(防过度工程化)

**R4 Reviewer 明确指示**:
> "If the gain is only marginal or only in Best, I would NOT add more modules; I would treat that as evidence that Plan A is already near ceiling in this regime."

换成白话:**如果 ES-IBND 失败,不要再发明新东西**。直接承认 Plan A 在 K=4 FedBN 下已到天花板,诚实写 negative result。

这是 refine 的**设计安全网**:有明确的失败判断标准,避免"又失败又加新模块"的无限循环(我们在 SCPR 时就差点这样)。

---

## ⚠️ 关键区分:ES-IBND vs SCPR(已证伪)

| | **SCPR**(刚证伪 EXP-095) | **ES-IBND**(READY) |
|---|---|---|
| 操作位置 | **分类端**(原型检索 alignment)| **特征端前端**(IBN + 正交)|
| 改什么 | InfoNCE 对齐目标 | backbone normalization + 新 loss |
| 碰 classifier? | 间接(通过 prototype)| **不碰** |
| 正交性 | 两者**完全正交**,可共存 | — |
| Lesson | 4 客户端下分类端 routing 不 work | — |

**两者完全正交**:SCPR 改下游 alignment target,ES-IBND 改上游 backbone normalization。即使 SCPR 失败,ES-IBND 仍有独立价值。

---

## 📎 相关文件

| 文件 | 内容 |
|------|------|
| `refine_logs/2026-04-19_hierarchical-decouple_v1/FINAL_PROPOSAL.md` | canonical 最终方案 |
| `refine_logs/2026-04-19_hierarchical-decouple_v1/REVIEW_SUMMARY.md` | 4 轮演化总结 |
| `refine_logs/2026-04-19_hierarchical-decouple_v1/score-history.md` | 评分轨迹 6.7→8.0→8.7→9.0 |
| `FDSE_CVPR25/algorithm/feddsa_scheduled.py` | 待添加 IBN 类和 L_orth_layer |

---

## 🚀 下一步 roadmap

```
1. 实现 ~80 行代码(IBN + decorrelation loss + config)
2. 单测(~4 个测试覆盖形状/数值/梯度/兼容)
3. codex exec 代码 review(抓 bug)
4. M0 Sanity R=20(10 min)看前几轮趋势
5. M1 Full R=200 × 3 configs × 3 seeds(24h 并行)
6. 结果判决:
   - ✅ A.3 AVG Last ≥ 81.67 且 A.3 > A.2 → 写论文
   - ❌ 失败 → 接受 Plan A ceiling,不加新模块
```

---

*记录时间:2026-04-19 18:00*
*来源:Claude Code 对话 + 4 轮 GPT-5.4 xhigh refine*
