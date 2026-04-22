# 大白话 FedDSA-Identifiable 方案

**时间**: 2026-04-21
**一句话**: 给图片换风格,然后逼模型"认出换风格之前和之后是同一个内容",这样 z_sem 就只学到"是什么",风格被挤到 z_sty。

## 核心比喻

> **"同一只狗,不管拍成照片、画成素描、做成卡通、涂成油画,模型都得认出这是同一只狗。"**

- 如果 z_sem 能把这 4 个风格版本"聚成一团",其他狗和别的类"推远",
- 那数学上 z_sem **必然**只记录了"是什么"(class),不记录"什么风格"(domain),
- 风格信息被逼到 z_sty 里 → 自动解耦。

## 为什么之前(orth_only)不够

- 之前只有 `L_orth = cos²(z_sem, z_sty)`,让两个向量**几何方向垂直**。
- 但"几何垂直"不等于"信息互斥",两个向量完全可以同时含 class 信息,只是方向不同。
- **实测**: linear probe(z_sty→class) = 0.24 看起来成功,MLP-256 = 0.71 一挖就暴露。

## 新方案怎么做 (5 步)

### 步骤 1: 风格仓库 (符合 FedDSA 原叙事)
- 每个 client 把自己域的风格参数 `(μ_k, σ_k)` 上传(就是特征的均值和方差)
- 服务器合成全局 `style_pool = {(μ_k, σ_k)}_{k=1..K}`
- 通信极轻 — 只传 2 × 128 维向量

### 步骤 2: 跨风格数据增强 (AdaIN)
训练时,随机做:
```
x_styled = AdaIN(x, μ_j, σ_j)   # μ_j, σ_j 从风格仓库随机采样
```
拿 client A 的图,贴上 client B 的风格 — 相当于"让狗从照片变素描"。

### 步骤 3: 双 view 编码
```
h_original = encoder(x)         → z_sem_A, z_sty_A
h_styled   = encoder(x_styled)  → z_sem_A_styled, z_sty_A_styled
```

### 步骤 4: InfoNCE 强制对齐 ⭐核心
```
L_InfoNCE: z_sem_A 和 z_sem_A_styled 拉近 (同一张图的两个风格版本)
           z_sem_A 和 z_sem_B 推远    (不同类)
```
这一项就是逼"换风格后 z_sem 不能变" → z_sem 只能编码 class。

### 步骤 5: 完整 loss
```
L_total = L_CE(z_sem, y)              ← 主任务
        + L_CE(z_sem_styled, y)        ← dual view 分类
        + 1.0 · L_InfoNCE              ← ★ 强制跨风格对齐
        + 0.1 · L_orth                 ← 软正交 (辅助,权重降低)
```

## 为什么 paper 能过 reviewer

**von Kügelgen NeurIPS 2021 Theorem 4.4** 有数学证明:
- 只要训练中用 style-only augmentation + 对比损失
- 学到的 z_sem **可证明 block-identify 到 content 变量**,up to 光滑可逆变换
- 意思就是: z_sem 数学上等同于真正的"内容变量",风格信息被完全挤出去

**Matthes NeurIPS 2023** 延伸到 SupCon (我们 PACS 有 label 可用)

## 预期效果

| 指标 | 现在 (orth_only) | 预期 (新方案) |
|------|:---------------:|:-----------:|
| probe_sty_class linear | 0.24 | **0.05-0.15** |
| probe_sty_class MLP-64 | 0.69 | **0.20-0.30** |
| probe_sty_class MLP-256 | 0.71 | **0.30-0.45** |
| PACS AVG Best | 80.64 | **81.5-83.0** |

## 和 FedDSA 原叙事的关系

| 原 FedDSA 要素 | 新方案对应 |
|--------------|----------|
| 双头解耦 z_sem / z_sty | ✅ 保留 |
| 风格仓库跨 client 共享 | ✅ 完全保留 (更突出) |
| AdaIN 增强 | ✅ 保留 (现在升级为 image-space 操作) |
| 语义原型 | ✅ z_sem 经 Theorem 4.4 可证是 "class prototype" 的良好载体 |
| 正交损失 L_orth | ⚠️ 权重降到 0.1,作为 regularizer |
| **新增**: InfoNCE 跨风格对齐 | ★ 理论核心 |

**没改题目方向**,只是把"为什么双头解耦有用"从"靠正交损失"(失败) 改成"靠对比学习"(有定理证明)。

## 关键风险 & 应对

| 风险 | 应对 |
|------|------|
| PACS photo↔sketch 大跨度 AdaIN 可能改 class 边界 | 先在同风格族内 AdaIN,再逐步扩大 |
| PACS 只 9k 样本,InfoNCE 需大 batch | memory bank 技巧 |
| AdaIN 位置错了 (深层会伤 acc) | 必须在**图像空间**或浅层 (像 PARDON 做法) |

## 实现时间

| 周 | 内容 |
|---|------|
| W1 | AdaIN image-space + InfoNCE 实现 + 单 seed smoke |
| W2 | 跨 client 风格仓库 + 跑 PACS seed=2 |
| W3 | 3-seed + Moyer 0/1/2/3 层 probe sweep |
| W4 | Ablation (去 AdaIN / 去 InfoNCE) + FediOS baseline 对照 |

## 论文 novelty 定位

- **方法**: FL 首次基于 von Kügelgen identifiability 理论设计的双头解耦
- **评估**: FL 首次用 Moyer 2018 gold-standard probe sweep (0/1/2/3 层)
- **诊断**: 证实 FediOS / CDANN 类方法在严格 probe 下都破 → 方法 + 评估双 contribution

---

## 一句话总结记住

> **"用风格增强 + 对比学习,逼 z_sem 只认'是什么'、不认'什么风格'。风格被迫进 z_sty,数学上可证解耦。"**
