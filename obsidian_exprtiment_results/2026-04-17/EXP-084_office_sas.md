# EXP-084 | Office 方案 A: Style-Aware Semantic Head Aggregation

## 基本信息
- **日期**: 2026-04-17 启动 / 2026-04-18 结果
- **算法**: feddsa_scheduled (mode=0 + sas=1)
- **服务器**: Lab-lry GPU 1
- **状态**: 🔄 运行中

## 动机

Office 上 orth_only LR=0.05 AVG Best 89.44 vs FDSE 90.58（-1.14%），差距主要在 **Caltech 域**（74.1% vs 77.7%，-3.6%）。
Caltech 是 "in-the-wild 自然图像"，风格与 Amazon/DSLR/Webcam 差异大 → 是 style outlier。

**假设**：服务器端按 style similarity 个性化 `semantic_head` → Caltech 收到的聚合权重偏向自己 + 风格相近的 client，而非被均值稀释 → Caltech 受益。

## 方案 A 机制（服务器端 only，不改损失）

每轮 Server.pack() 时：
1. 收集所有 client 的 `semantic_head` state（已有）
2. 对 target client i，计算其 style_proto 与其他 client style_proto 的 cosine 相似度 `sim(i, j)`
3. `softmax(sim / τ)` 得到权重 `w_{ij}`
4. `personalized_sem[i] = Σ_j w_{ij} · semantic_head[j]` 替换 global model 的 semantic_head
5. 把 personalized model 发给 client i

**其他参数（DFE backbone, classifier）仍做标准 FedAvg**。

## 关键区别（vs 之前失败方案）

| 维度 | 078a (MSE anchor) | 078c (MSE+alpha) | **本方案 A** |
|------|-------------------|------------------|-------------|
| 改动位置 | 损失函数 | 损失函数 | **服务器聚合 only** |
| 引入梯度冲突 | ✅ 是（R120+ 崩） | ✅ 是 | ❌ 否（client 训练不变）|
| 原型是否漂移 | ✅ 是（MSE 锚点追漂原型） | ✅ 是 | — （不用原型做 loss） |
| 之前结果 | R142 kill 76.7% | R200 崩 75.21% | 待验证 |

## 配置

| 参数 | 值 | 说明 |
|------|---|------|
| seed | 2, 15, 333 | 对齐 FDSE EXP-051 |
| LR | 0.05 | 已证实稳定 |
| R | 200 | - |
| sm | 0 | orth_only 基础 |
| **sas** | **1** | **启用方案 A** |
| **sas_tau** | **0.3** | softmax 温度，待 ablation |
| se | 1 | 保存 best checkpoint |

## 预期结果

**成功标准（任一满足即有效）**：
1. Caltech per-domain AVG ≥ orth_only baseline **+2%**（>= 76.1）→ 方案 A 真正帮了 outlier 域
2. Office 整体 AVG Best 3-seed **≥ 90.0**（超 baseline 89.44，逼近 FDSE 90.58）
3. 稳定性 drop < 1%（保持 LR=0.05 水平）

**可能的失败模式**：
- Caltech style 本身变化太大，聚合 outlier 反而降噪不足 → 无提升
- sas_tau=0.3 不够 sharp → 等同均值聚合（需要 ablation）

## 部署

| 服务器 | GPU | seeds | ETA |
|--------|-----|-------|-----|
| Lab-lry | GPU 1 | 2, 15, 333 | ~2h (Office E=1) |

## 结果（待填）

### Office 3-seed (2/15/333) R200

| seed | ALL Best | ALL Last | AVG Best | AVG Last | drop |
|------|---------|---------|---------|---------|------|
| 2 | - | - | - | - | - |
| 15 | - | - | - | - | - |
| 333 | - | - | - | - | - |
| **mean** | - | - | - | - | - |

### vs baseline (orth_only LR=0.05)

| 指标 | orth_only (baseline) | +sas (方案 A) | Δ |
|------|---------------------|--------------|---|
| ALL Best | 83.87 | - | - |
| ALL Last | 83.34 | - | - |
| AVG Best | 89.44 | - | - |
| AVG Last | 88.71 | - | - |

### Per-domain AVG Best (最关键 — Caltech 是否提升?)

| 方法 | Caltech | Amazon | DSLR | Webcam |
|------|---------|---------|------|--------|
| FDSE (EXP-051) | 77.7 | 89.5 | 99.2 | 93.3 |
| orth_only baseline | 74.1 | 88.8 | 100.0 | 91.9 |
| **+ sas (方案 A)** | - | - | - | - |

## 下一步

- 训练完 → 提取 per-domain acc → 决定是否扩到 PACS
- 如果 Caltech 提升 > 2% → 扩到 PACS（看 Art 是否也受益）
- 如果无提升 → ablation sas_tau ∈ {0.1, 1.0} 再看
