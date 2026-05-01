# Functional 视角看 FL global 塌缩 — 叙事方案

> 2026-04-26 创建
> 目的: 提供一个跟 F2DC (CVPR'26) 不撞车的 narrative, 用 functional 指标证明 "global model 不可用",
> 而不是 F2DC 的几何 (SV decay) 视角.

---

## 1. 缘起 — 为什么要换 narrative

### 已有数据 (PACS, EXP-129 v7 per-client F2DC-style 诊断)

| 方法 | mode | ER_full | i/i | Sil | acc |
|------|:----:|:----:|:----:|:----:|:----:|
| FedBN | client local mean | 5.9 | 7.7 | 0.75 | **96.6%** |
| FedBN | global | 7.1 | 2.32 | 0.34 | 78.4% |
| orth_only | client local mean | 9.9 | 3.9 | 0.58 | 96.1% |
| **orth_only** | **global** | **1.1** | **0.19** | 0.00 | **17.6%** ⚠ |
| FDSE | client local mean | 10.7 | 3.2 | 0.52 | 96.0% |
| **FDSE** | **global** | **3.2** | 0.58 | -0.12 | **10.6%** ⚠ |

### 问题
- **F2DC paper Fig.1 同款几何指标 (SV decay)** 已被人做过, 我们再做就是复刻
- 但我们的数据有更直接的发现: **orth_only / FDSE 的 global model 在 PACS 上 acc 仅 17%/10% (随机=14%)**
- 这是真实的 functional failure, 比"几何塌缩"更有冲击力

### 核心区分
| 维度 | F2DC narrative | 我们的 narrative |
|------|:----:|:----:|
| 测什么 | 几何 (SV decay) | functional (能不能直接用 / 收敛 / 适应) |
| 量化 | 1 个图 | 4 个独立指标 |
| 论证强度 | "看起来塌" | "实际不能用" |

---

## 2. 4 个 functional 指标定义

### 指标 1: Direct Inference Accuracy Drop (DIAD) ★ 最直接

**定义**:
```
DIAD_method = mean over clients [
    acc(client_i_local_model on test_data_i)
    - acc(global_model directly on test_data_i)
]
```

**意义**: global model 能不能直接拿来推理 (不 fine-tune).

**PACS 实测 (从 v7 已有数据)**:
| 方法 | client local acc | global direct acc | DIAD |
|------|:----:|:----:|:----:|
| FedBN | 96.6 | 78.4 | **+18.2pp** |
| orth_only | 96.1 | 17.6 | **+78.5pp** ⚠️ |
| FDSE | 96.0 | 10.6 | **+85.4pp** ⚠️ |

**结论**: orth_only / FDSE 的 global 是"假"模型, 不能直接用; FedBN global 至少能跨域跑一下.

---

### 指标 2: New Client Adaptation Speed (NCAS)

**定义**:
```python
模拟新 client 加入:
  起点: 加载 global model
  数据: 一个全新 domain (held-out)
  fine-tune 到 acc >= threshold (比如 80%) 需要多少 round
NCAS_method = rounds_to_threshold
```

**意义**: 量化 F2DC 口头说的 "塌缩让 global 难 generalize 到新 domain", 用真实 fine-tune 速度证明.

**预期** (待跑):
- FedBN ~5 round (global 还行, 小修就好)
- orth_only ~15 round (global 塌, 要重新教)
- FDSE ~12 round

**实施代价**: 需要重新 fine-tune, 3 小时 (3 方法 × leave-one-out × ~30 round each).

---

### 指标 3: Server Aggregation Convergence (SAC)

**定义**:
```python
最后 50 round 的每一轮:
  4 个 client 上传的 backbone 参数
  inter_client_param_variance[round] = mean over layers [
      var across 4 clients of layer.weight
  ]
SAC_method = mean(inter_client_param_variance over last 50 rounds)
```

**意义**: client 上传的参数方差越大 → 聚合越 inconsistent → consensus 难达成.
F2DC 说 "great difficulty to obtain consensus updates", 我们用方差量化.

**实施代价**: 需要保留训练每 round 的 client snapshot (我们没保, 需要重训 1 个).

---

### 指标 4: Cross-Client Transferability Matrix (CCTM)

**定义**:
```
matrix[i][j] = acc(client_i's local_model on client_j's test_data)

4×4 矩阵:
  对角线: 各 client 自己域 acc (健康基准)
  非对角: 跨域可传递性
```

**意义**: 直接证明 "local model 也不跨域", 支持 "global 是没人用的中转站".

**预期 PACS**:
- FedBN: 对角 96-99, 非对角 ~50-70 (FedBN 强本地化, 跨域中等)
- orth_only: 对角 96, 非对角 ~30-50 (强解耦, 跨域差)
- FDSE: 对角 96, 非对角 ~30

**实施代价**: 现有 ckpt 直接能算, 1 小时.

---

## 3. 4 指标 vs F2DC 对比

| 指标 | 我们说啥 | F2DC 怎么说同样的事 | 我们的优势 |
|------|------|------|:----:|
| DIAD | "global 直接推理已死" | (没说, F2DC 没量化这个) | ⭐⭐⭐ 完全独家 |
| NCAS | "新 client 加入要费劲适应" | "塌缩让 global 没法 generalize" (口头, 没量化) | ⭐⭐⭐ 量化 vs 口头 |
| SAC | "聚合 inconsistent" | "great difficulty to obtain consensus" (口头) | ⭐⭐ 量化 vs 口头 |
| CCTM | "local 也不跨域, 不只是 global 问题" | (F2DC 没做) | ⭐⭐⭐ 独家发现 |

**4 个指标都比 F2DC 更具体, 更可量化, 更有 functional 解读力**.

---

## 4. 完整 narrative 逻辑链

```
Step 1 [观察]: 
  PACS orth_only AVG Best 80.64 胜 FDSE 79.91 +0.73
  Office orth_only AVG Best 89.09 输 FDSE 90.58 -1.49
  → 数字看着不错, 但训练动力学有异常 (best round 早 + 后期退化)

Step 2 [怀疑]:
  global model 内部出了问题, 但传统 acc 数字看不出
  
Step 3 [诊断 — 4 个 functional 指标]:
  - DIAD: orth_only global 推理仅 17%, FedBN 78%
  - NCAS: orth_only 新 client 适应慢 3×
  - SAC: orth_only 聚合时 client 参数方差 2× FedBN
  - CCTM: orth_only local 跨域 acc 仅 30%
  
Step 4 [结论]:
  "elimination-based" FL 方法 (orth_only, FDSE) 的 global model 是
  "functional dead end" — 只是 client 之间共享参数的中转站, 
  本身没有可用 representation.
  
Step 5 [提出 fix]: (我们要试的方向)
  - Anti-collapse server-side regularization
  - Domain-aware aggregation
  - Calibration-based (不丢 domain feature)

Step 6 [验证]: fix 后 4 指标全面改善 + acc 涨
```

---

## 5. Novelty 论证 (跟现有工作差异)

| 已有工作 | 它做了什么 | 我们的差异 |
|------|------|------|
| F2DC (CVPR'26) | SV decay 量化几何塌缩 | 我们用 4 个 functional 指标量化"实际不可用" |
| FDSE (CVPR'25) | 提出层分解 elimination | 我们指出 elimination 路线的 functional cost |
| FedBN (ICLR'21) | BN 本地化避免聚合冲突 | 我们用 DIAD 量化 FedBN 实际 best 的 reason |
| FedSeProto (ECAI'24) | 信息瓶颈擦风格 | 我们的 functional 指标暴露这类方法 global 失功 |

**我们独家**: 4 个 functional 指标 + "global model is dead but local works" 这个反直觉发现.

---

## 6. 实施路线图 (按 ROI)

| 阶段 | 内容 | 时间 | 产出 |
|------|------|:----:|------|
| **P1** | DIAD 现有 ckpt 直接算 (PACS + Office) | 1h | 6 个数字 |
| **P2** | CCTM 现有 ckpt 算 4×4 矩阵 (PACS + Office) | 1h | 2 张矩阵图 |
| **P3** | 写 functional analysis 章节, 整理 4 指标定义 | 2h | paper 用素材 |
| **P4** | NCAS 重训 (PACS leave-one-out × 3 方法) | 3-4h | 适应曲线图 |
| **P5** | SAC 重训 1 个 PACS run, 保留 50 round client snapshot | 6h | 方差曲线 |

**P1+P2+P3 = 4h** 可以拿到主表 + 主图 + 章节素材, 足够 frame paper.

---

## 7. 关键决策点

### 风险 1: F2DC 没量化的指标, 不代表论文 reviewer 不知道
- 比如 DIAD = "global model 直接推理 acc" 可能太基础, 没人专门 frame 但很多人知道
- 缓解: 把 4 指标合在一起作为 **diagnostic suite**, novelty 在于"系统化地用 functional 视角"

### 风险 2: 4 指标都正向, fix 也涨, 但 venue 可能不买账
- "诊断+小幅 fix" 类 paper 通常发 workshop / second-tier
- top venue 还需要一个**新的 method contribution** (比如 anti-collapse loss 大涨 acc)

### 备选 narrative (如果 4 指标 narrative 不够强)
- 把 4 指标当成 **附录的 ablation tool**, 主 contribution 还是新 method
- 类似 paper "我们提出 method X (主), 用 functional 指标证明它不只是 acc 涨, 还修复了 global 失功 (附录支持)"

---

## 8. 对应代码 / 文件

- 现有 ckpt 路径 (lab-lry):
  - PACS: `task/PACS_c4/record/*.pth` (FedBN, orth_only, FDSE 各 1 seed)
  - Office: `fl_checkpoints/feddsa_s{2,15,333}_R200_best*` (orth_only 3 seed, sas / SAS variants)
- 现有诊断脚本: `FDSE_CVPR25/scripts/diagnostic/f2dc_diag_pacs.py` (v7, per-client + global)
- 待写脚本:
  - `f2dc_diag_office.py` (复制 pacs 版本, 改 dataset)
  - `functional_diag_diad_cctm.py` (新, P1+P2)
  - `functional_diag_ncas.py` (新, P4)
  - `functional_diag_sac.py` (新, P5, 需重训)

---

## 9. 跟之前 95 个实验的关系

- 不需要重训之前的 baseline (FedBN/orth_only/FDSE 的 R200 ckpt 已有)
- functional 诊断是**对已有 ckpt 的复用**, 不增加 GPU 预算
- 如果 functional 指标支持当前结论, 可写成 "method validation paper"
- 如果不支持 (比如 orth_only 也没 functional 优势), 转向找新方向

---

## 10. 立即下一步

1. 写 `functional_diag_diad_cctm.py` (P1+P2 合一脚本), 1 小时
2. 在 lab-lry 跑 PACS (现有 ckpt) → 6 个 DIAD 数字 + 3 个 CCTM 矩阵
3. 等 Office FedBN/FDSE ckpt 跑完 (lab-lry 上还在跑), 跑 Office 同样的诊断
4. 整理结果决定下一步 fix 方向 (anti-collapse / domain-aware agg / calibration)
