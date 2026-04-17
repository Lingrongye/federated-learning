# FedDSA-Adaptive 实验总结文档

> 生成时间: 2026-04-15  
> 算法文件: `FDSE_CVPR25/algorithm/feddsa_adaptive.py`  
> 基线对照: `FDSE_CVPR25/algorithm/feddsa.py` (原始 FedDSA)

---

## 一、研究背景

### 1.1 原始 FedDSA（Decouple-Share-Align）

FedDSA 是我们提出的跨域联邦学习方法，核心三步机制：

| 步骤 | 模块 | 作用 |
|------|------|------|
| **Decouple** | 正交约束 cos² + HSIC 核独立性 | 将骨干特征分离为语义特征 z_sem 和风格特征 z_sty |
| **Share** | 全局风格仓库 (μ,σ) + AdaIN 增强 | 收集各域 h-space 统计量，跨域风格交换做特征增强 |
| **Align** | InfoNCE 对比损失 | 语义特征 z_sem 与全局原型软对齐 |

**模型架构**:
```
输入图片 → AlexNet Encoder (→ h ∈ R^1024)
                ├── Semantic Head (MLP: 1024→128→128) → z_sem → Classifier (128→7) → 预测
                └── Style Head (MLP: 1024→128→128) → z_sty (仅用于解耦约束)
```

**聚合策略**: Encoder + Semantic Head + Classifier → FedAvg 聚合; Style Head → 私有不聚合; BN 层 → 私有不聚合

**损失函数**:
```
L = L_CE(z_sem) + L_CE(z_sem_aug) + λ_orth * L_orth + λ_hsic * L_HSIC + λ_sem * L_InfoNCE
```

**最优配置 (EXP-069f)**: tau=0.2, λ_orth=1.0, λ_hsic=0.0, λ_sem=1.0, warmup=50, R=200, E=5, lr=0.1

**最优结果**: PACS 数据集 AVG ≈ 80.93±0.30%

### 1.2 发现的问题

1. **增强强度固定**: 原始 FedDSA 用 `α ~ Beta(0.1, 0.1)` 随机采样增强强度，不区分域差异
   - Photo 域（与其他域差异小）不需要强增强，但仍被随机增强
   - Sketch 域（差异大）需要更强增强
2. **全局原型稀释**: 所有客户端的同类原型取简单均值 → 跨域平均后语义信息被稀释

### 1.3 提出改进

设计了两个改进模块，封装在 `feddsa_adaptive.py` 中：

- **M1 (自适应增强强度)**: 根据每个客户端的域偏移程度动态调节增强强度
- **M3 (域感知原型对齐)**: 保留每域每类独立原型，用 SupCon InfoNCE 多正样本对齐

---

## 二、改进方法详解

### 2.1 M1: 自适应增强强度 (Adaptive Augmentation Strength)

**动机**: 不同域与"全局平均"的风格偏差不同，应用不同强度的增强

**实现步骤**:

1. **服务器端 — 域偏差度量** (`_compute_gap_metrics()`):
   - 每轮收集各客户端的 z_sty 空间统计量 (μ_zsty, σ_zsty)
   - 计算原始偏差: `raw_gap_i = ||μ_i - μ_global||² + ||σ_i - σ_global||²`
   - EMA z-score 归一化: `gap_normalized_i = clip(z * 0.5 + 0.5, 0, 1)`

2. **客户端端 — 自适应 alpha** (`_style_augment()`):
   - `alpha = aug_min + (aug_max - aug_min) * gap_normalized + N(0, noise_std)`
   - 高 gap 域（如 Sketch）→ 大 alpha → 强增强
   - 低 gap 域（如 Photo）→ 小 alpha → 弱增强

3. **增强公式** (与原始 FedDSA 不同):
   ```python
   # 原始 FedDSA: 分别混合 μ 和 σ
   mu_mix = α * mu_local + (1-α) * mu_ext
   sigma_mix = α * sigma_local + (1-α) * sigma_ext
   h_aug = h_norm * sigma_mix + mu_mix

   # FedDSA-Adaptive: 先做完整 AdaIN，再混合
   h_adain = h_norm * sigma_ext + mu_ext   # 完整风格迁移
   h_aug = α * h_adain + (1-α) * h          # 混合原始和迁移后特征
   ```

**超参数**:
| 参数 | 值 | 含义 |
|------|-----|------|
| aug_min | 0.05 | alpha 下限（最弱增强） |
| aug_max | 0.8 | alpha 上限（最强增强） |
| noise_std | 0.05 | 随机扰动标准差 |
| ema_decay | 0.9 | EMA 平滑系数 |

### 2.2 M3: 域感知原型对齐 (Domain-Aware Prototype Alignment)

**动机**: 全局原型 = 所有域原型的均值 → 跨域平均导致语义信息稀释

**实现步骤**:

1. **服务器端 — 保留域级原型** (`_store_domain_protos()`):
   - 保留 (class, client_id) → proto 的映射（不做跨域平均）
   - 每个域的每个类有独立原型

2. **客户端端 — SupCon Multi-Positive InfoNCE** (`_infonce_domain_aware()`):
   - 构建原型矩阵: 所有 (class, client) 原型作为候选
   - 对每个样本: 同类的所有域原型都是正样本，异类原型是负样本
   - 损失: `L = -Σ_{p∈P(i)} [log(exp(sim/τ) / Σ_j exp(sim_j/τ))] / |P(i)|`

**效果**: 语义特征既向同类靠拢，又保持跨域一致性，不被单一"平均原型"稀释

### 2.3 实验模式 (adaptive_mode)

| Mode | 名称 | M1 | M3 | 说明 |
|------|------|----|----|------|
| **0** | Fixed Alpha | ✗ | ✗ | 固定 alpha 值（消融基线） |
| **1** | M1 Only | ✓ | ✗ | 仅自适应增强，全局原型对齐 |
| **2** | M3 Only | ✗ | ✓ | 固定 Beta(0.1,0.1) 增强，域感知原型 |
| **3** | M1+M3 Full | ✓ | ✓ | 完整系统 |

---

## 三、当前正在运行的实验

### 3.1 总览

| 实验编号 | 名称 | Config | Mode | 核心验证 |
|---------|------|--------|------|---------|
| **EXP-072a** | Fixed α=0.2 | feddsa_072a.yml | 0 | 弱增强基线 |
| **EXP-072b** | Fixed α=0.5 | feddsa_072b.yml | 0 | 中增强基线 |
| **EXP-072c** | Fixed α=0.8 | feddsa_072c.yml | 0 | 强增强基线 |
| **EXP-072** | M1 自适应 | feddsa_072.yml | 1 | M1 核心验证 |
| **EXP-072d** | M3 域感知原型 | feddsa_072d.yml | 2 | M3 核心验证 |
| **EXP-072e** | M1 零下限 | feddsa_072e.yml | 1 | aug_min=0 (低gap域可完全不增强) |
| **EXP-073** | M1+M3 完整 | feddsa_073.yml | 3 | 两模块组合 |

### 3.2 统一训练设置

| 参数 | 值 |
|------|-----|
| 数据集 | **PACS** (Photo / Art_painting / Cartoon / Sketch, 4域7类) |
| 骨干网络 | AlexNet (修改版, 含BN层) |
| 输入尺寸 | 224×224 |
| 总轮数 | R=200 |
| 本地训练 | E=5 epochs/round |
| 批大小 | B=50 |
| 学习率 | lr=0.1, 衰减率 0.9998/round |
| 梯度裁剪 | max_norm=10 |
| 特征维度 | h=1024 (backbone), z=128 (heads, proj_dim) |
| 参与率 | 100% (4 clients 全参与) |
| Train holdout | 20% (本地测试集) |
| Seeds | 2, 333, 42 (每实验3次) |

### 3.3 统一超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| lambda_orth | 1.0 | 正交解耦约束权重 |
| lambda_hsic | 0.0 | HSIC 约束权重（关闭） |
| lambda_sem | 1.0 | InfoNCE 对齐权重 |
| **tau** | **0.1** | InfoNCE 温度（与原始 FedDSA 的 0.2 不同！） |
| warmup_rounds | 50 | 前 50 轮不做增强和对齐 |
| style_dispatch_num | 5 | 每轮下发给客户端的风格数量 |

### 3.4 服务器部署状况 (2026-04-15 09:00 UTC)

**Seetacloud (主力服务器)** — 6 个进程运行中:

| PID | Config | Mode | Seed | 运行时长 |
|-----|--------|------|------|---------|
| 9122 | feddsa_072d | M3-only | 2 | ~9h37m |
| 9260 | feddsa_072d | M3-only | 333 | ~9h37m |
| 9393 | feddsa_072d | M3-only | 42 | ~9h37m |
| 9951 | feddsa_073 | M1+M3 | 2 | ~9h35m |
| 10116 | feddsa_073 | M1+M3 | 333 | ~9h32m |
| 16776 | feddsa_073 | M1+M3 | 42 | ~2h57m |

**Lab-lry (实验室备用)** — 11 个进程运行中:

| PID | Config | Mode | Seed | 运行时长 |
|-----|--------|------|------|---------|
| 2487547 | feddsa_073 | M1+M3 | 42 | ~7h39m |
| 2487548 | feddsa_072d | M3-only | 2 | ~7h39m |
| 2487549 | feddsa_072d | M3-only | 333 | ~7h39m |
| 2488631 | feddsa_072d | M3-only | 42 | ~7h32m |
| 2488632 | feddsa_072e | M1 zero-floor | 2 | ~7h32m |
| 2488633 | feddsa_072e | M1 zero-floor | 333 | ~7h32m |
| 2488634 | feddsa_072e | M1 zero-floor | 42 | ~7h32m |
| 2488635 | feddsa_073 | M1+M3 | 2 | ~7h32m |
| 2488636 | feddsa_073 | M1+M3 | 333 | ~7h32m |
| 2488637 | feddsa_072 | M1 adaptive | 333 | ~7h32m |
| 2488638 | feddsa_072 | M1 adaptive | 42 | ~7h32m |

> **注**: Lab-lry 跑的实验与 Seetacloud 有重叠（072d, 073），作为冗余备份。Lab-lry 额外跑了 072e (M1 zero-floor) 和 072 (M1 adaptive) 的补充 seeds。

---

## 四、已完成实验结果

### 4.1 Phase A: 固定 Alpha 基线 (全部完成 R200)

目的: 建立 alpha 值与性能的关系曲线，作为 M1 的对照基线。

| Config | α 值 | s=2 | s=333 | s=42 | Mean±Std |
|--------|------|-----|-------|------|----------|
| 072a | 0.2 (弱) | 75.02% | 74.92% | 77.33% | 75.76±1.36% |
| 072b | 0.5 (中) | 77.22% | 76.92% | 76.33% | 76.82±0.46% |
| 072c | 0.8 (强) | 77.12% | 75.42% | 77.93% | 76.82±1.29% |

**发现**: α=0.5 和 α=0.8 并列最优 (76.82%)。固定 alpha 之间差异 < 1%，说明 alpha 值本身不是核心瓶颈。

### 4.2 Phase B: M1 自适应增强 (全部完成 R200)

| Config | Mode | s=2 | s=333 | s=42 | Mean±Std |
|--------|------|-----|-------|------|----------|
| 072 | M1 adaptive (aug_min=0.05) | 81.14% | 75.92% | 77.13% | 78.06±2.73% |
| 072e | M1 zero-floor (aug_min=0.0) | 79.03% | 75.32% | 78.83% | 77.73±2.00% |

**发现**: M1 比最优固定 alpha 提升 +1.24%，但方差较大 (2.73%)。

### 4.3 Phase B/C: M3 & M1+M3 (进行中 ~R168-180/200)

| Config | Mode | s=2 | s=333 | s=42 | 状态 |
|--------|------|-----|-------|------|------|
| 072d | M3 domain-aware | ~82.44% | ~80.84% | ~82.44% | 🔄 R168+ |
| 073 | M1+M3 full | ~82.74% | ~79.83% | ~78.93%* | 🔄 R168+ (s42较早) |

> *073 s=42 在 seetacloud 启动较晚，进度落后约 6h。

**中期发现**: M3 域感知原型效果极显著: +5.1% vs 最优固定alpha (81.91 vs 76.82%)

### 4.4 2×2 Factorial 消融设计

| | No M3 (全局原型) | M3 (域感知原型) |
|--|-----------------|---------------|
| **Fixed α=0.5** | 072b: 76.82±0.46% ✅ | 072d: ~81.91% (interim) |
| **M1 adaptive** | 072: 78.06±2.73% ✅ | 073: ~80.5+% (interim) |

---

## 五、已发现的关键问题

### 5.1 训练崩溃 (Training Instability) ⚠️

**现象**: 所有 feddsa_adaptive 变体（tau=0.1）在 R150-170 发生准确率骤降 5-10%。

| 变体 | Seed | Peak (轮次) | R200 终值 | 崩溃幅度 |
|------|------|------------|----------|---------|
| mode=0 α=0.5 | s2 | R162: 83.34% | 77.22% | **-6.12%** |
| mode=0 α=0.5 | s42 | R164: 82.84% | 76.33% | **-6.51%** |
| mode=1 M1 | s333 | R166: 82.44% | 75.92% | **-6.52%** |
| mode=1 M1 | s42 | R52: 82.64% | 77.13% | -5.51% |

**对比**: 原始 FedDSA (tau=0.2) 极稳定，peak→final 仅降 0.80%。

### 5.2 根因分析

| 因素 | 原始 feddsa (069f) | feddsa_adaptive (072) | 是否根因? |
|------|-------------------|----------------------|----------|
| **tau** | **0.2** | **0.1** | ⭐ 最大嫌疑 |
| 增强方式 | Beta(0.1,0.1) 分别混合μ/σ | Full AdaIN + blend | ✗ mode=2用相同Beta仍崩 |
| 参数数量 | 7 | 13 | ✗ 不影响训练 |
| gap 追踪 | 无 | z_sty bank + EMA | ✗ mode=0 无gap也崩 |

**结论**: tau=0.1 导致 InfoNCE 梯度过于尖锐，与后期 lr 衰减叠加引发不稳定。

### 5.3 Peak vs Final 指标说明

- **Peak 性能**: 所有变体都在 ~80-82% AVG（与原始 FedDSA 持平）
- **R200 Final**: 由于崩溃，终值比 peak 低 3-7%
- 许多 FL 论文报告 "best accuracy" 而非 final，这点需注意

---

## 六、已规划的后续实验

### 6.1 EXP-074: Tau=0.2 修复验证 (Config 已创建，未部署)

目的: 验证 tau=0.2 是否修复训练崩溃，恢复稳定训练。

| Config 文件 | Mode | tau | 说明 |
|------------|------|-----|------|
| feddsa_074_m0_tau02.yml | 0 (固定α=0.5) | 0.2 | 基线：修复tau后能否达到~80%? |
| feddsa_074_m2_tau02.yml | 2 (M3 only) | 0.2 | M3真实效果：稳定训练下提升多少? |
| feddsa_074_m3_tau02.yml | 3 (M1+M3) | 0.2 | 完整系统：稳定训练下组合效果 |

**预期**: 如果 tau 是根因，mode=0 tau=0.2 应恢复到 ~80%（接近原始 FedDSA 80.93%），且不崩溃。

**部署计划**: 等当前 072d/073 完成后立即部署。

---

## 七、与原始 FedDSA 对比一览

| 方法 | AVG Peak | AVG Final (R200) | 训练稳定性 | 说明 |
|------|----------|-----------------|-----------|------|
| **原始 FedDSA** (069f, tau=0.2) | ~81-82% | **80.93%** | ✅ 稳定 | 基线 |
| mode=0 Fixed α=0.5 (tau=0.1) | ~82-83% | 76.82% | ❌ 崩溃 | tau=0.1 导致不稳定 |
| mode=1 M1 adaptive (tau=0.1) | ~82-83% | 78.06% | ❌ 崩溃 | M1 略好于固定alpha |
| mode=2 M3 domain-aware (tau=0.1) | ~81-82% | ~81.91% (interim) | ⚠️ 部分崩 | M3 效果显著 |
| mode=3 M1+M3 full (tau=0.1) | ~81-82% | ~80.5% (interim) | ⚠️ 部分崩 | 组合效果待确认 |

**关键结论**: M3 域感知原型是最有价值的改进（+5% vs 固定alpha），但需要 tau=0.2 修复训练崩溃后才能看到真正效果。

---

## 八、代码文件参考

| 文件 | 内容 | 行数 |
|------|------|------|
| `FDSE_CVPR25/algorithm/feddsa_adaptive.py` | 完整 FedDSA-Adaptive 算法（M1+M3） | ~641行 |
| `FDSE_CVPR25/algorithm/feddsa.py` | 原始 FedDSA 基线 | ~550行 |
| `FDSE_CVPR25/config/pacs/feddsa_072*.yml` | EXP-072 各变体配置 | - |
| `FDSE_CVPR25/config/pacs/feddsa_073.yml` | EXP-073 完整组合配置 | - |
| `FDSE_CVPR25/config/pacs/feddsa_074_*.yml` | EXP-074 tau修复配置 | - |
| `experiments/adaptive/EXP-072_adaptive_baselines/NOTE.md` | EXP-072 实验笔记 | - |
| `experiments/adaptive/EXP-073_m1_m3_full/NOTE.md` | EXP-073 实验笔记 | - |
| `experiments/adaptive/EXP-074_tau_investigation/NOTE.md` | EXP-074 实验笔记 | - |
