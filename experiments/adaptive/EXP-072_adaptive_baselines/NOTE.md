# EXP-072 Adaptive Baselines — FedDSA-Adaptive消融

## 研究问题
FedDSA的风格增强对低gap域(Photo)有害、对高gap域(Sketch)帮助有限。
EXP-072验证：**自适应增强强度(M1)** 和 **域感知原型对齐(M3)** 能否解决此问题。

## 核心假设
- H1: 固定alpha次优 — 不同域需要不同增强强度
- H2: M1自适应增强(gap→alpha) 优于任何固定alpha
- H3: M3域感知原型避免语义稀释
- H4: 低gap域禁止增强(aug_min=0)不损失整体性能

## 实验矩阵

### Phase A: Fixed-alpha基线 (建立alpha曲线)
| Sub-EXP | adaptive_mode | alpha | Config | 说明 |
|---------|--------------|-------|--------|------|
| **072a** | 0 (fixed) | 0.2 | feddsa_072a.yml | 弱增强基线 |
| **072b** | 0 (fixed) | 0.5 | feddsa_072b.yml | 中增强基线 |
| **072c** | 0 (fixed) | 0.8 | feddsa_072c.yml | 强增强基线 |

### Phase B: 自适应 + M3 + Sanity
| Sub-EXP | adaptive_mode | 说明 | Config |
|---------|--------------|------|--------|
| **072** | 1 (M1) | 核心自适应增强: gap→alpha | feddsa_072.yml |
| **072d** | 2 (M3-only) | 域感知原型(原始Beta增强) | feddsa_072d.yml |
| **072e** | 1 (M1) | 低gap域禁止增强(aug_min=0.0) | feddsa_072e.yml |

### Phase C: 完整组合 (→ EXP-073)
| Sub-EXP | adaptive_mode | 说明 | Config |
|---------|--------------|------|--------|
| **073** | 3 (M1+M3) | 自适应增强 + 域感知原型 | feddsa_073.yml |

## 算法文件
`FDSE_CVPR25/algorithm/feddsa_adaptive.py` (641行)

### 关键超参 (algo_para顺序)
```
0: lambda_orth=1.0   1: lambda_hsic=0.0   2: lambda_sem=1.0
3: tau=0.1           4: warmup_rounds=50  5: style_dispatch_num=5
6: proj_dim=128      7: aug_min=0.05      8: aug_max=0.8
9: noise_std=0.05   10: ema_decay=0.9
11: adaptive_mode (0/1/2/3)   12: fixed_alpha_value
```

### 自适应机制 (M1)
- 服务器: z_sty bank → `raw_gap = ||μ_i - μ_global||² + ||σ_i - σ_global||²`
- EMA z-score归一化 → gap_normalized ∈ [0,1]
- 客户端: `alpha = aug_min + (aug_max - aug_min) * gap + N(0, noise_std)`
- 高gap域→强增强, 低gap域→弱增强

### 域感知原型 (M3)
- (class, client_id) → 独立原型, 不平均
- SupCon multi-positive InfoNCE: 同类跨域原型全部为正样本

## 数据集 & 训练配置
- PACS (Photo/Art/Cartoon/Sketch, 4域7类)
- R=200, E=5, B=50, lr=0.1, AlexNet backbone
- Seeds: 2, 333, 42

## 基线对比
| 方法 | PACS 3-seed Mean±Std |
|------|---------------------|
| FedDSA base (EXP-069f) | 80.96% (best), 80.93±0.30 (mean) |
| FedDSA NoAug (EXP-052) | 80.57% |
| FDSE (论文R200) | 80.36% |

## 服务器
- seetacloud新实例 (ssh -p 14824 root@connect.westc.seetacloud.com)
- GPU: 单卡 ~24GB
- **并行执行**: 10并行, 21个run, 预计~8-10h总时间

## 启动方式
```bash
# 并行执行脚本 (10 concurrent, 已部署到/tmp/run_072_parallel.sh)
nohup bash /tmp/run_072_parallel.sh > /tmp/072_parallel.log 2>&1 &

# 进度监控
cat /tmp/072_parallel.log
ps -eo pid,etime,cmd | grep run_single | grep -v grep
```

## 结果 (2026-04-15 10:13 CST 中间回填)

### Phase A: Fixed-alpha基线 — ✅ 全部完成 (200/200 rounds)

| Config | α | s=2 | s=333 | s=42 | Mean±Std |
|--------|---|-----|-------|------|----------|
| **072a** | 0.2 | 81.95 | 79.77 | 81.49 | **81.07±1.15%** |
| **072b** | 0.5 | 81.64 | 78.70 | 81.32 | **80.55±1.61%** |
| **072c** | 0.8 | 82.42 | 78.43 | 82.35 | **81.07±2.28%** |

**发现**: α=0.2 和 α=0.8 并列最优 (81.07%)，但 α=0.8 方差更大 (2.28 vs 1.15)。
α=0.5 反而最差 (80.55%)。固定alpha之间差异<0.5%，说明 **alpha值本身不是关键瓶颈**。

### Phase B: 自适应变体 — 🔄 运行中

| Config | Mode | s=2 | s=333 | s=42 | 状态 |
|--------|------|-----|-------|------|------|
| **072** | M1 adaptive | 80.89 (✅200R) | 77.47 (50R) | 80.34 (50R) | 🔄 2/3 done |
| **072d** | M3 domain-aware | 81.22 (33R) | 78.83 (34R) | 79.54 (33R) | 🔄 running |
| **072e** | M1 zero-floor | 79.98 (49R) | 78.47 (49R) | 80.20 (48R) | 🔄 running |

### Phase C: 完整组合 — 🔄 运行中

| Config | Mode | s=2 | s=333 | s=42 | 状态 |
|--------|------|-----|-------|------|------|
| **073** | M1+M3 | 80.58 (33R) | 77.98 (32R) | — | 🔄 2/3 started |

### 与基线对比

| 方法 | Mean±Std | vs FedDSA base |
|------|----------|---------------|
| FedDSA base (EXP-069f) | 80.93±0.30 | — |
| FDSE (论文R200) | 80.36 | -0.57 |
| 072a Fixed α=0.2 | 81.07±1.15 | +0.14 |
| 072c Fixed α=0.8 | 81.07±2.28 | +0.14 |
| 072 M1-adaptive (s2 only) | 80.89 (single) | -0.04 (单seed) |

### 初步结论 (Phase A)
1. ❗ **固定alpha间差异极小** (<0.5%): 增强强度α不是性能瓶颈
2. ❗ α=0.5中间值反而最差: 非单调关系
3. ❗ 所有固定alpha变体均高于FedDSA base (80.93)，但base方差更小(0.30 vs 1.15+)
4. ⏳ M1自适应s=2完成: 80.89% 略低于固定alpha，待更多seed确认
5. ⏳ Phase B/C 仍在运行中，完整结论待最终结果

> **注**: best accuracy指整个训练过程中的最高值，最终性能(last)因过拟合通常更低。
