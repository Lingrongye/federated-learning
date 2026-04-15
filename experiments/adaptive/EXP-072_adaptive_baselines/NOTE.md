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

## 结果 (2026-04-15 更新 — R200最终值)

> **重要说明**: 所有结果均为R200最终轮准确率 (local_test_accuracy)。
> 由于 feddsa_adaptive 分类头从128d z_sem特征分类(原始feddsa从1024d)，
> 绝对值低于原始feddsa基线，但内部组间对比有效。
> Phase A原先NOTE中81%数字为训练过程最高值(非R200)，此处已修正。

### Phase A: Fixed-alpha基线 — ✅ 全部完成 (200/200 rounds)

| Config | α | s=2 | s=333 | s=42 | Mean±Std |
|--------|---|-----|-------|------|----------|
| **072a** | 0.2 | 75.02 | 74.92 | 77.33 | **75.76±1.36%** |
| **072b** | 0.5 | 77.22 | 76.92 | 76.33 | **76.82±0.46%** |
| **072c** | 0.8 | 77.12 | 75.42 | 77.93 | **76.82±1.29%** |

**发现**: α=0.5 和 α=0.8 并列最优 (76.82%)，固定alpha之间差异<1%，说明 **alpha值本身不是核心瓶颈**。

### Phase B/C: 自适应变体 — ✅/🔄 进行中

| Config | Mode | s=2 | s=333 | s=42 | 状态 | Mean |
|--------|------|-----|-------|------|------|------|
| **072** | M1 adaptive (aug_min=0.05) | 81.14 | 75.92 | 77.13 | ✅ done | **78.06±2.73%** |
| **072e** | M1 zero-floor (aug_min=0.0) | 79.03 | 75.32 | 78.83 | ✅ done | **77.73±2.00%** |
| **072d** | M3 domain-aware protos | 82.44 | 80.84 | 82.44 | 🔄 R146/200 | **~81.91** |
| **073** | M1+M3 full | 82.74 | 79.83 | 78.93* | 🔄 R145/R14* | **~80.5+** |

*073 s=42 刚启动(R14), 数字不具代表性; 072d/073 预计~18:30(s2/s333)和次日00:30(073 s42)完成。

### 阶段性发现
1. ✅ **M3域感知原型效果极显著**: +5.1% vs 最好的固定alpha (+81.91 vs 76.82%)
2. ✅ **M1自适应增强**: +1.2% vs 最好固定alpha (78.06 vs 76.82%)
3. ✅ **M1+M3组合**: 072d/073对比待最终数据确认
4. ⚠️ M1高方差(std=2.73): s2=81.14 vs s333=75.92，稳定性不佳
5. ⏳ 待确认: M1+M3是否优于单独M3

### 内部组对比 (vs 固定alpha最优072b/c=76.82%)
| 改进模块 | Mean | Δ |
|---------|------|---|
| 072b Fixed α=0.5 (control) | 76.82 | — |
| 072 M1 adaptive | 78.06 | **+1.24%** |
| 072d M3 domain-aware (interim R146) | ~81.91 | **+5.09%** ★★★ |
| 073 M1+M3 (partial) | ~80+ | TBD |
