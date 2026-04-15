# EXP-073 M1+M3 Full — FedDSA-Adaptive完整版

## 研究问题
验证M1(自适应增强) + M3(域感知原型) 的组合效果是否叠加。

## 核心假设
- H5: M1+M3的改进具有正交叠加性（两个机制解决不同问题）
  - M1解决: 不同域需要不同增强强度
  - M3解决: 全局原型平均导致语义稀释

## 实验配置
| 参数 | 值 |
|------|-----|
| adaptive_mode | 3 (M1+M3) |
| aug_min | 0.05 |
| aug_max | 0.8 |
| noise_std | 0.05 |
| ema_decay | 0.9 |
| Config | feddsa_073.yml |

## 依赖
EXP-072完成后执行。需要072的结果来判断M1和M3各自的独立效果。

## 2×2 Factorial设计（来自R2审稿建议）
| | No M3 (global protos) | M3 (domain protos) |
|--|----------------------|--------------------|
| **Fixed alpha** | FedDSA base | 072d |
| **M1 adaptive** | 072 | **073 (本实验)** |

## 算法文件
`FDSE_CVPR25/algorithm/feddsa_adaptive.py` (adaptive_mode=3)

## 数据集 & 训练配置
- PACS (4域7类), R=200, E=5, B=50, lr=0.1
- Seeds: 2, 333, 42

## 基线对比
| 方法 | 来源 | 说明 |
|------|------|------|
| FedDSA base | EXP-069f | 80.93±0.30 |
| M1-only | EXP-072 | 自适应增强 |
| M3-only | EXP-072d | 域感知原型 |

## 启动方式
EXP-072并行脚本的Phase C部分自动执行 (run_072_parallel.sh)。

## 结果 (2026-04-15 更新 — 中间值, R200最终待回填)

> **重要说明**: 所有准确率为 local_test_accuracy (R200 final)，除非特别注明"interim RN"。

### 2×2 Factorial 结果 — 内部消融 (feddsa_adaptive)

| | No M3 (global protos) | M3 (domain protos) |
|--|----------------------|--------------------|
| **Fixed α=0.5** | 072b: 76.82±0.46% ✅ | 072d: **~81.91** (R146 interim) |
| **M1 adaptive** | 072: 78.06±2.73% ✅ | **073: 🔄 R145/R14** |

**注**: 2×2完整填充后可计算M1效果和M3效果的加法性。

### 当前进度 (2026-04-15 15:52 UTC)

| Seed | Rounds | local_acc | mean_acc | 状态 |
|------|--------|-----------|----------|------|
| s=2 | 145/200 | 82.74% | 80.74% | 🔄 running |
| s=333 | 144/200 | 79.83% | 76.92% | 🔄 running |
| s=42 | 14/200 | 78.93% | 79.03% | 🔄 running (just started) |

预期完成时间: s2/s333 ~18:30 UTC, s42 ~次日00:30 UTC

### 初步发现 (基于interim R145数据)
- 073 (M1+M3) s=2: 82.74% > 072d (M3-only) s=2: 82.44% → M1在M3基础上提供微弱增益
- 073 (M1+M3) s=333: 79.83% ≈ 072d (M3-only) s=333: 80.84% → 接近
- H5 (正交叠加) 的验证需等最终R200数据

### 待最终数据回填后补充
```
| Config | s=2 | s=333 | s=42 | Mean±Std |
|--------|-----|-------|------|----------|
| 073 (M1+M3) | ? | ? | ? | ? |
```

> 完整结论和2×2 interaction analysis待R200数据后补充
