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

## 结果 (2026-04-15 10:13 CST 中间回填)

### 当前进度 — 🔄 2/3 seeds started

| Seed | Rounds | Best Acc | 状态 |
|------|--------|----------|------|
| s=2 | 33/200 | 80.58% | 🔄 running |
| s=333 | 32/200 | 77.98% | 🔄 running |
| s=42 | — | — | ⏳ queued |

### 2×2 Factorial 中间结果

| | No M3 (global protos) | M3 (domain protos) |
|--|----------------------|--------------------|
| **Fixed α=0.2** | 072a: 81.07±1.15% ✅ | — |
| **Fixed α=0.5** | 072b: 80.55±1.61% ✅ | — |
| **Fixed α=0.8** | 072c: 81.07±2.28% ✅ | — |
| **M1 adaptive** | 072: 80.89 (s2, 单seed) | **073: 🔄 running** |
| **M3-only** | — | 072d: 🔄 running |

### 初步观察
- Phase A 完成后，固定alpha之间差异极小 (<0.5%)
- H5 (正交叠加) 的验证需等073全部完成

> 完整结论待所有实验结束后回填
