# EXP-121 | Digits-5 baseline 3-seed R100 — 首次跨 5-domain 强异质验证

## 基本信息
- **日期**: 2026-04-23 启动
- **服务器**: seetacloud2 GPU 0 (RTX 4090 24GB, EXP-115 完成后空闲)
- **数据**: FedBN (ICLR 2021) 官方标准 Digits-5 (MNIST/MNIST_M/SVHN/SynthDigits/USPS)
  - 5 client = 5 domain, 每 client train 7430, test subsample 到 1860 per domain
  - 输入统一 (3, 32, 32), 范围 [-1, 1]
- **模型**: FedBN paper 标准 5-layer CNN (3 Conv + 2 FC, 约 20M 参数)
- **状态**: 🟡 待启动

## 这个实验做什么 (大白话)

**验证 orth_uc1 / FedDSA-SGPA 在第 5 个 (Digits-5) 强风格异质数据集上的表现**.

之前 3 个数据集结果:
| 数据集 | Regime | orth_uc1 vs FDSE |
|---|:---:|:---:|
| PACS | 强风格 | +0.73 ✅ |
| DomainNet | 强风格 | +0.28 ✅ |
| Office-Caltech10 | 弱风格 | -1.49 ❌ |

Digits-5 (MNIST/SVHN/USPS 等差异极大) 是强风格异质 regime. 预期结论:
- orth_uc1 > FDSE ≥ FedBN → 强化 "regime-dependent, 强异质有效" 的 paper 叙事 (跨 3 强异质数据集都胜)
- orth_uc1 < FDSE → 叙事需要 nuance (方法上有限制)

## 变体

| 变体 | 算法 | algo_para 关键 flag |
|---|:---:|---|
| FedAvg | fedavg | — (基线) |
| FedBN | fedbn | BN running stats 本地 (baseline 对标 FedBN paper) |
| FDSE | fdse | 层分解 DSEConv/DSELinear, lmbd=0.01 |
| **FedDSA-SGPA (uw1_uc1)** | feddsa_sgpa | lo=1 uw=1 uc=1 ue=0 (= orth_uc1 等价) |

## 实验配置

| 参数 | 值 |
|------|:--:|
| Task | digit5_c5 (5 client × 5 domain) |
| R / E / B / LR | 100 / 1 / 32 / 0.01 (对齐 FedBN ICLR 2021) |
| WD | 1e-5 |
| Seeds | {2, 15, 333} |
| Configs | `config/digit5/{fedavg,fedbn,fdse,feddsa_sgpa_uw1_uc1}_r100.yml` |
| 并行 | 12 runs greedy launcher, MIN_FREE_MB=1600 |

## 预期

- 单 run wall: R=100 约 **20 min** (smoke R=3 = 40s, 按比例)
- 12 runs 并行: 24GB GPU / 1.5-2 GB per run, 12 × 1.8 = 21.6 GB ✅
- 总 wall: **~25-30 min** (早期 gpu 未饱和, 后期并行)
- 预期完成: **2026-04-23 17:30 前**

## 胜负判决

| 阈值 | 意义 |
|:-:|---|
| `feddsa_sgpa_uw1_uc1 > fedbn + 0.3` | ✅ 跨 3 强异质数据集都胜 FedBN, 主叙事强化 |
| `feddsa_sgpa_uw1_uc1 > fdse + 0.3` | ✅ 对 FDSE 也胜 (不止是 FedBN) |
| 两个都胜 | 🎯 最佳结果, paper 主表完整 |
| 都输 / 持平 | ⚠️ 需 nuance: "在 3 个数据集上有效 (PACS/DomainNet), 1 个上不适用 (Digits-5)" |

## 📎 相关文件
- Task: `FDSE_CVPR25/task/digit5_c5/`
- Configs: `FDSE_CVPR25/config/digit5/*.yml`
- Algorithm patches: `FDSE_CVPR25/algorithm/fdse.py` (+ Digit5ModelDSE), `algorithm/feddsa_sgpa.py` (+ DigitEncoder + backbone='digit')
- Smoke: R=3 fedbn 91.94% mean_local_test_accuracy ✅
