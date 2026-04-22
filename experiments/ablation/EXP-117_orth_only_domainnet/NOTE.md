# EXP-117 | orth_only × DomainNet R200 3-seed — 跨数据集验证

## 基本信息
- **日期**: 2026-04-22 启动
- **算法**: `feddsa_scheduled` mode=0 (纯 orth_only, 对齐 EXP-080)
- **服务器**: lab-lry GPU 1
- **状态**: 🟡 运行中 (3 seeds 并行, 预计 10-20h 完成)

## 这个实验做什么 (大白话)

**把 EXP-080 的 "纯 orth_only" (只加正交头) 扩展到 DomainNet**, 验证跨 3 数据集一致性.

已有数据:
- PACS orth_only (EXP-080): **80.41** vs FDSE 79.91 (+0.50 ✅)
- Office orth_only (EXP-080): **89.44** vs FDSE 90.58 (-1.14 ❌)
- **DomainNet 未做** ← 本实验补齐

另外, EXP-115 正在 seetacloud2 跑 **orth_uc1 × DomainNet** (SGPA 复杂版本). 本实验跟它是**不同方案的对照**:
- EXP-115 orth_uc1: feddsa_sgpa 算法 (orth + whitening + centers + 差异化聚合)
- EXP-117 orth_only: feddsa_scheduled mode=0 (**只有** orth, 无其他组件)

两者对比能揭示: SGPA 的附加组件在 DomainNet 上贡献多少.

## 变体通俗解释

| 变体 | 算法 | 组件 |
|------|:----:|------|
| FDSE (baseline) | fdse | 层分解 DFE+DSE, 差异化聚合 |
| **EXP-117 orth_only** (本) | feddsa_scheduled mode=0 | **纯**正交头 (cos² on z_sem, z_sty) + CE |
| EXP-115 orth_uc1 (对比) | feddsa_sgpa | orth + pooled whitening + class_centers + Fixed ETF classifier |

## 实验配置

| 参数 | 值 |
|------|:--:|
| Task | domainnet_c6 (10 类子集, 6 client: clipart/infograph/painting/quickdraw/real/sketch) |
| 算法 | feddsa_scheduled (mode=0) |
| Backbone | AlexNet + 双 128d 头 |
| R / E / B / LR | 200 / 5 / 50 / 0.05 (对齐 EXP-080 PACS/Office + FDSE DomainNet) |
| algo_para | lo=1.0 lh=0 ls=1 tau=0.2 pd=128 sm=0 |
| WD / lr_decay | 1e-3 / 0.9998 |
| Seeds | {2, 15, 333} |
| Config | `FDSE_CVPR25/config/domainnet/feddsa_orth_only_r200.yml` |

**与 EXP-080 配置的区别**: 只改 `--task domainnet_c6`, 其他**完全一致** (公平对照).

## 数据准备记录 (实施细节)

seetacloud2 RAW_DATA 19GB 全量太大, lab-lry 磁盘不够 + 下载太慢. 解决:
1. domainnet_c6 只用 **10 类子集** (bird/feather/headphones/ice_cream/teapot/tiger/whale/windmill/wine_glass/zebra)
2. seetacloud2 打包 10 类图像 + txt 列表 = tar.gz **743MB**
3. seetacloud2 → Windows → lab-lry (scp 中转, ~10min)
4. lab-lry 解压到 `/home/lry/conda/envs/pfllib/lib/python3.11/site-packages/flgo/benchmark/RAW_DATA/domainnet/` (845MB)
5. flgo 加载时检测到 domain 目录存在, 跳过下载直接用

磁盘: lab-lry 28GB 剩余 (充足, 845MB 不占空间).

## 🏆 结果 (待回填)

### 3-seed AVG Best/Last

| 方法 | seeds | DomainNet AVG Best | vs FDSE 72.21 |
|------|:----:|:----:|:----:|
| **FDSE R200** (baseline) | {2,15,333} | **72.21** | 0 |
| 老 FedDSA (EXP-065) | {2,15,333} | ~72.4 | ≈ 0 |
| **EXP-115 orth_uc1** (seetacloud2 正在跑) | {2,15,333} | **R91 已 72.49 (+0.28)** 🟡 | ✅ 接近完成 |
| **EXP-117 orth_only** (本实验) | {2,15,333} | 待填 | 待填 |

### Per-seed × per-domain × Best/Last 完整矩阵 (待回填)

Client order: [clipart, infograph, painting, quickdraw, real, sketch] (待核实)

```
(待回填, 格式同 EXP-113/115 NOTE.md)
```

## 📋 部署快照 (2026-04-22 启动)

| Seed | PID | 状态 |
|:---:|:---:|:---:|
| 2 | 387302 | 🟡 R0 local client 1/6 (启动 03:28 elapsed) |
| 15 | 388603 | 🟡 启动 (01:00 elapsed) |
| 333 | 388700 | 🟡 启动 (00:50 elapsed) |

**GPU**: 10GB / 24GB (41%), 和 6 个 lo=0 runs 共存

## 胜负判决

| Scenario | 判决 | 意义 |
|----------|:---:|------|
| orth_only DomainNet > 72.21 | ✅ 跨 3 数据集 2-3 个胜 FDSE | paper 主卖点稳 |
| orth_only ≈ 72.21 ± 0.3 | 🟡 打平, 不显著胜 | paper 需讨论 |
| orth_only < 72.21 - 0.3 | ❌ DomainNet 输 | regime-dependent (强-弱异质) |

## 下一步

1. 等 3 seeds 完成 (10-20h)
2. 对比 EXP-115 orth_uc1 DomainNet 结果 → 揭示 SGPA 附加组件贡献
3. 汇总 PACS + Office + DomainNet 跨数据集 orth_only vs orth_uc1 vs FDSE 主表
4. 连 EXP-116 (lo=0 对照) 一起完成 → paper 消融章节成熟

## 📎 相关文件

- Config: `FDSE_CVPR25/config/domainnet/feddsa_orth_only_r200.yml`
- 算法: `FDSE_CVPR25/algorithm/feddsa_scheduled.py` mode=0 (orth_only base)
- 数据: `/home/lry/conda/envs/pfllib/lib/python3.11/site-packages/flgo/benchmark/RAW_DATA/domainnet/`
- 上游: EXP-080 (PACS/Office orth_only baseline 80.41/89.44), EXP-065 (老 FedDSA DomainNet), EXP-115 (orth_uc1 DomainNet 对比)
- 平行: EXP-116 (lo=0 × PACS/Office, 也在 lab-lry 跑)
