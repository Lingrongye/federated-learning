# EXP-117 | orth_only × DomainNet R200 3-seed — 跨数据集验证

## 基本信息
- **日期**: 2026-04-22 启动 → 2026-04-23 01:11-01:19 全部完成 ✅
- **算法**: `feddsa_scheduled` mode=0 (纯 orth_only, 对齐 EXP-080)
- **服务器**: lab-lry GPU 1
- **状态**: ✅ **3 seeds R=201/200 全部完成**

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

## 🏆 结果 (2026-04-23 01:11-01:19 全部完成, R=201/200)

### 3-seed 最终结果

| 方法 | seeds | DomainNet AVG Best/Last | vs FDSE 72.21/70.37 |
|------|:----:|:---:|:---:|
| **FDSE R200 本地复现** (baseline) | {2,15,333} | 72.21/70.37 | — |
| 老 FedDSA (EXP-065, 历史) | {2,15,333} | ~72.4/~70.7 | ≈ +0.2 |
| **EXP-115 orth_uc1** (R=201) | {2,15,333} | **72.49/70.68** | **+0.28** / +0.31 |
| **EXP-117 orth_only** (R=201, 本实验) | {2,15,333} | **72.23/70.57** | **+0.02** / +0.20 (基本持平) |

### Per-seed 最终结果

| seed | ALL B/L | AVG B/L |
|:---:|:---:|:---:|
| 2 | 74.55/73.23 | 72.13/70.67 |
| 15 | 74.59/73.13 | 72.26/70.71 |
| 333 | 74.86/73.02 | 72.31/70.34 |
| **Mean** | **74.67/73.13** | **72.23/70.57** |

### 🔑 关键判决: **orth 头在 DomainNet 贡献 ≈ 0**

| 对比 | AVG Best | AVG Last | 说明 |
|---|:---:|:---:|---|
| FDSE 本地 | 72.21 | 70.37 | baseline |
| orth_only (EXP-117) | 72.23 | 70.57 | **+0.02 B / +0.20 L** (打平) |
| orth_uc1 (EXP-115) | 72.49 | 70.68 | +0.28 B / +0.31 L |
| Δ (uc1 − only) | **+0.26** | **+0.11** | **pooled whitening + Fixed ETF 带来的真实增益** |

**解读**: EXP-115 orth_uc1 +0.28 增益中, orth 头本身**只贡献 0** (orth_only 几乎等于 FDSE), 增益主要来自 **pooled whitening + Fixed ETF classifier**. 这和 EXP-116 (PACS/Office lo=0 vs lo=1 差异 ≈ 0) 的结论**一致**: **正交头在 3 个数据集都对 Best 无实质贡献**.

**Paper 叙事更正**: 不能再卖"正交头", 真正 accuracy 来源是:
1. **pooled whitening** (特征白化)
2. **Fixed ETF classifier** (分类器等角紧框架)
3. **SGPA 架构** (双头 + 差异化聚合)

## 📋 部署完成

| Seed | 完成时间 | Round |
|:---:|:---:|:---:|
| 2 | 2026-04-23 01:11 | 201 |
| 333 | 2026-04-23 01:14 | 201 |
| 15 | 2026-04-23 01:19 | 201 |

**GPU**: lab-lry GPU 1 (与 EXP-116 6 runs 共存, 总 ~10-20 GB 显存)

## 胜负判决 (最终)

| Scenario | 阈值 | 实际 AVG B | 判决 |
|----------|:---:|:---:|:---:|
| orth_only DomainNet > 72.21 (FDSE) | > 72.21 | **72.23** | ✅ **打平 / 小过 +0.02** |
| orth_only DomainNet > 72.51 (显著 +0.3) | > 72.51 | 72.23 | ❌ **未达显著阈值** |

**跨 3 数据集 orth_only vs FDSE 本地**:

| 数据集 | orth_only AVG B | FDSE 本地 | Δ |
|---|:---:|:---:|:---:|
| PACS (EXP-080) | 80.41 | 79.91 | +0.50 ✅ |
| Office (EXP-080) | 89.44 | 90.58 | -1.14 ❌ |
| DomainNet (本) | **72.23** | 72.21 | **+0.02** 🟡 打平 |

**叙事**: orth_only 跨 3 数据集行为**不一致**, 不能作主卖点. orth_uc1 (+ whitening + ETF) 才是真正的增益来源.

## 下一步

1. ✅ 数据收集完毕, 对比 EXP-115 orth_uc1 清晰揭示 whitening+ETF 才是增益源
2. ✅ 连同 EXP-116 (lo=0 对照) 印证正交头对 Best 无贡献
3. **Paper 叙事重写**: 砍掉"正交头"作为主卖点, 改为 "SGPA 双头架构 + pooled whitening + Fixed ETF" 三元贡献
4. **Office 仍是弱点**, 需要单独攻关

## 📎 相关文件

- Config: `FDSE_CVPR25/config/domainnet/feddsa_orth_only_r200.yml`
- 算法: `FDSE_CVPR25/algorithm/feddsa_scheduled.py` mode=0 (orth_only base)
- 数据: `/home/lry/conda/envs/pfllib/lib/python3.11/site-packages/flgo/benchmark/RAW_DATA/domainnet/`
- 上游: EXP-080 (PACS/Office orth_only baseline 80.41/89.44), EXP-065 (老 FedDSA DomainNet), EXP-115 (orth_uc1 DomainNet 对比)
- 平行: EXP-116 (lo=0 × PACS/Office, 也在 lab-lry 跑)
