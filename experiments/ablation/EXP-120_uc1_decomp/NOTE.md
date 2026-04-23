# EXP-120 | orth_uc1 分解消融 — SGPA 架构 / whitening / centers 三组件贡献拆分

## 基本信息
- **日期**: 2026-04-23 启动
- **算法**: `feddsa_sgpa` (不同 uw/uc flag 组合)
- **服务器**: lab-lry GPU 1 (RTX 3090 24GB, EXP-117 后空闲)
- **状态**: 🟡 待启动

## 这个实验做什么 (大白话)

**分解 EXP-115 orth_uc1 在 DomainNet 上 +0.28 胜 FDSE 的增益来源**.

EXP-117 (orth_only, feddsa_scheduled 单路径) AVG B **72.23** ≈ FDSE 72.21 (打平 +0.02), 说明"纯正交头"无贡献.
EXP-115 (orth_uc1 = feddsa_sgpa + uw=1 + uc=1) AVG B **72.49** (+0.28 vs FDSE).

orth_uc1 - orth_only = **+0.26** (SGPA 架构 + whitening + centers 三者合计).

本实验**把三个组件分开跑**, 看谁是真正的贡献者:
- SGPA only (uw=0 uc=0): 只有 SGPA 双头架构, 没 whitening 没 centers
- SGPA + w (uw=1 uc=0): 加 whitening, 没 centers
- SGPA + c (uw=0 uc=1): 加 centers, 没 whitening
- SGPA + w + c (uw=1 uc=1) = EXP-115 已有

## 变体通俗解释

| 变体 | 算法 | uw | uc | 含义 |
|------|:----:|:--:|:--:|------|
| orth_only (EXP-117) | feddsa_scheduled | — | — | 单路径 + 正交头 (无 SGPA 双头架构) |
| **sgpa_only** (本 EXP-120) | feddsa_sgpa | 0 | 0 | **SGPA 双头架构 + 正交, 无 whitening 无 centers** |
| **sgpa_w** (本 EXP-120) | feddsa_sgpa | 1 | 0 | SGPA + orth + **pooled whitening** |
| **sgpa_c** (本 EXP-120) | feddsa_sgpa | 0 | 1 | SGPA + orth + **class centers** 对齐 |
| orth_uc1 (EXP-115) | feddsa_sgpa | 1 | 1 | SGPA + orth + whitening + centers (全套) |

## 实验配置

| 参数 | 值 |
|------|:--:|
| Task | domainnet_c6 (10 类, 6 client) |
| Backbone | AlexNet |
| R / E / B / LR | 200 / 5 / 50 / 0.05 |
| WD / lr_decay | 1e-3 / 0.9998 |
| Seeds | {2, 15, 333} |
| algo_para | lo=1 pd=128 wr=10 es=1e-3 mcw=2 dg=1 ue=0 se=0 lp=0 ca=0, 仅 uw/uc 变化 |
| Configs | `feddsa_sgpa_{only,w,c}_r200.yml` |

**总 runs**: 3 变体 × 3 seeds = **9 runs**

## 预期结果分析

| 情景 | 解读 |
|---|---|
| sgpa_only ≈ orth_only (72.2) | SGPA 架构本身无贡献, whitening 和/或 centers 带来增益 |
| sgpa_only > orth_only + 0.3 | **SGPA 架构本身就有贡献** (双头 > 单路径) |
| sgpa_w ≈ sgpa_c | whitening 和 centers 效果相当, 可选其一 |
| sgpa_w > sgpa_c by >0.2 | whitening 是主力 |
| sgpa_c > sgpa_w by >0.2 | centers 是主力 |
| sgpa_w + sgpa_c ≈ 72.49 | 两者**累加**, 都有独立贡献 |
| sgpa_w ≈ sgpa_c ≈ 72.49 | 两者**冗余** (随便一个就够) |

## 预期 GPU + 时间

- lab-lry GPU 1 RTX 3090 24GB 完全空闲 (24129 MiB free)
- DomainNet E=5 单 run 显存 ~2-3 GB (历史数据), 9 并行 ~22 GB ✅
- 3090 单 run ~5-6h (比 4090 慢), 9 并行 wall ≈ **6-8h**
- 预计完成 2026-04-23 **18:00-20:00**

## 部署步骤

1. 本地创建 3 个 config + NOTE → git push
2. lab-lry: `cd /home/lry/code/federated-learning && git pull origin main`
3. lab-lry: 写 greedy launcher 脚本 (按 GPU 1 显存动态 launch, 阈值 2500 MB)
4. 启动 9 runs 并行, nohup 后台
5. 每 30min-1h 抽查进度
6. 完成后提取 record 回填 NOTE

## 📎 相关文件

- Configs: `FDSE_CVPR25/config/domainnet/feddsa_sgpa_{only,w,c}_r200.yml`
- 算法: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (uw/uc 在 init_algo_para 中控制)
- 基线数据: EXP-115 (orth_uc1 72.49), EXP-117 (orth_only 72.23), FDSE 本地 72.21
