# EXP-115 | DomainNet 基线扩展 — orth_uc1 + 5 种基线 × 3-seed R200

## 基本信息
- **日期**: 2026-04-22 凌晨 02:58 启动
- **算法**: feddsa_sgpa (orth_uc1), fedbn, fedavg, fedprox, ditto (待启), moon (待启)
- **服务器**: seetacloud2 GPU 0 (单 4090 12+ runs 并行)
- **状态**: 🟡 运行中 (预计 2026-04-22 早上 6-10 点完成)

## 这个实验做什么 (大白话)

把**所有还没在 DomainNet 上跑过的基线**部署一遍, 为 paper 补齐"多数据集跨任务"证据:

1. **orth_uc1 × 3-seed R200** (本课题主实验, 必须跑) — 验证 "PACS 上胜 FDSE 的 orth_uc1" 在 DomainNet 6-domain 更复杂场景下是否仍有效
2. **FedBN / FedAvg / FedProx / Ditto / MOON × 3-seed R200** — 补齐 DomainNet 基线, 方便 paper 主表

**FDSE DomainNet R200 3-seed (已有)**: AVG Best = **72.21%** (s=2 72.53 / s=15 72.59 / s=333 71.52)
**老 FedDSA DomainNet R200 3-seed (EXP-065 已有)**: AVG Best ≈ 72.4% (s=2 72.48 / s=15 72.43 / s=333 72.30)

## 动机

- EXP-113 PACS + Office 完成后发现:
  - orth_uc1 在 **PACS 胜 FDSE +0.73** ✅
  - A VIB 在 **Office 胜 orth_uc1 +0.76**, 但仍 -0.65 输 FDSE ⚠️
  - Office 上 regime-dependent (弱域异质) 和 PACS (强域异质) 行为不一致
- **DomainNet** (6 个域覆盖 real/stylized 整个光谱) 是理想的 regime-verification 场景:
  - sketch/quickdraw/clipart: 极高风格差 (预期 orth_uc1 帮 FDSE)
  - real/painting/infograph: 低风格差 (预期 orth_uc1 持平或略差)
- EXP-065 老 FedDSA 在 DomainNet "+0.19 vs FDSE" 太小 (1-seed noise 范围), 本实验用 3-seed 严格验证

## 变体通俗解释

| 变体 | 机制 | 一句话 |
|------|------|-------|
| **orth_uc1** (主) | feddsa_sgpa + L_orth + pooled whitening + Fixed ETF classifier + uc=1 | 正交双头 + 数据白化 + 对齐分类器, 我们的主方案 |
| FedBN | BN 层本地, 其他 FedAvg | 最简单的 domain adaptation 基线 |
| FedAvg | 所有参数 FedAvg 平均 | 最弱基线, 不考虑域异质 |
| FedProx | FedAvg + 近端正则 | 标准基线, μ=0.1 |
| Ditto | 双模型 (global + personal) | 个性化 FL 基线 |
| MOON | 模型级对比学习 | 纠正 client drift 的对比基线 |

## 实验配置

| 参数 | 值 |
|------|:--:|
| Task | domainnet_c6 (10 类, 6 clients: clipart, infograph, painting, quickdraw, real, sketch) |
| Backbone | AlexNet |
| R / E / B / LR | 200 / 5 / 50 / 0.05 |
| WD / lr_decay | 1e-3 / 0.9998 |
| Seeds | {2, 15, 333} |
| orth_uc1 config | `FDSE_CVPR25/config/domainnet/feddsa_orth_uc1_r200.yml` (lo=1 pd=128 wr=10 uw=1 uc=1 ca=0 se=0) |
| 基线 config | `FDSE_CVPR25/config/domainnet/{fedbn,fedavg,fedprox,ditto,moon}_r200.yml` |

## 🏆 结果 (待回填)

### 3-seed AVG Best / AVG Last (对齐 FDSE Table 1 口径)

| 方法 | seeds | AVG Best | AVG Last | sketch B/L | quickdraw B/L | clipart B/L | painting B/L | infograph B/L | real B/L |
|------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **FDSE R200** (基线, EXP-065 已有) | {2,15,333} | **72.21** | **70.37** | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 老 FedDSA R200 (EXP-065 已有) | {2,15,333} | ~72.4 | ~70.7 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| **orth_uc1** (本实验 ⭐) | {2,15,333} | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| FedBN | {2,15,333} | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| FedAvg | {2,15,333} | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| FedProx | {2,15,333} | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Ditto | {2,15,333} | 待填 (待启动) | — | | | | | | |
| MOON | {2,15,333} | 待填 (待启动) | — | | | | | | |
| **Δ orth_uc1 − FDSE** | — | 待填 | 待填 | | | | | | |

### Per-seed × per-client × Best/Last 矩阵 (待回填, 格式同 EXP-113)

```
client order: [clipart, infograph, painting, quickdraw, real, sketch] (按 alphabetical, 需核实)
每方法 × 3 seed × 6 client × B/L + @R
```

## 📋 部署状态快照 (2026-04-22 03:05 时刻)

| 方法 | 状态 | PID | 进度 |
|------|:---:|:---:|:---:|
| orth_uc1 s=2 | 🟡 running | 335237 | R? |
| orth_uc1 s=15 | 🟡 running | 335371 | R? |
| orth_uc1 s=333 | 🟡 running | 335503 | R? |
| fedbn s=2 | 🟡 running | 335669 | R? |
| fedbn s=15 | 🟡 running | 335801 | R? |
| fedbn s=333 | 🟡 running | 335935 | R? |
| fedavg s=2 | 🟡 running | 336203 | R? |
| fedavg s=15 | 🟡 running | 336335 | R? |
| fedavg s=333 | 🟡 running | 336469 | R? |
| fedprox s=2 | 🟡 running | 336626 | R? |
| fedprox s=15 | 🟡 running | 336758 | R? |
| fedprox s=333 | 🟡 running | 336892 | R? |
| ditto × 3 | ⏸ 待 GPU 资源 | — | — |
| moon × 3 | ⏸ 待 GPU 资源 | — | — |

**GPU 使用**: 15/24.5 GB (60%)

**预计完成时间**: DomainNet 6 domains × R=200 × E=5 × batch=50 约 3-4h/run, 12 并行约 6-8h wall. 预计 2026-04-22 早 9-11 点完成第一批.

## 胜负判决 (对齐 CLAUDE.md 0 节)

| 指标 | 阈值 | 本实验最佳 | 判决 |
|------|:---:|:---:|:---:|
| DomainNet AVG Best | > 72.21 (FDSE R200) | 待填 | ⏳ |

## 下一步

1. 等所有 12 runs R200 完成
2. 如 GPU 资源允许, 追加 ditto × 3 + moon × 3
3. 收集结果回填主表
4. 分析: 如果 orth_uc1 胜 FDSE → 跨 3 数据集都赢, paper novelty 成立
5. 如果 orth_uc1 持平/输 → regime-dependent 假说加强, paper 改为 "orth_uc1 在 strong-gap PACS 专属优势"

## 📎 相关文件

- DomainNet Configs: `FDSE_CVPR25/config/domainnet/*_r200.yml`
- 算法代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (orth_uc1), `fedbn.py`, flgo 自带 (fedavg/fedprox/ditto/moon)
- 上游依赖: EXP-109 (PACS orth_uc1 baseline 80.64), EXP-110 (Office orth_uc1 89.09), EXP-113 (VIB/VSC/SupCon), EXP-065 (老 FedDSA DomainNet 72.4)
- 对照数据: MASTER_RESULTS.md (PACS/Office 对照), FDSE DomainNet R200 record JSON (seetacloud2 task/domainnet_c6/record/)
