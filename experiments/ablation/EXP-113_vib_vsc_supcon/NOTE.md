# EXP-113 | FedDSA-VIB (A) + FedDSA-VSC (B) + orth+SupCon (C) × 2 数据集 × 3-seed R200

## 基本信息
- **日期**: 2026-04-21 代码完成 / 2026-04-21 18:49 启动部署 / 2026-04-22 01:40 PACS 全完成 / Office 2026-04-21 晚完成
- **算法**: `feddsa_sgpa_vib` (2×2 ablation)
- **服务器**: seetacloud2 GPU 0 (单 4090 21 runs 并行)
- **状态**: ✅ R200 全部完成, 12 Office probe JSON 已出, PACS probe 待跑

## 这个实验做什么 (大白话)

Round-2 Codex pivot 后的核心方案: 抛弃 CDANN (EXP-108 anchor 被 EXP-109 反事实证伪), 改用**信息瓶颈 VIB + EMA-lagged 语义原型 prior**解决 EXP-111 暴露的"非线性 probe 泄漏没被正交根治" (MLP-256 仍 0.71 即使 lo=10)。

2×2 ablation 矩阵: `{无/有 VIB} × {无/有 SupCon}`:
- **orth_uc1** (已有 baseline): 无 VIB + 无 SupCon (只有 CE + L_orth + InfoNCE)
- **A FedDSA-VIB**: 有 VIB + 无 SupCon (信息瓶颈压语义)
- **B FedDSA-VSC**: 有 VIB + 有 SupCon (VIB+多正例 contrastive)
- **C orth+SupCon**: 无 VIB + 有 SupCon (只换对比损失)

## 变体通俗解释

| 变体 | 机制 | 一句话 |
|------|------|-------|
| orth_uc1 | baseline | 正交双头 + pooled whitening + Fixed ETF + 差异化聚合, R200 |
| **A VIB** | q(z_sem\|x)=N(μ,σ²), 闭式 KL 到 EMA-lagged 类原型 prior, 可学 per-class σ_prior, λ_IB warmup R0→50 线性 | 给语义头加一个"按原型对齐的噪声注入器", 希望把域信息挤出去 |
| **B VSC** | A + 把 InfoNCE 换 SupCon (Khosla 2020 多正例 supervised contrastive), σ-head FedBN 本地化 | VIB 和 SupCon 一起上 |
| **C SupCon** | baseline + 把 InfoNCE 换 SupCon, 保持 orth+whitening 不变 | 只换对比损失 |

## 技术细节
- **VIB 闭式 KL**: `0.5 Σ(σ²+(μ-μ_proto)² - 2logσ - 1)` (Alemi 2017 式), μ_proto = EMA β=0.99 stop-grad 防止 chicken-and-egg
- **可学 log_sigma_prior**: `nn.Parameter(torch.zeros(num_classes))`, clamp [-2, 2] 防退化
- **σ-head 本地化**: `log_var_head` + `log_sigma_prior` 加入 private_keys (FedBN 模式), 不参加 FedAvg 聚合
- **λ_IB warmup**: R0-20 = 0, R20-50 线性 0→1, R50+ = 1.0
- **lib (β) = 0.01** (Alemi 2017 标准值, Round-2 review 修正: 原 1.0 过大 100×)
- **Client.unpack override**: 跳过 log_var_head/log_sigma_prior 避免 Server 每轮覆盖 (Codex review 发现的 critical bug)

## 实验配置

| 参数 | PACS | Office |
|------|:----:|:------:|
| Task | PACS_c4 (7 类, 4 client) | office_caltech10_c4 (10 类, 4 client) |
| Backbone | AlexNet + 双 128d 头 + Fixed ETF classifier |  同 |
| R / E / B / LR | 200 / 5 / 50 / 0.05 | 200 / 1 / 50 / 0.05 |
| λ_orth | 1.0 | 1.0 |
| λ_IB (lib) | 0.01 (Alemi 2017) | 0.01 |
| λ_SupCon (lsc) | 1.0 | 1.0 |
| VIB warmup (vws/vwe) | 20/50 | 20/50 |
| SupCon τ | 0.07 | 0.07 |
| Seeds | {2, 15, 333} | {2, 15, 333} |
| Configs | `FDSE_CVPR25/config/pacs/feddsa_{vib,vsc,supcon}_pacs_r200.yml` | `FDSE_CVPR25/config/office/feddsa_{vib,vsc,supcon}_office_r200.yml` |

## 🏆 PACS_c4 R200 3-seed 完整结果

### AVG Best / AVG Last (对齐 FDSE Table 1 口径)

| 方法 | seeds | AVG Best | AVG Last | Art B/L | Cartoon B/L | Photo B/L | Sketch B/L |
|------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **FDSE R200** (基线) | {2,15,333} | **79.91** | **77.55** | 69.6/63.7 | 85.6/81.8 | 82.6/77.0 | 89.2/87.7 |
| FedBN R200 | {2,15,333} | 79.01 | 76.92 | 64.5/57.7 | 86.6/83.8 | 81.8/78.0 | 89.5/88.2 |
| FedAvg R200 | {2,15,333} | 72.31 | 68.71 | 58.0/52.1 | 79.3/71.9 | 71.1/66.7 | 86.4/84.1 |
| **orth_uc1 (EXP-109)** 🏆 | {2,15,333} | **80.64** | **79.98** | 64.5/62.4 | 87.7/86.6 | 83.4/82.0 | 89.5/88.9 |
| **A VIB** (本实验) | {2,15,333} | **79.92** | 77.51 | 65.8/61.4 | 86.8/83.5 | 82.4/76.4 | 90.7/88.7 |
| **B VSC** (本实验) | {2,15,333} | 79.42 | 76.62 | 62.7/56.9 | 86.5/82.6 | 82.4/77.4 | 91.0/89.5 |
| **C SupCon** (本实验) | {2,15,333} | 79.77 | 79.00 | 62.9/59.6 | 87.0/85.5 | 83.4/81.4 | 90.5/89.5 |
| **Δ A VIB − orth_uc1** | — | **−0.72** ❌ | −2.47 | +1.3/−1.0 | −0.9/−3.1 | −1.0/−5.6 | +1.2/−0.2 |
| **Δ orth_uc1 − FDSE** | — | **+0.73** ✅ | +2.43 | −5.1/−1.3 | +2.1/+4.8 | +0.8/+5.0 | +0.3/+1.2 |

### PACS per-seed (每 seed 的 AVG Best / Last)

| 方法 | s=2 B/L | s=15 B/L | s=333 B/L |
|------|:-------:|:--------:|:---------:|
| orth_uc1 | 82.23/81.41 | 80.35/80.23 | 79.35/78.30 |
| A VIB | 81.15/79.35 | 79.08/75.86 | 79.52/77.32 |
| B VSC | 80.88/77.77 | 78.26/76.76 | 79.13/75.32 |
| C SupCon | 81.90/81.84 | 79.05/78.62 | 78.36/76.54 |

## 🎯 Office-Caltech10 R200 3-seed 完整结果

Client 顺序 = [**Caltech, Amazon, DSLR, Webcam**] (非字母序, 对齐 MASTER_RESULTS.md)

### AVG Best / AVG Last

| 方法 | seeds | AVG Best | AVG Last | Caltech B/L | Amazon B/L | DSLR B/L | Webcam B/L |
|------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **FDSE R200** (基线) | {2,15,333} | **90.58** | **89.22** | 78.9/77.7 | 92.3/89.5 | 100.0/97.8 | 95.4/92.0 |
| FedBN R200 (s=333) | {333} | 88.68 | 85.82 | 75.9/73.2 | 94.7/90.5 | 100.0/93.3 | 89.7/86.2 |
| **orth_uc1** (EXP-110) | {2,15,333} | 89.17 | 87.56 | 73.2/70.2 | 90.5/89.1 | 100.0/97.8 | 94.3/93.1 |
| **A VIB** (本实验) 🏆 | {2,15,333} | **89.93** | 88.01 | 76.2/72.0 | 91.9/89.1 | 100.0/97.8 | 95.4/93.1 |
| **B VSC** (本实验) | {2,15,333} | 89.55 | **88.46** | 76.8/72.3 | 92.6/89.5 | 97.8/97.8 | 95.4/94.3 |
| **C SupCon** (本实验) | {2,15,333} | 89.20 | 87.73 | 74.7/70.5 | 90.9/88.4 | 100.0/100.0 | 96.6/92.0 |
| **Δ A VIB − orth_uc1** | — | **+0.76** ✅ | +0.45 | +2.9/+1.8 | +1.4/±0 | ±0/±0 | +1.1/±0 |
| **Δ A VIB − FDSE** | — | **−0.65** ⚠️ | −1.21 | −2.7/−5.7 | −0.4/−0.4 | ±0/±0 | ±0/+1.1 |

### Office per-seed

| 方法 | s=2 B/L | s=15 B/L | s=333 B/L |
|------|:-------:|:--------:|:---------:|
| orth_uc1 | 88.84/86.53 | 89.28/87.98 | 89.40/88.18 |
| A VIB | 89.74/88.91 | 89.77/88.24 | 90.29/86.87 |
| B VSC | 89.88/88.77 | 90.14/88.68 | 88.63/87.92 |
| C SupCon | 88.46/87.39 | 89.06/87.11 | 90.07/88.68 |

## 🔬 Office Probe 3-seed Mean (capacity probe, 4 探针 × 5 容量)

**关键诊断指标**: 判决 "VIB +0.76 accuracy 是不是通过更好解耦实现的"

### probe_sty_class — 风格里的类别信息 (↓ 越低越好, 期望 VIB < orth_uc1)

| 方法 | linear | mlp_16 | mlp_64 | mlp_128 | mlp_256 |
|------|:------:|:------:|:------:|:-------:|:-------:|
| orth_uc1 | 96.43 | 95.84 | 96.13 | 95.77 | 96.21 |
| **A VIB** | **95.84** | **94.24** | **95.26** | **95.04** | **95.33** |
| B VSC | 96.13 | 93.22 | 95.11 | 95.70 | 95.84 |
| C SupCon | 96.28 | 94.38 | 96.13 | 95.92 | 94.31 |
| **Δ A − orth_uc1** | −0.58 | **−1.60** | −0.88 | −0.73 | −0.88 |
| 判决 | ✅ 5/5 容量都降 → VIB 让风格少含类别信息 (支持解耦方向) |

### probe_sem_domain — 语义里的域信息 (↓ 越低越好, 期望 VIB < orth_uc1)

| 方法 | linear | mlp_16 | mlp_64 | mlp_128 | mlp_256 |
|------|:------:|:------:|:------:|:-------:|:-------:|
| orth_uc1 | 50.62 | 53.25 | 55.14 | 59.15 | 57.04 |
| **A VIB** | 55.22 | 46.24 | **68.78** | **72.87** | **74.62** |
| B VSC | 54.85 | 51.57 | 63.68 | 68.34 | 68.13 |
| C SupCon | 60.61 | 52.59 | 59.30 | 64.04 | 62.07 |
| **Δ A − orth_uc1** | **+4.60** | −7.00 | **+13.64** | **+13.71** | **+17.58** |
| 判决 | ❌❌ 4/5 容量反向 → VIB 让语义**更**含域信息 (反方向!) |

### probe_sem_class — 语义判别力 (↑ 越高越好)

| 方法 | linear | mlp_16 | mlp_64 | mlp_128 | mlp_256 |
|------|:------:|:------:|:------:|:-------:|:-------:|
| orth_uc1 | 96.35 | 96.43 | 96.21 | 96.21 | 96.50 |
| A VIB | 95.84 | 94.46 | 94.89 | 95.48 | 95.33 |
| B VSC | 96.13 | 94.24 | 95.26 | 95.40 | 95.26 |
| C SupCon | **97.23** | **96.57** | **96.79** | **96.72** | **96.94** |
| **Δ A − orth_uc1** | −0.51 | −1.97 | −1.31 | −0.73 | −1.17 |

### probe_sty_domain — 风格里的域信息 (↑ 越高越好)

| 方法 | linear | mlp_16 | mlp_64 | mlp_128 | mlp_256 |
|------|:------:|:------:|:------:|:-------:|:-------:|
| orth_uc1 | 90.44 | 88.11 | 91.90 | 93.29 | 93.14 |
| A VIB | 90.15 | 90.01 | 90.59 | 92.05 | 92.49 |
| B VSC | 93.22 | 89.93 | 93.58 | 93.22 | 93.07 |
| C SupCon | 91.03 | 89.13 | 91.25 | 91.61 | 92.27 |

## 🔑 关键发现

### 1. PACS: orth_uc1 (80.64) 仍是最强, A/B/C 全部输给它 (−0.7 ~ −1.2pp)

- A VIB PACS −0.72pp: VIB 正则化在 PACS 这种**强域异质**场景反而伤害, 把有用的域信号也压掉了
- B VSC 最差 (−1.22pp): VIB + SupCon 组合在 PACS 上叠加伤害, s=15 尤其不稳 (79.08 vs orth_uc1 80.35)
- C SupCon (−0.87pp) 接近 orth_uc1, 说明单纯换对比损失影响小

### 2. Office: A VIB (89.93) +0.76 赢 orth_uc1, 但仍 −0.65 输 FDSE

- A VIB 在 Office **显著**涨 +0.76 (主要来自 Caltech +2.9pp 这个最难域)
- Caltech 是 Office outlier (FDSE 78.9, 其他三个都 ≥92), VIB 对这种域外数据有稳定化作用
- 但 Caltech 仍 -2.7 ~ -5.7 pp 输 FDSE (FDSE 对 Caltech 有特殊优势, 可能跟 FDSE 的层分解 DSE 有关)

### 3. Probe 揭示: VIB 的 accuracy 涨点**不是**通过"更好解耦"实现的 ⚠️

- ✅ 唯一支持解耦的证据: probe_sty_class 5/5 容量都降 (−0.93pp avg, 风格少泄漏类别)
- ❌ 反向证据:
  - probe_sem_domain 反而涨 +8.51 avg (**语义混入更多域信息**, mlp_256 上 +17.58!)
  - probe_sem_class 略降 −1.14 avg (语义判别力变弱)
  - probe_sty_domain 略降 −0.32 avg (风格反而少带域)
- **真实机制猜测**: VIB 的涨点来自**正则化效应 + EMA 原型锚定**, 而非"解耦更纯净"

### 4. 跨数据集模式: VIB 在弱域异质 (Office) 帮上忙, 在强域异质 (PACS) 反而伤害

- Office 4 域都是办公室物品 (弱异质), VIB 的"压缩语义信息"刚好去掉无用 noise, 涨 +0.76
- PACS 4 域完全不同风格 (强异质), VIB 压缩把**有用的域信号**也压没了, 伤害 −0.72
- 建议: paper 叙事强调"regime-dependent", VIB 是 Office 专属 trick, 不通用

### 5. orth_uc1 的 probe 本身很差 (Office linear probe_sty_class 96%)

- 所有方法都在 94-96%, 说明正交损失只是几何正交, 没实现**信息论解耦**
- 这对 paper 是警钟: 如 reviewer 用 probe 打, 我们的"解耦"叙事很脆弱
- 但 PACS 上 orth_only 的 linear probe_sty_class 只有 0.24 (EXP-109), 说明 **PACS 上解耦真做到了**

## 🎯 胜负判决 (对齐 CLAUDE.md 0 节硬标准)

| 指标 | 阈值 | 本实验最佳 | 判决 |
|------|:---:|:-----:|:---:|
| PACS AVG Best | > 79.91 | orth_uc1 **80.64** (+0.73) | ✅ 已胜 (不是本实验 VIB) |
| Office AVG Best | > 90.58 | A VIB **89.93** (−0.65) | ❌ 未达 |

**现状**: PACS 通过 orth_uc1 已达标 (EXP-109 就已完成), Office 仍差 0.65pp. 本实验 A VIB 把 Office 从 89.17 推到 89.93, 但仍未过 FDSE.

## 下一步

1. **Office 仍 -0.65 差 FDSE**: 尝试 A+C 组合 (VIB + SupCon 但不走 B 的 σ-head 混合路径) 或 R500
2. **PACS probe 待跑**: 12 个 PACS ckpt 已保存, 能验证 "PACS 上 VIB 是否真的压了 probe"
3. **重要**: 决定 paper 叙事方向 — 诚实的 "VIB 通过正则化 + 原型锚定涨点" vs 强行卖"更好解耦"
4. **跨数据集迁移验证**: 下一步 DomainNet orth_uc1 × 3-seed R200 (用户已决定今晚部署)

## 📎 相关文件

- Configs: `FDSE_CVPR25/config/{pacs,office}/feddsa_{vib,vsc,supcon}_*_r200.yml`
- 新算法代码: `FDSE_CVPR25/algorithm/feddsa_sgpa_vib.py` + `algorithm/common/{vib,supcon,diagnostic_ext}.py`
- 单元测试: 50/50 全绿 (tests/test_{vib,supcon,diagnostic_ext,integration_smoke,fl_pack_unpack}.py)
- Office Probe: `experiments/ablation/EXP-113_vib_vsc_supcon/office/probes/*.json` (12 个)
- 设计文档: `obsidian_exprtiment_results/2026-04-21/EXP-113_FedDSA-VIB-VSC_流程_完整技术版.md`
- 上游依赖: EXP-109 (PACS orth_only baseline), EXP-110 (Office orth_only), EXP-111 (lo=3/10 强正交 probe 观察)
