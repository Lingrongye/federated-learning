# EXP-108: FedDSA-CDANN Office + PACS R200 3-seed — 方案 v4 首次完整验证

**日期**: 2026-04-20 设计 / 待部署 / 待完成
**算法**: `feddsa_sgpa` (ca=1, **CDANN 首次启用**)
**服务器**: seetacloud2 GPU 0 (6 runs 并行, 预估 ~12h wall)
**状态**: 🟡 部署中

## 这个实验做什么 (大白话)

> FedDSA-CDANN 是 2026-04-20 完成的 5 轮 research-refine (Codex gpt-5.4 xhigh) 产出方案 (最终 8.75/10 proposal-complete, near ceiling). 锁定的 **one-sentence novelty**:
>
> "**Shared non-adversarial domain discriminator + asymmetric encoder-gradient supervision (GRL on z_sem path only) is the minimal repair for whitening-induced style collapse in client=domain FedDG when style carries class signal.**"
>
> 即: 在 FedDSA-SGPA Linear+whitening 的基础上, 加 1 个 MLP (dom_head, 9K 参数) + 1 个 GRL 层, domain 监督信号用 client id 作 label, **让 encoder 学会** "z_sem 看不出域 (GRL 反向梯度), z_sty 看得出域 (正向梯度)". dom_head 自身非对抗 (两路都 minimize CE).
>
> **目的**: 修 EXP-098 PACS Linear+whitening 的 -1.49pp regression (诊断证据: z_sty_norm 塌 95%), 同时维持 EXP-102 Office Linear+whitening 的 +6.20pp gain. 一个机制测两个数据集.

## Claim 和成功标准 (from FINAL_PROPOSAL.md R4)

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C-main (Primary)**: CDANN 修 PACS 不伤 Office | PACS 3-seed AVG Best **≥ 82.2** (回到 Plan A 81.69) **且** Office ≥ 88.0 | 方案无法 address 诊断发现的 failure mode |
| **C-probe (Evidence)**: z_sty 保留 class 判别信号 | PACS `probe_sty_class` ≥ 40% (≫ random 14%); baseline ≈ 15% | 没有 representation-level 证据支持 anchor |
| **C-domain (Mechanism)**: GRL 有效分离 | probe_sem_domain ≈ 25% (random 1/4); probe_sty_domain ≥ 95% | GRL 或正向监督至少一边失效 |
| **C-ablate (GPU 余量)**: 正向监督必要性 | V3(full) > V2(z_sem-only only) 显著 | 单向 DANN 已够用, 正向监督多余 |

## 配置 (严格对齐 EXP-102/098 baseline, 只改 ca)

### Office CDANN (vs EXP-102 对照)

| 参数 | EXP-102 baseline | **EXP-108 Office CDANN** | 说明 |
|------|-----------------|-------------------------|------|
| Task | office_caltech10_c4 | office_caltech10_c4 | 同 |
| Backbone | AlexNet + 双 128d 头 | 同 | |
| Algorithm | feddsa_sgpa | feddsa_sgpa | 同 |
| R / E / LR | 200 / 1 / 0.05 | 同 | |
| λ_orth / λ_hsic | 1.0 / 0.1 | 同 | |
| use_etf | 0 | 0 | Linear |
| use_whitening | 1 | 1 | 保留 +6.20pp 基础 |
| use_centers | 0 | 0 | |
| **diag** | 0 | **1** | **必须 ON 用于 probe** |
| **ca (CDANN)** | 0 | **1** | **唯一核心改动** |
| λ_adv schedule | — | R0-20 off, R20-40 linear, R40+ full=1.0 | |
| Seeds | {2, 15, 333} | 同 | 严格对齐对比 |
| Config | `feddsa_whiten_only_office_r200.yml` | `feddsa_cdann_office_r200.yml` | |

### PACS CDANN (vs EXP-098 Linear 对照)

| 参数 | EXP-098 Linear baseline | **EXP-108 PACS CDANN** | 说明 |
|------|------------------------|------------------------|------|
| Task | PACS_c4 | PACS_c4 | 同 |
| R / E / LR | 200 / 5 / 0.05 | 同 | PACS 惯例 E=5 |
| use_etf | 0 | 0 | |
| use_whitening | 1 (implicit 默认) | 1 | |
| use_centers | 1 (implicit 默认) | 1 | 严格对齐 EXP-098 |
| diag | 1 | 1 | |
| **ca** | 0 | **1** | |
| Seeds | {2, 15, 333} | 同 | |
| Config | (EXP-098 早期 8-param) | `feddsa_cdann_pacs_r200.yml` | |

### 新 trainable 组件 (R4 锁定)

- `dom_head = MLP(128 → 64 → 4)` + Dropout(0.1), ~**9K params**, 参与 FedAvg 聚合
- `GradientReverseLayer`: forward 恒等, backward 乘 -λ_adv, **无参数**
- `λ_adv(r)`: R=0-20=0 / 20-40 linear / ≥40=1.0

### Loss

```
L = L_task + λ_orth · L_orth + λ_hsic · HSIC + L_dom_sem + L_dom_sty
  = CE(y, sem_classifier(z_sem))
  + 1.0 · cos²(z_sem, z_sty) + 0.1 · HSIC(z_sem, z_sty)
  + CE(d, dom_head(GRL(z_sem, λ_adv)))
  + CE(d, dom_head(z_sty))
```

- d = client id (每 client 内 batch 内所有样本同 domain label)
- Inference 只走 z_sem → sem_classifier

## 🏆 完整结果 (3-seed mean) — 待回填

### C-main 主对比

**per-domain (Caltech/Amazon/DSLR/Webcam 或 Art/Cart/Photo/Sketch) 格式 Best/Last:**

#### Office

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Caltech | Amazon | DSLR | Webcam |
|------|------|---------|---------|---------|---------|---------|--------|------|--------|
| **Plan A orth_only** (EXP-083) | **mean** | 88.61 | 87.30 | **82.55** | 81.35 | 72.6/70.2 | 90.9/88.1 | 100.0/97.8 | 94.3/93.1 |
| **Linear+whitening** (EXP-102, baseline) | **mean** | 82.81 | 81.09 | **88.75 ± 0.86** | 86.91 | 72.3/70.5 | 88.4/87.4 | 100.0/97.8 | 94.3/92.0 |
| **CDANN (本实验)** | **mean** | 待填 | 待填 | **待填** | 待填 | — | — | — | — |
|  | 2 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
|  | 15 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
|  | 333 | 待填 | 待填 | 待填 | 待填 | — | — | — | — |
| **Δ CDANN − Linear+whitening** | — | 待填 | 待填 | **待填** | 待填 | — | — | — | — |

#### PACS

| 配置 | seed | AVG Best | AVG Last | Art | Cart | Photo | Sketch |
|------|------|---------|---------|-----|------|-------|--------|
| **Plan A orth_only** (EXP-080) | **mean** | **81.69** | 73.87 | — | — | — | — |
| **SGPA (use_etf=1)** (EXP-098) | **mean** | 78.96 ± 0.37 | 73.77 | 62.6/54.6 | 85.0/78.9 | 80.0/74.3 | 88.2/87.3 |
| **Linear+whitening (baseline)** (EXP-098) | **mean** | 80.20 ± 0.94 | 79.36 | 63.4/61.4 | 86.0/84.0 | 81.8/82.4 | 89.5/89.5 |
| **CDANN (本实验)** | **mean** | **待填** | 待填 | — | — | — | — |
|  | 2 | 待填 | 待填 | — | — | — | — |
|  | 15 | 待填 | 待填 | — | — | — | — |
|  | 333 | 待填 | 待填 | — | — | — | — |
| **Δ CDANN − Linear+whitening** | — | 待填 | 待填 | — | — | — | — |
| **Δ CDANN − Plan A** | — | 待填 | 待填 | — | — | — | — |

## 🔬 诊断指标对比 (Layer 1+2+CDANN, R200 mean)

**这是本实验的关键诊断预留** — CDANN 的所有新 metrics 都会被 diag jsonl 记录, 对比 baseline 就能量化 CDANN 机制是否生效.

### Layer 1 (Train-time, 4-client mean at R200)

| 指标 | Linear+whitening (EXP-098/102) | **CDANN (本)** | Δ 预期 |
|------|-------------------------------|---------------|--------|
| intra_cls_sim R200 | Office 0.954 / PACS 0.999 | 待填 | ≈ 保持 |
| inter_cls_sim R200 | Office 0.051 / PACS -0.172 | 待填 | ≈ 保持 |
| **z_sem_norm** R200 | Office 12.39 / PACS 5.79 | 待填 | 期望稳定 (不因 GRL 坍塌) |
| **z_sty_norm** R200 | Office 2.21 / PACS **0.15** | 待填 | **PACS 期望 ≥ 1.5** (风格保留) |
| loss_task R200 | Office 0.0018 / PACS 0.0019 | 待填 | ≈ 保持 |
| loss_orth R200 | — | 待填 | ≈ 保持 |

### CDANN 训练指标 (首次记录, R0-200 trajectory)

| 指标 | 含义 | 期望轨迹 |
|------|------|---------|
| **lambda_adv** | GRL 反向梯度强度 | R0-20=0, R20-40 ramp, R40+=1.0 |
| **loss_dom_sem** | dom_head 在 z_sem 上的 CE | R0-20 随机 (~log N=1.4), R40+ 由于 GRL 对抗应**上升**至 ~log N |
| **loss_dom_sty** | dom_head 在 z_sty 上的 CE | R0-20 随机, R40+ 由于正向监督应**下降**接近 0 |
| **dom_sem_acc_train** | z_sem → domain 训练 acc | R40+ 期望接近 25% (随机) |
| **dom_sty_acc_train** | z_sty → domain 训练 acc | R40+ 期望接近 100% |

### Layer 2 (Server-side, R200)

| 指标 | Linear+whitening | **CDANN** | Δ 预期 |
|------|------------------|-----------|--------|
| client_center_var | Office 待查 / PACS 0.00078 | 待填 | 接近 |
| param_drift | Office 0.0032 / PACS 0.007 | 待填 | 接近 |

### Frozen Post-hoc Probes (训完后单独跑, C-probe + C-domain 核心证据)

| Probe | Linear+whitening baseline | **CDANN** | 期望 |
|-------|---------------------------|-----------|------|
| probe_sem_domain (z_sem → domain) | 高 (无 GRL 压制) | **待填** | ≈ 0.25 (1/N=4) random |
| probe_sty_domain (z_sty → domain) | 中等 | **待填** | **≈ 0.95** (正向监督成功) |
| **probe_sty_class** (z_sty → class, PACS) | **≈ 15%** (whitening 磨) | **待填** | **≥ 40%** (风格保留类信号) |
| probe_sty_class (z_sty → class, Office) | ≈ 15-20% | **待填** | 20-30% (Office 风格不强, neutral) |

**关键 claim 证据**: PACS 的 `probe_sty_class` 差距 (CDANN - baseline) ≥ 25pp 则 C-probe 成立.

## 🔍 Verdict Decision Tree

```
PACS AVG Best ≥ 82.2 且 Office ≥ 88.0 且 PACS probe_sty_class ≥ 40%
  → ✅ C-main + C-probe + C-domain 全部成立, CDANN 方案验证
  → 下一步: /experiment-plan 详细 ablation + paper draft

PACS AVG Best ∈ [80, 82.2) 但 probe_sty_class 显著提升 (≥30pp gap)
  → ⚠️ mechanism 生效 (representation-level) 但 empirical gain 不足
  → 可能是 λ_adv schedule 调校问题, pilot seed=2 可试 λ_adv 提前/延后启动

PACS AVG Best < 80 (掉 Linear baseline)
  → ❌ CDANN 反向拉低 Office + PACS
  → 必须检查: dom_head 是否被 FedAvg 聚合正确 / GRL 是否反向
  → 可能需要 pivot 到 Option I1 (Style-Aware Selective Whitening)

Office AVG Best < 87 (伤 Office > 1.75pp)
  → ⚠️ 机制在 style-weak 数据集副作用
  → λ_adv 降到 0.5, 或只在 PACS 启用 (有数据集 router)
```

## 📋 消融 (GPU 余量做 C-ablate)

| Variant | 配置 | seed 策略 | 目的 |
|---------|------|----------|------|
| V1 baseline | Linear+whitening (EXP-098 已有) | 复用 | baseline |
| V2 z_sem-only | 只 L_dom_sem (标准 DANN) | 2 seeds | 看正向监督必要性 |
| V3 full CDANN (本实验) | L_dom_sem + L_dom_sty | 3 seeds | 完整方案 |

V2 需要加 config `feddsa_cdann_zsemonly_pacs_r200.yml` (ca=2 或单独 flag); 若 GPU 紧张可省略作 future work.

## 📊 实验统计

- **C-main 总 runs**: 6 (Office 3 seeds + PACS 3 seeds)
- **预估 GPU·h**: ~12h wall (6 runs 并行)
- **启动**: 2026-04-20 待
- **完成**: 待

## 📎 相关文件

- 方案文档: `refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md` (8.75/10 after 5 rounds)
- 文献综述: `LITERATURE_SURVEY_DANN_AND_DECOUPLING.md`
- Brainstorm: `IDEA_REPORT_2026-04-20.md`
- 知识笔记 (大白话): `obsidian_exprtiment_results/知识笔记/大白话_FedDSA-CDANN.md`
- 知识笔记 (学术): `obsidian_exprtiment_results/知识笔记/FedDSA-CDANN_技术方案.md`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (GRL + dom_head + L_dom, ~80 行新增)
- Configs: `FDSE_CVPR25/config/{office,pacs}/feddsa_cdann_*_r200.yml`
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (55/55 全绿, 14 新 CDANN tests)
- Probe 脚本: `FDSE_CVPR25/scripts/run_frozen_probes.py`
- Baseline 对照 (Office): `experiments/ablation/EXP-102_full_diag0_office_r200/` (whitening only)
- Baseline 对照 (PACS): `experiments/ablation/EXP-098_sgpa_pacs_r200/` (Linear+whitening + SGPA)
