# EXP-095: SCPR — Self-Masked Style-Weighted Multi-Positive InfoNCE

**日期**:2026-04-19 启动
**算法**:`feddsa_scheduled` (scpr=1/2, sas=0)
**服务器**:seetacloud2(单卡 24GB,主力);seetacloud 失联暂不依赖
**状态**:🔄 代码已就位,待部署

## 这个实验做什么(大白话)

> 把 M3 "所有同类域原型等权拉近"改成"按风格相似度加权拉近",自己不算(self-mask)。
>
> - **scpr=0**:关闭,等价 Plan A orth_only(无 InfoNCE)
> - **scpr=1**:uniform 多原型拉近(= M3 下界,不用 style)
> - **scpr=2**:按风格相似度加权(SCPR 主方法,OURS)
>
> 数学保证:`scpr_tau → ∞` 时 scpr=2 严格等于 scpr=1(M3 +5.09% 下界)。

## 跟之前几个实验的关系

| 标识 | 跟 SCPR 的关系 | 数据位置 |
|------|--------------|---------|
| Plan A (EXP-080 PACS / EXP-083 Office) | Claim A.1 / B.1 baseline | 复用 seetacloud2 record |
| SAS (EXP-084) | Claim B.2 baseline | 复用 seetacloud2 record |
| M3 (EXP-072 feddsa_adaptive 128d) | **不可直接 cite**,架构不同;A.2 在 feddsa_scheduled 1024d 重跑 | 新跑 scpr=1 × 3 seeds |
| FDSE | 论文主表对照 | 独立跑 |

## 技术细节

**SCPR 核心公式**(详见 `obsidian/.../2026-04-18_SCPR_v1/FINAL_PROPOSAL.md`):
- Self-masked style attention:`w_{k→j} = softmax_{j≠k}(cos(s_k, s_j) / τ_SCPR)`
- Per-class renormalize:只在有 `p_c^j` 的 client 上归一
- Style-weighted multi-positive SupCon with positives in denominator (standard SupCon)
- 所有 `p_c^j`、`s_j` 均 `.detach()`(软锚点,梯度不回 bank)

**Formal Derivation**(为什么 softmax 是推导结果不是设计):在 imperfect decoupling + 线性噪声近似下,softmax-over-cosine 是 entropy-regularized noise-minimization 的唯一 Boltzmann 最优解。

**Codex review 修复**(2026-04-19):
1. [IMPORTANT] scpr=1 uniform 模式不再依赖 style bank(M3 style-free)
2. [IMPORTANT] scpr>0 时 disable legacy InfoNCE block,保证"scpr=1 = M3"契约
3. [MINOR] docstring 更新为标准 SupCon 公式(positives in denominator)

## 配置矩阵

| 配置 | scpr | scpr_tau | sas | 用途 | Config 文件 |
|------|------|----------|-----|------|-----------|
| Plan A | 0 | — | 0 | A.1/B.1 baseline(复用) | 已有 |
| SAS | 0 | — | 1 | B.2 baseline(Office,复用) | 已有 |
| **SCPR uniform** | **1** | — | 0 | A.2(M3 in-codepath) | `pacs/feddsa_scpr_uniform.yml` |
| **SCPR τ=0.3** | **2** | **0.3** | 0 | A.3 / B.3 主方法 | `pacs,office/feddsa_scpr_tau03.yml` |
| SCPR τ=0.1 | 2 | 0.1 | 0 | τ 扫(sharp) | `pacs/feddsa_scpr_tau01.yml` |
| SCPR τ=1.0 | 2 | 1.0 | 0 | τ 扫 | `pacs/feddsa_scpr_tau10.yml` |
| SCPR τ=3.0 | 2 | 3.0 | 0 | τ 扫(near uniform) | `pacs/feddsa_scpr_tau30.yml` |
| SCPR + SAS | 2 | 0.3 | 1 | 附录 composability | `office/feddsa_scpr_sas.yml` |

**共 8 个 config × 3 seeds = 24 runs**(不含 sanity)。实际新增 22 runs(见 TRACKER)。

## 共享超参(所有 SCPR configs)

| 参数 | 值 |
|------|-----|
| lo (λ_orth) | 1.0 |
| lh (λ_hsic) | 0.0 |
| ls (λ_sem) | 1.0 |
| tau (InfoNCE 温度) | 0.2 |
| sm (schedule_mode) | 0 (orth_only, SCPR 是唯一对齐 loss) |
| se (save best) | 1 |
| LR | 0.05 |
| R | 200 |
| E(PACS) | 5 |
| E(Office) | 1 |
| B | 50 |
| seeds | 2, 15, 333 |

## 🏆 结果槽位(Phase 5 回填)

### Claim A:PACS 全 outlier,SCPR > M3 uniform

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Art | Cartoon | Photo | Sketch |
|------|------|---------|---------|---------|---------|-----|---------|-------|--------|
| **A.1 Plan A orth_only**(EXP-080,复用) | 2   | — | — | — | — | — | — | — | — |
|                                           | 15  | — | — | — | — | — | — | — | — |
|                                           | 333 | — | — | — | — | — | — | — | — |
|                                           | **mean** | — | — | — | — | — | — | — | — |
| **A.2 SCPR uniform**(scpr=1, 新跑) | 2   | — | — | — | — | — | — | — | — |
|                                     | 15  | — | — | — | — | — | — | — | — |
|                                     | 333 | — | — | — | — | — | — | — | — |
|                                     | **mean** | — | — | — | — | — | — | — | — |
| **A.3 SCPR τ=0.3**(scpr=2, OURS) | 2   | — | — | — | — | — | — | — | — |
|                                    | 15  | — | — | — | — | — | — | — | — |
|                                    | 333 | — | — | — | — | — | — | — | — |
|                                    | **mean** | — | — | — | — | — | — | — | — |
| **Δ A.3 − A.2**(核心 claim)      | —   | — | — | — | — | — | — | — | — |
| **Δ A.3 − A.1**                    | —   | — | — | — | — | — | — | — | — |

### Claim B:Office 单 outlier,SCPR ≥ SAS

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Caltech | Amazon | DSLR | Webcam |
|------|------|---------|---------|---------|---------|---------|--------|------|--------|
| **B.1 Plan A orth_only**(EXP-083,复用) | 2/15/333 mean | — | — | — | — | — | — | — | — |
| **B.2 SAS τ=0.3**(EXP-084,复用)        | 2/15/333 mean | — | — | 89.82 | 88.28 | 75.0 | — | — | — |
| **B.3 SCPR τ=0.3**(scpr=2, OURS)       | 2/15/333 mean | — | — | — | — | — | — | — | — |
| **Δ B.3 − B.2**                         | —   | — | — | — | — | — | — | — | — |

### Claim C:τ 敏感性 + outlier-ness 机制诊断

| τ_SCPR | PACS AVG Best 3-seed mean | H(w_k) 末轮均值 | Spearman ρ(iso_k, gain_k) |
|--------|---------------------------|---------------|--------------------------|
| 0.1 | — | — | — |
| 0.3(= A.3) | — | — | — |
| 1.0 | — | — | — |
| 3.0 | — | — | — |

### 附录:SCPR + SAS Composability (Office)

| 配置 | AVG Best 3-seed mean | Δ vs SAS only (B.2) | Δ vs SCPR only (B.3) |
|------|---------------------|--------------------|---------------------|
| B.2 SAS only | 89.82 | — | — |
| B.3 SCPR only | — | — | — |
| SCPR + SAS | — | — | — |

## Runs 状态(Tracker 同步)

共 22 新 runs(含 1 sanity)。详见 `obsidian/../2026-04-18_SCPR_v1/EXPERIMENT_TRACKER.md`。

## 启动顺序

1. **M0 Sanity**:R001 PACS scpr=2 τ=0.3 s=2 R=20 × ~30min → 无 NaN 再继续
2. **M1 补齐 A.2**:R002-R004 PACS scpr=1 × {2,15,333}
3. **M2 主实验**:R005-R010 PACS/Office scpr=2 τ=0.3 × {2,15,333}
4. **M3 τ 扫**:R011-R019 PACS scpr=2 × τ{0.1,1.0,3.0} × {2,15,333}
5. **M4 附录**:R020-R022 Office SCPR+SAS × {2,15,333}

## 监控节奏(CLAUDE.md 要求)

- T+1min:ps 进程 + nvidia-smi + log 无立即崩
- T+5min:log 前 10 轮 round 递增 + 无 NaN
- T+1h 每小时:抽查 log 末尾 round + NaN 检测 + OOM 检测
- 完成后:collect_results.py → 回填 NOTE.md

## 结论

(实验完成后填)

## 变更记录

- 2026-04-19 初版:基于 GPT-5.4 refine READY 9.1/10 设计 + codex code review MUST_FIX 3 issue 修复
