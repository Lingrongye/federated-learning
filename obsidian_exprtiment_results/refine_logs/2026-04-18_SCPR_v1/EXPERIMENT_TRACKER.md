# Experiment Tracker — SCPR

**日期**:2026-04-19
**计划**:`EXPERIMENT_PLAN.md`
**算法**:feddsa_scheduled.py(将新增 scpr / scpr_tau 参数)
**实验 ID 前缀**:EXP-095_scpr
**seeds**:{2, 15, 333} 严格对齐
**服务器**:seetacloud2(主力,24GB 空闲)

---

## 新增 Runs 清单(按执行顺序)

| Run ID | Milestone | 用途 | Config / 变体 | Task | Seed | GPU | Priority | Status | Notes |
|--------|-----------|------|--------------|------|------|-----|----------|--------|-------|
| R001 | M0 sanity | SCPR 初检 | scpr=2 τ=0.3 R=20 | PACS_c4 | 2 | 0 | MUST | TODO | 前 20 轮验证代码不崩 |
| R002 | M1 补齐 A.2 | M3 in-codepath | scpr=1 | PACS_c4 | 2 | 0 | MUST | TODO | — |
| R003 | M1 补齐 A.2 | M3 in-codepath | scpr=1 | PACS_c4 | 15 | 0 | MUST | TODO | — |
| R004 | M1 补齐 A.2 | M3 in-codepath | scpr=1 | PACS_c4 | 333 | 0 | MUST | TODO | — |
| R005 | M2 主 A.3 | SCPR τ=0.3 | scpr=2 τ=0.3 | PACS_c4 | 2 | 0 | MUST | TODO | — |
| R006 | M2 主 A.3 | SCPR τ=0.3 | scpr=2 τ=0.3 | PACS_c4 | 15 | 0 | MUST | TODO | — |
| R007 | M2 主 A.3 | SCPR τ=0.3 | scpr=2 τ=0.3 | PACS_c4 | 333 | 0 | MUST | TODO | — |
| R008 | M2 主 B.3 | SCPR τ=0.3 | scpr=2 τ=0.3 | office_caltech10_c4 | 2 | 0 | MUST | TODO | — |
| R009 | M2 主 B.3 | SCPR τ=0.3 | scpr=2 τ=0.3 | office_caltech10_c4 | 15 | 0 | MUST | TODO | — |
| R010 | M2 主 B.3 | SCPR τ=0.3 | scpr=2 τ=0.3 | office_caltech10_c4 | 333 | 0 | MUST | TODO | — |
| R011 | M3 τ 扫 | SCPR τ=0.1 | scpr=2 τ=0.1 | PACS_c4 | 2 | 0 | MUST | TODO | 低 τ 风险:可能 NaN |
| R012 | M3 τ 扫 | SCPR τ=0.1 | scpr=2 τ=0.1 | PACS_c4 | 15 | 0 | MUST | TODO | — |
| R013 | M3 τ 扫 | SCPR τ=0.1 | scpr=2 τ=0.1 | PACS_c4 | 333 | 0 | MUST | TODO | — |
| R014 | M3 τ 扫 | SCPR τ=1.0 | scpr=2 τ=1.0 | PACS_c4 | 2 | 0 | MUST | TODO | — |
| R015 | M3 τ 扫 | SCPR τ=1.0 | scpr=2 τ=1.0 | PACS_c4 | 15 | 0 | MUST | TODO | — |
| R016 | M3 τ 扫 | SCPR τ=1.0 | scpr=2 τ=1.0 | PACS_c4 | 333 | 0 | MUST | TODO | — |
| R017 | M3 τ 扫 | SCPR τ=3.0 | scpr=2 τ=3.0 | PACS_c4 | 2 | 0 | MUST | TODO | 高 τ:近似 uniform |
| R018 | M3 τ 扫 | SCPR τ=3.0 | scpr=2 τ=3.0 | PACS_c4 | 15 | 0 | MUST | TODO | — |
| R019 | M3 τ 扫 | SCPR τ=3.0 | scpr=2 τ=3.0 | PACS_c4 | 333 | 0 | MUST | TODO | — |
| R020 | M4 附录 | SCPR+SAS | scpr=2 sas=1 τ=0.3 | office_caltech10_c4 | 2 | 0 | NICE | TODO | composability |
| R021 | M4 附录 | SCPR+SAS | scpr=2 sas=1 τ=0.3 | office_caltech10_c4 | 15 | 0 | NICE | TODO | — |
| R022 | M4 附录 | SCPR+SAS | scpr=2 sas=1 τ=0.3 | office_caltech10_c4 | 333 | 0 | NICE | TODO | — |

**新增 runs 总数**:22(含 R001 sanity)
**预计 GPU·h**:R001 ≈ 0.2h + 其余 21 runs × ~1.5h ≈ 32.2h wall-clock(不并行);并行 3 约 **12h**。

---

## 复用(不重跑)的 Runs

| 来源 | 用途 | 数据位置 |
|------|------|---------|
| EXP-080 PACS orth_only LR=0.05 3-seed | Claim A.1 baseline | seetacloud2 `task/PACS_c4/record/` |
| EXP-083 Office orth_only LR=0.05 3-seed | Claim B.1 baseline | seetacloud2 `task/office_caltech10_c4/record/` |
| EXP-084 Office SAS τ=0.3 3-seed | Claim B.2 baseline | seetacloud2 `task/office_caltech10_c4/record/` |

**验收**:Phase 1-B 完成时要 SSH 确认这三组每 seed 的 JSON record 都在,并提取 ALL Best/Last + AVG Best/Last + per-domain 数据到对照行。

---

## Fallback / Abort 规则

| 触发 | 动作 |
|------|------|
| R001 sanity 前 20 轮 NaN | **abort 全部**,查 code,重新 review |
| R001 正常但 R005-R007(A.3 主)任一 seed < 75% at R=50 | 暂停 τ 扫和附录,先查 attention entropy,定位问题 |
| 3-seed mean A.3 < A.2 | **abort τ 扫**,降级论文 claim 为 Office 专属,跑 qualitative |
| OOM 3+ 次 | 并行度降 1,单 run 排查显存 leak |

---

## 进度更新(每小时刷新)

> 启动后每小时在此追加一行:`[HH:MM] {完成 Runs} / 22, GPU{占用情况}, 任何异常`

```
[待开始]
```
