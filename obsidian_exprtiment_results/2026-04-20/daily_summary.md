# 2026-04-20 每日总结

**主题**: PACS 复刻验证 Office 结论 + SGPA 推理 C3 证伪 + λ_pull 平衡探索 + rescue 工具链

## 一句话概要

- **PACS 再次证伪 ETF**: Linear 全面胜 SGPA, Last +5.59pp (ETF 后期严重退化)
- **SGPA 推理完全失败**: fallback_rate=1.00, proto 机制 0 触发, C3 死
- **λ_pull 软约束有戏**: seed=2 pilot λ=0.01 AVG Last +0.72pp 赢 baseline, λ=0.03 ALL Best 持平
- **发现 bug**: 12 algo_para 拼接文件名 278 > 255 字节, EXP-106 record JSON 未保存
- **Rescue 工具**: 从 flgo log 倒推 record JSON, 精度 PASS (max diff 5e-5 即 0.005pp)

---

## 完成实验列表 (4 个)

| EXP | 数据集 | seeds | 状态 | 关键结果 |
|-----|-------|-------|------|---------|
| [EXP-098](EXP-098_pacs_sgpa_linear.md) | PACS | 3 | ✅ DONE | SGPA AVG 78.96 / Linear 80.20, Δ Last +5.59pp |
| [EXP-099](EXP-099_sgpa_inference.md) | Office | 1 (推理) | 🔴 证伪 | fallback_rate=1.00, proto_vs_etf_gain=0 |
| [EXP-105](EXP-105_linear_whitening_saveckpt.md) | Office | 1 (s=2) | ✅ DONE | checkpoint 5 artifact 保存, 精度 ALL Best 81.36 |
| [EXP-106](EXP-106_lambda_etf_pull_pilot.md) | Office | 1 (s=2 pilot) | 🟢 pilot (rescued) | λ=0.01 AVG Last +0.72pp, λ=0.03 ALL Best 持平 |

---

## 🏆 完整结果汇总 (四指标)

### Office seed=2 EXP-106 pilot 对比 (rescue from log, 精度 <0.005pp)

| 方案 | ALL Best | ALL Last | AVG Best | AVG Last | AVG Max@R |
|------|----------|----------|----------|----------|-----------|
| **EXP-102 baseline (whiten only)** | **81.36** | **79.77** | **88.13** | **86.74** | @R45 |
| EXP-106 λ=0.003 | 80.57 | 77.37 | 87.98 | **83.88** ❌ | @R34 |
| EXP-106 λ=0.01  | 80.96 | **79.78** | 88.09 | **87.46** ✅ | @R42 |
| EXP-106 λ=0.03  | **81.36** | 79.77 | **88.39** | 86.74 | @R53 |

**Δ 矩阵 vs EXP-102 baseline**:

| λ | Δ ALL Best | Δ ALL Last | Δ AVG Best | Δ AVG Last |
|---|-----------|-----------|-----------|-----------|
| 0.003 | -0.79 | -2.40 | -0.15 | **-2.86** |
| 0.01  | -0.40 | +0.01 | -0.04 | **+0.72** |
| 0.03  | **±0.00** | **±0.00** | **+0.26** | **±0.00** |

**小结**:
- λ=0.003 全面劣化 → 小 λ 梯度噪声干扰训练, **不再扩 seed**
- λ=0.01 AVG Last **+0.72pp** 最稳, ALL Last 近持平
- λ=0.03 **ALL Best 81.36 与 baseline 完全持平** (同 round weighted 一致), AVG Best +0.26pp
- ALL 差异 <0.5pp vs AVG 差异 <3pp, 说明 λ_pull 主要影响 **小 client (DSLR 15 / Webcam 29 样本, AVG 权重高)**, 对大 client (Caltech 112 / Amazon 95, ALL 权重高) 影响小

### PACS EXP-098 3-seed mean

| 配置 | AVG Best | AVG Last | Art B/L | Cart B/L | Photo B/L | Sketch B/L |
|------|----------|----------|---------|----------|-----------|------------|
| Plan A orth_only (EXP-080) | 81.69 | 73.87 | — | — | — | — |
| SGPA (use_etf=1) | 78.96±0.37 | 73.77 | 62.6/54.6 | 85.0/78.9 | 80.0/74.3 | 88.2/87.3 |
| **Linear+whitening** | **80.20±0.94** | **79.36** | 63.4/61.4 | 86.0/84.0 | 81.8/82.4 | 89.5/89.5 |
| Δ Linear − SGPA | **+1.24** | **+5.59** | +0.8/+6.8 | +1.0/+5.1 | +1.8/+8.1 | +1.3/+2.2 |

### Office + PACS 双数据集对照

| 方法 | Office AVG Best | PACS AVG Best | Office AVG Last | PACS AVG Last |
|------|-----------------|---------------|-----------------|---------------|
| Plan A orth_only | 82.55 | **81.69** | 81.35 | 73.87 |
| SGPA (use_etf=1) | 86.97 | 78.96 | 85.44 | 73.77 |
| **Linear+whitening** | **88.75** 🔥 | 80.20 | **86.91** | **79.36** |
| Δ Linear − SGPA | **+1.78** | **+1.24** | +1.47 | **+5.59** |
| Δ Linear − Plan A | **+6.20** | -1.49 | +5.56 | +5.49 |

---

## 核心发现

### 1. Linear 双数据集碾压 Hard ETF

**6 组独立证据 (2 数据集 × 3 seeds)**:
- Office Linear AVG Best 88.75 vs SGPA 86.97 (Δ +1.78)
- PACS Linear AVG Best 80.20 vs SGPA 78.96 (Δ +1.24)
- PACS 上 Linear 甚至把 Last 超出 +5.59pp — **ETF 在 PACS 后期严重退化** (Photo Last -8.1pp 最惨)

### 2. SGPA 推理机制在 Office 场景下 0 触发

- EXP-099 seed=2 × 4 clients × ~251 query ≈ 1000+ 样本, **0 个通过双 gate**
- 默认 τ_H_q=0.5 / τ_S_q=0.3 warmup 校准下完全失效
- 论文 "dominant contribution = 推理端原型校正" 故事死

### 3. λ_pull 软约束 seed=2 初步 promise

- **λ=0.01 AVG Last +0.72pp** 最稳, peak 不输 baseline
- **λ=0.03 Peak AVG +0.26pp** 最高, Last 持平
- **所有 max 都在 R34-53 早期** → pull 给早期类间分离, 后期被 CE 抵消
- ALL 差异 <0.5pp 说明主要影响小 client

### 4. PACS 暴露 whitening broadcast 的数据集边界

| 数据集 | Linear+whitening vs Plan A |
|--------|---------------------------|
| Office E=1 | +6.20pp 🔥 (whitening 广播主导) |
| PACS E=5 | **-1.49pp** ⚠️ (被本地 BN drift 覆盖) |

---

## Rescue 工具链准确性验证

### 场景

EXP-106 3 runs 因文件名 278 > ext4 NAME_MAX 255 触发 Errno 36, flgo save record 失败 (但 R200 训练完整完成, log 里每 round 都有 20 个标量 eval metrics).

### 脚本

- `FDSE_CVPR25/scripts/rescue_record_from_log.py` — 从 flgo log 重建 20 标量序列兼容 JSON
- `FDSE_CVPR25/scripts/verify_rescue_vs_orig.py` — 对比 rescue 与原生 record

### 严格验证 (EXP-102 seed=2 whitening only, 既有原生 record 也有 log)

```
========================================================================
Metric           Original      Rescued          Diff    Diff_pp
========================================================================
ALL Best         0.813565     0.813600   +3.5355e-05   +0.0035pp OK
ALL Last         0.797686     0.797700   +1.3530e-05   +0.0014pp OK
AVG Best         0.881276     0.881300   +2.4073e-05   +0.0024pp OK
AVG Last         0.867392     0.867400   +7.9207e-06   +0.0008pp OK
ALL@R100         0.805657     0.805700   +4.2763e-05   +0.0043pp OK
AVG@R100         0.878245     0.878200   -4.4912e-05   -0.0045pp OK
std@R200         0.113472     0.113500   +2.8461e-05   +0.0028pp OK
min@R200         0.696429     0.696400   -2.8571e-05   -0.0029pp OK
max@R200         1.000000     1.000000    0.0000e+00   +0.0000pp OK
loss@R200        0.733942     0.733900   -4.2208e-05   -0.0042pp OK

Per-round max|Δ| across 20 scalar metrics × 201 rounds: 5.00e-05 (at max_local_val_accuracy)
VERDICT: PASS  (rescue faithful to <=1e-4)
```

**关键文件**:
- 原生 record (对照基准): `FDSE_CVPR25/task/office_caltech10_c4/record/feddsa_sgpa_..._diag0_use_etf0_use_whitening1_use_centers0_...S2_...R200*.json`
- 原 flgo log (数据源): `FDSE_CVPR25/task/office_caltech10_c4/log/2026-04-20-02-06-15*|0|0|1|0M*_S2_*.log`
- Rescue 脚本: `FDSE_CVPR25/scripts/rescue_record_from_log.py`
- 对比报告: 上面 20 标量每一个 201 round 的 max|Δ| = **5.00e-05 (即 0.005pp)**

### 结论

- **Rescue 完全可信**: 4020 数据点 (20 × 201) max |Δ| = 5e-5, 远优于 `1e-4` 阈值
- 精度损失**来自 flgo log 输出时的 4 位小数舍入**, 非脚本 bug
- **业务判断精度 0.005pp 完全不影响结论** — EXP-106 的 AVG Best/Last/ALL Best/Last 差异都 ≥ 0.01pp 级别

### 无法恢复的字段

`local_val_accuracy_dist`, `local_val_loss_dist`, `local_test_accuracy_dist`, `local_test_loss_dist` — flgo `log_once` 不打印 list 字段, 所以 per-client per-round 的 4 维向量丢失. 影响 per-domain R-by-R 追踪, 不影响 AVG/ALL 级别的 Best/Last 分析.

---

## 发现的 Bug

### bug-1: 文件名长度 278 > ext4 NAME_MAX 255

**现象**: EXP-106 3 runs R200 完成但 record JSON 未保存 (flgo Errno 36)
**根因**: `feddsa_sgpa.py` 的 `hparam_names_in_filename` 包含 12 个全名参数, 拼接后溢出
**修复方案** (待做): 压缩 alias
- `lambda_orth` → `lo` (已用)
- `use_whitening` → `w`
- `lambda_etf_pull` → `lp`
- `eps_sigma` → `eps`
- `min_clients_whiten` → `mcw`
- `use_centers` → `uc`
- `use_etf` → `ue`

预计压缩后 150-180 字节, 安全余量足够.

### 临时对策 (已落地)

已跑 `rescue_record_from_log.py` 生成 3 个 rescued JSON 提交到服务器 record 目录 (commit `5df69f0`), 可供 `collect_results.py` 等后续脚本直接使用.

---

## 论文叙事影响

### 已死的 claim (三个接连证伪)

| Claim | 证据 | Verdict |
|-------|-----|---------|
| C1: SGPA > Plan A | PACS AVG Best -2.73pp | ❌ |
| C2: ETF 本身是 gain 来源 | Office Δ Linear-SGPA +1.78, PACS +1.24 | ❌ |
| C3: SGPA 推理端双 gate + proto 校正 | EXP-099 fallback_rate=1.00, proto 0 触发 | ❌ |

### 唯一还活着的 (但有数据集边界)

| Claim | 证据 | Verdict |
|-------|-----|---------|
| "Pooled source-style broadcast" 是主贡献 | Office +6.20pp vs Plan A | ✅ Office |
| ... 在 PACS 上也有效 | PACS -1.49pp 输 Plan A | ⚠️ 数据集敏感 |

### 新骨架 (两条路)

**路 A: Office-specific 叙事**
```
Dominant:    Pooled whitening broadcast (Office +6.20pp)
Subordinate: L_orth decouple, λ_etf_pull soft reg (pilot)
Dead:        Fixed ETF, SGPA double-gate inference, class_centers 收集
```

**路 B: 找一个 PACS 也生效的变体**
- λ_etf_pull soft pull 是当前候选 (Office seed=2 λ=0.01 +0.72pp)
- 必须 PACS 跑 EXP-106 等价验证才能下结论

---

## 下一步待做

1. [ ] **修 `hparam_names_in_filename` alias** — 压缩到 150-180 字节, 解除 bug-1 阻塞
2. [ ] **扩 EXP-106 λ={0.01, 0.03} × seed={15, 333}** 4 runs on seetacloud2 (~3h)
3. [ ] **PACS 跑 EXP-102 等价** (use_whitening=1, use_centers=0, use_etf=0) 3 seeds — 确认 whitening 在 PACS 是否也有 gain (对应 Office +6.20pp)
4. [ ] **PACS 跑 EXP-106 等价** (若 PACS 有 pull 收益, 新骨架路 B 成立)
5. [ ] **回填 EXP-097/098 诊断数据** (若需要 per-domain R-by-R 曲线)

---

## Commit 记录 (04-20)

- `c11f812` feat: EXP-106 lambda_etf_pull 软性 ETF regularization (Codex REVISE 实施完成)
- `011b160` feat: EXP-099 run_sgpa_inference.py 完整版 (接入 flgo.init task loader)
- `1d0b1d3` 结果: EXP-099 SGPA 证伪 + EXP-106 λ_pull pilot rescue from log
- `c0241d0` 结果: EXP-098 PACS (SGPA+Linear) + EXP-099/106 Office 结果回填
- `1b395de` docs: 04-20 Obsidian 同步 EXP-098/099/105/106 NOTE + daily summary + 总览追加
- `037040f` feat: rescue_record_from_log.py 从 flgo log 倒推 record JSON
- `5df69f0` rescue: EXP-106 3 runs record 从 flgo log 倒推
- `acab4d3` docs: EXP-106 NOTE + Obsidian 补 ALL Best/Last (rescue JSON 提全量 4 指标)
- `8e389f9` tool: rescue 准确性验证脚本
