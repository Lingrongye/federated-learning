# 2026-04-20 每日总结

**主题**: PACS 复刻验证 Office 结论 + SGPA 推理 C3 证伪 + λ_pull 平衡探索

## 一句话概要

- **PACS 再次证伪 ETF**: Linear 全面胜 SGPA, Last +5.59pp (ETF 后期严重退化)
- **SGPA 推理完全失败**: fallback_rate=1.00, proto 机制 0 触发, C3 死
- **λ_pull 软约束有戏**: seed=2 pilot λ=0.01 Last +0.72pp 赢 baseline
- **发现 bug**: 12 algo_para 拼接文件名 278 > 255 字节, EXP-106 record JSON 未保存

---

## 完成实验列表 (4 个)

| EXP | 数据集 | seeds | 状态 | 关键结果 |
|-----|-------|-------|------|---------|
| EXP-098 | PACS | 3 | ✅ DONE | SGPA 78.96 / Linear 80.20, Δ Last +5.59 |
| EXP-099 | Office | 1 (推理) | 🔴 证伪 | fallback_rate=1.00, proto_vs_etf_gain=0 |
| EXP-105 | Office | 1 (s=2) | ✅ DONE | checkpoint 5 artifact 保存, 精度 87.56 |
| EXP-106 | Office | 1 (s=2 pilot) | 🟢 pilot | λ=0.01 +0.72pp, λ=0.03 peak +0.26pp |

---

## 核心发现

### 1. Linear 双数据集碾压 Hard ETF

| 方法 | Office AVG Best | PACS AVG Best | Office Last | PACS Last |
|------|-----------------|---------------|-------------|-----------|
| Plan A orth_only | 82.55 | 81.69 | 81.35 | 73.87 |
| SGPA (use_etf=1) | 86.97 | 78.96 | 85.44 | 73.77 |
| **Linear+whitening** | **88.75** 🔥 | **80.20** | **86.91** | **79.36** |
| Δ Linear − SGPA | **+1.78** | **+1.24** | +1.47 | **+5.59** |

两个数据集 × 3 seeds × R200 × Linear 对照 **共 6 组独立证据, 全部证伪 ETF 的几何贡献 claim**.

### 2. SGPA 推理机制在 Office 场景下 0 触发

- EXP-099 seed=2 × 4 clients × ~251 query ≈ **1000+ 样本, 0 个通过双 gate**
- 默认 τ_H_q=0.5 / τ_S_q=0.3 warmup 校准下完全失效
- 论文 "dominant contribution = 推理端原型校正" 故事死

### 3. λ_pull 软约束显示 promise (但需 3-seed 验证)

seed=2 pilot:
- **λ=0.003 崩盘** (Last -2.86, 推翻"小 λ 保守"假设)
- **λ=0.01 Last 最稳** (+0.72pp vs baseline)
- **λ=0.03 Peak 最高** (+0.26pp vs baseline)

所有 max 都在 R34-53 早期 → 说明 pull 早期给类间分离, 后期被 CE 抵消.

### 4. PACS 暴露 whitening broadcast 的数据集边界

| Office | PACS |
|--------|------|
| Linear+whitening 88.75 赢 Plan A 82.55 +6.20pp 🔥 | Linear+whitening 80.20 **输** Plan A 81.69 -1.49pp ⚠️ |

**猜测**: Office E=1 短本地训练 → 广播一致 whitening 有效; PACS E=5 长本地训练 → whitening 被本地 BN drift 覆盖.

---

## 发现的 Bug

### bug-1: 文件名长度 278 > ext4 NAME_MAX 255

**现象**: EXP-106 3 runs R200 完成但 record JSON 未保存 (flgo Errno 36)
**根因**: `feddsa_sgpa.py` 的 `hparam_names_in_filename` 包含 12 个全名参数, 拼接后溢出
**修复方案** (待做): 压缩 alias
- `lambda_orth` → `lo`
- `use_whitening` → `w`
- `lambda_etf_pull` → `lp`
- `eps_sigma` → `eps`
- `min_clients_whiten` → `mcw`

预计压缩后 150-180 字节, 安全余量足够.

### 临时对策
从 flgo log 的 `mean_local_test_accuracy` 序列 rescue 结果 (见 `rr/extract_exp106.sh`).

---

## 论文叙事影响

### 已死的 claim

- ❌ **C1**: SGPA > Plan A (PACS 上 -2.73pp 证伪)
- ❌ **C2**: ETF 本身是 gain 来源 (Office/PACS 都证伪)
- ❌ **C3**: SGPA 推理端双 gate + proto 校正 (EXP-099 0 触发)

### 唯一还活着的

- ✅ **"Pooled source-style broadcast"**: Office +6.20pp dominant, 但 PACS 边界未明

### 新骨架 (Office-only 版)

```
Dominant Contribution:  Pooled whitening broadcast (Office-specific)
Subordinate:            L_orth decouple, λ_etf_pull soft regularizer (pilot)
Dead (delete):          Fixed ETF classifier, SGPA double-gate inference,
                        class_centers 收集 (待 EXP-103 确认)
```

### 风险

PACS 上 Linear+whitening **输 Plan A**, 意味着如果只做 Office, 审稿人会问 "why not PACS?". 两个出路:
1. 把 whitening 限定为 Office 场景的 ablation, 主论文用 Plan A + 新 mechanism
2. 找一个 PACS 也能用的变体 (λ_pull 是候选)

---

## 下一步待做

1. [ ] **修 hparam_names_in_filename alias** (阻塞 EXP-106 扩 seed)
2. [ ] **扩 EXP-106 λ={0.01, 0.03} × seed={15, 333} = 4 runs** (seetacloud2, ~3h)
3. [ ] **PACS 跑 EXP-102 等价** (use_whitening=1, use_centers=0, use_etf=0) 3 seeds 确认 whitening 在 PACS 的 ablation
4. [ ] **回填 EXP-097/098 per-seed per-domain 诊断数据** (若需要)

---

## Commit 记录

- `c11f812` feat: EXP-106 lambda_etf_pull 软性 ETF regularization (Codex REVISE 实施完成)
- `011b160` feat: EXP-099 run_sgpa_inference.py 完整版 (接入 flgo.init task loader)
- `1d0b1d3` 结果: EXP-099 SGPA 证伪 + EXP-106 λ_pull pilot rescue from log
- `c0241d0` 结果: EXP-098 PACS (SGPA+Linear) + EXP-099/106 Office 结果回填
