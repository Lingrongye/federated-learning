# EXP-106: Lambda ETF-Pull Office R200 — 平衡类间分离 + 类内紧密

**日期**: 2026-04-20 启动 / 2026-04-20 04:27 完成 (seed=2 pilot)
**状态**: 🟢 Pilot 完成 (3 λ × s=2), rescue from log (JSON save 失败: 文件名 >255 字节)
**前置**: Codex REVISE verdict 实施:
  - ✅ use_etf=0 guard 加了
  - ✅ n_c < 2 singleton class 跳过
  - ✅ λ 范围降到 {0.003, 0.01, 0.03} (Codex 推荐 ≤0.1)
  - ✅ 41/41 单测全绿 (新增 7 个 lambda_etf_pull tests)

## 大白话

诊断发现 ETF vs Linear 各有所长: ETF 类间分离好, Linear 类内紧密. 能不能**兼得**?

**做法**: 在 Linear classifier 基础上, 加一个**软性 regularizer** 让每类的特征中心 **温柔地朝 ETF 顶点方向靠**, 但不强制对齐. λ 控制强度:
- λ=0 → 纯 Linear+whitening = EXP-102 baseline 89.26%
- λ 小 → 保留 Linear 紧密度的同时拿到一点 ETF 类间分离
- λ 大 → 可能退化成 ETF-like (失去 Linear 优势)

## 实验配置

| Config | λ | 目的 |
|--------|----|------|
| EXP-106_pull003 | 0.003 | 最保守 |
| EXP-106_pull01  | 0.01  | Codex 推荐起点 |
| EXP-106_pull03  | 0.03  | 中等 |
| EXP-102 baseline | 0 | 已有 3-seed mean 89.26% |

**采样策略** (Codex 建议): **3 seeds {2, 15, 333}** × 最优 λ, 而非 5λ × 1 seed (FL 方差高)

**Pilot (先跑)**: 3 个 λ × seed=2 R200 = 3 runs, 找 λ* 后扩 3 seeds

## 新机制代码 (Codex compliance)

```python
# feddsa_sgpa.py Client.train loss 定义里
if lambda_pull > 0 and not model.use_etf:  # guard: 仅 Linear 模式
    pulls = []
    for c in range(K_cls):
        mask = y == c
        n_c = int(mask.sum().item())
        if n_c < 2:  # skip singleton (gradient noise)
            continue
        z_c_mean = z_sem[mask].mean(dim=0)
        cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
        pulls.append(1.0 - cos.squeeze())
    if pulls:
        loss_etf_pull = torch.stack(pulls).mean()

loss = loss_task + λ_orth * loss_orth + λ_pull * loss_etf_pull
```

## Claim

| λ | 目标 | 失败含义 |
|---|------|---------|
| 0.003 | AVG Best ≥ 89.26 (≥ EXP-102) | pull 无贡献 |
| 0.01  | AVG Best ≥ 89.50 (+0.2 over EXP-102) | λ 范围不对 |
| 0.03  | 可能最好或最差 (边界探索) | ETF 约束太强 |

## 结果 (2026-04-20 rescue from log)

**⚠️ 保存 record 失败**: 文件名长度 278 字节 > ext4 NAME_MAX 255 字节
- flgo log 完整 (有每个 round 的 `mean_local_test_accuracy`), 从 log rescue 出结果
- 需要修 `feddsa_sgpa.py` 的 `hparam_names_in_filename` 压缩 alias (见 **待办** 段)

### Seed=2 Pilot 结果 (从 rescued JSON 提全量指标)

| λ | seed | ALL Best | ALL Last | AVG Best | AVG Last | AVG Max@R | vs EXP-102 AVG Last |
|---|------|----------|----------|----------|----------|-----------|--------------------|
| **EXP-102 whiten only** | 2 | **81.36** | **79.77** | **88.13** | **86.74** | @R45 | (baseline) |
| 0.003 | 2 | 80.57 | 77.37 | 87.98 | 83.88 | @R34 | **-2.86** ❌ 后期崩 |
| 0.01  | 2 | 80.96 | **79.78** | 88.09 | **87.46** | @R42 | **+0.72** ✅ |
| 0.03  | 2 | **81.36** | 79.77 | **88.39** | 86.74 | @R53 | **+0.00** 持平 |

**Δ vs EXP-102 (四指标)**:

| λ | Δ ALL Best | Δ ALL Last | Δ AVG Best | Δ AVG Last |
|---|-----------|-----------|-----------|-----------|
| 0.003 | -0.79 | -2.40 | -0.15 | -2.86 |
| 0.01  | -0.40 | +0.01 | -0.04 | **+0.72** |
| 0.03  | **±0.00** | **±0.00** | **+0.26** | **±0.00** |

**观察**:
1. **λ=0.01 last 最稳** — 后期不崩反升 0.72pp
2. **λ=0.03 peak 最高** — 88.39 超 baseline peak 88.13 (+0.26pp)
3. **λ=0.003 崩盘** — Last 83.88, 反证 "λ 越小越保守" 错误 (小 λ 梯度噪声反而干扰训练)
4. 所有变体 max 都出现在 R34-53 早期 — 后期 CE 拉扯把 pull 的早期收益抵消

### vs 其他 ablation (已有 3-seed mean)

| 方法 | AVG Best (R200) | AVG Last | 解释 |
|------|-----------------|----------|------|
| EXP-097 hard ETF (use_etf=1) | 86.97 | — | 硬对齐, 牺牲 intra tightness |
| EXP-102 whitening only | **89.26** | — | 3-seed mean, 当前 best |
| EXP-106 λ=0.03 (s=2 only) | 88.39 | 86.74 | **peak 超 seed=2 baseline**, 但只 1 seed |

### 下一步决策

**方案 A**: 扩 λ=0.01 和 λ=0.03 到 3 seeds (s=15, s=333)
- λ=0.01 last 最稳, 值得看是否泛化
- λ=0.03 peak 最高
- 估时: 6 runs × 46min ≈ 4.6h (seetacloud2 单卡 4090)

**方案 B**: 承认 pull 机制 marginal, 聚焦 SGPA 诊断 (EXP-099 fallback_rate=1.00 问题)

**倾向**: 方案 A, 但先修文件名问题避免 rescue

## 待办

- [ ] **修 `feddsa_sgpa.py` 的 hparam_names_in_filename**: 压缩 alias, 避免 >255 字节 (此 bug 影响所有带 12 algo_para 的实验)
- [ ] 方案 A: 扩 λ=0.01 + λ=0.03 × s={15, 333} 4 runs
- [ ] 同步 Obsidian

## 📎 相关

- Codex 审核: `/tmp/codex_lambda_pull_out.txt` (VERDICT: REVISE → 所有 MUST_FIX 实施)
- 前置诊断: `obsidian_exprtiment_results/2026-04-19/诊断数据深度分析_ETF失败机制.md`
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (TestLambdaETFPull 7 tests)
