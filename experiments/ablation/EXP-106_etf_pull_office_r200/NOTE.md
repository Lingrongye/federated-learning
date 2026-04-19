# EXP-106: Lambda ETF-Pull Office R200 — 平衡类间分离 + 类内紧密

**日期**: 2026-04-20 启动 / 待完成
**状态**: 🟡 部署中
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

## 结果 (待回填)

| λ | seed | AVG Best | AVG Last | intra_cls_sim R200 | etf_align R200 | loss_etf_pull R200 |
|---|------|----------|----------|--------------------|-----------------|---------------------|
| 0.003 | 2 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 0.01  | 2 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 0.03  | 2 | 待填 | 待填 | 待填 | 待填 | 待填 |

**对照 (已有)**:
- EXP-102 λ=0 mean 89.26 (intra=0.941 ± Linear; etf_align=-0.002)
- EXP-097 use_etf=1 (硬 ETF) mean 86.97 (intra=0.909; etf_align=0.951)

## 📎 相关

- Codex 审核: `/tmp/codex_lambda_pull_out.txt` (VERDICT: REVISE → 所有 MUST_FIX 实施)
- 前置诊断: `obsidian_exprtiment_results/2026-04-19/诊断数据深度分析_ETF失败机制.md`
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (TestLambdaETFPull 7 tests)
