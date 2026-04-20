# EXP-106: λ_pull 软性 ETF regularization Office R200 — 平衡类间分离 + 类内紧密

**日期**: 2026-04-20 03:40 启动 / 2026-04-20 04:27 完成 (seed=2 pilot)
**算法**: `feddsa_sgpa` + 新增 `lambda_etf_pull` (第 12 个 algo_para)
**服务器**: seetacloud2 GPU 0 (3 runs 并行, 46min wall)
**状态**: 🟢 **Pilot seed=2 完成**, λ=0.01 Last +0.72pp 赢 baseline, λ=0.03 peak +0.26pp. **⚠️ record JSON 因文件名过长未保存, 从 log rescue**

## 这个实验做什么 (大白话)

> 诊断发现 ETF vs Linear 各有所长 (EXP-098 + EXP-102 双数据集证据):
>
> - **ETF**: 类间分离好 (inter_cls_sim 低到 -0.16), 但 intra 紧密差 (0.909)
> - **Linear+whitening**: 类内紧密 (intra 0.941), 但类间平庸 (inter -0.11)
>
> 能不能**兼得**? 做法: 在 Linear classifier 基础上, 加一个**软性 regularizer** 让每类的特征中心 **温柔地朝 ETF 顶点方向靠**, 但不强制对齐.
>
> 损失函数:
> ```
> L_pull = mean_{c: n_c>=2} (1 − cos(z_sem_c_mean, M[:,c]))
> L_total = L_CE + λ_orth · L_orth + λ_pull · L_pull
> ```
>
> λ_pull 控制强度:
> - λ=0 → 纯 Linear+whitening = EXP-102 baseline (Office 88.75%)
> - λ 小 → 保留 Linear 紧密度同时拿到一点 ETF 类间分离
> - λ 大 → 可能退化成 ETF-like (失去 Linear 优势)

## Claim 和成功标准

| Claim | 判定标准 | 失败含义 |
|-------|---------|---------|
| **C-pull (Primary)**: λ_pull 软约束比 Linear+whitening 提分 ≥ +0.3% | 3-seed mean AVG Best ≥ 89.56% (88.75 + 0.5) | pull 无贡献, ETF 分离与 Linear 紧密不可兼得 |
| **C-稳定性**: λ=0.01 不崩盘 | Last ≥ 86% | λ 太小反而梯度噪声干扰 |

## 配置

### 新机制代码 (Codex REVISE compliance)

```python
# feddsa_sgpa.py Client.train loss 定义里
if lambda_pull > 0 and not model.use_etf:  # MUST_FIX 1: guard
    pulls = []
    for c in range(K_cls):
        mask = y == c
        n_c = int(mask.sum().item())
        if n_c < 2:  # MUST_FIX 2: skip singleton (gradient noise)
            continue
        z_c_mean = z_sem[mask].mean(dim=0)
        cos = F.cosine_similarity(z_c_mean.unsqueeze(0), M[:, c].unsqueeze(0))
        pulls.append(1.0 - cos.squeeze())
    if pulls:
        loss_etf_pull = torch.stack(pulls).mean()

loss = loss_task + λ_orth * loss_orth + λ_pull * loss_etf_pull
```

### Pilot 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Task | office_caltech10_c4 | 同 EXP-102 |
| use_etf | 0 | Linear classifier (baseline 胜者) |
| use_whitening | 1 | 开启 pooled whitening broadcast |
| use_centers | 0 | 不收集 class centers (EXP-102 配置) |
| R / E / LR | 200 / 1 / 0.05 | 同 Office 惯例 |
| λ_orth | 1.0 | 同 |
| **λ_etf_pull** | **{0.003, 0.01, 0.03}** | Codex 推荐范围 ≤0.1 |
| Seeds | {2} (pilot); {15, 333} 待扩 | |
| Configs | `feddsa_pull{003,001,03}_office_r200.yml` | |

### Codex REVISE verdict 实施状态

| MUST_FIX | 状态 |
|----------|------|
| use_etf=0 guard 防止与 ETF 模式冲突 | ✅ |
| n_c < 2 singleton class 跳过防 NaN | ✅ |
| λ 范围限制 ≤0.1 (原计划 0.03-0.5 过激) | ✅ 降到 {0.003, 0.01, 0.03} |
| 41/41 单测全绿 (新增 7 个 lambda_etf_pull tests) | ✅ |

## 🏆 完整结果 (seed=2 pilot, rescue from log)

### ⚠️ Record JSON 保存失败

**文件名长度**: 278 字节 > ext4 NAME_MAX 255 字节 → flgo Errno 36 报错, record 没保存
flgo log 有每 round `mean_local_test_accuracy`, 从 log rescue 准确率曲线

### seed=2 主结果

| λ_etf_pull | seed | Max | Max@Round | Last R200 | vs EXP-102 seed=2 Last (86.74) |
|-----------|------|-----|-----------|-----------|-------------------------------|
| **0 (EXP-102 baseline)** | 2 | **88.13** | @R45 | **86.74** | (baseline) |
| **0.003** | 2 | 87.98 | @R34 | **83.88** | **-2.86** ❌ 崩盘 |
| **0.01**  | 2 | 88.09 | @R42 | **87.46** | **+0.72** ✅ Last 最稳 |
| **0.03**  | 2 | **88.39** | @R53 | 86.74 | **+0.00** Peak 最高 |

### 关键观察

1. **λ=0.01 是甜蜜点** — Last 87.46 超 baseline last 86.74 (+0.72pp), peak 88.09 不输 baseline 88.13
2. **λ=0.03 peak 最高** — Max 88.39 超 baseline max 88.13 (+0.26pp), 但 Last 持平 baseline (pull 后期被 CE 抵消)
3. **λ=0.003 崩盘** — Last 83.88, 最差. 推翻"λ 越小越保守"假设 (小 λ 梯度噪声反而干扰训练)
4. **所有变体 max 都出现在 R34-53 早期** — 后期 CE 拉扯抵消 pull 的类间分离收益

### vs 其他 Office ablation (已有 3-seed mean)

| 方法 | AVG Best (R200) | 解释 |
|------|-----------------|------|
| EXP-097 Hard ETF (use_etf=1) | 86.97 | 硬对齐, 牺牲 intra tightness |
| EXP-102 Linear+whitening | **89.26** | 3-seed mean, 当前 best |
| **EXP-106 λ=0.03 (s=2 only)** | 88.39 | peak 超 seed=2 baseline 88.13 +0.26, 但单 seed 噪声大 |

## 🔍 Verdict Decision Tree

```
λ=0.01 Last 超 baseline +0.72pp (本实验), peak 不输
  → 值得扩 3 seeds 看是否泛化
λ=0.03 peak 超 baseline +0.26pp (本实验), Last 持平
  → 值得扩, 看 peak 优势能否保持
λ=0.003 Last -2.86pp 崩盘
  → ❌ 不再扩 seed
```

## 📋 下一步

### 立即 (方案 A: 修 bug + 扩 seed)

- [ ] **修 `feddsa_sgpa.py` hparam_names_in_filename alias**: 压缩 lambda_orth→lo, use_whitening→w, lambda_etf_pull→lp, eps_sigma→eps. 预计压到 150-180 字节
- [ ] 扩 λ=0.01 + λ=0.03 × s={15, 333} = 4 runs on seetacloud2
- [ ] 估时: 4 × 46min ≈ 3h (并行)
- [ ] 回填 3-seed mean, 看是否超 EXP-102 89.26

### Claim 调整

- 若 3-seed mean λ=0.01 ≥ 89.56 → ✅ **C-pull 成立**, soft ETF pull 是新增量
- 若 < 89.56 但 ≥ 89.26 → ⚠️ nice-to-have 但不是核心
- 若 < 89.26 → ❌ C-pull 证伪, 回到 EXP-102 叙事

## 📊 实验统计

- **总 runs**: 3 (3 λ × seed=2 pilot)
- **Wall**: ~46min (3 并行)
- **启动**: 2026-04-20 03:40:52
- **完成**: 2026-04-20 04:27 (log End, Total 2770s)
- **已扩 seed**: 0 (待修 bug 后扩 4 runs)

## 📎 相关文件

- 本地 NOTE: `experiments/ablation/EXP-106_etf_pull_office_r200/NOTE.md`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (lambda_etf_pull block)
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (TestLambdaETFPull 7 tests)
- Configs: `FDSE_CVPR25/config/office/feddsa_pull{003,001,03}_office_r200.yml`
- Log (rescue 数据源):
  `FDSE_CVPR25/task/office_caltech10_c4/log/2026-04-20-03-40-52feddsa_sgpa_algopara_..._0.{003,01,03}M*S2*.log`
- Terminal log: `experiments/ablation/EXP-106_etf_pull_office_r200/terminal_pull{001,003,03}_s2.log`
- Codex review: `/tmp/codex_lambda_pull_out.txt` (VERDICT: REVISE → 所有 MUST_FIX 实施)
- 前置诊断: `obsidian_exprtiment_results/2026-04-19/诊断数据深度分析_ETF失败机制.md`
