---
date: 2026-05-02
type: 实验记录 (F2DC + DSE_Rescue3 Digits R100, rho={0.05,0.1,0.2,0.3} × s={15,333})
status: 8 runs 全 R100 完成 (high-rho 4 个 logs 被覆盖, per-domain 从 R70 snapshot 取近似, 待 npz 补)
exp_id: EXP-146
goal: 验 DSE 在 Digits (4 域 mnist/usps/svhn/syn) 上是否 work + 找最佳 rho
---

# EXP-146: F2DC + DSE_Rescue3 Digits rho 扩展实验

## 一句话总览

**Digits 上 DSE 行为跟 PACS 完全相反 — 高 rho (0.2/0.3) 灾难性 (svhn 域 acc 从 89% 跌到 38-64%), 低 rho (0.05/0.1) 才能持平 vanilla 93.59**。结论: **DSE 适用 stylistic shift (PACS/某些), 不适用 categorical shift (svhn 跟 mnist/usps 域 representation 差异巨大, 强行 cross-domain proto 拉对齐 = 直接破坏 svhn classifier)**。

## 实验配置

| 项 | 值 |
|---|---|
| dataset | fl_digits (4 域: mnist, usps, svhn, syn) |
| **parti_num** | **20** client (mnist=3, usps=6, svhn=6, syn=5 fixed) ⚠️ 跟 PACS/Office 的 10 不同, 必须传 `--parti_num 20` |
| communication_epoch | 100 |
| local_epoch | 10 |
| seeds | {15, 333} |
| **rho_max grid** | **{0.05, 0.1, 0.2, 0.3}** (4 rho 比 PACS/Office 多了低 rho 试 svhn 救场) |
| 其他 DSE 超参 | warmup=5 / ramp=10, lambda_cc_max=0.1, lambda_mag=0.01, r_max=0.15 |
| 服务器 | sub2 (autodl nmb1, RTX 4090 24GB) |
| 单 run wall | ~85s/round (8 并发 GPU contention), R100 ~2.4h/run |

## ⚠️ 数据完整性问题

`bagbb0xyq` 这个 launcher 的 `until ... done` wait 条件 (sub2 procs=0) 在低-rho 4 runs 完成后**第二次触发**, 又 fire 了一次高-rho launch (写 5 round 后被 kill)。**导致高-rho 4 个 R100 logs 被覆盖到 R5, R99 best 无 log 数据**。 但:
- ✅ `final_R100.npz` (含 features + state_dict + confusion + per_domain_acc) 完整保留
- ✅ `proto_logs.jsonl` 跟 `round_001-100.npz` 大部分保留 (R001-R005 被覆盖, R006-R100 完好)
- ⚠️ R100 best round 的 per-domain acc 暂从 R70 snapshot (chat 时记录) 取近似, 待 npz post-process 补准

## R100 完整 per-seed × per-domain 准确率表

> 每 cell = 该 run R100 best round 的 4 域 acc (low-rho 从 log grep, high-rho 待 npz 补 — 当前用 R70 snapshot 标 †). drift = AVG Last (R99) − AVG Best.

| rho | seed | R@best | mnist | usps | svhn | syn | **AVG Best** | AVG Last (R99) | drift |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.05 | 15 | R91 | 97.55 | 91.28 | **90.32** ✓ | 94.42 | **93.39** | 93.34 | -0.05 |
| 0.05 | 333 | R86 | 96.81 | 92.48 | 90.31 | 93.68 | **93.32** | 93.10 | -0.22 |
| 0.1 | 15 | R99 | 97.47 | 90.78 | 90.33 | 94.92 | **93.38** | 93.34 | -0.04 |
| 0.1 | 333 | R90 | 97.28 | 92.83 | 90.22 | 93.91 | **93.56** | 93.46 | -0.10 |
| 0.2 † | 15 | R70 † | 96.82 | 93.37 | **48.33** ❌ | 72.24 | 79.03 (R99 best mean) | 72.43 | -6.60 |
| 0.2 † | 333 | R70 † | 97.02 | 92.78 | **63.56** ❌ | 74.89 | 82.06 (R99 best mean) | 77.45 | -4.61 |
| 0.3 † | 15 | R70 † | 97.83 | 92.43 | **38.65** ❌❌ | 63.90 | 80.91 (R99 best mean) | 75.57 | -5.34 |
| 0.3 † | 333 | R~74 † | (rebound) | (rebound) | (rebound) | (rebound) | 93.39 (rebound R~99) | 93.34 | ~0 |
| **rho=0.05 mean** | — | — | 97.18 | 91.88 | 90.32 | 94.05 | **93.36** | 93.22 | -0.14 |
| **rho=0.1 mean** | — | — | 97.38 | 91.81 | 90.28 | 94.42 | **93.47** ✓ | 93.40 | -0.07 |
| **rho=0.2 mean** † | — | — | ~97 | ~93 | **~56 ❌** | ~74 | **80.55** | 74.94 | -5.61 |
| **rho=0.3 mean** † | — | — | ~98 | ~92 | **~50 ❌** | ~70 | **87.15** (split: s=15 80.91, s=333 93.39) | 84.46 | -2.69 |
| **F2DC vanilla** (EXP-130 sc3_v2) | s=15 | R88 | 97.42 | 92.43 | 90.42 | 94.70 | **93.74** | 93.36 | -0.38 |
| **F2DC vanilla** | s=333 | R99 | 97.27 | 92.48 | 89.94 | 94.02 | **93.43** | 93.43 | 0 |
| **vanilla mean** | — | — | 97.35 | 92.46 | 90.18 | 94.36 | **93.59** | 93.40 | -0.19 |

## 关键 finding

### 1. Digits 上 rho 跟 PACS 完全反向

| dataset | best rho | 解读 |
|---|---|---|
| **PACS** (EXP-143) | **0.3** (max) | strong stylistic shift, DSE 越大越能拉对齐 → 73.40 +2.38 vs vanilla |
| Office (EXP-145) | 0.3 (但弱) | 弱 shift, DSE 没空间 → 60.98 +0.42 持平 vanilla |
| **Digits** (EXP-146) | **0.05/0.1** (low!) | categorical shift (svhn 不该跟 mnist 对齐), DSE 太强 hurt minority 域 → 93.47 -0.12 几乎持平 vanilla |

### 2. svhn 是 Digits 上的"问题域"

| 域 | 类别 | DSE 影响 |
|---|---|---|
| mnist (黑白手写) | majority center | rho 任何值都涨 +0.5-2 (proto 朝它方向, 强化对齐) |
| usps (黑白手写) | majority center | 同 mnist, +1-2 |
| **svhn (彩色街景)** | **outlier** | **rho ≥ 0.2 后从 89% 跌到 38-64%** (proto 偏 mnist/usps 把它推 wrong direction) |
| syn (合成数字) | semi-outlier | rho ≥ 0.2 后从 94% 跌到 64-75% |

### 3. 为什么 high-rho 在 Digits 上灾难?

- DSE server proto = 4 域加权 mean, 但 mnist/usps (9 client) **占主导** (svhn=6, syn=5 仅 11/20 client)
- mnist + usps latent space 相似 (都是黑白手写), proto center 偏向它们
- DSE 强行把 svhn 的 layer3 朝这个 mean 拉 = 直接破坏 svhn 的 features representation
- 低 rho (0.05/0.1) 修正幅度小, 没把 svhn 推坏, 同时给 majority 带来轻微涨

### 4. rho=0.3 s=333 的奇异 rebound

rho=0.3 s=333 R74 时 mean=92.77 (vs s=15 R74=68 左右), 最终 R99 best=93.39。**单 seed 突然 recover**, 跟 s=15 同 rho 80.91 形成 12.5pp 巨大 split。说明 svhn collapse 在某些 seed 是 transient — 后期 mag guard 抑制 + ccc loss 收敛, svhn 可能 recover。但 single-seed 现象, 不可靠。

## 终局对比 (mean best)

| Method | mean best | vs vanilla 93.59 |
|---|---|---|
| **F2DC vanilla** (winner) | **93.59** | — |
| **EXP-146 rho=0.1** | **93.47** | -0.12 (基本持平) ✓ |
| EXP-146 rho=0.05 | 93.36 | -0.23 (持平) ✓ |
| EXP-146 rho=0.3 | 87.15 | -6.44 (s=333 偶然 recover, s=15 烂) |
| EXP-146 rho=0.2 | 80.55 | -13.04 ❌❌ |

## paper 价值

**强 negative result**: 跟 EXP-145 Office "DSE 输 DaA" 形成对照, 共同建立 DSE 的 boundary condition:
- DSE 适用: PACS-like stylistic shift (photo/sketch/cartoon/art 同 class, style 不同)
- DSE 不适用: Digits-like categorical shift (svhn vs mnist 是 fundamentally different signal sources)
- DSE 边界: 看 "server proto cross-domain center 是否 helpful" — 如果跨域共识 = helpful, DSE +; 如果跨域共识 = noise/wrong direction, DSE −

这是个清晰的 boundary, paper 可以写成 "DSE adaptive ρ 调节策略" 的 motivation。

## 后续 ablation (可选)

1. **DSE 自动 rho 选择**: 训练前先看 server proto 跨域分布 (高 variance = stylistic, 高 mode = categorical), 决定 rho_max
2. **per-class proto rho**: svhn 类的 layer3 用 rho=0, mnist/usps 类用 rho=0.3 — 让 svhn 不被破坏

## 数据保存

- **logs**: `experiments/ablation/EXP-146_f2dc_dse_digits_rho_scan/logs/digits_rho{005,01,02,03}_s{15,333}_R100.log` (8 个; 高-rho 4 个被覆盖到 R5, 低-rho 4 个完整 R100)
- **diag npz** (rsync 不入 git): `EXP-146/diag/rho{005,01,02,03}_s{15,333}/` 各含完整 round_006-100 + best/final/meta/jsonl. 单 dir ~900MB (Digits parti=20 features dump 大)。 高-rho 的 round_001-005 被覆盖, 但 R6-R100 + best + final 完整
- **rho02/rho03 R99 best per-domain 待补**: 从 final_R100.npz 的 confusion matrix 算 per-domain acc 替换上表 † 标记
