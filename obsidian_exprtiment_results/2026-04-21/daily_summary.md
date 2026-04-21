# 2026-04-21 daily summary

## 今晚任务总览 (你睡觉期间)

用户决定: **CLUB Stage-1 pilot 不跑** (Codex 两次 RETHINK,加 MLP probe 结果已证 CLUB 无望)。改做双头解耦围绕的验证性实验,GPU 铺满。

## 已完成 ✅

1. **Codex R2 review** Purify-then-Share 提案 → 6.8/10 RETHINK (refine-logs/2026-04-21_SymmetricCDANN/round-2-review-purifyshare.md)
2. **扩展 probe 脚本**: linear + MLP hidden=64, 加 probe_sem_class (第 4 个 target)
3. **EXP-108 MLP probe 6/6 完成**:
   - Office CDANN s=2 sty_class: lin=0.950, mlp=0.906
   - Office CDANN s=15 sty_class: lin=0.956, mlp=0.908
   - (其余 4 个 s=333 Office / s=2,15,333 PACS 待汇总数字)
4. **Capacity probe 脚本**: hidden ∈ {16,64,128,256} + per-domain 分解
5. **3 个新 config**: feddsa_pacs_orth3/orth10_saveckpt_r200.yml

## 进行中 ⏳ (8 训练 run + 1 probe batch 并发)

一张 4090 24GB 并发 8 run,GPU 13GB util 65%。

| EXP | run | PID | 启动 | Est Done |
|-----|-----|-----|------|---------|
| EXP-109 PACS orth_only s=2 | 247627 | ~00:15 | ~07:00 |
| EXP-109 PACS orth_only s=15 | 255317 | 02:45 | ~07:00 |
| EXP-109 PACS orth_only s=333 | 255604 | 02:48 | ~07:00 |
| EXP-110 Office orth_only s=2 | 255844 | 02:50 | ~06:00 |
| EXP-110 Office orth_only s=15 | 256308 | 03:00 | ~06:00 |
| EXP-110 Office orth_only s=333 | 256522 | 03:02 | ~06:00 |
| EXP-111 PACS lo=3 s=2 | 256060 | 02:52 | ~07:00 |
| EXP-111 PACS lo=10 s=2 | 256736 | 03:05 | ~07:00 |
| Capacity probe batch (EXP-108) | 257001 | 03:05 | ~03:45 |

## 明早起床后要做 ☀️

1. `bash rr/check_tonight_progress.sh` 看全盘进度
2. 如果所有 8 run 完成 → 在新 checkpoint 上跑 capacity probe batch 得到 3-seed probe 对比
3. 回填 EXP-108/109/110/111 NOTE.md 的 3-seed 结果表
4. 更新本文件 "核心结论" 段

## 核心结论 (部分已回填 2026-04-21 早上)

**Q1: CDANN 真的压了 probe_sty_class 的非线性泄漏吗?** — **答案: NO,相反**

PACS seed=2 对比:
- orth_only (ca=0) probe_sty_class: lin=0.240, MLP-256=0.813
- CDANN (ca=1) probe_sty_class: lin=0.963, MLP-256=0.962
- **Δ = CDANN 比 orth_only 多 +72pp linear probe,-0.15 MLP-256**

**颠覆性发现**: CDANN 不是"保留 class" 而是"**错误灌入 class 到 z_sty**"。
- orth_only 下 z_sty 已经几乎不含 class (0.24 ≈ random 0.14)
- CDANN 把 class 从 z_sem 挤进 z_sty (设计缺陷,不是设计优势)

**Office 对比** (3-seed):
- orth_only mean: lin=0.962, MLP-128=0.955
- CDANN mean: lin=0.957, MLP-128=0.887
- CDANN 略低 (但 seed 方差内),Office 无 probe effect

**Q2: 强正交 lo=10 能压下去吗?** — 等 EXP-111 run 完成 (预计 08-09 点)

**Q3: capacity probe 稳健性?** — ✅ 已证实
- EXP-108 CDANN PACS: linear 0.963 ≈ MLP-256 0.960 (饱和)
- EXP-109 orth_only s=2: linear 0.240 << MLP-256 0.813 (非线性有少量 bonus)
- 说明 z_sty 的 class 信息在 CDANN 下是线性暴露的,在 orth_only 下是部分非线性残留

## Accuracy 结果 (2026-04-21 早上)

**Office 3-seed**:
| 配置 | AVG Best | 备注 |
|------|---------|------|
| EXP-102 whiten_only (老的) | 89.26±0.83 | |
| EXP-108 Office CDANN | 89.54±0.49 | |
| **EXP-110 Office orth_only (本晚)** | **89.09** | CDANN 无增益 |

**PACS (部分 seed 完成)**:
| 配置 | seed=2 | seed=15 | seed=333 | mean |
|------|--------|---------|----------|------|
| EXP-108 PACS CDANN | 80.87 | 79.99 | 79.40 | 80.08±0.60 |
| **EXP-109 PACS orth_only** | **82.23** | TBD | TBD | TBD |
| EXP-111 PACS lo=3 | TBD | — | — | — |
| EXP-111 PACS lo=10 | TBD | — | — | — |

**single-seed 初步**: PACS orth_only s=2 (82.23) > CDANN s=2 (80.87) **+1.36pp**。CDANN 确实在 PACS 上微幅拖 accuracy。

## 方向判断 (2026-04-21 早上)

✅ **确定死路**: CDANN 方向 (anchor 虚假 + accuracy 略低 + Office 无差异)
✅ **已验证路**: orth_only 是 clean baseline,PACS ~82%,Office ~89%,无需 CDANN
🔍 **待验证**: 强正交 lo=10 能否把 probe_sty_class 继续压到 ≤ 0.2
⚠️ **重要问题**: **accuracy 提升需要新的方向**,光靠双头解耦正交 + whitening 已经到 82% 天花板

## 关键实验发现备忘

见: `关键实验发现备忘.md`
