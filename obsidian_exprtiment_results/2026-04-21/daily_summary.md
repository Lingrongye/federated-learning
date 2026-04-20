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

## 核心结论 (等数据回填)

**Q1: CDANN 真的压了 probe_sty_class 的非线性泄漏吗?**
- 对比 EXP-108 CDANN (MLP ≈ 0.91) vs EXP-109 orth_only (MLP = ?)
- 预期: orth_only MLP ≈ 0.91 → CDANN 无贡献,全架构造成

**Q2: 强正交 lo=10 能压下去吗?**
- 对比 lo=1 vs lo=3 vs lo=10 的 probe_sty_class MLP
- 预期: 看轨迹是否单调下降,还是饱和在 0.91

**Q3: capacity probe 稳健性?**
- hidden=16/64/128/256 的 probe acc 是否趋于饱和
- 如果 hidden=256 仍 0.91 → probe 不是瓶颈,信息真的在

## 关键实验发现备忘

见: `关键实验发现备忘.md`
