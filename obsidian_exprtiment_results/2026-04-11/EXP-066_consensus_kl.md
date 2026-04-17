# EXP-066 | Consensus Aggregation + KL Consistency Regularization

## 基本信息
- **日期**:2026-04-11
- **算法**:feddsa_consensus_kl
- **核心改进**:在 EXP-064 Consensus 基础上,加入 FDSE 的 consistency regularization (C 模块)
- **状态**:✅ Office 3-seed 完成; ⏳ PACS s15/s333 跑中

## 背景与动机

### EXP-064 结果(seed=2)
- PACS: **83.04** (超 baseline +0.80, 超 FDSE +2.23) 🎯
- Office: **89.40** (比 baseline -0.55, 但是所有变体里最好)

### 问题
Consensus 在 PACS 上突破,但 Office 仍差 baseline。GPT-5.4 Research Review 指出:
> FDSE Table 3 显示 Office 上 C 模块(Consistency Reg)单独贡献 +1.20,需要组合 A+B+C 才能到 91.58。

### 解决方案 (EXP-066)
在 Consensus 聚合基础上,**加入 FDSE 的 BN 一致性正则化**:
1. 在每个 encoder BN 层前注册 forward hook,捕获 per-batch μ/σ
2. 每个训练 step 计算:
   ```
   L_kl = Σ_l w_l · [||μ_local - μ_global||² + ||log σ²_local - log σ²_global||²]
   ```
3. 层权重 w_l = softmax(β·l),后面层权重更大
4. 加到总 loss: `L = ... + aux_w · λ_kl · L_kl`

## 与 EXP-064 的差异

| 组件 | EXP-064 Consensus | EXP-066 Consensus+KL |
|---|---|---|
| 服务器聚合 | Consensus-max (QP) | 同 |
| 客户端 task/orth/InfoNCE loss | 同原版 | 同 |
| 客户端 style aug | 同原版 | 同 |
| **客户端 BN 一致性 loss** | ❌ 无 | ✅ **新增** |

实现: `feddsa_consensus_kl.py` 继承 `feddsa_consensus.Server`,改写 `Client.train()` 加 hook 和 loss。

## 配置

| 参数 | 值 |
|---|---|
| λ_orth | 1.0 |
| λ_hsic | 0.0 |
| λ_sem | 1.0 |
| **λ_kl** | **0.01** (FDSE 默认范围 0.01-0.05) |
| **kl_beta** | **0.1** (layer weighting) |
| warmup_rounds | 50 |
| style_dispatch_num | 5 |

## 单元测试 (10/10 PASS)
- ✅ BN layer detection (7 layers in encoder)
- ✅ Forward hook captures stats (14 pairs per forward)
- ✅ Method inheritance from Consensus Server
- ✅ Integration runs without NaN

## 运行命令
```bash
# Office
nohup python run_single.py --task office_caltech10_c4 --algorithm feddsa_consensus_kl --gpu 0 \
  --config ./config/office/feddsa_consensus_kl.yml --logger PerRunLogger --seed $S \
  > /tmp/exp066_office_s${S}.out 2>&1 &

# PACS
nohup python run_single.py --task PACS_c4 --algorithm feddsa_consensus_kl --gpu 0 \
  --config ./config/pacs/feddsa_consensus_kl.yml --logger PerRunLogger --seed $S \
  > /tmp/exp066_pacs_s${S}.out 2>&1 &
```

## 结果

### Office-Caltech10 (3-seed ✅ COMPLETE)

| seed | 服务器 | AVG Best | Last | Gap |
|---|---|---|---|---|
| 2 | SC2 | 88.95 | 87.42 | 1.53 |
| 15 | lab-lry | 87.37 | 83.64 | 3.73 |
| **333** | lab-lry | **91.52** 🎯 | 90.74 | 0.78 |
| **Mean ± Std** | | **89.28 ± 2.10** | 87.27 | 2.01 |

**vs baseline 89.13**: +0.15
**vs FDSE 90.58**: -1.30

### PACS (部分完成)

| seed | 服务器 | AVG Best | Last | 状态 |
|---|---|---|---|---|
| 2 | SC2 | 80.79 | 68.72 | DONE |
| 15 | lab-lry | 79.76 (158) | 78.66 | ⏳ 158/200 |
| **333** | lab-lry | **84.15 (158)** 🎯 | 84.08 | ⏳ 158/200 |

**seed=333 初步 84.15** 远超 baseline 81.05 和 FDSE 79.93!

## 关键发现

### 1. **seed=333 双数据集爆发**
- PACS s333: baseline 81.05 → Cons+KL 84.15+ (**+3.10**)
- Office s333: baseline 86.35 → Cons+KL 91.52 (**+5.17**)
- 这是 baseline 最弱的 seed,Cons+KL 在此 seed 下帮助模型**从训练陷阱中爬出**

### 2. **seed=2/15 轻微下降**
- seed=2 Office: 88.95 (-1.00)
- seed=15 Office: 87.37 (-3.71)
- baseline 本身已经在 good local optimum 时,KL 的强制约束反而扰动

### 3. **seed 敏感性极高**
- Office std = 2.10 (几乎等于 baseline 2.42)
- 均值提升 +0.15 (Office),但单 seed 变化范围 -3.71 ~ +5.17

### 4. **PACS 上 Cons+KL 比纯 Cons 弱**
- EXP-064 PACS s2: 83.04
- EXP-066 PACS s2: 80.79
- 说明对于 PACS 大风格差数据集,KL 约束是**多余的**

## 结论

**Consensus+KL 是一个 "seed 敏感的高方差" 方法**:
- 在低基准 seed (s333) 下有巨大收益
- 在高基准 seed (s2/s15) 下反而有害
- 平均而言,对 Office 微增 (+0.15), 对 PACS 当前均值不清(等 s15/s333 跑完)

**PAPER 含义**: 这可以作为"方法有效性依赖训练陷阱"的 ablation 证据。对 paper 主 claim:
- 如果走"稳定超越 FDSE"路线 → Cons+KL 不够
- 如果走"分析不同 seed 行为"路线 → Cons+KL 提供了一个有意思的 seed-conditional 结果

## 下一步
- 等 PACS s15/s333 跑完计算完整 3-seed 均值
- 如果 PACS 均值 > 81.29 (baseline),则 Cons+KL 是 PACS 的最优变体
- 考虑是否组合 EXP-064 Consensus (PACS 更强) + EXP-066 KL (Office 潜力) 的 dataset-specific 选择
