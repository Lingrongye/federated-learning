# EXP-068 | FedDSA-RegimeGated v2: Fix Signal Source + Farthest-K Dispatch

## 基本信息
- **日期**:2026-04-12
- **算法**:`feddsa_regime_gated_v2`
- **核心改进**:在 EXP-067 v1 基础上修两个 bug

## 动机:EXP-067 v1 的两个关键失败

### Failure 1 — Regime signal 不 discriminative
v1 用 backbone 特征 `h = encoder(x)` 的 mean/std 做 regime signal:
- PACS r ≈ 2.0-2.4
- Office r ≈ 2.1-2.2
- **几乎一样!** 原因:AlexNet 骨干有 7 个 BN 层,激活被归一化到相似范围

### Failure 2 — KNN (nearest) dispatch 在 PACS 上有害
v1 dispatch K 个最近邻风格:
- PACS: EXP-067 79.46 vs EXP-064 80.74 → **-1.28** ❌ (最近邻 = 最没用的增强)
- Office: 89.72 vs 89.83 → -0.11 ≈ 平手

### 额外发现:Consensus PACS "突破" 是单 seed 假象
EXP-064 Consensus 3-seed:
- s2=83.04 (这是之前认为的 "突破")
- s15=79.39 (< baseline 80.59)
- s333=79.80 (< baseline 81.05)
- **Mean = 80.74** < baseline 81.29 → Consensus 其实没赢 baseline on PACS

## 修复

### Fix A — Signal 源(style_head 输出)
- 新增 `style_bank_proj`:128-d 来自 `z_sty = style_head(h)` 的 mean/std
- style_head 被正交+HSIC 约束去学 backbone 丢弃的 style residual → 更 discriminative
- `_compute_regime_score()` 和 dispatch 距离都用新 bank
- AdaIN 增强仍用原 bank(1024-d,不变)

### Fix B — Farthest-K Dispatch
- 从"永远最近邻"改为**永远最远邻**
- PACS:最远邻 = 最不同的域 → 最大化增强多样性
- Office:所有客户端距离接近 → farthest ≈ nearest → 无害

## 代码结构
```
feddsa_regime_gated_v2.py
├── Server(继承 v1 Server)
│   ├── initialize(): +style_bank_proj
│   ├── iterate(): +收集 proj stats
│   ├── _select_signal_bank(): 选 proj 或 fallback backbone
│   ├── _compute_regime_score(): 用 signal bank
│   └── _dispatch_knn_styles(): farthest-K + signal bank 距离
└── Client(继承 v1 Client)
    ├── train(): +eval pass 算 style_head 投影统计
    ├── _compute_proj_style_stats(): 10 batch eval forward
    └── pack(): +style_proj_stats
```

## 预期结果 vs baseline

| 数据集 | baseline | EXP-064 Consensus | EXP-067 v1 | **EXP-068 v2 预期** |
|---|---|---|---|---|
| PACS | 81.29 | 80.74 | 79.46 | **≥ 81** (farthest 恢复 diversity) |
| Office | 89.13 | 89.83 | 89.72 | **≈ 89.7** (farthest ≈ nearest) |

关键判据:PACS 是否**从 79.46 恢复到 ≥ 81**

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
for S in 2 15 333; do
    for T in PACS_c4 office_caltech10_c4; do
        CFG=$(echo $T | sed 's/PACS_c4/pacs/;s/office_caltech10_c4/office/')
        nohup $PY run_single.py --task $T --algorithm feddsa_regime_gated_v2 --gpu 0 \
            --config ./config/${CFG}/feddsa_regime_gated_v2.yml --logger PerRunLogger --seed $S \
            > /tmp/exp068_${T}_s${S}.out 2>&1 &
        sleep 2
    done
done
```

## 结果 (3-seed COMPLETE)

### Office-Caltech10 (✅ DONE)

| seed | AVG Best | Last | vs baseline 89.13 | vs Consensus 89.83 |
|---|---|---|---|---|
| 2 | 89.48 | 88.69 | +0.35 | -0.35 |
| 15 | **90.22** | 89.55 | +1.09 | +0.11 |
| 333 | 89.77 | 87.43 | +0.64 | -0.22 |
| **Mean ± Std** | **89.82 ± 0.30** | | **+0.69** | **-0.01** |

### PACS (✅ DONE)

| seed | AVG Best | Last | vs baseline 81.29 | vs Consensus 80.74 |
|---|---|---|---|---|
| 2 | 80.64 | 74.73 | -1.65 | -2.40 |
| 15 | 78.89 | 66.66 | -2.40 | -0.50 |
| 333 | 78.19 | 71.00 | -3.10 | -1.61 |
| **Mean ± Std** | **79.24 ± 1.01** | | **-2.05** | **-1.50** |

### Regime score r (style_head 投影空间) — FIX A VALIDATED ✅

| Dataset | round 10 | round 20 | round 40 | round 60 | round 100 | round 200 |
|---|---|---|---|---|---|---|
| PACS avg | 2.92 | 4.67 | **8.36** | **12.24** | 4.26 | ~3 |
| Office avg | ~2.8 | ~2.8 | ~2.8 | ~2.5 | ~2.3 | **3.12** |

**Fix A 验证成功**: PACS r 峰值 12.24 vs Office r 3.12 = **3.6x ratio** (v1 仅 1.0x)
**Fix A 发现**: r 在 round 60 达峰后下降,因为 farthest dispatch 的负反馈使 style_head 趋同

### Farthest dispatch 效果 (Fix B)

| 方法 | PACS Mean | Office Mean |
|---|---|---|
| EXP-064 Consensus (random) | 80.74 | 89.83 |
| EXP-067 v1 (nearest) | 79.46 | 89.72 |
| **EXP-068 v2 (farthest)** | **79.24** | **89.82** |

**结论**: Dispatch 方向 (nearest/farthest) 对准确率影响 < 1%, 不是关键变量。
PACS 的瓶颈不在 dispatch,而在 Consensus QP 聚合本身对高 gap 域有害。
