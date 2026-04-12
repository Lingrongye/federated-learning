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

## 结果(待填)

### 3-seed

| Dataset | s=2 | s=15 | s=333 | Mean ± Std | vs baseline |
|---|---|---|---|---|---|
| PACS | - | - | - | - | - |
| Office | - | - | - | - | - |

### Regime score r (style_head 投影空间)

| Dataset | r range | 是否 discriminative? |
|---|---|---|
| PACS | - | - |
| Office | - | - |
