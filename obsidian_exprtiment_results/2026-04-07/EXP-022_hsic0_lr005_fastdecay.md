# EXP-022 | HSIC=0 + lr=0.05 + Fast LR Decay (稳定性优化)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (稳定性优化)
- **方法**：FedDSA V4 去HSIC + lr=0.05 + 快速衰减
- **算法**：feddsa (原版)
- **状态**：⏳ 待执行

## 目的
EXP-017(SOTA 82.24%)虽然Best超过FDSE，但Best→Last掉6.78%，稳定性差。
FDSE在R200时Best→Last只掉1.98%。本实验用更低的lr和更快的lr衰减，
让后期学习率快速降低，避免震荡。

## 与EXP-017的差异
| 参数 | EXP-017 | EXP-022 |
|------|---------|---------|
| learning_rate | 0.1 | **0.05** |
| learning_rate_decay | 0.9998 | **0.995** (快40倍) |
其他相同 (HSIC=0, orth=1.0, sem=1.0, warmup=50)

## 学习率变化对比
```
Round 200时的LR:
  EXP-017: 0.1 * 0.9998^200 = 0.0960  (几乎没降)
  EXP-022: 0.05 * 0.995^200 = 0.0183  (降到初始36%)
```

## 假设
- 后期lr降到很小 → 无法"跳出"最优解 → Last接近Best
- 目标: Last ≥ 79%, Best-Last gap ≤ 3%

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_exp022.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp022.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| Best-Last gap | |
| Drops>5% | |
| vs EXP-017 | |

## 结论
