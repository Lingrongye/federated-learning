# EXP-023 | HSIC=0 + Server-Side EMA (EMA=0.9)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (稳定性优化 - 算法级)
- **方法**：FedDSA-Stable 启用EMA
- **算法**：feddsa_stable (新建)
- **状态**：⏳ 待执行

## 目的
深度学习中EMA(Exponential Moving Average)是经典稳定技巧，能平滑模型更新，
减少高频震荡。在FedAvg聚合后应用EMA，让全局模型参数变化更平滑。

## 算法改进
在 `_aggregate_shared()` 中加入EMA：
```python
new_params = (1 - ema_momentum) * aggregated + ema_momentum * prev_global
```
- ema_momentum=0.9 → 90%保留旧参数，10%吸收新更新
- 每轮的变化更小 → 不会"冲高回落"

## 与EXP-017的差异
| 参数 | EXP-017 | EXP-023 |
|------|---------|---------|
| Algorithm | feddsa | **feddsa_stable** |
| ema_momentum | N/A | **0.9** |
其他相同

## 假设
- Best可能略降(因为参数更新慢)
- 但Last应该更接近Best
- Drops应该显著减少
- 目标: Last ≥ 80%, Drops ≤ 3

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_stable --gpu 0 \
    --config ./config/pacs/feddsa_exp023.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp023.out 2>&1 &
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
