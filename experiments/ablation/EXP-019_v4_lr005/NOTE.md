# EXP-019 | FedDSA V4 with lr=0.05

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (学习率调优)
- **方法**：FedDSA V4 + lr=0.05
- **状态**：⏳ 待执行

## 目的
FDSE原论文FedAvg/FedBN用lr=0.1，FDSE自己用lr=0.05(更低)。我们所有FedDSA变体都用lr=0.1。
降低学习率可能减少训练振荡，提高稳定性和最终精度。

## 与EXP-014的差异
| 参数 | EXP-014 | EXP-019 |
|------|---------|---------|
| learning_rate | 0.1 | **0.05** |
其他相同

## 假设
- 更低LR → 更小梯度步长 → 更稳定的辅助损失更新
- 可能牺牲收敛速度（Best出现更晚），但Last更接近Best

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4_lr005.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp019.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| vs EXP-014 | |

## 结论
