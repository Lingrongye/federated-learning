# EXP-007 | FedAvg in FDSE Framework (PACS, AlexNet, 500轮)

## 基本信息
- **日期**：2026-04-06
- **类型**：baselines
- **方法**：FedAvg
- **模型**：AlexNet from scratch
- **框架**：FDSE (flgo)
- **状态**：⏳ 待执行

## 目的
FDSE论文报告FedAvg在PACS上74.3%，复现验证。

## 结果

| 指标 | 值 |
|------|---|
| mean_local_test_accuracy | |

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm fedavg --gpu 0 \
    --config ./config/pacs/fedavg.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp007.out 2>&1 &
```

## 结论
