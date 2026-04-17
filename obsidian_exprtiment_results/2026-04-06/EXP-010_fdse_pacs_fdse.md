# EXP-010 | FDSE in FDSE Framework (PACS, AlexNet+DSE, 500轮)

## 基本信息
- **日期**：2026-04-06
- **类型**：baselines
- **方法**：FDSE (论文原方法)
- **模型**：AlexNet + DSEConv/DSELinear
- **框架**：FDSE (flgo)
- **状态**：⏳ 待执行

## 目的
复现FDSE论文PACS结果(83.8%)，作为最强基线与FedDSA对比。

## 结果

| 指标 | 值 |
|------|---|
| mean_local_test_accuracy | |

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm fdse --gpu 0 \
    --config ./config/pacs/fdse.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp010.out 2>&1 &
```

## 结论
