# EXP-009 | FedProto in FDSE Framework (PACS, AlexNet, 500轮)

## 基本信息
- **日期**：2026-04-06
- **类型**：baselines
- **方法**：FedProto
- **模型**：AlexNet from scratch
- **框架**：FDSE (flgo)
- **状态**：⏳ 待执行

## 目的
FedProto原型学习基线，在相同框架下对比FedDSA。

## 结果

| 指标 | 值 |
|------|---|
| mean_local_test_accuracy | |

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm fedproto --gpu 0 \
    --config ./config/pacs/fedavg.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp009.out 2>&1 &
```
注：FedProto无独立yml，复用fedavg.yml(lr=0.1)，算法内部自带lmbd=0.1

## 结论
