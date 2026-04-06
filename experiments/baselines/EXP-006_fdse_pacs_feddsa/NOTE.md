# EXP-006 | FedDSA in FDSE Framework (PACS, AlexNet, 500轮)

## 基本信息
- **日期**：2026-04-06
- **类型**：baselines (公平对比)
- **数据集**：PACS（4域x7类，每域=1客户端）
- **方法**：FedDSA (移植到FDSE/flgo框架)
- **模型**：AlexNet from scratch (与FDSE论文一致)
- **框架**：FDSE (flgo)
- **服务器**：seetacloud RTX 3080 Ti
- **状态**：🔄 运行中

## 目的
在与FDSE论文完全相同的实验条件下测试FedDSA，结果可直接与论文Table对比。

## 与之前实验的差异
| 参数 | EXP-002 (PFLlib) | EXP-006 (FDSE框架) |
|------|-------------------|---------------------|
| 框架 | PFLlib | FDSE (flgo) |
| 模型 | ResNet-18 pretrained | **AlexNet from scratch** |
| Rounds | 50 | **500** |
| LR | 0.005 | **0.1** |
| LR decay | 无 | **0.9998/round** |
| Batch size | 32 | **50** |
| Weight decay | 1e-5 | **1e-3** |
| Grad clip | 无 | **10** |
| Train/Test | 75/25 | **80/20** |

## 结果

| 指标 | 值 |
|------|---|
| mean_local_test_accuracy | |
| std_local_test_accuracy | |
| 各域准确率 | |
| 最佳轮次 | |

## 结论

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp006.out 2>&1 &
```
- **服务器**���seetacloud (RTX 3080 Ti)
- **Python**：/root/miniconda3/bin/python

## 问题与备注
- 已在Round 0成功运行，mean_local_test_accuracy=0.1601 (随机水平)
