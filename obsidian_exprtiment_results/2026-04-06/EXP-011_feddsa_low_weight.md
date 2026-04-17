# EXP-011 | FedDSA Low Weight (诊断: 降低辅助损失权重)

## 基本信息
- **日期**：2026-04-06
- **类型**：ablation (诊断性调参)
- **方法**：FedDSA V1: λ_orth=0.1, λ_hsic=0.01, λ_sem=0.5
- **模型**：AlexNet from scratch
- **框架**：FDSE (flgo)
- **状态**：⏳ 待执行

## 目的
EXP-006发现FedDSA有5次major accuracy drops，怀疑辅助损失权重太大导致梯度冲突。降低权重验证。

## 与EXP-006的差异
| 参数 | EXP-006 | EXP-011 |
|------|---------|---------|
| lambda_orth | 1.0 | **0.1** |
| lambda_hsic | 0.1 | **0.01** |
| lambda_sem | 1.0 | **0.5** |
| num_rounds | 500 | **200** |

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v1.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp011.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| mean_local_test_accuracy | |
| major drops | |

## 结论
