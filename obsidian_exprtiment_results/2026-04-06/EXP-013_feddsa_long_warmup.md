# EXP-013 | FedDSA Long Warmup (诊断: 延长warmup到50轮)

## 基本信息
- **日期**：2026-04-06
- **类型**：ablation (诊断性调参)
- **方法**：FedDSA V3: warmup=50, λ_orth=0.5, λ_hsic=0.05, λ_sem=0.5
- **模型**：AlexNet from scratch
- **框架**：FDSE (flgo)
- **状态**：⏳ 待执行

## 目的
给backbone更长的纯CE训练期(50轮)，让特征稳定后再引入辅助约束，降低早期振荡。

## 与EXP-006的差异
| 参数 | EXP-006 | EXP-013 |
|------|---------|---------|
| warmup_rounds | 10 | **50** |
| lambda_orth | 1.0 | **0.5** |
| lambda_hsic | 0.1 | **0.05** |
| lambda_sem | 1.0 | **0.5** |
| num_rounds | 500 | **200** |

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v3.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp013.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| mean_local_test_accuracy | |
| major drops | |

## 结论
