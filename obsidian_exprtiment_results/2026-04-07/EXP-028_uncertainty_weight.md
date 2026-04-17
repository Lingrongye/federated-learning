# EXP-028 | FedDSA with Uncertainty Weighting (Kendall et al. CVPR 2018)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (loss权重自动学习)
- **方法**：FedDSA-Auto 自动学习loss权重
- **算法**：feddsa_auto (新建)
- **状态**：⏳ 待执行

## 目的
手动调lambda_orth/lambda_sem很痛苦，而且5个loss的梯度方向可能冲突。
Kendall et al. 2018提出用可学习的log-variance自动平衡多任务loss。
这是学术界的经典方案。

## 算法核心
```python
log_sigma_task_aug = nn.Parameter(torch.zeros(1))
log_sigma_orth = nn.Parameter(torch.zeros(1))
log_sigma_sem = nn.Parameter(torch.zeros(1))

total = loss_task + \
    exp(-log_sigma_task_aug) * loss_task_aug + log_sigma_task_aug + \
    exp(-log_sigma_orth) * loss_orth + log_sigma_orth + \
    exp(-log_sigma_sem) * loss_sem + log_sigma_sem
```
- precision高(方差小)的loss权重大
- 优化log_sigma使得loss间自动平衡

## 与EXP-017的差异
| 维度 | EXP-017 | EXP-028 |
|------|---------|---------|
| lambda_orth | 1.0 (hardcoded) | **可学习** |
| lambda_sem | 1.0 (hardcoded) | **可学习** |
| 额外可学参数 | 无 | **3个 log_sigma** |

## 假设
- 自动学习的权重可能比人拍的1.0/1.0更好
- 训练过程中权重动态调整，能缓解后期震荡
- 目标: Best ≥ 82%, Last ≥ 79%

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_auto --gpu 0 \
    --config ./config/pacs/feddsa_exp028.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp028.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| 学到的权重 | |

## 结论
