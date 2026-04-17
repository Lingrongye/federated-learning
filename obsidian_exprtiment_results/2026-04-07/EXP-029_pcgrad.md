# EXP-029 | FedDSA with PCGrad (Projecting Conflicting Gradients)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (梯度投影)
- **方法**：FedDSA-PCGrad 消除梯度冲突
- **算法**：feddsa_pcgrad (新建)
- **状态**：⏳ 待执行

## 目的
5个loss可能有冲突的梯度方向（尤其是orth vs InfoNCE）。
PCGrad (NeurIPS 2020)是解决多任务梯度冲突的经典方法：
对每对loss梯度，如果方向相反(cos<0)，就把冲突部分投影掉。

## 算法核心
```python
# 对每个loss分别backward得到gradient
g_cls = grad(loss_task + loss_task_aug)
g_orth = grad(loss_orth)
g_sem = grad(loss_sem)

# 两两投影
for each pair (g_i, g_j):
    if cos(g_i, g_j) < 0:
        g_i = g_i - (g_i·g_j / ||g_j||²) * g_j

# 投影后求和
final_grad = g_cls + g_orth + g_sem
```

## 与EXP-017的差异
| 维度 | EXP-017 | EXP-029 |
|------|---------|---------|
| Loss定义 | 相同 | 相同 |
| Backward方式 | 一次total backward | **每个loss单独backward** |
| 梯度处理 | 直接相加 | **PCGrad投影冲突部分** |

## 假设
- 如果冲突严重: PCGrad会显著提升
- 如果冲突不严重: 效果接近EXP-017
- 代价: 每step需要3次backward，速度慢3x
- 目标: 如果Last ≥ 80% 证明了冲突是根源

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_pcgrad --gpu 0 \
    --config ./config/pacs/feddsa_exp029.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp029.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| 训练速度 | |
| vs EXP-017 | |

## 结论
