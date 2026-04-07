# EXP-015 | FedDSA+ : Three-Stage Training with Adaptive Weights

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (算法层面改进)
- **方法**：FedDSA+ (新算法 feddsa_plus.py)
- **模型**：AlexNet from scratch
- **框架**：FDSE (flgo)
- **状态**：⏳ 待执行

## 算法改进 (vs 原版FedDSA)
1. **三阶段训练**:
   - Stage 1 (round < 50): 纯CE，让backbone学到稳定特征
   - Stage 2 (50 ≤ round < 100): 加入解耦约束(orth + HSIC)
   - Stage 3 (round ≥ 100): 加入风格增强 + InfoNCE

2. **Sigmoid自适应权重**: 平滑过渡 vs 原版的硬线性warmup
   `weight(r) = sigmoid((r - threshold) / transition_width)`
   - 在阶段切换边界处用sigmoid平滑过渡(width=10轮)
   - 避免突然引入新损失导致的振荡

3. **辅助损失梯度裁剪**: aux_grad_clip=1.0
   - 对semantic_head + style_head参数单独梯度裁剪
   - 防止辅助损失主导更新方向

## 与原版FedDSA的对比
| 维度 | 原版 | FedDSA+ |
|------|------|---------|
| 阶段划分 | 单一阶段(线性warmup=10) | **3阶段显式划分** |
| 权重过渡 | 线性 (round/warmup) | **Sigmoid平滑** |
| 梯度裁剪 | 全模型一次 | **辅助损失单独裁剪** |
| 假设 | warmup后所有约束同时生效 | **逐步引入降低耦合** |

## 关键参数
```yaml
stage1_end: 50         # 阶段1结束
stage2_end: 100        # 阶段2结束
transition_width: 10   # sigmoid过渡宽度(轮数)
aux_grad_clip: 1.0     # 辅助损失梯度裁剪
lambda_orth: 1.0       # 同原版
lambda_hsic: 0.1       # 同原版
lambda_sem: 1.0        # 同原版
```

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_plus --gpu 0 \
    --config ./config/pacs/feddsa_plus.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp015.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| Drops>5% | |
| 与V4差距 | |
| 与FDSE差距 | |

## 结论
