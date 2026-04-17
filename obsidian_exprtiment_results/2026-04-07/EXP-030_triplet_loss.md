# EXP-030 | FedDSA with Triplet Loss (replacing InfoNCE)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (替换loss形式)
- **方法**：Triplet Margin Loss 代替 InfoNCE
- **算法**：feddsa_triplet (新建)
- **状态**：⏳ 待执行

## 目的
InfoNCE (tau=0.1) 梯度锐利，对原型噪声敏感。FISC/PARDON论文用的是triplet loss，
更平滑更稳定。验证替换后的效果。

## Loss对比
```
InfoNCE (EXP-017):
  logits = z @ proto^T / tau   (tau=0.1 锐利)
  loss = CE(logits, target)

Triplet (EXP-030):
  d_pos = 1 - cos(z, same_class_proto)    (相同类距离)
  d_neg = 1 - cos(z, closest_diff_proto)  (最近异类距离)
  loss = max(0, d_pos - d_neg + margin)   (margin=0.3)
```

## 与EXP-017的差异
| 维度 | EXP-017 | EXP-030 |
|------|---------|---------|
| Sem loss | InfoNCE (tau=0.1) | **Triplet (margin=0.3)** |
| 其他 | — | 完全相同 |

## 假设
- Triplet梯度更平滑 → 训练更稳定
- 但triplet收敛更慢 → Best可能略低
- 目标: Last ≥ 79%

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_triplet --gpu 0 \
    --config ./config/pacs/feddsa_exp030.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp030.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| vs EXP-017 | |

## 结论
