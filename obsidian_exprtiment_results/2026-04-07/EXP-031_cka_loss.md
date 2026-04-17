# EXP-031 | FedDSA with CKA Loss (replacing cos² orthogonality)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (替换orth loss形式)
- **方法**：CKA 代替 cos² 作为解耦约束
- **算法**：feddsa_cka (新建)
- **状态**：⏳ 待执行

## 目的
cos²(z_sem, z_sty) 是对单个样本的约束，没有考虑 batch 分布。
CKA (Centered Kernel Alignment) 是 batch 级的相似度度量，梯度更平滑。

## Loss对比
```
cos² (EXP-017):
  per-sample: sim_i = cos(z_sem_i, z_sty_i)²
  loss = mean(sim_i)

CKA (EXP-031):
  X = z_sem - mean  (centered)
  Y = z_sty - mean
  CKA = ||Y^T X||²_F / (||X^T X||_F * ||Y^T Y||_F)
  loss = CKA  (minimizing drives decorrelation at batch level)
```

## 与EXP-017的差异
| 维度 | EXP-017 | EXP-031 |
|------|---------|---------|
| Orth loss | cos² (per-sample) | **CKA (batch-level)** |

## 假设
- CKA更平滑 → 梯度方差小 → 稳定性提升
- 但batch级约束强度弱 → 解耦可能不够彻底
- 目标: 更稳定的训练，Drops减少

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_cka --gpu 0 \
    --config ./config/pacs/feddsa_exp031.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp031.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| Drops>5% | |
| vs EXP-017 | |

## 结论
