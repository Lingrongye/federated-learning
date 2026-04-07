# EXP-016 | FedDSA V4 with seed=15 (Variance Check)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (seed鲁棒性)
- **方法**：FedDSA V4 (warmup=50 + 强权重)
- **Seed**：15 (vs EXP-014的seed=2)
- **状态**：⏳ 待执行

## 目的
单seed结果可能不稳定。用seed=15重跑V4，配合EXP-014(seed=2)能得到方差估计，
为论文级结果提供基础。

## 与EXP-014的差异
| 参数 | EXP-014 | EXP-016 |
|------|---------|---------|
| seed | 2 | **15** |
其他全部相同 (warmup=50, λ=1.0/0.1/1.0)

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4.yml \
    --logger PerRunLogger --seed 15 \
    > /tmp/exp016.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| 与EXP-014差距 | |

## 结论
