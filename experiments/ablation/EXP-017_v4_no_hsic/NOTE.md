# EXP-017 | FedDSA V4 without HSIC

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (HSIC贡献验证)
- **方法**：FedDSA V4 + λ_hsic=0
- **状态**：⏳ 待执行

## 目的
EXP-012(V2 去HSIC + warmup=10)Best=80.76%，原版(warmup=10)Best=81.15%。
HSIC的贡献是+0.39%，但HSIC计算开销大且梯度不稳。
本实验在最优配置(warmup=50)下去掉HSIC，看影响有多大。

## 假设
1. 如果HSIC贡献小: V4-no-HSIC ≈ V4 → 可以彻底移除HSIC，简化算法
2. 如果HSIC贡献大: V4-no-HSIC < V4 → 需要保留但找更稳定的实现

## 与EXP-014的差异
| 参数 | EXP-014 | EXP-017 |
|------|---------|---------|
| lambda_hsic | 0.1 | **0.0** |
其他全部相同

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4_no_hsic.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp017.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| 与EXP-014差距 | |

## 结论
