# EXP-021 | FedDSA V4 with λ_orth=2.0 (Stronger Orthogonality)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (正交约束强度)
- **方法**：FedDSA V4 + λ_orth=2.0 (vs 1.0)
- **状态**：⏳ 待执行

## 目的
V1实验(λ_orth=0.1)表现最差(80.15%)。V4的warmup=50提供了稳定基础。
既然warmup保护了早期，能否用更强的正交约束让解耦更彻底？

## 与EXP-014的差异
| 参数 | EXP-014 | EXP-021 |
|------|---------|---------|
| lambda_orth | 1.0 | **2.0** |

## 假设
- 更强的正交约束 → 更彻底的语义/风格分离
- 因为有warmup=50保护，不会在早期破坏backbone
- 如果提升: 证明FedDSA的瓶颈在解耦充分性
- 如果下降: 证明过度正交会损害特征能力

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4_strong_orth.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp021.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| vs EXP-014 | |

## 结论
