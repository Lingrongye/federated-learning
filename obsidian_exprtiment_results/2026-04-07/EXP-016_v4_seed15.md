# EXP-016 | FedDSA V4 with seed=15 (Variance Check)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (seed鲁棒性)
- **方法**：FedDSA V4 (warmup=50 + 强权重)
- **Seed**：15 (vs EXP-014的seed=2)
- **状态**：✅ 已完成

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
| Best acc | **79.79%** @Round 35 |
| Last acc | 77.96% |
| Drops>5% | 4 |
| 最弱域@best | 61.27% |
| 最强域@best | 89.80% |
| 稳定性 | 0.0046 |
| 与EXP-014(seed=2)差距 | -0.14% |
| 总轮次 | 200 |

## 结论

**Seed鲁棒性确认通过**：
- seed=2 Best=79.93% vs seed=15 Best=79.79%，差距仅0.14%
- 说明FedDSA V4的结果是可复现的，不是偶然

但由于V4本身不是最优配置(EXP-017才是)，这个seed验证仅能说明V4的鲁棒性。
后续需要用同样seed验证EXP-017(去HSIC)的鲁棒性，那才是真正的主结果。
