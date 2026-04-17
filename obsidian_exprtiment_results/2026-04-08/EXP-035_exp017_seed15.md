# EXP-035 | EXP-017 (Best Config) with seed=15 - Multi-seed validation

## 基本信息
- **日期**：2026-04-08
- **类型**：multi-seed validation
- **方法**：FedDSA V4 no HSIC (EXP-017配置)
- **Seed**：15
- **状态**：⏳ 待执行

## 目的
论文需要多seed取平均±方差。EXP-017是当前SOTA(82.24%)，需要验证不同seed下结果稳定。
配合EXP-017(seed=2)和EXP-036(seed=333)，得到3-seed结果。

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4_no_hsic.yml \
    --logger PerRunLogger --seed 15 \
    > /tmp/exp035.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| vs seed=2 (EXP-017 82.24%) | |

## 结论
