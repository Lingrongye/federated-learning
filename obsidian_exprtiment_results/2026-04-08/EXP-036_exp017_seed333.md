# EXP-036 | EXP-017 (Best Config) with seed=333 - Multi-seed validation

## 基本信息
- **日期**：2026-04-08
- **类型**：multi-seed validation
- **方法**：FedDSA V4 no HSIC (EXP-017配置)
- **Seed**：333
- **状态**：⏳ 待执行

## 目的
配合EXP-017(seed=2)和EXP-035(seed=15)，得到3-seed结果。
论文报告 "FedDSA: 82.x ± y%"

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4_no_hsic.yml \
    --logger PerRunLogger --seed 333 \
    > /tmp/exp036.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|

## 结论
