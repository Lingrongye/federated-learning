# EXP-072 Adaptive Baselines

## 实验说明
FedDSA-Adaptive消融实验系列：

| Sub-EXP | Mode | Config | 说明 |
|---------|------|--------|------|
| 072a | fixed_alpha=0.2 | feddsa_072a.yml | 弱固定增强基线 |
| 072b | fixed_alpha=0.5 | feddsa_072b.yml | 中固定增强基线 |
| 072c | fixed_alpha=0.8 | feddsa_072c.yml | 强固定增强基线 |
| 072d | M3-only | feddsa_072d.yml | 域感知原型(无自适应增强) |
| 072e | aug_min=0.0 | feddsa_072e.yml | 低gap域禁止增强(sanity) |
| 072 | M1 adaptive | feddsa_072.yml | 核心自适应增强 |

## 算法文件
`FDSE_CVPR25/algorithm/feddsa_adaptive.py`

## 数据集
PACS (4域, 7类)

## Seeds
2, 333, 42 (每个配置3个seed)

## 基线对比
- EXP-069f (FedDSA base, PACS=80.96%)
- EXP-052 (NoAug, PACS=80.57%)

## 启动命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

# Phase A: Fixed-alpha baselines
for SEED in 2 333 42; do
  nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_adaptive --gpu 0 \
    --config ./config/pacs/feddsa_072a.yml --seed $SEED \
    > ../../experiments/adaptive/EXP-072_adaptive_baselines/logs/072a_s${SEED}.log 2>&1 &
done
```

## 结果
(待填)
