# EXP-056 | FedProx 基线

## 基本信息
- **目的**:补齐 FDSE 论文主表基线 FedProx
- **算法**:fedprox (flgo 内置)
- **超参**:mu=0.1 (FDSE 论文默认)
- **状态**:⏳ 待执行

## 配置
| 参数 | PACS | Office |
|---|---|---|
| Config | fedprox_r200.yml | fedprox_r200.yml |
| R | 200 | 200 |
| E | 5 | 1 |
| LR | 0.1 | 0.1 |
| mu | 0.1 | 0.1 |

## FDSE 论文参考 (R500, 5-seed)
| 数据集 | ALL | AVG |
|---|---|---|
| PACS | 74.38 ± 1.55 | 72.33 ± 1.53 |
| Office | 82.69 ± 1.52 | 87.36 ± 1.87 |

## 运行命令
```bash
# PACS
nohup python run_single.py --task PACS_c4 --algorithm fedprox --gpu 0 \
  --config ./config/pacs/fedprox_r200.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp056_pacs.out 2>&1 &

# Office
nohup python run_single.py --task office_caltech10_c4 --algorithm fedprox --gpu 0 \
  --config ./config/office/fedprox_r200.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp056_office.out 2>&1 &
```

## 结果
| 数据集 | ALL Best | AVG Best | AVG Last | Gap |
|---|---|---|---|---|
| PACS | | | | |
| Office | | | | |

## 结论
