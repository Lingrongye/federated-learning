# EXP-057 | MOON 基线

## 基本信息
- **目的**:补齐 FDSE 论文主表基线 MOON
- **算法**:moon (flgo 内置)
- **超参**:mu=0.1, tau=0.5 (FDSE 论文默认)
- **状态**:⏳ 待执行

## 配置
| 参数 | PACS | Office |
|---|---|---|
| Config | moon_r200.yml | moon_r200.yml |
| R | 200 | 200 |
| E | 5 | 1 |
| LR | 0.1 | 0.1 |
| mu | 0.1 | 0.1 |
| tau | 0.5 | 0.5 |

## FDSE 论文参考 (R500, 5-seed)
| 数据集 | ALL | AVG |
|---|---|---|
| PACS | 75.00 ± 0.32 | 72.13 ± 0.32 |
| Office | 80.12 ± 1.86 | 82.48 ± 1.71 |

## 运行命令
```bash
# PACS
nohup python run_single.py --task PACS_c4 --algorithm moon --gpu 0 \
  --config ./config/pacs/moon_r200.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp057_pacs.out 2>&1 &

# Office
nohup python run_single.py --task office_caltech10_c4 --algorithm moon --gpu 0 \
  --config ./config/office/moon_r200.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp057_office.out 2>&1 &
```

## 结果
| 数据集 | ALL Best | AVG Best | AVG Last | Gap |
|---|---|---|---|---|
| PACS | | | | |
| Office | | | | |

## 结论
