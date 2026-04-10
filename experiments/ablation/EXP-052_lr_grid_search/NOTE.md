# EXP-052~055 | FedDSA LR Grid Search

## 基本信息
- **目的**:FDSE 论文对每个方法独立 grid search LR ∈ {0.001, 0.01, 0.05, 0.1, 0.5}。我们之前固定 LR=0.1,可能不是最优。
- **状态**:⏳ 待执行

## 实验矩阵
| Exp | 数据集 | LR | decay | Config | Seed |
|---|---|---|---|---|---|
| EXP-052 | PACS | **0.05** | 0.9998 | feddsa_lr005.yml | 2 |
| EXP-053 | PACS | **0.2** | 0.9998 | feddsa_lr02.yml | 2 |
| EXP-054 | PACS | 0.1 | **0.998** (强衰减) | feddsa_decay0998.yml | 2 |
| EXP-055 | Office | **0.05** | 0.9998 | feddsa_office_lr005.yml | 2 |

## 对照基线
| 数据集 | LR=0.1, decay=0.9998 (当前) | 来源 |
|---|---|---|
| PACS | AVG Best 82.24, ALL Best 83.75 | EXP-017 s2 |
| Office | AVG Best 89.95, ALL Best 84.13 | EXP-051 s2 |

## 配置一致性检查
- algo_para 与 EXP-017 完全一致(orth=1, hsic=0, sem=1, tau=0.1, warmup=50, dispatch=5, proj=128)
- 仅改 learning_rate 或 learning_rate_decay
- PACS: E=5, B=50, R=200 ✅
- Office: E=1, B=50, R=200 ✅

## LR decay 说明
- FDSE 论文写 "decay ratio 0.998",但 FDSE 官方 repo 所有 config 用 0.9998
- 我们的 FDSE R500 用 0.9998 复现出 AVG=82.16,论文报 82.17 → 0.9998 正确
- EXP-054 测试 0.998 作为 FedDSA 调优探索(更强衰减可能缓解后期震荡)

## 运行命令
```bash
# EXP-052 PACS LR=0.05
nohup python run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 \
  --config ./config/pacs/feddsa_lr005.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp052.out 2>&1 &

# EXP-053 PACS LR=0.2
nohup python run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 \
  --config ./config/pacs/feddsa_lr02.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp053.out 2>&1 &

# EXP-054 PACS decay=0.998
nohup python run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 \
  --config ./config/pacs/feddsa_decay0998.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp054.out 2>&1 &

# EXP-055 Office LR=0.05
nohup python run_single.py --task office_caltech10_c4 --algorithm feddsa --gpu 0 \
  --config ./config/office/feddsa_office_lr005.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp055.out 2>&1 &
```

## 结果
| Exp | 数据集 | 变量 | ALL Best | AVG Best | ALL Last | AVG Last | Gap(AVG) |
|---|---|---|---|---|---|---|---|
| baseline | PACS | LR=0.1 | 83.75 | 82.24 | 77.53 | 75.46 | 6.78 |
| EXP-052 | PACS | LR=0.05 | | | | | |
| EXP-053 | PACS | LR=0.2 | | | | | |
| EXP-054 | PACS | decay=0.998 | | | | | |
| baseline | Office | LR=0.1 | 84.13 | 89.95 | 80.55 | 85.86 | 4.09 |
| EXP-055 | Office | LR=0.05 | | | | | |

## 决策规则
- 如果某 LR 的 AVG Best > 当前 82.24 (PACS) 或 89.95 (Office) → 用新 LR 跑 multi-seed
- 如果所有 LR 都不如 0.1 → 确认 0.1 是最优,继续用
- 如果 decay=0.998 显著改善 gap → 可作为默认 decay

## 结论
