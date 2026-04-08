# EXP-032 | Triple Combo: HSIC=0 + orth=2.0 + PCGrad

## 基本信息
- **日期**：2026-04-08
- **类型**：ablation (终极组合)
- **方法**：PCGrad + 强正交 + 无HSIC
- **算法**：feddsa_pcgrad
- **状态**：⏳ 待执行

## 目的
合并我们3个独立验证有效的改进：
1. **HSIC=0** (EXP-017): +2.31% Best
2. **orth=2.0** (EXP-021): +1.65% Best
3. **PCGrad** (EXP-029): Best-Last gap几乎为0 (最稳定)

## 与已有实验对比
| 实验 | HSIC | orth | PCGrad | Best | Last |
|------|------|------|--------|------|------|
| EXP-017 | 0 | 1.0 | 否 | **82.24%** | 75.46% |
| EXP-021 | 0.1 | 2.0 | 否 | 81.58% | 75.80% |
| EXP-029 | 0 | 1.0 | 是 | 80.08% | **79.91%** |
| **EXP-032** | **0** | **2.0** | **是** | **? 期望83%** | **? 期望81%** |

## 期望
- Best ≥ 82.5% (突破SOTA)
- Last ≥ 80% (稳定性达标)
- Best-Last gap ≤ 3%

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_pcgrad --gpu 0 \
    --config ./config/pacs/feddsa_exp032.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp032.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| Best-Last gap | |

## 结论
