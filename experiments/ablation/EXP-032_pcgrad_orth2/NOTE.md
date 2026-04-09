# EXP-032 | Triple Combo: HSIC=0 + orth=2.0 + PCGrad

## 基本信息
- **日期**：2026-04-08
- **类型**：ablation (终极组合)
- **方法**：PCGrad + 强正交 + 无HSIC
- **算法**：feddsa_pcgrad
- **状态**:✅ 已完成(199/200 轮,服务器 01:12 挂掉,log 恢复)

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

## 结果 (199/200 轮, log 恢复)
| 指标 | 值 |
|------|---|
| Best acc | **80.82** |
| Last acc | 75.96 |
| Best-Last gap | 4.86 |

## 结论
- vs EXP-017 (原版 82.24):**未突破 SOTA**, -1.42%
- vs EXP-029 (单 PCGrad 80.72):仅 +0.10%,**三合一无叠加效应**
- vs EXP-021 (orth=2.0 81.58):-0.76%
- Gap 4.86 > 期望 3%,稳定性未达标
- **判定**:三个正向因子不会线性叠加,可能触及同一优化瓶颈
- **pivot**:放弃 loss 组合路线,转向多数据集 + R500 验证
