# EXP-047A | loss_aug Ramp-Down (修复 Best-Last Gap Bug)

## 基本信息
- **日期**:2026-04-09
- **类型**:bug fix ablation
- **算法**:feddsa_augdown
- **配置**:feddsa_exp047a.yml
- **状态**:✅ 已完成

## 代码证据(Bug)
`FDSE_CVPR25/algorithm/feddsa.py` L320-323:
```python
loss = loss_task + loss_aug + \
       aux_w * self.lambda_orth * loss_orth + \
       aux_w * self.lambda_hsic * loss_hsic + \
       aux_w * self.lambda_sem * loss_sem
```
- `loss_orth/hsic/sem` 都带 `aux_w` 衰减包装
- **`loss_aug` 裸 weight=1.0**,warmup 后永远满权重
- 结果:CE 梯度后期被"clean + style-aug 双倍"持续扰动
- 证据链:FedDSA gap ~5% vs FDSE gap ~2%

## 修复方案(A)
引入独立 `aug_w` 线性 ramp-down:
```
aug_w = 0.0                               if round < warmup (=50)
aug_w = 1.0                               if warmup <= round <= aug_peak_round (=100)
aug_w = max(aug_min, 1 - (r - peak)/span) if round > aug_peak_round
```
- `aug_peak_round=100`:round 50-100 保持满权重(让模型充分学习风格鲁棒)
- `aug_decay_span=80`:round 100-180 线性降到 `aug_min`
- `aug_min=0.1`:保留微弱 aug,防止完全脱离多样性

## 假设
- Best 保持或略降(aug peak phase 仍 active)
- **Last 显著提升**:round 150+ 无扰动,模型能 fine-tune 到稳定点
- **Gap 从 ~5% 降到 <3%**

## 运行命令
```bash
nohup python run_single.py --task PACS_c4 --algorithm feddsa_augdown --gpu 0 \
  --config ./config/pacs/feddsa_exp047a.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp047a.out 2>&1 &
```

## 结果
| 指标 | EXP-017 (baseline s2) | EXP-047A |
|---|---|---|
| Best | 82.24 | 81.33 |
| Last | 75.46 | 73.83 |
| Gap | 6.78 | **7.50** |

## 结论
- ❌ **假设不成立**:ramp-down loss_aug 反而让 gap 从 6.78 增加到 7.50
- Best 降了 0.91,Last 降了 1.63,全面劣化
- **分析**:style augmentation 在后期仍有正向泛化作用,降权导致模型域鲁棒性下降
- **判定**:`loss_aug` 没有 `aux_w` 不是 bug 而是合理设计;Aug 对训练后期仍必要
- 与 EXP-047D 共同否定了"后期 aug 有害"的假设
