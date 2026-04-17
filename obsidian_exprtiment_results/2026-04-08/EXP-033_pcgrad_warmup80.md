# EXP-033 | PCGrad + warmup=80 + orth=2.0

## 基本信息
- **日期**：2026-04-08
- **类型**：ablation (PCGrad + 长warmup)
- **方法**：feddsa_pcgrad with warmup=80
- **状态**:✅ 已完成(200/200,log 恢复)

## 目的
基于EXP-032，进一步延长warmup让backbone更充分预训练。
PCGrad + 长warmup = 双重稳定保障。

## 与EXP-032的差异
| 参数 | EXP-032 | EXP-033 |
|------|---------|---------|
| warmup_rounds | 50 | **80** |
其他相同

## 期望
- 比EXP-032更稳定
- Best可能略低（因为辅助loss延后引入）
- 但Last应该更接近Best

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_pcgrad --gpu 0 \
    --config ./config/pacs/feddsa_exp033.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp033.out 2>&1 &
```

## 结果 (log 恢复)
| 指标 | 值 |
|------|---|
| Best acc | 80.18 |
| Last acc | 75.09 |
| Gap | 5.09 |

## 结论
- vs EXP-032 (warmup=50 同 PCGrad+orth2, 80.82): **-0.64%**, 长 warmup **反而更差**
- 长 warmup 没帮助:decouple/对齐 loss 延后引入的损失大于 backbone 预训练收益
- **判定**:warmup=50 已是足够,不需要更长
