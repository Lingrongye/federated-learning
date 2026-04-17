# EXP-020 | FedDSA V4 with tau=0.5 (Softer InfoNCE)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (InfoNCE温度)
- **方法**：FedDSA V4 + tau=0.5 (vs 0.1)
- **状态**：✅ 已完成

## 目的
InfoNCE温度tau控制对比损失的锐利度。tau=0.1很锐利，对噪声原型敏感；
tau=0.5更平滑，可能在早期原型不准确时更稳定。

## 与EXP-014的差异
| 参数 | EXP-014 | EXP-020 |
|------|---------|---------|
| tau | 0.1 | **0.5** |

## 假设
- tau=0.1：强锐化，对noise敏感 → 容易过拟合到早期的噪声原型
- tau=0.5：软化 → 对原型误差容忍度更高
- 如果Best更高: 说明原版tau=0.1太aggressive
- 如果Last更稳: 说明锐利损失是振荡根源之一

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4_tau05.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp020.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| Best acc | **80.65%** @Round 23 (极早) |
| Last acc | 74.27% |
| Drops>5% | 6 |
| 最弱域@best | 65.20% |
| 最强域@best | 87.76% |
| **稳定性** | **0.0747** (最差之一) |
| vs EXP-014 | +0.72% Best |
| 总轮次 | 200 |

## 结论

**softer InfoNCE (tau=0.5) 的双面效果**：
- ✅ Best提高+0.72%，说明tau=0.1确实太锐利
- ❌ 但Round 23就到峰值后**严重震荡**，稳定性0.0747最差
- ❌ Last掉到74.27%，过拟合严重

**解释**：更软的对比损失让模型更快找到好的特征空间（早期Best高），
但软损失的梯度信号更弱，后期无法精调，反而被其他损失拉走。

**建议**：tau=0.5可以作为warmup阶段的初始值，后期逐步降到0.1（类似curriculum learning）。
本实验作为tau敏感性分析的数据点。
