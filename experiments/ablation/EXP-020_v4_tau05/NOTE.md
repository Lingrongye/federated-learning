# EXP-020 | FedDSA V4 with tau=0.5 (Softer InfoNCE)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (InfoNCE温度)
- **方法**：FedDSA V4 + tau=0.5 (vs 0.1)
- **状态**：⏳ 待执行

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
| Best acc | |
| Last acc | |
| vs EXP-014 | |

## 结论
