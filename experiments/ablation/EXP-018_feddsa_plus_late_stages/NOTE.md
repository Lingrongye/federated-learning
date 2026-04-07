# EXP-018 | FedDSA+ Later Stages (More Conservative)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (Stage时机优化)
- **方法**：FedDSA+ with stage1_end=80, stage2_end=150
- **状态**：⏳ 待执行

## 目的
EXP-015(FedDSA+ stage1=50/stage2=100)假设stage1=50足够。
本实验更保守，让backbone学得更充分(stage1=80)再引入解耦，
风格增强延后到Round 150。验证"更晚=更稳定"假设。

## 假设
- 更晚的stage1结束 → backbone已收敛到较好状态
- 减少早期stage切换时的扰动
- 200轮内可能无法看到stage3完整效果，但能验证stage2质量

## 与EXP-015的差异
| 参数 | EXP-015 | EXP-018 |
|------|---------|---------|
| stage1_end | 50 | **80** |
| stage2_end | 100 | **150** |
| transition_width | 10 | **15** (更平滑) |

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_plus --gpu 0 \
    --config ./config/pacs/feddsa_plus_late.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp018.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| 与EXP-015差距 | |

## 结论
