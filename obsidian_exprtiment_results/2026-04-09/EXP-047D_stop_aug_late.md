# EXP-047D | Hard Stop Style Aug at Round 150

## 基本信息
- **日期**:2026-04-09
- **类型**:hard ablation
- **算法**:feddsa_noaug_late
- **配置**:feddsa_exp047d.yml
- **状态**:✅ 已完成

## 目的
EXP-047A 的硬消融对照:**完全停止** style augmentation 从 round 150 开始。
- 如果有效 → 证明 aug 后期有害(假设成立)
- 如果无效 → 说明 Best-Last gap 另有原因

## 与 047A 的关系
| 方案 | round<150 | round 150-200 |
|---|---|---|
| 047A (ramp-down) | aug_w=1→0.1 线性降 | aug_w=0.1(微弱保留) |
| **047D (hard stop)** | aug_w=1.0 | **aug_w=0** 完全关闭 |

047D 提供对照:如果 047D 比 047A 还好,说明"彻底停"比"ramp 降"更合适。

## 配置
- `aug_stop_round = 150` (R200 的最后 25% 完全关闭 style aug)
- 其他一切同 EXP-017 配置

## 运行命令
```bash
nohup python run_single.py --task PACS_c4 --algorithm feddsa_noaug_late --gpu 0 \
  --config ./config/pacs/feddsa_exp047d.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp047d.out 2>&1 &
```

## 结果
| 指标 | EXP-017 baseline | EXP-047D |
|---|---|---|
| Best | 82.24 | **82.32** |
| Last | 75.46 | 73.81 |
| Gap | 6.78 | **8.51** |

## 结论
- Best 82.32 ≈ 基线(微高 0.08,无意义)
- ❌ **Last 降到 73.81**(比基线 75.46 还低 1.65)
- ❌ **Gap 增到 8.51**,比基线 6.78 更差
- **解读**:round 150 后关闭 aug → 最后 50 轮模型丧失跨域泛化 → Last 反而崩
- 与 EXP-047A 共同结论:**style augmentation 在全程都有正向作用,不可缩减**
- Aug 导致的"震荡"不是 bug,而是泛化能力的代价;停掉 aug 反而更差
