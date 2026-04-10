# EXP-058 | Detach-Style: 阻断 orth 梯度污染 encoder

## 基本信息
- **目的**:修复审查发现的设计缺陷——orth loss 的梯度通过 style_head 回传到 encoder,干扰 encoder 学习有用特征
- **修改**:feddsa.py L297 `z_sty = model.get_style(h.detach())`
- **算法**:feddsa (base, 就地修改)
- **状态**:⏳ 待执行

## 代码证据(审查验证)
```
未 detach 时:
  CE only     -> style_head grad: NONE
  orth only   -> style_head grad: 66.61
  orth loss   -> encoder.fc1 grad: 3594.02  ← 很大! orth 在改 encoder

detach 后:
  orth loss   -> encoder.fc1 grad: 2680.87  ← 仅从 z_sem 路径回传
```

## 设计逻辑
- `z_sem = model.get_semantic(h)` → encoder 通过 task + InfoNCE 学语义特征
- `z_sty = model.get_style(h.detach())` → style_head 学正交于 z_sem 的方向,但不影响 encoder
- orth loss 仍然约束 z_sem 和 z_sty 正交,但只从 z_sem 侧影响 encoder
- encoder 不再被"学会与无任务信号的 style 正交"的梯度干扰

## 附加发现
- style_head 的 z_sty [128-d] 从未被用于风格增强(AdaIN 用的是 h [1024-d] 的 mu/sigma)
- style_head 实际上是"反语义锚",不是真正的风格提取器
- detach 后 style_head 仍能学习(orth gradient 仍流入 style_head 自身)

## 运行命令
```bash
# PACS (LR=0.1, 当前最优)
nohup python run_single.py --task PACS_c4 --algorithm feddsa --gpu 0 \
  --config ./config/pacs/feddsa_v4_no_hsic.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp058_pacs.out 2>&1 &

# Office (LR=0.05, 当前最优)
nohup python run_single.py --task office_caltech10_c4 --algorithm feddsa --gpu 0 \
  --config ./config/office/feddsa_office_lr005.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp058_office.out 2>&1 &
```

## 对照
| 数据集 | 旧版 (无 detach) | EXP-058 (detach) | Delta |
|---|---|---|---|
| PACS AVG Best (s2) | 82.24 | | |
| PACS ALL Best (s2) | 83.75 | | |
| Office AVG Best (s2) | 90.82 (LR=0.05) | | |

## 结果
| 数据集 | ALL Best | AVG Best | AVG Last | Gap |
|---|---|---|---|---|
| PACS | | | | |
| Office | | | | |

## 结论
