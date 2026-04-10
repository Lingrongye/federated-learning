# EXP-059 | StyleHead→StyleBank 连接

## 基本信息
- **目的**:修复审查发现的第二个设计缺陷——style_head 的输出 z_sty 与 style bank/AdaIN 增强完全脱节
- **算法**:feddsa_stylehead_bank
- **包含 EXP-058 的 detach 修复**
- **状态**:⏳ 待执行

## 设计缺陷(审查发现)
原版:
```
style_head: h [1024] → z_sty [128]  → 仅用于 orth loss,从不参与增强
style bank: 存 h [1024] 的 (mu, sigma) → AdaIN 在 h 空间操作
```
style_head 和 style bank 是两个独立的"风格"概念,完全没连接。

## 本次修复
```
style_head: h [1024] → z_sty [128]  → orth loss + style bank 统计量来源
style bank: 存 z_sty [128] 的 (mu, sigma)
AdaIN: 在 z_sem [128] 空间操作,使用 z_sty 的 style bank 统计量
loss_aug: 分类 AdaIN(z_sem, sty_stats) → head → CE
```

改动:
1. style bank 改存 z_sty 统计量(128-d)而非 h 统计量(1024-d)
2. AdaIN 在 z_sem 空间操作(128-d),而非 h 空间(1024-d)
3. 包含 EXP-058 的 `h.detach()` fix

## 预期效果
- style_head 现在有**真正的任务信号**(通过 loss_aug → AdaIN → z_sem 路径)
- style bank 和 style_head 统一,论文叙事更一致
- AdaIN 在更低维空间(128 vs 1024)操作,可能更高效

## 运行命令
```bash
# PACS
nohup python run_single.py --task PACS_c4 --algorithm feddsa_stylehead_bank --gpu 0 \
  --config ./config/pacs/feddsa_v4_no_hsic.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp059_pacs.out 2>&1 &

# Office
nohup python run_single.py --task office_caltech10_c4 --algorithm feddsa_stylehead_bank --gpu 0 \
  --config ./config/office/feddsa_office_lr005.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp059_office.out 2>&1 &
```

## 对照
| 数据集 | 原版 | EXP-058 detach | EXP-059 detach+bank |
|---|---|---|---|
| PACS AVG (s2) | 82.24 | 待测 | 待测 |
| Office AVG (s2) | 90.82 | 待测 | 待测 |

## 结果
| 数据集 | ALL Best | AVG Best | AVG Last | Gap |
|---|---|---|---|---|
| PACS | | | | |
| Office | | | | |

## 结论
