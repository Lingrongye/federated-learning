# EXP-061~063 | Style Sharing 诊断实验套件

## 基本信息
- **日期**:2026-04-11
- **目的**:系统性验证 GPT-5.4 review 的核心诊断——Office 负迁移源自 Beta(0.1,0.1) 过激 AdaIN
- **共 3 个变体 + EXP-060 gated = 4 维度对照**
- **状态**:⏳ 待执行

## 假设
**H1**: FedDSA 在 Office 上输 FDSE,根本原因是 Beta(0.1,0.1) 对低风格差域制造 off-manifold 噪声。

## 诊断矩阵 (Office seed=2)

| 实验 | 假设测试 | 预期 Office AVG |
|---|---|---|
| FedDSA 原版 (baseline) | | 89.95 |
| **EXP-060 Gated** | 通过距离门限阻止小差异 style dispatch | > 90? |
| **EXP-061 NoAug** | 完全禁用 style aug,看 Office 是否变好 | > 89.95? (若 > 则 H1 成立) |
| **EXP-062 SoftBeta** | Beta(1,1) 均匀混合,而非 Beta(0.1,0.1) 极端 | 介于 baseline 和 NoAug 之间 |
| **EXP-063 AugSchedule** | loss_aug 在后 1/3 cosine decay 到 0 | 比 baseline 略好 |

## 推理逻辑

```
如果 NoAug (061) > baseline → 证实 H1: style aug 是负迁移源头
  如果 Gated (060) ≈ NoAug → gated 等于禁用,需要更严格的自适应
  如果 Gated (060) > NoAug → gated 保留了少量有效 dispatch
  如果 SoftBeta (062) > baseline → Beta 参数是关键调参
  如果 AugSchedule (063) > baseline → 时序调度是关键

如果 NoAug (061) ≤ baseline → H1 错误,style aug 本身有益,问题在别处
  可能原因: aggregation conflict (FDSE consensus max 更重要)
  需要 EXP-064 实现 consensus-aware aggregation
```

## PACS sanity check

每个变体也跑 PACS seed=2, 确保改动不会 **破坏原本赢的场景**:

| 实验 | PACS 预期 |
|---|---|
| 原版 | 82.24 |
| EXP-060 Gated | ≈ 82 (所有距离大,基本不 gate) |
| EXP-061 NoAug | ≈ 80 (失去 style aug 收益, 可能跌) |
| EXP-062 SoftBeta | ≈ 81 (混合变弱,可能跌) |
| EXP-063 AugSchedule | ≈ 82 (前 2/3 满权重) |

## 代码变动位置

| 实验 | 文件 | 改动 |
|---|---|---|
| EXP-060 | feddsa_gated.py | Server.pack() 加距离门限 |
| EXP-061 | feddsa_noaug.py | Server.pack() 返回 style_bank=None |
| EXP-062 | feddsa_softbeta.py | Client._style_augment 改 Beta(1,1) |
| EXP-063 | feddsa_augschedule.py | Client.train() 加 w_aug 余弦 schedule |

## 运行命令

```bash
# Office 4 个实验
for algo in feddsa_gated feddsa_noaug feddsa_softbeta feddsa_augschedule; do
  nohup python run_single.py --task office_caltech10_c4 --algorithm $algo --gpu 0 \
    --config ./config/office/${algo}.yml --logger PerRunLogger --seed 2 \
    > /tmp/exp06x_${algo}_office.out 2>&1 &
done

# PACS 4 个实验 (sanity check)
for algo in feddsa_gated feddsa_noaug feddsa_softbeta feddsa_augschedule; do
  nohup python run_single.py --task PACS_c4 --algorithm $algo --gpu 0 \
    --config ./config/pacs/${algo}.yml --logger PerRunLogger --seed 2 \
    > /tmp/exp06x_${algo}_pacs.out 2>&1 &
done
```

## 结果矩阵

### Office-Caltech10 (seed=2, AVG Best)

| Method | ALL Best | AVG Best | AVG Last | Gap | vs baseline |
|---|---|---|---|---|---|
| FedDSA baseline | 84.13 | 89.95 | 85.86 | 4.09 | — |
| FDSE R200 | 88.10 | 92.39 | 91.90 | 0.49 | — |
| EXP-060 Gated | | | | | |
| EXP-061 NoAug | | | | | |
| EXP-062 SoftBeta | | | | | |
| EXP-063 AugSchedule | | | | | |

### PACS (seed=2, AVG Best)

| Method | ALL Best | AVG Best | AVG Last | Gap | vs baseline |
|---|---|---|---|---|---|
| FedDSA baseline | 83.75 | 82.24 | 75.46 | 6.78 | — |
| FDSE R200 | 82.04 | 80.81 | 78.09 | 2.72 | — |
| EXP-060 Gated | | | | | |
| EXP-061 NoAug | | | | | |
| EXP-062 SoftBeta | | | | | |
| EXP-063 AugSchedule | | | | | |

## 决策树

- **最佳 Office 表现 > 90** → 找到了 fix,补多 seed 验证
- **最佳表现 88-90** → 方向对但不够,考虑组合 (gated + augschedule)
- **最佳表现 < 89.95** → H1 部分或完全错误,需转向 aggregation 改进

## 结论
