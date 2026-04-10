# EXP-051 | Office-Caltech10 数据集扩展

## 基本信息
- **目的**:在第二个数据集上验证 FedDSA vs FDSE,增强 paper claim
- **数据集**:Office-Caltech10 (4 域: Caltech, Amazon, DSLR, Webcam; 10 类)
- **算法**:FedDSA 原版 + FDSE R200
- **状态**:✅ FedDSA+FDSE 已完成;基线待跑

## 为什么选 Office-Caltech10
1. FDSE 论文报告的 OC10 结果:FDSE **87.15%**, FedAvg 82.60%, FedBN 83.08%
2. 代码管线完全就绪(task/config/model 全有)
3. 4 域 4 clients,与 PACS 架构一致
4. 数据量小(~2500),训练极快

## 数据划分(data.json)
| Client | 域 | 样本数 |
|---|---|---|
| Client0 | Caltech | 1123 |
| Client1 | Amazon | 958 |
| Client2 | DSLR | **157** (极小) |
| Client3 | Webcam | 295 |

⚠️ Client2(DSLR)只有 157 样本,batch_size=50 下每 epoch 只 3 步。可能导致原型统计不稳定。

## 配置对齐
| 参数 | FDSE Office (论文) | FedDSA Office | 差异 |
|---|---|---|---|
| R | **500** | **200** | 同预算对比 |
| E | **1** | **1** | ✅ 对齐 |
| B | 50 | 50 | ✅ |
| LR | 0.1 | 0.1 | ✅ |
| decay | 0.9998 | 0.9998 | ✅ |
| FDSE lmbd | 0.05 | N/A | FDSE Office 用 0.05(vs PACS 0.5) |

## 实验矩阵
| 方法 | seeds | 预计时间/run |
|---|---|---|
| FedDSA R200 | 2, 15, 333 | ~30min (数据小+E=1) |
| FDSE R200 | 2, 15, 333 | ~30min |

6 runs 总计 ~3 小时,GPU 足够并行全部。

## 运行命令
```bash
# FedDSA
for s in 2 15 333; do
  nohup python run_single.py --task office_caltech10_c4 --algorithm feddsa --gpu 0 \
    --config ./config/office/feddsa_office.yml --logger PerRunLogger --seed $s \
    > /tmp/exp051_feddsa_s${s}.out 2>&1 &
  sleep 2
done

# FDSE
for s in 2 15 333; do
  nohup python run_single.py --task office_caltech10_c4 --algorithm fdse --gpu 0 \
    --config ./config/office/fdse_office_r200.yml --logger PerRunLogger --seed $s \
    > /tmp/exp051_fdse_s${s}.out 2>&1 &
  sleep 2
done
```

## 结果
### FedDSA (R200, E=1)
| seed | Best | Last | Gap |
|---|---|---|---|
| 2 | 89.95 | 85.86 | 4.09 |
| 15 | **91.08** | **90.18** | **0.90** |
| 333 | 86.35 | 83.53 | 2.82 |
| **3-seed mean** | **89.13 ± 2.42** | **86.52** | **2.60** |

### FDSE (R200, E=1)
| seed | Best | Last | Gap |
|---|---|---|---|
| 2 | **92.39** | **91.90** | **0.49** |
| 15 | 91.24 | 89.96 | 1.28 |
| 333 | 88.11 | 85.80 | 2.31 |
| **3-seed mean** | **90.58 ± 2.22** | **89.22** | **1.36** |

### 参考:FDSE 论文 OC10 R500 结果
| 方法 | Acc |
|---|---|
| FedAvg | 82.60 |
| FedBN | 83.08 |
| Ditto | 84.12 |
| **FDSE** | **87.15** |

### 方法对比总结
| 方法 | OC10 R200 3-seed mean | vs FDSE 论文 R500 |
|---|---|---|
| **FedDSA (ours)** | **89.13 ± 2.42** | +1.98 (R200 已超 R500!) |
| **FDSE R200** | **90.58 ± 2.22** | +3.43 |
| FDSE 论文 R500 | 87.15 | — |

## 结论
- **FDSE R200 (90.58) > FedDSA R200 (89.13)**:Office 上 FDSE 赢 1.45%
- 与 PACS 相反(PACS 上 FedDSA 微赢 0.50%)
- 两者均**大幅超越 FDSE 论文 R500 报告值 87.15%**,差距达 +2~3.4%
  - 可能原因:数据划分差异 / 训练策略差异 / 论文用 5-seed 但我们只用 3-seed
- FDSE 在 Office 上 gap 更小(1.36 vs 2.60):稳定性也更好
- **FedDSA 在 Office 上弱于 FDSE**:可能因为 Office 数据量极小(DSLR 只 157 样本)
  导致 style bank 统计量不稳定,AdaIN 增强噪声大于收益
- **paper 影响**:需要在 Office 上找到 FedDSA 的优势角度,或承认 FDSE 在小数据域上更稳定

## 待补基线
- FedAvg Office R200 × 3 seeds
- FedBN Office R200 × 3 seeds
- Ditto Office R200 × 3 seeds
