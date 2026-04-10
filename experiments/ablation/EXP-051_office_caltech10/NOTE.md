# EXP-051 | Office-Caltech10 数据集扩展

## 基本信息
- **目的**:在第二个数据集上验证 FedDSA vs FDSE,增强 paper claim
- **数据集**:Office-Caltech10 (4 域: Caltech, Amazon, DSLR, Webcam; 10 类)
- **算法**:FedDSA 原版 + FDSE R200
- **状态**:✅ FedDSA+FDSE 已完成;基线待跑

## 为什么选 Office-Caltech10
1. FDSE 论文报告的 OC10 结果:FDSE ALL=**87.15%** / AVG=**91.58%**, FedAvg ALL=82.60/AVG=86.26, FedBN ALL=83.08/AVG=87.01
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

> **⚠️ 指标说明**:论文同时报 ALL(按样本量加权)和 AVG(client 简单平均)两个指标。
> 我们的 `local_test_accuracy` = ALL,`mean_local_test_accuracy` = AVG。
> 之前误将论文的 ALL 与我们的 AVG 直接比,已修正。

### FedDSA (R200, E=1)
| seed | ALL Best | ALL Last | AVG Best | AVG Last |
|---|---|---|---|---|
| 2 | 84.13 | 80.55 | 89.95 | 85.86 |
| 15 | 86.91 | 85.33 | 91.08 | 90.18 |
| 333 | 82.12 | 78.95 | 86.35 | 83.53 |
| **mean** | **84.39 ± 2.40** | 81.61 | **89.13 ± 2.42** | 86.52 |

### FDSE (R200, E=1)
| seed | ALL Best | ALL Last | AVG Best | AVG Last |
|---|---|---|---|---|
| 2 | 88.10 | 87.31 | 92.39 | 91.90 |
| 15 | 86.92 | 84.93 | 91.24 | 89.96 |
| 333 | 84.12 | 82.91 | 88.11 | 85.80 |
| **mean** | **86.38 ± 2.01** | 85.05 | **90.58 ± 2.22** | 89.22 |

### 参考:FDSE 论文 OC10 R500 结果 (5-seed mean±std)
| 方法 | ALL | AVG |
|---|---|---|
| FedAvg | 82.60 ± 3.14 | 86.26 ± 2.54 |
| FedBN | 83.08 ± 1.84 | 87.01 ± 1.30 |
| Ditto | 84.12 ± 1.32 | 88.72 ± 1.28 |
| **FDSE** | **87.15 ± 2.06** | **91.58 ± 2.01** |

### 修正后的对比(同指标)
| 方法 | ALL (R200 我们) | ALL (论文 R500) | AVG (R200 我们) | AVG (论文 R500) |
|---|---|---|---|---|
| **FedDSA** | 84.39 | — | 89.13 | — |
| **FDSE** | 86.38 | **87.15** | 90.58 | **91.58** |
| 差距 FedDSA vs FDSE R200 | **-1.99** | | **-1.45** | |

## 结论
- **Office 上 FDSE 赢 FedDSA**: ALL -1.99%, AVG -1.45% (R200 同预算)
- 之前错误:~~"FedDSA 89.13 超越 FDSE 87.15"~~ → 87.15 是 ALL,89.13 是 AVG,指标不同不能比
- 同指标修正:FedDSA ALL 84.39 < FDSE 论文 ALL 87.15 (输 2.76%)
- 我们的 FDSE R200 ALL=86.38 接近论文 R500 ALL=87.15,复现基本成功
- **FedDSA 在 Office 上弱于 FDSE**:可能因为 DSLR 只 157 样本导致 style bank 不稳定
- **paper 影响**:Office 上不能 claim FedDSA 优于 FDSE,需另找优势或分析原因

## 待补基线
- FedAvg Office R200 × 3 seeds
- FedBN Office R200 × 3 seeds
- Ditto Office R200 × 3 seeds
