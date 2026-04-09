# EXP-051 | Office-Caltech10 数据集扩展

## 基本信息
- **目的**:在第二个数据集上验证 FedDSA vs FDSE,增强 paper claim
- **数据集**:Office-Caltech10 (4 域: Caltech, Amazon, DSLR, Webcam; 10 类)
- **算法**:FedDSA 原版 + FDSE R200
- **状态**:⏳ 准备就绪,待服务器恢复后启动

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
| 2 | | | |
| 15 | | | |
| 333 | | | |
| **3-seed mean** | | | |

### FDSE (R200, E=1)
| seed | Best | Last | Gap |
|---|---|---|---|
| 2 | | | |
| 15 | | | |
| 333 | | | |
| **3-seed mean** | | | |

### 参考:FDSE 论文 OC10 R500 结果
| 方法 | Acc |
|---|---|
| FedAvg | 82.60 |
| FedBN | 83.08 |
| Ditto | 84.12 |
| **FDSE** | **87.15** |

## 结论
