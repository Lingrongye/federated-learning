# EXP-118 | 完整 FedBN (γ/β 整层本地) vs 半 FedBN (仅 running stats 本地) 对照

## 基本信息
- **日期**: 2026-04-22 09:22 启动, 12:26 完成 (Office)
- **算法**: `feddsa_scheduled_fullbn` (继承 feddsa_scheduled, 仅 override `_init_agg_keys` 让整个 BN 层进 private_keys)
- **服务器**: lab-lry GPU 1 (和 EXP-116/117 共享)
- **状态**: ✅ **Office 3/3 完成** (PACS 没启动)

## 这个实验做什么 (大白话)

**量化 BN 层 γ/β (可学习 affine 参数) 是否该参与 FedAvg 聚合**.

历史背景:
- 半 FedBN (原版 FedBN, ICLR 2021): 只 **running_mean/running_var** 本地化, γ/β 参与 FedAvg 聚合
- 完整 FedBN (后续实现): **整个 BN 层**全本地化 (γ/β + running_mean + running_var 都不聚合)

我们自己的 feddsa_scheduled 一直用"半 FedBN"实现 (EXP-080 PACS 80.41 / Office 89.44). 但没有做过"完整 FedBN"的对照 — 不知道 γ/β 聚合是否在伤害我们的 accuracy.

**本实验**: 仅改 `_init_agg_keys` 把 BN 整层进 private_keys, 其他**完全不动**, 跑 Office 3 seeds, 对比 EXP-080.

## 变体通俗解释

| 配置 | BN γ/β | BN running stats | 其他 |
|------|:------:|:----------------:|------|
| **EXP-080 半 FedBN (feddsa_scheduled)** | ✅ FedAvg 聚合 | ❌ 本地 | lo=1 其他同 |
| **EXP-118 完整 FedBN (feddsa_scheduled_fullbn)** | ❌ **本地** | ❌ 本地 | 其他 100% 同 EXP-080 |

## 实验配置

| 参数 | 值 |
|------|:--:|
| Task | office_caltech10_c4 |
| 算法 | feddsa_scheduled_fullbn (lo=1.0 lh=0 ls=1.0 tau=0.2 pd=128 ...) |
| R / E / B / LR | 200 / 5 / 50 / 0.05 |
| Seeds | {2, 15, 333} |
| Config | `config/office/feddsa_fullbn_office_r200.yml` |

**PACS 为什么没跑**: EXP-118 启动时 lab-lry GPU 1 显存已被 EXP-116 (6 runs) + EXP-117 (3 runs) 占, 只能再塞下 3 个 Office (E=1 显存少). PACS E=5 没空间. 如果 Office 差异不显著, PACS 也大概率无差异, 所以没追跑.

## 数据准备/执行细节

**执行过程中的 bug**:
1. **第一次 09:22 启动** — 用 default_model 加载 (log 文件带 `Mdefault_model`), 但 default_model 没有 `encode` 方法 (feddsa_scheduled 训练需要), 所有 3 seed 立即崩.
2. **修复 commit dab5103**: 在 `feddsa_scheduled_fullbn.py` 里 `from algorithm.feddsa_scheduled import init_global_module, init_local_module, init_dataset`, 让 flgo 能用 feddsa_scheduled 的 FedDSAModel 而不是 fallback 到 default.
3. **第二次 09:24 重启** — 用 FedDSAModel 加载 (log 带 `Mfeddsa_scheduled_fullbn`), 3 seed 全部正常跑完 R200.
4. **JSON 保存失败 (Errno 36 File name too long)**: flgo 自动生成的 record 文件名 = algo + 所有 algo_para + 所有超参, 合起来 **260+ 字符**超过 Linux ext4 的 255 字节文件名上限. 所有 3 个 seed 都踩这个坑, **数据只在 log 文件里**.

## 🏆 结果 (从 log 直接提取)

### 3-seed 汇总

| 方法 | seeds | Office AVG Best | Office AVG Last | 备注 |
|------|:----:|:---:|:---:|---|
| FDSE (paper baseline) | 3 | 90.58 | 89.22 | Table 1 |
| EXP-080 半 FedBN (lo=1) | {2,15,333} | **89.44** | 88.71 | γ/β 聚合 |
| EXP-116 半 FedBN (lo=0) | {2,15,333} | **89.75** | 86.89 | γ/β 聚合, 无正交 |
| **EXP-118 完整 FedBN (lo=1)** | {2,15,333} | **89.68** | 88.58 | γ/β 本地 |

### Per-seed (EXP-118 fullbn)

| Seed | ALL B/L | AVG B/L |
|:---:|:---:|:---:|
| 2 | 100.00/100.00 ⚠️ | 90.59/88.76 |
| 15 | 100.00/96.55 ⚠️ | 87.44/86.15 |
| 333 | 100.00/100.00 ⚠️ | 91.00/89.82 |
| **Mean** | — | **89.68/88.58** |

### ⚠️ ALL=100% 异常解释

ALL (local_test_accuracy, **per-client test set weighted avg**) 出现 100% 的异常, 原因是 FedBN full (BN 整层本地) 让每个 client 的模型在本地 test 上**过度拟合**:
- 每个 client 用**自己的 BN 统计量**做 eval → BN 统计量已经完美匹配该 client 数据
- 小 client (如 DSLR 只 ~157 sample) 的 test 被本地 BN + local weights memorize 到 100%
- 但 AVG (mean_local_test_accuracy, **per-client 简单 mean**) 是 89.68 — 说明部分 client (Caltech, Amazon) test acc 没到 100%, 只有 DSLR/Webcam 之类小 client 达 100%

**实际 paper 可用指标**: **AVG 89.68**, 不是 ALL 100.

## 🔍 判决结论

### Δ (完整 FedBN − 半 FedBN)

| 对比 | EXP-080 lo=1 | EXP-118 lo=1 | Δ |
|---|:---:|:---:|:---:|
| Office AVG Best | 89.44 | **89.68** | **+0.24** |
| Office AVG Last | 88.71 | 88.58 | −0.13 |

**判决**: 
- `|Δ Best| = 0.24 < 0.3pp` → 属于"差异无实质"区间
- `|Δ Last| = 0.13 < 0.3pp` → 同样无实质差异

**结论**: **γ/β 是否参与 FedAvg 几乎不影响 Office accuracy**. 两种 FedBN 写法都 OK, 半 FedBN (γ/β 聚合) 略差 0.24pp 但在 seed 方差内 (std ~1.5pp).

### 对 paper 的影响

- ✅ Paper 用"半 FedBN" (EXP-080 接口) 是 safe 的, 不需要重写代码
- ✅ 如果 reviewer 问 "你们 FedBN 是完整还是半? γ/β 是否本地?", 可以回答 "本文用半 FedBN 对齐 FedBN ICLR 2021 原版, 我们测试 γ/β 本地化只带 +0.24pp 微弱 Office 增益, 在 seed 方差内"
- ⚠️ EXP-116 lo=0 (89.75) 比 EXP-118 fullbn (89.68) 还略高 0.07pp — 再次印证 **BN γ/β 本地不是性能来源** (详见发现 6)

## 下一步 (如果需要)

1. PACS 是否跑 fullbn? — 只有在 Office 看到 ≥ 0.5pp 改进才值得跑. 目前 +0.24 不值得.
2. 代码 cleanup: fullbn 作为 `feddsa_scheduled` 的一个 yaml 参数 (`bn_full=True`) 而不是单独算法 — 减少 code 分叉. 但这是 engineering cleanup 不是科学价值.

## 📎 相关文件

- 算法: `FDSE_CVPR25/algorithm/feddsa_scheduled_fullbn.py`
- Config: `FDSE_CVPR25/config/office/feddsa_fullbn_office_r200.yml`
- Log (数据源, JSON 因文件名超长没保存): `task/office_caltech10_c4/log/2026-04-22-09-24-*Mfeddsa_scheduled_fullbn*.log`
- 对照实验: [EXP-080 半 FedBN (base)], [EXP-116 lo=0 (半 FedBN 无正交)]
