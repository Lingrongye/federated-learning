---
date: 2026-04-28
type: 基线复现实验
status: deployed (sc5 12 runs + sc3 6 runs)
exp_id: EXP-132
goal: 在 F2DC 框架下复现 FedBN / FedProx / FedProto 三个经典 baseline 跟主表对齐
---

# EXP-132: F2DC 框架 FedBN + FedProx + FedProto baseline 复线

## 背景

主表 (PG-DFC vs baselines) 还差 FedBN / FedProto / FPL 三个经典 baseline 没跑过。
本次先复线 FedBN / FedProx / FedProto 三个 (FPL 后续看时间)。

## 算法实现

| 算法 | 文件 | 核心 idea | 关键改动 |
|---|---|---|---|
| FedBN [ICLR'21] | `F2DC/models/fedbn.py` | BN 层不聚合, 留本地 | `aggregate_nets_skip_bn` 用 strict=False 只 load non-BN keys |
| FedProx [MLSys'20] | `F2DC/models/fedprox.py` | FedAvg + 近端项 µ/2‖w-w_g‖² | `_train_net` 加 prox loss, 默认 µ=0.01 |
| FedProto [AAAI'22] | `F2DC/models/fedproto.py` | 上传/聚合 class prototype + MSE proto loss | 维护 local/global prototype dict, 加 µ=1.0 MSE 权重 |

## 实现关键 fix (sanity 抓到的 bug)

1. **fedbn.py**: ResNet 用 `_features = nn.Sequential([...])` 包装, BN buffer 在 state_dict
   有两个 alias key (`bn1.running_mean` 跟 `_features.1.running_mean`). 旧字符串 match
   只 capture 主路径名, alias key 被错聚合, load 时通过 alias 写回真 BN buffer.
   **修**: `collect_bn_keys` 用 `data_ptr()` 比对 storage 指针, 任何 tensor 指针落在 BN
   module set 里都判为 BN.

## Sanity test 结果

`F2DC/test_baselines_sanity.py`:
- FedBN: BN preserved 96/96 ✓ , non-BN 28/32 changed (聚合生效) ✓
- FedProx: prox term = 0.711 > 0 ✓
- FedProto: 4 clients local protos / 7 class global protos / dim=512 ✓ , 第二轮 MSE path 跑通 ✓

## 实验配置 (跟 EXP-130 / EXP-131 一致)

- Datasets: PACS / Office-Caltech10 / Digits
- Allocation: fixed (PACS photo:2/art:3/cartoon:2/sketch:3, Office c:3/a:2/w:2/d:3, Digits m:3/u:6/s:6/n:5)
- Seeds: {15, 333} (跟主表一致)
- R=100 communication rounds, E=10 local epochs
- Optimizer: SGD lr=0.01 momentum=0.9 wd=1e-5
- Batch: PACS 46 / Office 64 / Digits 64
- Backbone: resnet10 (跟 fedavg / moon 同 backbone, 跟 f2dc 的 resnet10_dc 不同)

## 部署 (2026-04-28 00:14 启)

总 18 runs = 3 algo × 3 dataset × 2 seed.

### sc5 (12 runs): PACS + Office × 3 algo × 2 seed
Launcher: `/tmp/launch_sc5_baselines.sh` (greedy, MIN_FREE=2500MB)

- fedbn_pacs_{15,333}, fedbn_office_{15,333}
- fedprox_pacs_{15,333}, fedprox_office_{15,333}
- fedproto_pacs_{15,333}, fedproto_office_{15,333}

Logs: `experiments/ablation/EXP-132_baselines_3algo/logs/`

### sc3 (6 runs): Digits × 3 algo × 2 seed
Launcher: `/tmp/launch_sc3_baselines.sh` (greedy, MIN_FREE=2000MB)

- fedbn_digits_{15,333}
- fedprox_digits_{15,333}
- fedproto_digits_{15,333}

Logs: `experiments/ablation/EXP-132_baselines_3algo/logs/`

## 预期 wall time

- 单 run R100 ~50s/round = ~83 min
- sc5 12 runs 分 3 批 (~5 并发) = ~3-4h 总
- sc3 6 runs 一批跑完 = ~1.5h 总

预计 04:00-05:00 (4 月 28 日上午) 全部完成。

## 跑完后预期回填的主表

PACS / Office / Digits 三表的 baseline 行 (FedBN / FedProx / FedProto), 跟 PG-DFC v3.2 比.

预期结果范围 (基于文献):
- FedBN PACS ~75-78 (跟 fedavg 接近, 略好)
- FedProto Office ~58-62 (跟 fedavg 接近)
- FedProto Digits ~88-92 (低于 F2DC)
