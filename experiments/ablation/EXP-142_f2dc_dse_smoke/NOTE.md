# EXP-142 F2DC + DSE_Rescue3 R30 Smoke

**日期**: 2026-05-01
**目的**: 验证新方案 f2dc_dse 在 PACS R30 短跑下:
  1. 不崩盘 (R30 acc vs F2DC vanilla R30 同 seed gap < 5pp)
  2. DSE diag 真实数据下健康:
     - `proto3_ema_delta_norm` 收敛 (前几轮大, 后期 < 0.1)
     - `mag_exceed_rate` 在 ramp 后 < 5% (warmup 阶段 0)
     - `mag_ratio_p95_mean` < r_max=0.15 (大部分样本不爆)
     - `ccc_improvement` > 0 (rescued cos > raw cos, 即 DSE 真在拉近 proto)
     - `proto3_valid_ratio` 接近 1.0 (大部分 class 每轮被 client 见到)
  3. 13 个单元测试在 sub1 (cuda 12.1) 已全过, 真实链路 server EMA + global_net sync
     在长跑中没出现 NaN / shape mismatch / dump 失败

## 变体通俗解释

之前 PG-DFC 跟 PG-DFC-ML 都死了 (attention entropy ~1.0, mask3 学不动)。这次回到纯 F2DC base, 只在 layer3/layer4 之间嵌一个 **DSE_Rescue3** 模块 (灵感来自 FDSE_CVPR25 的 DSEConv 但更轻):

- 1×1 reduce (256→32) → GroupNorm/ReLU → depthwise 3×3 → 1×1 expand (zero-init)
- `feat3_rescued = feat3_raw + rho_t * delta3` (rho_t warmup→ramp 调度)
- 再加 **CCC** (Class-Conditional Consistency): `1 - cos(rescued, server_proto3_unit[label])`
- **Magnitude guard**: `(max(0, ||rho*delta||/||feat|| - r_max))^2` 防止 delta 太大

主张: F2DC 在 layer4 解耦 robust/non-robust, 但 layer3 跨域偏移没纠 → DSE_Rescue3 在 layer3 把 cross-domain shift 校正到 server 累的 class proto3 方向, 让 layer4 拿到更纯净的输入。

## 技术细节

| 组件 | 公式 / 配置 |
|---|---|
| DSE_Rescue3 参数量 | ~45k (256→32 reduce + 3×3 dw + 32→256 expand) vs F2DC backbone ~5M = +0.9% |
| rho 调度 | warmup=5 (rho=0) → ramp 10 round (0→0.1) → full=0.1 |
| lambda_cc 调度 | warmup=5 (lambda=0, server 静聚 proto3) → ramp 10 (0→0.1) → full=0.1 |
| r_max | 0.15 (单 sample 的 ||rho*delta||/||feat|| 上限) |
| lambda_mag | 0.01 (mag loss 权重, 跟 main loss 比小 100×) |
| proto3 EMA β | 0.85 (server 累 raw feat3 GAP class mean, EMA 前后混) |
| CCC 诊断 | 每 (client, epoch) 固定前 2 batch 记 raw_cos / rescued_cos (改自 10% 随机) |

Loss: `L_total = L_F2DC_main + lambda_cc_t · CCC + lambda_mag · L_mag`

## 实验配置

| 项 | 值 |
|---|---|
| dataset | fl_pacs (4 域: photo, art, cartoon, sketch, fixed allocation) |
| parti_num | 10 client |
| communication_epoch | 30 (smoke, 不是 main R100) |
| local_epoch | 10 |
| seeds | 2, 15, 333 (3-seed) |
| 服务器 | sub1 (AutoDL nmb1, RTX 4090 24GB) |
| 路径 | /root/autodl-tmp/federated-learning/F2DC |

## 启动命令

```bash
cd /root/autodl-tmp/federated-learning/F2DC
PY=/root/miniconda3/bin/python
EXP_DIR=../experiments/ablation/EXP-142_f2dc_dse_smoke

# 3-seed PACS R30 smoke (rho_max=0.1, lambda_cc=0.1, mag r_max=0.15)
for SEED in 2 15 333; do
  nohup $PY main_run.py \
    --model f2dc_dse \
    --dataset fl_pacs \
    --seed $SEED \
    --communication_epoch 30 \
    --use_daa False \
    > $EXP_DIR/logs/pacs_s${SEED}_R30.log 2>&1 &
done
```

## 跑前 self-check

- [x] 13 个单元测试全过 (T1-T8 unit / T9 federated 7-round / T9b codex critical / T10 干预 / T11 N_CLASSES / T12 mag per-sample)
- [x] 模拟链路 F2DCDSE(...) → ini() → loc_update() → global_evaluate() OK (codex 验)
- [x] global_net.rho_t / global_proto3_unit_buf 跨 round sync OK
- [x] proto_logs persist 到 npz (proto_diag_*) + jsonl
- [x] global_proto3 tensor 落盘 (utils/diagnostic.py)
- [x] sub1 cuda 12.1 + torch 2.1.2 兼容性确认 (13 测试也过)

## 预期结果 (R30)

acc 大概不会比 R100 F2DC vanilla 高 (R30 太短没收敛), 关键看 diag 健康度:

| 指标 | 健康范围 | 异常红线 |
|---|---|---|
| acc R30 mean | > 65% (PACS, R30 没收敛) | < 50% 说明 DSE 把训练搞崩了 |
| proto3_ema_delta_norm | R5: 大 (~0.1+); R20+: < 0.05 | 一直不降 = proto3 没收敛 |
| mag_exceed_rate | R5-R15 ramp 阶段 < 10%; R15+: < 5% | > 30% 说明 r_max 太严或 rho 太大 |
| mag_ratio_p95_mean | < 0.15 (= r_max) | > 0.3 说明 mag guard 没起作用 |
| ccc_improvement | > 0 (rescued > raw) | < 0 持续 = DSE 反方向校正 |
| proto3_valid_ratio | > 0.9 | < 0.7 说明 client 数据稀疏 |

## 结果回填 (待跑完)

待 smoke 跑完后回填。
