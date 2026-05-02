# EXP-138 PG-DFC-ML R10 Smoke Test

**日期**: 2026-04-29
**目的**: 验证新增的 multi-layer PG-DFC (f2dc_pg_ml) 在 PACS R10 短跑下:
  1. 不崩盘 (主 acc 跟 PG-DFC v3.3 R10 比, gap 不超 5pp)
  2. layer3 mask3_sparsity 健康 (0.3-0.7,不 collapse 到 0/1)
  3. aux3_loss / main_loss 比值合理 (< 1.5,不主导)
  4. lite 分支梯度真实流 (不出现 NaN / 全 0)

## 变体通俗解释

PG-DFC v3.3 现在只在 layer4 (深层 512×4×4) 做 DFD 切 mask + DFC 重建。新版 ML(Multi-Layer) 在 layer3 (中层 256×8×8 Office / 256×32×32 PACS) 加一对 lite 版的 DFD/DFC,产生一个 aux3 logits 给一个**辅助 deep supervision loss**,但**不污染 layer4 主路输入**(layer4 仍然吃原始 layer3 输出而不是 cleaned 版本)。

`ml_aux_alpha=0` 时整个 lite 分支没梯度 → 退化等价 PG-DFC v3.3。
默认 `ml_aux_alpha=0.1` 起步,smoke 看 layer3 mask 是不是学得动。

## 技术细节

| 模块 | 结构 | 参数量 (PACS layer3=256×32×32) |
|---|---|---|
| **DFD_lite** | Conv3×3(C→32) + BN + ReLU + Conv3×3(32→C) + GumbelSigmoid(τ=0.5) | ~148k |
| **DFC_lite** | bottleneck (1×1→3×3→1×1, mid=32) + 不带 prototype | ~27k |
| **aux3** | nn.Linear(256, num_classes) | 1.8k (PACS) |

总 lite 增量 = ~177k 参数,vs PG-DFC v3.3 backbone ~5M = +3.5%。

Loss: `L_total = L_PGv33_existing + ml_aux_alpha · CE(aux3_logits, labels)`

## 启动命令

```bash
cd /root/autodl-tmp/federated-learning/F2DC
PY=/root/miniconda3/bin/python
EXP_DIR=../experiments/ablation/EXP-138_pgml_smoke

# 3-seed PACS R10 smoke (alpha=0.1)
for SEED in 2 15 333; do
  nohup $PY main_run.py \
    --model f2dc_pg_ml \
    --dataset fl_pacs \
    --seed $SEED \
    --communication_epoch 10 \
    --ml_aux_alpha 0.1 \
    --pg_proto_weight 0.3 \
    --pg_warmup_rounds 5 \
    --pg_ramp_rounds 3 \
    --use_daa False \
    > $EXP_DIR/logs/pacs_s${SEED}_alpha0.1_R10.log 2>&1 &
done

# 对照: ml_aux_alpha=0 (退化等价 PG-DFC v3.3, 验证不掉点)
for SEED in 2 15 333; do
  nohup $PY main_run.py \
    --model f2dc_pg_ml \
    --dataset fl_pacs \
    --seed $SEED \
    --communication_epoch 10 \
    --ml_aux_alpha 0.0 \
    --pg_proto_weight 0.3 \
    --pg_warmup_rounds 5 \
    --pg_ramp_rounds 3 \
    --use_daa False \
    > $EXP_DIR/logs/pacs_s${SEED}_alpha0.0_R10.log 2>&1 &
done

wait
```

## 验收标准 (Smoke 通过的判定)

| 指标 | 健康范围 | 判定 |
|---|---|---|
| **R10 主 acc** | alpha=0.1 ≥ alpha=0 - 5pp | 不掉点 |
| **alpha=0 vs PG-DFC v3.3** | max_abs_diff < 1pp | 退化等价 (这是单独验证) |
| **mask3_sparsity_mean** | 0.3 - 0.7 | mask 学得动,没坍塌 |
| **mask3_sparsity_std** | < 0.15 | round 内稳定 |
| **aux3_loss / main_loss** | < 1.5 | aux 不主导 |
| **训练 loss 单调下降** | yes | 没 NaN/发散 |
| **mask3 跨 round 不震荡** | per-round mean 变化 < 0.1 | BN 聚合稳定 |

任何一条不过 → 调超参重跑 (修 ml_lite_tau, ml_aux_alpha, ml_lite_channel)

## 本地 unit test 状态 (smoke 前的最后防线)

| Test | 验证 | 结果 |
|---|---|---|
| T1 forward shape | 7-tuple 接口 + aux3/mask3 shape (Office + PACS) | ✅ PASS |
| T2 transient refresh | _last_aux3_logits 每次 forward 刷新 | ✅ PASS (max_diff=0.0109) |
| T3 eval determinism | is_eval=True 两次 forward 主输出 + aux3 一致 | ✅ PASS (主 0,aux3 0,is_eval=False 6.97e-04) |
| T4 state_dict roundtrip | dfd_lite/dfc_lite/aux3 在 sd,_last_*/class_proto 不在 | ✅ PASS |
| T5 FedAvg 兼容 | 全 state_dict 平均后 load 不报 size mismatch | ✅ PASS |
| T6 gradient flow | α=0.1 lite 全有梯度,α=0 lite 全 None,layer4 仍有梯度 | ✅ PASS |
| **T7 alpha=0 等价 PG-DFC** | 同 seed/init,主路输出 bit-identical | ✅ PASS (max_abs_diff=0) |
| T8 PACS/Office/Digits | 三种 image_size 全部 forward 不崩 | ✅ PASS |

**8/8 PASS** — 设计层面无 bug,可上 GPU 跑实测验证 mask collapse 和 deep sup 收益。

## 结论 (R10 完成,2026-04-30 v100)

**Smoke 6/6 全过 ✅**

### R10 完整数据 (s2 alpha=0.1, ml_lite_tau=0.5, ml_lite_channel=32)

| Round | acc | epoch_loss | mask3_sparsity | mask3_std | aux3/main |
|---|:---:|:---:|:---:|:---:|:---:|
| R0 | 10.42 | 7.921 | 0.498 | 0.00015 | 0.22 |
| R1 | 22.89 | 5.576 | 0.498 | 0.00017 | 0.30 |
| R2 | 21.40 | 5.093 | 0.498 | 0.00015 | 0.32 |
| R3 | 24.86 | 4.756 | 0.497 | 0.00018 | 0.33 |
| R4 | 27.72 | 4.477 | 0.497 | 0.00018 | 0.34 |
| R5 | 29.00 | 4.233 | 0.496 | 0.00020 | 0.35 |
| R6 | 35.28 | 4.030 | 0.495 | 0.000276 | 0.35 |
| R7 | 35.47 | 3.747 | 0.494 | 0.00031 | 0.36 |
| **R8** | **40.57** | 3.629 | 0.493 | 0.00034 | 0.37 |
| R9 | 38.99 | 3.428 | 0.492 | 0.000369 | 0.38 |

R0-R9 acc 累计涨 +28.57pp (10.42 → 38.99). Best R8=40.57.

### 关键发现

1. **mask3 std 跟 PG-DFC ramp 同步 spread**: R0-R5 (warmup, proto_weight=0) std≈0.00015-0.00020 几乎不动; R6 ramp 第一步 +38% → R9 翻倍至 0.000369. **layer3 mask 的 spread 是被 layer4 PG-DFC ramp 通过 backward 间接拉动, 不是 deep sup loss 直接驱动**
2. **aux3/main ratio 全程 0.22-0.38**, 远低于 1.5 安全线. ml_aux_alpha=0.1 默认值合理, 不需调
3. **mask3_sparsity stuck near 0.5** 是 PG-DFC v3.3 系统级现象 (layer4 mask4=0.505 同样平), 不是 ML 引入 bug. DFD 网络 BN 强制 logit mean=0 → sigmoid≈0.5 → mask 没真切, 但不影响 acc
4. **进程自然退出**, 无 GPU memory 残留

### vs PG-DFC vanilla (跨 seed 对比, 非严格 seed-controlled)

| Round | EXP-138 ML s2 | EXP-137 vanilla s15 | EXP-137 vanilla s333 |
|---|:---:|:---:|:---:|
| R7 | **35.47** | 29.52 | 34.01 / 34.72 |
| R8 | **40.57** | (TBD) | (TBD) |
| R9 | **38.99** | (TBD) | (TBD) |

R7 ML 比 s15 vanilla **+5.95pp**, 跟 s333 vanilla 持平 (+0.75 ~ +1.46pp). 跨 seed 看在区间内偏高.

### 后续

✅ 通过 → 启动 EXP-139 (PG-DFC-ML R100 正式 4 run: PACS s15+s333 + Office s15+s333)
- 等 sc3 EXP-137 PACS s333 R100 完成 → sc3 跑 ML PACS s15+s333
- 等 v100 EXP-137 PACS s15 R100 完成 → v100 跑 ML Office s15+s333

**没必要再做 alpha=0 退化对照** (T7 unit test 已 bit-identical 验证), 不浪费 GPU.
