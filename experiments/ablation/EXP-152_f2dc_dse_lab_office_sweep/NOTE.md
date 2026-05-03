# EXP-152: F2DC + DSE_Rescue3 + LAB v4.2 Office sweep — 救 webcam 多 path 探索

**date**: 2026-05-03 启动
**status**: 6 runs R0/100 起步 (sub1 ×3 + sub2 ×3)
**exp_id**: EXP-152
**关联**: EXP-151 winner s=15 rho=0.3 webcam 60.34% vs DaA 67.24% **输 -6.9pp**, 诊断显示真凶是 DSE 信号扰动 webcam 干净特征空间 (LAB 权重已 18.58% 是 sample_share 2 倍, 不是权重不足)

---

## 问题诊断 (来自 EXP-151 best_R086.npz LAB 字段)

EXP-151 R86 winner per-domain 权重:

| 域 | sample_share | LAB ratio | w_proj | small_protect | val_loss_ema | q (LAB boost) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| caltech | 0.5306 | 0.80 | 0.4245 | No | 3.20 | 0.476 |
| amazon | 0.3024 | 0.80 | 0.2419 | No | 1.42 | 0.000 |
| webcam | 0.0929 | **2.00** | **0.1858** ⭐ | **Yes** | 1.62 | **0.000 ❗** |
| dslr | 0.0741 | 2.00 | 0.1482 | Yes | 3.29 | 0.524 |

**真凶**: DSE-only s=15 webcam 56.90% (DSE 单跑就压 webcam -10pp), EXP-151 LAB 救回 +3.45pp 但仍输 DaA -6.9pp。webcam 域差异低 ("acquisition-level"), DSE adapter 注入扰动反而打乱干净特征。

## 6 runs 配置 (3 path × 2 seed)

| host | seed | rho | LAB mode | tag | 假设验证 |
|---|:--:|:--:|:--:|:--:|:--:|
| sub1 | 15 | **0.2** | v2c (rmin=2.0/rmax=4.0) | rho02_v2c | DSE 弱化 → webcam 救回 65+ |
| sub1 | 15 | **0.1** | v2c | rho01_v2c | DSE 更弱 → 看 dslr 是否丢 |
| sub1 | 15 | 0.3 | **standard** (rmin=0.8/rmax=2.0) | rho03_std | 关 small_protect → LAB raw 公式 |
| sub2 | 333 | 0.2 | v2c | rho02_v2c | 同上 |
| sub2 | 333 | 0.1 | v2c | rho01_v2c | 同上 |
| sub2 | 333 | 0.3 | standard | rho03_std | 同上 |

## 启动命令模板

```bash
# Path A/B (rho={0.2, 0.1} v2c):
$PY -u main_run.py \
  --model f2dc_dse_lab --dataset fl_officecaltech --seed $SEED \
  --communication_epoch 100 --local_epoch 10 --use_daa false \
  --dse_rho_max $RHO --dse_lambda_cc 0.1 --dse_lambda_mag 0.01 \
  --lab_lambda 0.15 --lab_projection_mode office_small_protect \
  --lab_small_share_threshold 0.125 --lab_small_ratio_min 2.0 --lab_small_ratio_max 4.0 \
  --num_classes 10 \
  --dump_diag $EXP/diag_office_s${SEED}_rho0${RHO_TAG}_v2c \
  --dump_warmup 0 --dump_min_gain 0.01 --dump_min_interval 1

# Path C (rho=0.3 standard):
$PY -u main_run.py \
  --model f2dc_dse_lab --dataset fl_officecaltech --seed $SEED \
  --communication_epoch 100 --local_epoch 10 --use_daa false \
  --dse_rho_max 0.3 --dse_lambda_cc 0.1 --dse_lambda_mag 0.01 \
  --lab_lambda 0.15 --lab_projection_mode standard \
  --lab_ratio_min 0.80 --lab_ratio_max 2.00 \
  --num_classes 10 \
  --dump_diag $EXP/diag_office_s${SEED}_rho03_std \
  --dump_warmup 0 --dump_min_gain 0.01 --dump_min_interval 1
```

## 期望 / 决策树

| 假设结果 | 结论 |
|---|---|
| **Path A rho=0.2 webcam ≥ 64** | DSE 弱化救 webcam ✅, 新 winner = rho=0.2 (替代 EXP-151 rho=0.3) |
| **Path B rho=0.1 webcam ≥ 65 但 dslr ≤ 53** | DSE 拉太弱 dslr 丢, rho=0.2 才是 sweet |
| Path A/B 都救不回 webcam | DSE+LAB 在 Office 本质问题, paper 接受 trade-off |
| **Path C standard webcam 反而高** | small_protect 设计错了, standard mode 才对 — 完全推翻 EXP-144 P4 finding |
| Path C standard 跟 EXP-151 持平 | small_protect 跟 standard 相对 standard 优势小 (除非 dslr 帮大) |

## 当前进度 (待回填)

| host | seed | rho | LAB mode | round | best | best dumps |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| sub1 | 15 | 0.2 | v2c | R?/100 | — | — |
| sub1 | 15 | 0.1 | v2c | R?/100 | — | — |
| sub1 | 15 | 0.3 | std | R?/100 | — | — |
| sub2 | 333 | 0.2 | v2c | R?/100 | — | — |
| sub2 | 333 | 0.1 | v2c | R?/100 | — | — |
| sub2 | 333 | 0.3 | std | R?/100 | — | — |

## 数据保存 (CLAUDE.md 零零零规则)

- 6 个独立 dump_diag 路径
- per-round npz 含 169-181 proto_diag 字段 (DSE 22 + LAB 147-159)
- best dump 触发宽松 (`gain=0.01/interval=1`) → 预计 100 round 触发 15-25 次 dump

## 预计完成

Office R100 单 run ~4-5h (EXP-151 经验). 6 并发 sub1 + sub2 各 3 跑, 估 **~6h R100 全完** (00:00 启动 → ~05:00-06:00 完成)。