# EXP-139 PG-DFC-ML 正式 R100 (4 run)

**日期**: 2026-04-30
**前置**: EXP-138 R10 smoke (s2 alpha=0.1) 验收通过 (mask3 不坍塌, aux3/main 比合理)

**目的**: 跟 EXP-137 PG-DFC vanilla R100 同 seed 严格对比 PG-DFC-ML 是否有 acc 提升

## 4 个 Run

| label | dataset | seed | algo | ml_aux_alpha | server | 启动条件 |
|---|---|---|---|---|---|---|
| `pgml_pacs_s15_R100` | PACS | 15 | f2dc_pg_ml | 0.1 | sc3 | 等 sc3 EXP-137 跑完 |
| `pgml_pacs_s333_R100` | PACS | 333 | f2dc_pg_ml | 0.1 | sc3 | 等 sc3 EXP-137 跑完 |
| `pgml_office_s15_R100` | OfficeCaltech | 15 | f2dc_pg_ml | 0.1 | v100 | 等 v100 EXP-137 跑完 |
| `pgml_office_s333_R100` | OfficeCaltech | 333 | f2dc_pg_ml | 0.1 | v100 | 等 v100 EXP-137 跑完 |

## 跟 EXP-137 PG-DFC vanilla 的对比设计

| dataset | seed | EXP-137 PG-DFC vanilla R100 | EXP-139 PG-DFC-ML R100 (待跑) | Δ 期望 |
|---|:---:|:---:|:---:|:---:|
| PACS | 15 | (待 v100 EXP-137 完成) | TBD | ML 不掉点即可 |
| PACS | 333 | (待 sc3 EXP-137 完成) | TBD | 同上 |
| Office | 15 | 54.15% (R100 last,sc3 EXP-137) | TBD | ≥ 54.15 |
| Office | 333 | 55.60% (R100 last,sc3 EXP-137) | TBD | ≥ 55.60 |

## 启动命令模板

```bash
# 通用模板 (用 python -u unbuffered, smoke 验证过此方式可看实时诊断)
PY=/root/miniconda3/bin/python  # sc3 (autodl)
# PY=python                      # v100

EXP_DIR=/root/autodl-tmp/federated-learning/experiments/ablation/EXP-139_pgml_main
mkdir -p $EXP_DIR/logs

# PACS s15 (sc3, R100, ml_aux_alpha=0.1)
setsid $PY -u main_run.py \
  --model f2dc_pg_ml --dataset fl_pacs --seed 15 \
  --communication_epoch 100 \
  --ml_aux_alpha 0.1 --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
  --use_daa False --device_id 0 \
  > $EXP_DIR/logs/pgml_pacs_s15_R100.log 2>&1 < /dev/null & disown

# PACS s333 (sc3) — 同上换 seed 333

# Office s15 (v100) — 同上换 dataset fl_officecaltech + seed 15
setsid python -u main_run.py \
  --model f2dc_pg_ml --dataset fl_officecaltech --seed 15 \
  --communication_epoch 100 \
  --ml_aux_alpha 0.1 --pg_proto_weight 0.3 --pg_warmup_rounds 30 --pg_ramp_rounds 20 \
  --use_daa False --device_id 0 \
  > /workspace/federated-learning/experiments/ablation/EXP-139_pgml_main/logs/pgml_office_s15_R100.log 2>&1 < /dev/null & disown
```

## 资源规划

- **sc3** (4090D 24GB): EXP-137 PACS s333 ~R55 在跑,~2.5-3h 完成 → 释放后跑 ML PACS s15 + s333
  - PACS R100 需 ~5GB/run,sc3 24GB 能并跑 2 个
  - 单 run 估 4-6h,**两个并跑约 5-7h wall**
- **v100** (32GB): EXP-137 PACS s15 ~R37 在跑,~6h+ 完成 → 释放后跑 ML Office s15 + s333
  - Office R100 input 32×32,GPU 占用 ~4GB/run,能并跑 2 个
  - 单 run 估 2-3h(Office 数据小),**两个并跑约 2-3h wall**

## 验收标准

| 指标 | 要求 |
|---|---|
| **AVG Best mean acc (3 seed,但本实验只 s15+s333 两 seed)** | PACS ≥ EXP-137 PG-DFC vanilla 同 seed mean |
| **AVG Last mean acc** | PACS ≥ EXP-137 PG-DFC vanilla 同 seed mean |
| Office Best/Last | 同上 (vs EXP-137 同 seed) |
| mask3_sparsity_mean | 0.3-0.7 全程稳定 |
| aux3/main ratio | < 1.5 全程 |
| 训练 loss 单调下降 | yes |

## 配套诊断 (smoke 期间已写好)

每 round 末 print `[ML diag] {...}` 行,包含:
- mask3_sparsity_mean / mask3_sparsity_std (layer3 mask 健康度)
- aux3_loss_mean / main_loss_mean / aux3_over_main_ratio (deep sup 弱辅助是否成立)
- mask_sparsity_mean / attn_entropy_mean / proto_signal_ratio_mean (PG-DFC v3.3 原诊断)

## 当前依赖

- ⏳ sc3 EXP-137 PACS s333 跑完 (R55 → R100, 2.5-3h)
- ⏳ v100 EXP-137 PACS s15 跑完 (R37 → R100, 6h+)
- ✅ EXP-138 smoke s2 alpha=0.1 正在 R7+,等到 R10 完成确认 mask 行为后才启动 EXP-139

## 结论 (待回填)

(R100 完成后填)
