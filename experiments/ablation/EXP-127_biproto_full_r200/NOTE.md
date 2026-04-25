# EXP-127 — BiProto 完整 R200 Pipeline (3-seed × Office + PACS)

**创建日期**: 2026-04-24
**目标**: 覆盖 EXP-126 S0 gate 的 kill 结论, 跑完整 R200 from-scratch 训练看 unfrozen encoder 下 BiProto 是否有 signal.
**类型**: 直接走 S1+S2+S3 全流程 (绕过 S0 gate)
**部署**: lab-lry GPU 1
**预算**: 6 runs × 2-7h = ~30 GPU-h wall (6 并发)

---

## 一、为什么绕过 S0 Gate

EXP-126 S0 Gate 显示 Δ = 0 (frozen encoder + head-only), 但:
- S0 冻死主干, 不测 encoder 和 encoder_sty 协同端到端学习动态
- S0 schedule 压缩到 R30, 没覆盖标准 50/80/150/200 Bell schedule
- R4 reviewer 原话: "Negative C0 sufficient to kill" 但没覆盖"from-scratch 全程"场景
- 用户决定: 直接跑完整训练, 用实际 R200 结果做最终判决

---

## 二、实验配置

| 项 | 值 |
|---|---|
| Task | Office-Caltech10 × 3 seeds + PACS × 3 seeds = 6 runs |
| Seeds | 2, 15, 333 (对齐 FDSE baseline) |
| Rounds | 200 |
| Office config | `config/office/feddsa_biproto_office_r200.yml` |
| PACS config | `config/pacs/feddsa_biproto_pacs_r200.yml` |
| Office LR | 0.05 (E=1) |
| PACS LR | 0.1 (E=5) |
| BiProto schedule | warmup R0-50 / ramp R50-80 / peak R80-150 / ramp_down R150-200 |
| lp (lambda_sty_proto peak) | 0.5 |
| le (lambda_proto_excl peak) | 0.3 |
| mc (MSE coef) | 0.5 |
| Save best | se=1 (每 seed 存 best round ckpt) |
| Freeze encoder_sem | fz=0 (★ 完整训练, 不冻 encoder) |
| GPU | lab-lry GPU 1, greedy dispatch |

---

## 三、判决标准 (3-seed mean)

| 数据集 | baseline | BiProto 目标 | 判决 |
|---|:-:|---|:-:|
| Office-Caltech10 | FDSE 90.58 / orth_only 89.09 | 3-seed mean AVG Best ≥ 91.08 | 胜 |
| PACS | FDSE 79.91 / orth_only 80.64 | 3-seed mean AVG Best ≥ 80.91 | 保住 |

**同时成立** → BiProto 真正赢
**任一不满足** → 最终 kill, 回 Calibrator 兜底

---

## 四、可视化诊断 (跑完后)

所有 6 runs 结束后, 对 seed=2 ckpt 跑 3 套诊断:

### Vis-A: t-SNE 双面板
```bash
python FDSE_CVPR25/scripts/visualize_tsne.py \
    --ckpt ~/fl_checkpoints/feddsa_s2_R200_best*_<ts>/ \
    --task office_caltech10_c4 \
    --out experiments/ablation/EXP-127_biproto_full_r200/figs \
    --label biproto_office_s2
```
但现有 `visualize_tsne.py` 写死了 `feddsa_sgpa` 或 `feddsa_sgpa_vib`, 需要改支持 `feddsa_biproto` (TODO)

### Vis-B: Probe ladder
```bash
python FDSE_CVPR25/scripts/run_capacity_probes.py ...
```
同样需要改支持 `feddsa_biproto` (TODO)

### Vis-C: Weights 解耦诊断
```bash
python FDSE_CVPR25/scripts/visualize_decouple.py \
    --ckpt ~/fl_checkpoints/.../global_model.pt \
    --out experiments/ablation/EXP-127_biproto_full_r200/figs
```
这个直接从 state_dict 读 `semantic_head.0.weight` 和 `style_head.0.weight`, 但 BiProto 没有 `style_head`, 改成 `encoder_sty.net.0.weight` 再对比 (TODO 或直接改)

---

## 五、结果回填 (运行后)

待填.
