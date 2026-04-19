# Experiment Tracker — FedDSA-SGPA

| Run ID | Milestone | EXP | Purpose | System / Variant | Dataset | Seed | Rounds | Metric Target | Priority | Status | Notes |
|--------|-----------|-----|---------|------------------|---------|------|--------|---------------|----------|--------|-------|
| R001 | M1 | EXP-097 | C1 Primary | SGPA (use_etf=1) | Office | 2 | 200 | AVG Best ≥ 84% | MUST | TODO | 并行组 1 |
| R002 | M1 | EXP-097 | C1 Primary | SGPA (use_etf=1) | Office | 15 | 200 | 同上 | MUST | TODO | 并行组 1 |
| R003 | M1 | EXP-097 | C1 Primary | SGPA (use_etf=1) | Office | 333 | 200 | 同上 | MUST | TODO | 并行组 1 |
| R004 | M1 | EXP-100 | C2 Control | Linear (use_etf=0) | Office | 2 | 200 | AVG Best ≤ 83% | MUST | TODO | 并行组 1 |
| R005 | M1 | EXP-100 | C2 Control | Linear (use_etf=0) | Office | 15 | 200 | 同上 | MUST | TODO | 并行组 1 |
| R006 | M1 | EXP-100 | C2 Control | Linear (use_etf=0) | Office | 333 | 200 | 同上 | MUST | TODO | 并行组 1 |
| R007 | M2 | EXP-099 | C3 SGPA 推理 | SGPA infer script | Office | 2 (EXP-096 ckpt) | N/A | proto_vs_etf_gain > 0.5% | MUST | TODO | CPU 可做 |
| R008 | M4 | EXP-098 | C1 PACS | SGPA (use_etf=1) | PACS | 2 | 200 | AVG Best ≥ 82.5% | NICE | RUNNING | 08:54 deploy |
| R009 | M4 | EXP-098 | C1 PACS | SGPA (use_etf=1) | PACS | 15 | 200 | 同上 | NICE | RUNNING | 08:54 deploy |
| R010 | M4 | EXP-098 | C1 PACS | SGPA (use_etf=1) | PACS | 333 | 200 | 同上 | NICE | RUNNING | 08:54 deploy |
| R011 | M4 | EXP-098 | C2 PACS | Linear (use_etf=0) | PACS | 2 | 200 | AVG Best ≈ 80.4% | NICE | RUNNING | 08:54 deploy |
| R012 | M4 | EXP-098 | C2 PACS | Linear (use_etf=0) | PACS | 15 | 200 | 同上 | NICE | RUNNING | 08:54 deploy |
| R013 | M4 | EXP-098 | C2 PACS | Linear (use_etf=0) | PACS | 333 | 200 | 同上 | NICE | RUNNING | 08:54 deploy |

## 状态字段

- **TODO**: 等待部署
- **RUNNING**: 服务器在跑
- **DONE**: 已完成 + 数据回填
- **WAIT**: 等外部条件 (GPU / 其他实验完成)
- **KILL**: 因某原因主动终止
- **CRASH**: 意外崩溃需诊断

## 首批 6 runs 部署命令 (M1)

```bash
# seetacloud2 (GPU 0 与 SCPR v2 共享, 各 ~1.5GB)
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python

# SGPA (use_etf=1)
for s in 2 15 333; do
  EXP_DIR=../experiments/ablation/EXP-097_sgpa_office_r200
  mkdir -p $EXP_DIR/{results,logs}
  nohup $PY run_single.py --task office_caltech10_c4 --algorithm feddsa_sgpa --gpu 0 \
    --config ./config/office/feddsa_sgpa_office_r200.yml --logger PerRunLogger --seed $s \
    > $EXP_DIR/terminal_s${s}.log 2>&1 &
done

# Linear (use_etf=0)
for s in 2 15 333; do
  EXP_DIR=../experiments/ablation/EXP-100_linear_office_r200
  mkdir -p $EXP_DIR/{results,logs}
  nohup $PY run_single.py --task office_caltech10_c4 --algorithm feddsa_sgpa --gpu 0 \
    --config ./config/office/feddsa_linear_office_r200.yml --logger PerRunLogger --seed $s \
    > $EXP_DIR/terminal_s${s}.log 2>&1 &
done
```
