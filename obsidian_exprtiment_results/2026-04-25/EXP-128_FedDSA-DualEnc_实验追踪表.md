# FedDSA-DualEnc 实验追踪表

**Date**: 2026-04-25
**Plan**: refine-logs/2026-04-25_FedDSA-DualEnc/EXPERIMENT_PLAN.md
**Total runs**: 33 must-run + 12 nice-to-have = 45

---

## Stage M0: Sanity Check (单 GPU,~30min)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | 代码语法 + AST 解析 | feddsa_dualenc 全模块 | — | parse OK | MUST | TODO | python -c "import ast; ast.parse(...)" |
| R002 | M0 | 单元测试:E_sem/E_sty/Decoder 形状 | DualEncAlexNet | — | shape 正确 | MUST | TODO | 写 test_dualenc.py |
| R003 | M0 | 单元测试:4 loss 都有非零梯度 | full method | dummy batch | grad_norm > 0 | MUST | TODO | 单批前向+反向 |
| R004 | M0 | 单元测试:cycle GT 真的 detached | full method | — | z_sem.requires_grad=True, z_sem.detach().requires_grad=False | MUST | TODO | 防自循环 bug |
| R005 | M0 | 5 round Office sanity | full method | Office_c4, seed=2, R=5 | acc > 0, recon non-trivial, viz dump 输出 | MUST | TODO | seetacloud GPU |
| R006 | M0 | Codex code review | feddsa_dualenc.py + dualenc_alexnet.py | — | 0 critical/important issue | MUST | TODO | CLAUDE.md §17.4 第 2 步 |

---

## Stage M1: Baseline 复用 (无新 run)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R007 | M1 | 复用 orth_only 3-seed PACS (EXP-074?) | orth_only | PACS_c4 × 3 seeds | AVG Best 80.64 | MUST | DONE | 已有,见 EXP-074 |
| R008 | M1 | 复用 orth_only 3-seed Office | orth_only | Office_c4 × 3 seeds | AVG Best 89.09 | MUST | DONE | 已有 |
| R009 | M1 | 复用 FDSE 3-seed PACS | FDSE | PACS_c4 × 3 seeds | AVG Best 79.91 | MUST | DONE | 已有 |
| R010 | M1 | 复用 FDSE 3-seed Office | FDSE | Office_c4 × 3 seeds | AVG Best 90.58 | MUST | DONE | 已有 |
| R011 | M1 | 复用 BiProto 3-seed PACS | BiProto | PACS_c4 × 3 seeds | AVG Best 79.X | MUST | DONE | 已有,作为反例 |
| R012 | M1 | 复用 BiProto 3-seed Office | BiProto | Office_c4 × 3 seeds | AVG Best 87.X | MUST | DONE | 已有,作为反例 |
| R013 | M1 | 复用 FedAvg/BN/Proto/FPL 3-seed | 4 baselines | PACS+Office × 3 seeds | — | MUST | DONE | 已有 |

---

## Stage M2-M3: Pilot (~13 GPU-h)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R014 | M2 | DualEnc Office pilot seed=2 | Full method | Office_c4, seed=2, R=200 | AVG Best > 88 (gate 1) | MUST | TODO | 看是否能涨 90.58 |
| R015 | M3 | DualEnc PACS pilot seed=2 | Full method | PACS_c4, seed=2, R=200 | AVG Best > 78 (gate 1) | MUST | TODO | 看是否破坏 orth_only 现状 |

**★ Decision Gate 1**: R014 AND R015 都通过 → 进入 M4;任一不通过 → debug 或 kill

---

## Stage M4: Block 1 主表 3-seed (~50 GPU-h)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R016 | M4 | DualEnc PACS seed=15 | Full method | PACS_c4, seed=15, R=200 | per-domain B/L | MUST | TODO | |
| R017 | M4 | DualEnc PACS seed=333 | Full method | PACS_c4, seed=333, R=200 | per-domain B/L | MUST | TODO | |
| R018 | M4 | DualEnc Office seed=15 | Full method | Office_c4, seed=15, R=200 | per-domain B/L | MUST | TODO | |
| R019 | M4 | DualEnc Office seed=333 | Full method | Office_c4, seed=333, R=200 | per-domain B/L | MUST | TODO | |
| R020 | M4 | 收集 3-seed 主表 | (R014+R016+R017) PACS / (R015+R018+R019) Office | — | 3-seed mean AVG B/L | MUST | TODO | python collect_results.py |

**★ Decision Gate 2**:
- PACS 3-seed mean AVG Best > 79.91 ✅
- Office 3-seed mean AVG Best > 90.58 ✅

通过 → 进入 ablation;不通过 → 重新评估方向(参照 Failure interpretation)

---

## Stage M5: Block 2 - 4-Loss Ablation (~40 GPU-h)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R021 | M5 | -L_saac (no cycle) | Full method 关闭 L_saac | Office_c4, seed=2 | acc + P1 z_sty class probe | MUST | TODO | 期望 P1 飙到 > 50% |
| R022 | M5 | -L_rec (no recon) | Full method 关闭 L_rec | Office_c4, seed=2 | acc + recon PSNR | MUST | TODO | 期望 PSNR ↓ |
| R023 | M5 | -L_dsct (no contrastive) | Full method 关闭 L_dsct | Office_c4, seed=2 | acc + z_sty SVD ER | MUST | TODO | 期望 ER ↓ |
| R024 | M5 | -L_saac -L_rec | 仅留 CE+dsct | Office_c4, seed=2 | acc | MUST | TODO | 等价 orth_only + VAE |
| R025 | M5 | Decoder-only (CE+rec) | 关闭 saac+dsct | Office_c4, seed=2 | acc | MUST | TODO | 排除 A1 |
| R026 | M5 | + L_orth + HSIC (BiProto-style) | Full + 显式正交 | Office_c4, seed=2 | acc + probe | MUST | TODO | 看是否互补 |

(R027 = orth_only baseline 已在 R008 复用)

---

## Stage M6: Block 3 + 4 - Simplicity + Cross-Client (~35 GPU-h)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R027 | M6 | + R-Drop on z_sem | Full + R-Drop 一致性 | Office_c4, seed=2 | acc | MUST | TODO | Block 3 simplicity |
| R028 | M6 | + Image CutMix | Full + CutMix p=0.5 | Office_c4, seed=2 | acc | MUST | TODO | Block 3 simplicity |
| R029 | M6 | No swap (z_sty_swap=z_sty) | Full method 退化 | Office_c4, seed=2 | acc | MUST | TODO | Block 4 cross-client |
| R030 | M6 | Intra-client swap | Full,只在 client 内 swap | Office_c4, seed=2 | acc | MUST | TODO | Block 4 cross-client |
| R031 | M6 | Cross-client mean | Full,只用别 client (μ,σ) 均值 | Office_c4, seed=2 | acc | MUST | TODO | Block 4,FISC 对照 |

---

## Stage M7: Block 5 - R400 Stability (~30 GPU-h)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R032 | M7 | DualEnc R400 长跑 | Full method | Office_c4, seed=2, R=400 | z_sty SVD ER 全程 vs round | MUST | TODO | 必须 ER > 10 全程 |
| R033 | M7 | BiProto R400 反例 | BiProto | Office_c4, seed=2, R=400 | z_sty SVD ER 全程 | MUST | TODO | 已知 ER ~2.73 @ R200 |

---

## Stage M8: Block 6 + 7 - Visualization + Probes (无新 run, ~5 GPU-h 离线分析)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R034 | M8 | dump 4×4 swap grid (PACS) | best ckpt from R016/R017 | PACS val | qualitative | MUST | TODO | dump_style_swap_grid.py |
| R035 | M8 | dump 4×4 swap grid (Office) | best ckpt from R018/R019 | Office val | qualitative | MUST | TODO | |
| R036 | M8 | dump cycle 验证图 | best ckpt | PACS+Office val | PSNR (orig vs cycle) | MUST | TODO | 必须 cycle PSNR > 18 |
| R037 | M8 | 4-probe @ R200 (Full) | best ckpt | PACS+Office val | P1-P4 数字 | MUST | TODO | run_decouple_probes.py |
| R038 | M8 | 4-probe @ R200 (orth_only) | orth_only ckpt | PACS+Office val | P1-P4 数字 | MUST | TODO | 对照 |
| R039 | M8 | 4-probe @ R200 (BiProto) | BiProto ckpt | PACS+Office val | P1-P4 数字 | MUST | TODO | 反例对照 |

---

## Stage M9: Block 8 + 9 - Polish (~60 GPU-h, NICE-TO-HAVE)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R040 | M9 | AdaIN 浅层注入 | Full,只在 dec_block1 注入 | Office_c4, seed=2 | acc | NICE | TODO | Block 8 |
| R041 | M9 | AdaIN 中层注入 | Full,只在 dec_block2 注入 | Office_c4, seed=2 | acc | NICE | TODO | Block 8 |
| R042 | M9 | AdaIN 深层注入 | Full,只在 dec_block3 注入 | Office_c4, seed=2 | acc | NICE | TODO | Block 8 |
| R043-R045 | M9 | λ_saac grid {0.5, 1.0, 2.0} | Full | Office_c4, seed=2 | acc | NICE | TODO | Block 9 |
| R046-R048 | M9 | λ_rec grid {0.0005, 0.001, 0.005} | Full | Office_c4, seed=2 | acc | NICE | TODO | Block 9 |

---

## 状态字段说明

- **TODO**: 未启动
- **READY**: 配置已准备,等 GPU
- **RUNNING**: 实验进行中(ID 后括号写 server name + GPU id)
- **DONE**: 完成,结果已收集
- **DEAD**: 实验失败,需要 debug

## 进度统计

| Stage | Total | TODO | READY | RUNNING | DONE | DEAD |
|-------|------:|-----:|------:|--------:|-----:|-----:|
| M0 | 6 | 6 | 0 | 0 | 0 | 0 |
| M1 | 7 | 0 | 0 | 0 | 7 | 0 |
| M2-M3 | 2 | 2 | 0 | 0 | 0 | 0 |
| M4 | 5 | 5 | 0 | 0 | 0 | 0 |
| M5 | 6 | 6 | 0 | 0 | 0 | 0 |
| M6 | 5 | 5 | 0 | 0 | 0 | 0 |
| M7 | 2 | 2 | 0 | 0 | 0 | 0 |
| M8 | 6 | 6 | 0 | 0 | 0 | 0 |
| M9 (NICE) | 9 | 9 | 0 | 0 | 0 | 0 |
| **Total** | **48** | **41** | **0** | **0** | **7** | **0** |

---

## 启动命令模板 (CLAUDE.md §17.4.2)

```bash
# 标准启动
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
EXP_DIR=../../experiments/ablation/EXP-128_dualenc_main

mkdir -p $EXP_DIR/results $EXP_DIR/logs

nohup $PY run_single.py --task Office_c4 --algorithm feddsa_dualenc --gpu 0 \
  --config ./config/office/feddsa_dualenc.yml --logger PerRunLogger --seed 2 \
  > $EXP_DIR/terminal_office_s2.log 2>&1 &
```

```bash
# Greedy launcher (CLAUDE.md §17.8 多 run 并行)
TASKS=(
    "office_s2|Office_c4|feddsa_dualenc|config/office/feddsa_dualenc.yml|2"
    "office_s15|Office_c4|feddsa_dualenc|config/office/feddsa_dualenc.yml|15"
    "office_s333|Office_c4|feddsa_dualenc|config/office/feddsa_dualenc.yml|333"
    "pacs_s2|PACS_c4|feddsa_dualenc|config/pacs/feddsa_dualenc.yml|2"
    "pacs_s15|PACS_c4|feddsa_dualenc|config/pacs/feddsa_dualenc.yml|15"
    "pacs_s333|PACS_c4|feddsa_dualenc|config/pacs/feddsa_dualenc.yml|333"
)
MIN_FREE_MB=5000  # decoder + cycle 比 orth_only 重,留余量
for task in "${TASKS[@]}"; do
    IFS="|" read -r label t algo config seed <<< "$task"
    while true; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "$free_mb" -ge $MIN_FREE_MB ]; then
            echo "[$(date +%H:%M:%S)] LAUNCH $label (free=${free_mb}MB)"
            CUDA_VISIBLE_DEVICES=0 $PY run_single.py \
                --task $t --algorithm $algo --gpu 0 \
                --config ./$config --logger PerRunLogger --seed $seed \
                > $EXP_DIR/logs/${label}.log 2>&1 &
            sleep 25
            break
        fi
        sleep 45
    done
done
wait
```

```bash
# 收集结果 (CLAUDE.md §17.4.1)
python collect_results.py --exp EXP-128 --task Office_c4 --algorithm feddsa_dualenc --seed 2
python collect_results.py --exp EXP-128 --task Office_c4 --algorithm feddsa_dualenc --seed 15
python collect_results.py --exp EXP-128 --task Office_c4 --algorithm feddsa_dualenc --seed 333
```

---

## 提取与回填 NOTE.md 的字段

每个 Stage 完成后,必须回填 NOTE.md:

- **AVG Best / Last** (3-seed mean ± std)
- **ALL Best / Last** (3-seed mean)
- **Per-domain Best/Last** (检查是否有 outlier)
- **z_sty SVD ER** @ R50/R100/R150/R200
- **Probe P1-P4** @ R50/R100/R150/R200
- **训练时长** (min)
- **是否过 baseline 阈值** (PACS 79.91 / Office 90.58)
- **结论一句话** (有效 / 无效 / 部分有效)
