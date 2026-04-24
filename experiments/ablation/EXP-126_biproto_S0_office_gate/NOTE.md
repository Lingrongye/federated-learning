# EXP-126 — BiProto S0 Matched-Intervention Gate (Office + PACS)

**创建日期**: 2026-04-24
**目标**: BiProto 方案启动前置诊断, 双数据集判断 BiProto 支路是否值得完整 R200 投资.
**类型**: 兜底方案 BiProto 的 C0 pruning gate (cost-effective kill-or-go)
**预算**: 4 runs × ~1.5 GPU-h = ~6 GPU-h (Office A/B + PACS A/B 单 seed 各跑一次)
**部署服务器**: lab-lry GPU 1

## 数据集分工

| 数据集 | 当前 orth_only 战况 | S0 目标 | 判决意义 |
|---|:-:|---|---|
| Office-Caltech10 | 89.09 ❌ (输 FDSE -1.49) | BiProto-lite > Head-only +0.3pp | **kill-or-go**: BiProto 是否值得做 |
| PACS | 80.64 ✅ (赢 FDSE +0.73) | BiProto-lite ≥ Head-only -0.3pp | **保住**: BiProto 是否让 PACS 退步 |

---

## 一、变体通俗解释

> **拿个已经训好的 orth_only Office checkpoint, 把 AlexNet 主干冻死, 但 head 留着可训, 加上 BiProto 风格支路 (encoder_sty + 联邦域原型 Pd + 互斥 loss), 跑 R30 看 Office accuracy 能不能比 head-only fine-tune 涨 ≥0.3pp.**
>
> **如果涨不动 → BiProto 方向直接 kill, 改回 Calibrator 兜底或者聚合侧改造**.
> **如果涨了 → 进 S1 完整 R200 smoke**.

这个测试的关键设计 (R2 reviewer 修复点):
- **不冻 head** (R2 之前是冻死整个 inference path, 测不到 add-on 增量) — 现在 freeze encoder_sem only, semantic_head + sem_classifier 可训
- **完整 BiProto 支路加上**: encoder_sty + Pd + L_sty_proto + L_proto_excl (而不是只加部分)
- **Baseline 严格 matched**: 同 ckpt, 同 freeze scope, 但**不加** BiProto 支路 (只 head 自己 fine-tune)

---

## 二、技术细节

### 损失函数 (R30 short run, schedule 适配短跑)

```
L = L_CE + L_orth + λ_sty_proto(t) · L_sty_proto + λ_proto_excl(t) · L_proto_excl
```

Schedule:
| Round | λ_sty_proto | λ_proto_excl | MSE coef |
|:-:|:-:|:-:|:-:|
| 0-4 | 0 | 0 | 0 (warmup, EMA buffer 预热) |
| 5-9 | 0→0.5 | 0→0.3 | 0.5 (ramp-up) |
| 10-24 | 0.5 | 0.3 | 0.5 (peak) |
| 25-30 | 0.5→0 | 0.3 | 0.5 (ramp-down) |

### 关键参数
- `freeze_encoder_sem = 1` (S0 gate 关键 flag)
- `save_endpoint = 1` (保存 best-round ckpt)
- `tau_proto = 0.1` (InfoNCE temperature)
- `ema_decay = 0.9` (Pc/Pd EMA)

### 两条对照实验

#### 对照 A: BiProto-lite (本 config)
- Config: `config/office/feddsa_biproto_s0_gate.yml`
- 加载 EXP-105 orth_only Office R200 seed=2 ckpt
- Freeze `encoder.*`, 训 `semantic_head + head + encoder_sty`
- 加 L_orth + L_sty_proto + L_proto_excl 全部 loss
- 算法: `feddsa_biproto`

#### 对照 B: Head-only fine-tune (Baseline)
- Config: 沿用 `config/office/feddsa_orth_lr05_save.yml` (orth_only)
- 加载同 ckpt
- Freeze 同 scope (encoder_sem)
- 不加 BiProto 支路, 只 L_CE + L_orth
- 算法: `feddsa_scheduled` (sm=0)

---

## 三、实验配置

| 项 | 值 |
|---|---|
| Task | Office-Caltech10 (`office_caltech10_c4`) |
| Seed | 2 (单 seed, S0 是诊断不要 3-seed) |
| Rounds | 30 |
| Local epochs | 1 |
| Batch size | 50 |
| LR | 0.05 (Office optimal per EXP-052) |
| Algorithm | `feddsa_biproto` (对照 A) / `feddsa_scheduled` (对照 B baseline) |
| GPU | 1 (服务器 lab-lry) |
| 启动方式 | nohup + 后台 |
| 输出目录 | `experiments/ablation/EXP-126_biproto_S0_office_gate/` |
| Ckpt 来源 | `~/fl_checkpoints/feddsa_s2_R200_best*_*` (EXP-083 saved) |

---

## 四、判决规则

| Δ (BiProto-lite − Head-only) | 决策 |
|---|---|
| ≥ +1.0 pp | **强信号** → 进 S1 (Office seed=2 R200 smoke), 全 BiProto pipeline |
| +0.3 ~ +1.0 pp | **弱信号** → 进 S1 但降档预期 (BiProto 可能涨幅有限) |
| < +0.3 pp | **KILL BiProto** → 改投 Calibrator 兜底或聚合侧改造 (SAS τ tune / Caltech 权重) |

---

## 五、成功 / 失败标准

### 成功 (进 S1)
- 对照 A 完成 R30 不崩 (no NaN, no z_sty norm < 0.3 全程)
- 对照 A AVG > 对照 B + 0.3pp
- ckpt 保存成功 (`~/fl_checkpoints/feddsa_*_R30_best*` 存在)

### 失败 (kill)
- 对照 A AVG < 对照 B + 0.3pp
- 任一 run 崩 (NaN / loss inf / z_sty 坍缩)

---

## 六、预期结果

| 项 | 对照 B (head-only) | 对照 A (BiProto-lite) | Δ |
|---|:-:|:-:|:-:|
| AVG Best | ~88-89% | ~89-90% | +0.5~1.5 (best case) |
| Per-domain Caltech | ~76% | ~78% | +1~2 (BiProto 主要 expected gain) |

注: orth_only Office R200 EXP-083 seed=2 是 88.13%, R30 frozen-trunk fine-tune 应该接近同水平.

---

## 七、运行指令 (lab-lry)

### Step 1: 检查 EXP-083/EXP-105 orth_only ckpt
```bash
wsl bash -lc "ssh lab-lry 'ls -la ~/fl_checkpoints/ | grep -E \"feddsa.*s(2|15|333).*R200\" | tail -10'"
# 确认有: feddsa_s2_R200_best*_* (Office) 和 PACS 对应
```

### Step 2: 启动 4 个 run (greedy GPU launcher, lab-lry GPU 1)

按 CLAUDE.md 17.8 的 greedy launcher 模板, 单卡 24GB 可并行 ~3-4 runs (每 run ~4GB).
4 runs (Office A/B + PACS A/B), seed=2 单 seed.

```bash
PY=/home/lry/miniconda3/bin/python
EXP_DIR=experiments/ablation/EXP-126_biproto_S0_office_gate
mkdir -p $EXP_DIR/logs $EXP_DIR/results

# Office A (BiProto-lite, fz=1)
nohup $PY run_single.py --task office_caltech10_c4 --algorithm feddsa_biproto --gpu 0 \
    --config ./config/office/feddsa_biproto_s0_gate.yml --logger PerRunLogger --seed 2 \
    > $EXP_DIR/logs/office_A_biproto.log 2>&1 &

# Office B (Head-only baseline, fz=1, lp=0/le=0)
nohup $PY run_single.py --task office_caltech10_c4 --algorithm feddsa_biproto --gpu 0 \
    --config ./config/office/feddsa_biproto_s0_gate_baseline.yml --logger PerRunLogger --seed 2 \
    > $EXP_DIR/logs/office_B_baseline.log 2>&1 &

# PACS A (BiProto-lite)
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_biproto --gpu 0 \
    --config ./config/pacs/feddsa_biproto_s0_gate.yml --logger PerRunLogger --seed 2 \
    > $EXP_DIR/logs/pacs_A_biproto.log 2>&1 &

# PACS B (Head-only baseline)
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_biproto --gpu 0 \
    --config ./config/pacs/feddsa_biproto_s0_gate_baseline.yml --logger PerRunLogger --seed 2 \
    > $EXP_DIR/logs/pacs_B_baseline.log 2>&1 &

wait
```

### Step 3: 烟测 (5-round 集成测试, 验证完整联邦流程)

为避免直接跑 R30 后才发现 bug, 先跑 R5 烟测确认 pipeline 完整可跑:

```bash
# 临时改 num_rounds: 30 → 5 在 config 顶部加 -c num_rounds=5 (run_single 不支持, 复制一份 config)
cp ./config/office/feddsa_biproto_s0_gate.yml ./config/office/_smoke_5round.yml
sed -i 's/num_rounds: 30/num_rounds: 5/' ./config/office/_smoke_5round.yml

$PY run_single.py --task office_caltech10_c4 --algorithm feddsa_biproto --gpu 0 \
    --config ./config/office/_smoke_5round.yml --logger PerRunLogger --seed 2 \
    > $EXP_DIR/logs/smoke_5round.log 2>&1

# 检查日志
tail -30 $EXP_DIR/logs/smoke_5round.log
# 期望: 5 round 结束, 没有 NaN/inf, AVG 合理
```

### Step 4: 收集 results
```bash
$PY collect_results.py --exp EXP-126_biproto_S0_office_gate \
    --task office_caltech10_c4 --algorithm feddsa_biproto --seed 2
$PY collect_results.py --exp EXP-126_biproto_S0_office_gate \
    --task PACS_c4 --algorithm feddsa_biproto --seed 2
```

---

## 八、可视化诊断 (S0 通过后跑)

S0 gate 输出 best ckpt 后, 跑 3 套诊断脚本验证 "解耦真的做成了":

```bash
# Vis-A: t-SNE 双面板
python scripts/visualize_tsne.py \
    --ckpt ~/fl_checkpoints/feddsa_biproto_s2_*  \
    --task office_caltech10_c4 \
    --out experiments/ablation/EXP-126_biproto_S0_office_gate/figs \
    --label biproto_s0_office_s2

# Vis-B: Probe ladder
python scripts/run_capacity_probes.py \
    --ckpt ~/fl_checkpoints/feddsa_biproto_s2_* \
    --task office_caltech10_c4 \
    --config ./config/office/feddsa_biproto_s0_gate.yml \
    --output experiments/ablation/EXP-126_biproto_S0_office_gate/probe.json \
    --gpu 0

# Vis-C: Decouple weights + prototype quality (新写)
# TODO: 加新脚本 visualize_biproto_protos.py (Pc/Pd separation, Pd⊥bc trajectory)
```

---

## 九、TODO (S0 启动前)

- [x] 实现 `algorithm/feddsa_biproto.py` (~330 lines)
- [x] 写 config `config/office/feddsa_biproto_s0_gate.yml`
- [x] 写 config `config/office/feddsa_biproto_office_r200.yml`
- [x] 写 config `config/pacs/feddsa_biproto_pacs_r200.yml`
- [x] AST + model 创建 + forward 本地 smoke
- [x] ckpt 加载兼容性测试 (orth_only ckpt → BiProto strict=False)
- [ ] 写 `feddsa_biproto_s0_gate_baseline.yml` (对照 B, lp=0/le=0)
- [ ] codex review
- [ ] commit + push
- [ ] lab-lry pull
- [ ] 检查 EXP-083 / EXP-105 ckpt 在 lab-lry 可用
- [ ] 启动对照 A + 对照 B
- [ ] 回填本 NOTE.md 结果

---

## 十、结果回填 (运行后)

待跑.
