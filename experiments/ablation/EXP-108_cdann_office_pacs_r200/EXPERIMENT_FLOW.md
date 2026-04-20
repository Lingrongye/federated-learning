# EXP-108 FedDSA-CDANN 完整实验流程文档

**实验**: EXP-108 — Constrained Dual-Directional DANN for Federated Domain Generalization
**日期**: 2026-04-20
**方案来源**: 5 轮 Codex gpt-5.4 xhigh research-refine, 最终 8.75/10 proposal-complete (见 `refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md`)
**诊断脚本**: ✅ 已加 (训练端 `diag=1` + 训练完 `run_frozen_probes.py`)

---

## 1. 方案 (One Sentence)

**"Shared non-adversarial domain discriminator + asymmetric encoder-gradient supervision (GRL on z_sem path only) is the minimal repair for whitening-induced style collapse in client=domain FedDG when style carries class signal."**

---

## 2. 流程总览 (8 大阶段)

```
Phase 0: 背景问题 (Office +6.20pp / PACS -1.49pp, z_sty_norm 塌 95%)
  ↓
Phase 1: Landscape 文献调研 (30 篇 2024-2026)
  ↓
Phase 2: Idea brainstorm (12 候选 → 3 top)
  ↓
Phase 3: Novelty check (对比 30 篇, 零直接 prior)
  ↓
Phase 4: Research-refine (5 轮 Codex 精炼, 7.1 → 8.75)
  ↓
Phase 5: 代码实现 (本阶段) ← **CURRENT**
  ↓
Phase 6: 单测 + Codex code review
  ↓
Phase 7: 部署训练 (6 runs) + 诊断监控
  ↓
Phase 8: frozen probe + 回填 NOTE + Obsidian summary
```

**当前位置**: Phase 5-6 已完成, 准备进 Phase 7.

---

## 3. 代码实现详情 (Phase 5, ✅ 完成)

### 3.1 改动的文件

| 文件 | 改动 | 作用 |
|------|------|------|
| `FDSE_CVPR25/algorithm/feddsa_sgpa.py` | **+120 行** (主文件) | CDANN 机制实现 |
| `FDSE_CVPR25/tests/test_feddsa_sgpa.py` | **+14 tests** | 单测覆盖 (55/55 全绿) |
| `FDSE_CVPR25/scripts/run_frozen_probes.py` | **新文件** | 训练后 frozen probe 评估 |
| `FDSE_CVPR25/config/office/feddsa_cdann_office_r200.yml` | **新文件** | Office 实验配置 |
| `FDSE_CVPR25/config/pacs/feddsa_cdann_pacs_r200.yml` | **新文件** | PACS 实验配置 |
| `experiments/ablation/EXP-108_cdann_office_pacs_r200/NOTE.md` | **新文件** | 实验记录 (按 EXP-100 模板) |

### 3.2 核心组件

#### A. Gradient Reversal Layer (新增)

```python
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)                # 恒等, 不改变 forward 值
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None  # 梯度反向乘 -lam
```

#### B. 共享 dom_head (non-adversarial)

```python
# FedDSASGPAModel 里, ca=1 时添加
self.dom_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(0.1),
    nn.Linear(64, num_clients),   # 输出 N_clients 维, 每位 = 一个域
)  # 约 9K 参数
```

**关键**: dom_head **自身不做对抗**, 两路都 minimize standard CE.

#### C. λ_adv 三段 schedule

```python
CDANN_WARMUP_START = 20   # R < 20: L_dom_* 完全禁用
CDANN_WARMUP_END = 40     # R >= 40: λ_adv = 1.0

def compute_lambda_adv(r):
    if r < 20: return 0.0          # warmup, CDANN 关
    if r < 40: return (r-20)/20    # linear ramp
    return 1.0
```

#### D. Client.train 里的 CDANN loss (带 warmup gate)

```python
cdann_active = (ca_flag == 1 
                and model.dom_head is not None
                and self.current_round >= CDANN_WARMUP_START)  # warmup gate
if cdann_active:
    lambda_adv = compute_lambda_adv(self.current_round)
    d_label = torch.full((B,), client.id, dtype=torch.long)
    # z_sem 路径: GRL 反转 encoder 端梯度
    loss_dom_sem = F.cross_entropy(
        model.dom_head(grl(z_sem, lambda_adv)), d_label)
    # z_sty 路径: 正向 encoder 梯度
    loss_dom_sty = F.cross_entropy(
        model.dom_head(z_sty), d_label)
    loss = loss + loss_dom_sem + loss_dom_sty
```

### 3.3 Codex Code Review 修复 (见 `refine-logs/.../codex-code-review-result.md`)

| Level | Issue | Fix |
|-------|-------|-----|
| CRITICAL | warmup 不是真 baseline-equivalent (R<20 时 L_dom_sty 仍更新) | 加 `cdann_active` gate, R<20 完全跳过 L_dom_* |
| IMPORTANT | `_TASK_NUM_CLIENTS` heuristic vs runtime 不匹配 | Server.initialize 动态 rebuild dom_head 到 `len(clients)` |
| IMPORTANT | `ca=1 + ue=1` 无 enforcement | Server.initialize raise ValueError |
| MINOR | `c.num_clients_total` dead state | 删除 |

### 3.4 单测结果

```
55 passed in 10.87s
(41 原有 + 14 新 CDANN tests)
```

测试覆盖:
- GRL forward/backward 正确性 (4 tests)
- λ_adv schedule 三段式 (4 tests)
- dom_head 存在性/shape/参数量/state_dict/非对称梯度 (5 tests)
- ca=0 backward compat (1 test)

---

## 4. 诊断指标 (本实验的关键)

### ✅ 已加的诊断 — 两层

#### 4.1 训练端诊断 (`diag=1` 开关, 已有框架 + 新增 5 个 CDANN 指标)

**开关**: config `algo_para[6] = dg = 1` → 激活 `SGPADiagnosticLogger`

**原有 21 指标** (继续生效):
- Layer 1 (Client, 每 5 round):
  - `orth`, `etf_align_mean`, `intra_cls_sim`, `inter_cls_sim`
  - `loss_task`, `loss_orth`
  - `z_sem_norm_{mean,std,min,max}`, `z_sty_norm_{mean,std,min,max}` (**本次关键**: PACS z_sty_norm 是否保留)
- Layer 2 (Server, 每 round):
  - `client_center_var`, `param_drift`, `n_valid_classes`

**新增 5 个 CDANN 专属指标** (ca=1 时写入):
| 指标 | 含义 | 期望 |
|------|------|------|
| `loss_dom_sem` | dom_head(GRL(z_sem)) CE | R<20=0 (禁用), R40+=↑ (encoder 反抗 GRL) |
| `loss_dom_sty` | dom_head(z_sty) CE | R<20=0, R40+=↓ (正向监督起效) |
| `lambda_adv` | GRL 反向梯度系数 | 三段 schedule: 0 → ramp → 1.0 |
| `dom_sem_acc_train` | z_sem → domain 训练 acc | R40+ → ≈ 25% (random 1/N) |
| `dom_sty_acc_train` | z_sty → domain 训练 acc | R40+ → ≈ 100% |

**输出位置**: `FDSE_CVPR25/task/<TASK>/diag_logs/R200_S<seed>_cdann/`
- `diag_train_client{0,1,2,3}.jsonl` — 各 client 每 5 round 记录
- `diag_aggregate_client-1.jsonl` — server 每 round 记录

**CDANN variant 标记**: diag_root 自动加 `_cdann` 后缀避免 jsonl 污染 (与 baseline `_linear` / `_etf` 区分).

#### 4.2 训练后诊断 — Frozen Post-Hoc Linear Probe (新写脚本)

**脚本**: `FDSE_CVPR25/scripts/run_frozen_probes.py`
**依赖**: 训练末保存的 checkpoint (需要 config `se=1`, **已改**)

**流程**:
1. Load checkpoint (global_model + client_models + whitening + source_style_bank)
2. 所有 client 的 train/test data 过模型一次, 收集 **post-whitening** z_sem / z_sty
3. Train probe **ON TRAIN features** (严格不 leak 到 test)
4. Evaluate probe **ON HELD-OUT TEST features**

**3 个 probe (全部在 post-whitening feature space)**:
| Probe | 输入 | 输出 | 期望 | Claim 角色 |
|-------|------|------|------|------------|
| `probe_sem_domain` | z_sem | domain | ≈ 25% (random 1/4), GRL 成功 | C-domain |
| `probe_sty_domain` | z_sty | domain | ≈ 95%+, 正向监督成功 | C-domain |
| **`probe_sty_class`** | **z_sty** | **class** | **PACS ≥ 40%**, Office ~ 20-30% | **C-probe 核心** |

**关键**: `probe_sty_class` 的 **PACS CDANN vs PACS Linear+whitening baseline** 差距 ≥ 25pp 就直接证明 **anchor claim "whitening 磨掉了 class-relevant style, CDANN 保留了它"**.

---

## 5. 实验配置 (严格对齐 baseline)

### Office (vs EXP-102 对照)

| 参数 | EXP-102 baseline | **EXP-108 CDANN** |
|------|------------------|-------------------|
| algo_para | `[1.0, 0.1, 128, 10, 1e-3, 2, 0, 0, 1, 0]` (10 参数) | `[1.0, 0.1, 128, 10, 1e-3, 2, **1**, 0, 1, 0, **1**, 0, **1**]` (13 参数) |
| diag (dg) | 0 | **1** (ON, 用于 probe 和 CDANN metrics) |
| use_etf (ue) | 0 | 0 |
| use_whitening (uw) | 1 | 1 (保留 +6.20pp 基础) |
| use_centers (uc) | 0 | 0 |
| **se** | 0 | **1** (保存 ckpt 供 probe) |
| **ca** | — | **1** (CDANN ON) |
| R/E/LR | 200/1/0.05 | 同 |
| λ_orth / λ_hsic | 1.0/0.1 | 同 |
| Seeds | {2,15,333} | 同 (严格对齐) |
| Config | `feddsa_whiten_only_office_r200.yml` | `feddsa_cdann_office_r200.yml` |

### PACS (vs EXP-098 Linear 对照)

| 参数 | EXP-098 Linear | **EXP-108 CDANN** |
|------|----------------|-------------------|
| algo_para | `[..., diag=1, ue=0]` (8 参数, 早期版本) | 13 参数 (ca=1, se=1) |
| uc | 1 (default) | 1 (严格对齐) |
| R/E/LR | 200/5/0.05 | 同 |
| Seeds | {2,15,333} | 同 |

---

## 6. 部署计划 (Phase 7, 待执行)

### 6.1 GPU 资源

- seetacloud2 单 4090 24GB, 并行 6 runs (每 run ~2-6GB)
- Office E=1, 单 run ~1h, 3 seeds 并行 ~1h wall
- PACS E=5, 单 run ~3h, 3 seeds 并行 ~3h wall
- **总预估**: ~12h wall (PACS 主要)

### 6.2 部署脚本

`rr/deploy_exp108_cdann.sh`:
1. 服务器 git pull
2. 检查 GPU 空闲
3. `nohup` 后台启动 6 runs (3 PACS + 3 Office)
4. 输出到 `experiments/ablation/EXP-108_cdann_office_pacs_r200/terminal_*.log`

### 6.3 启动命令 (每 run)

```bash
python run_single.py \
  --task {PACS_c4 | office_caltech10_c4} \
  --algorithm feddsa_sgpa \
  --gpu 0 \
  --config ./config/{pacs|office}/feddsa_cdann_{pacs|office}_r200.yml \
  --logger PerRunLogger \
  --seed {2, 15, 333}
```

### 6.4 监控点

- 启动后 2 min: 看 terminal log 有无 crash
- R=20 时: 看 `lambda_adv` 从 0 开始 ramp (diag jsonl)
- R=40 时: 看 `lambda_adv` 到 1.0
- R=50 时: 看 z_sty_norm 是否保留 (PACS 关键)
- R=100 时: 看 AVG Best 趋势
- R=200 End: record JSON + checkpoint 均保存

---

## 7. 验收标准 (Phase 8, 待完成)

### 7.1 Training metrics (从 record JSON 提)

| Claim | 判定 | 决策 |
|-------|------|------|
| **C-main Primary** | PACS 3-seed AVG Best ≥ 82.2 **且** Office ≥ 88.0 | ✅ 成功 / ⚠️ 部分 / ❌ 失败 |
| C-stability | 3 seed std ≤ 1.5 | 方法稳定性 |

### 7.2 Training-time diagnostics (从 diag jsonl 提)

- `z_sty_norm R200`: PACS CDANN ≥ 1.5 (baseline 0.15) → 风格保留 ✓
- `loss_dom_sty R100`: 接近 0 → 正向监督起效
- `dom_sty_acc_train R100`: ≥ 95% → z_sty 携带 domain 信息
- `loss_dom_sem R100`: ≈ log(N_clients) = 1.39 → GRL 对抗有效

### 7.3 Frozen probe (跑 `run_frozen_probes.py`)

- `probe_sem_domain test`: ≈ 0.25 (random 1/4, GRL 成功)
- `probe_sty_domain test`: ≥ 0.95 (正向监督生效)
- **`probe_sty_class test` (PACS)**: **≥ 0.40** (anchor 证据)
- `probe_sty_class test` (Office): 0.20-0.30 (Office 风格不强, neutral)

### 7.4 回填流程

1. 从 record JSON 提 AVG Best/Last, per-domain → NOTE.md 主表
2. 从 diag jsonl 提 z_sty_norm / dom_head metrics → NOTE.md 诊断表
3. 跑 probe 脚本 → NOTE.md probe 表
4. 同步 Obsidian: `obsidian_exprtiment_results/2026-04-20/EXP-108_cdann.md` 按 EXP-100 格式
5. 更新 `已做实验总览.md`
6. 更新 `daily_summary.md`

---

## 8. 风险与兜底

| 风险 | 概率 | Mitigation |
|------|------|------------|
| DANN 对抗训练发散 | 中 | warmup schedule (已实现) + grad clip=10 (config) |
| Office 掉分 > 0.75pp | 中 | scope 诚实限定 PACS-like, Office parity 即可 |
| PACS `probe_sty_class` 不够高 (< 30%) | 中 | 可能需要调 λ_adv schedule (R5 建议通过实验结果决定) |
| dom_head FedAvg 聚合泄露 domain 分布 | 低 | 只聚合参数非 data, 可接受 |
| checkpoint 磁盘占用 (6 × 270MB = 1.6GB) | 低 | seetacloud2 磁盘够用 |

---

## 9. 相关文件链接 (当前 commit `ca83d91`)

### 方案层
- Problem Anchor + Final Proposal: `refine-logs/2026-04-20_FedDSA-CDANN/FINAL_PROPOSAL.md`
- 5 轮 Review Summary: `refine-logs/2026-04-20_FedDSA-CDANN/REVIEW_SUMMARY.md`
- Score Evolution (7.1→8.75): `refine-logs/2026-04-20_FedDSA-CDANN/score-history.md`

### 文献层
- Landscape (30 papers): `LITERATURE_SURVEY_DANN_AND_DECOUPLING.md`
- Idea Report: `IDEA_REPORT_2026-04-20.md`

### 知识层
- 大白话版 (比喻 + 对比表): `obsidian_exprtiment_results/知识笔记/大白话_FedDSA-CDANN.md`
- 学术版 (完整公式): `obsidian_exprtiment_results/知识笔记/FedDSA-CDANN_技术方案.md`

### 代码层
- 算法: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (+120 lines)
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (55/55)
- Probe 脚本: `FDSE_CVPR25/scripts/run_frozen_probes.py`
- Configs: `FDSE_CVPR25/config/{office,pacs}/feddsa_cdann_*.yml`

### 实验层
- NOTE: `experiments/ablation/EXP-108_cdann_office_pacs_r200/NOTE.md`
- 部署脚本: `rr/deploy_exp108_cdann.sh`

### 审查层
- Codex Code Review: `refine-logs/2026-04-20_FedDSA-CDANN/codex-code-review-result.md` (REVISE, 已全部修复)
- Codex Refine R1-R5: `refine-logs/2026-04-20_FedDSA-CDANN/round-{1..5}-review.md`

### Baseline 对比
- Office baseline (EXP-102): `experiments/ablation/EXP-102_full_diag0_office_r200/` — 88.75 AVG
- PACS baseline (EXP-098): `experiments/ablation/EXP-098_sgpa_pacs_r200/` — Linear 80.20 / SGPA 78.96
- Plan A Office (EXP-083): AVG 82.55
- Plan A PACS (EXP-080): AVG 81.69

---

## 10. 当前状态 (2026-04-20 时刻)

- ✅ Phase 0-6 完成 (问题→调研→brainstorm→novelty→refine→代码→单测→codex review)
- ✅ 训练端诊断指标 **已加**:
  - 原 21 指标 (diag=1) ✓
  - 新 5 个 CDANN 指标 ✓
- ✅ 训练后诊断脚本 **已写**:
  - `run_frozen_probes.py` ✓
- ✅ Config `se=1` **已改** (保存 checkpoint 供 probe)
- ⏳ Phase 7 部署: 待启动 (deploy_exp108_cdann.sh 已准备)
- ⏳ Phase 8 回填: 待训练完成 ~12h 后

**下一步**: 确认部署, 启动 6 runs on seetacloud2.
