# EXP-119 | FedPTR Sanity Smoke Phase 1 — 验证 3 个组件是否值得投入

## 基本信息
- **日期**: 2026-04-22
- **目标**: 在写 FedPTR 全量代码前, 用最小实验验证 3 个核心组件是否真 work
- **服务器**: seetacloud2 + lab-lry (GPU 1)
- **状态**: 🟡 设计完成, 待部署
- **决策依据**: 接受 CVPR reviewer review (Score 4/10) 的建议, 不堆组件, 先验证单点

---

## 0. Claim Map & Hypothesis

### 需要回答的 3 个 Go/No-Go 问题

| # | Claim 被验证 | 反向假设 | 最小证据 |
|:-:|---|---|---|
| **C1** | AlexNet on Office/PACS 有 method-space 到 FDSE+2pp | "AlexNet 已到 capacity 天花板, method 做不上" | Centralized > 93 on Office |
| **C2** | Class-conditional BN stats bank + feature AdaIN work | "AdaIN 家族第 5 次失败 (EXP-040/059/061/047 历史阴影)" | Office 3-seed > FedBN 88.68 + 0.5 |
| **C3** | Prototype trajectory prediction 有真实贡献 | "FL E=5 noise 下 signal<<noise, 预测等于放大噪声" | η=0.5 vs η=0 差 > 0.5pp |

### Paper 储备 (if sanity 都通过)

- **Primary Claim (dominant)**: 在 FedBN 基础上, class-conditional style prototype 的 cross-client 共享 + trajectory prediction 使跨域 accuracy 超 FDSE +2pp
- **Supporting Claim**: Learnable per-client α 作为 difficulty-aware curriculum 救 outlier 域 (Caltech)
- **Anti-Claim to rule out**: "Gain 仅来自 FedFA/FedCA 的增量组合"

### Paper 储备 (if sanity 部分失败)

**Fallback 方案**:
- C1 失败 → 换 backbone 到 ResNet-18 (或降 venue 到 WACV/BMVC, target +1pp)
- C2 失败 → 砍 CC-Bank, 转向 **logit-level calibration** (类似 FedDr+ class-conditional 变体, 不碰 feature AdaIN)
- C3 失败 → 砍 trajectory, 保留 simple prototype alignment

---

## 1. 实验 Block 列表

### Block A: Centralized AlexNet Capacity Test

**Claim tested**: C1 — method space 还有 +2pp 的上限吗?

**Why this block exists**: FDSE 已经在 Office 上 90.58. 如果 centralized AlexNet 都打不到 93, 我们 FL 无论怎么调都冲不上 +2pp.

**Dataset / split**: Office-Caltech10 + PACS + DomainNet (10-class subset)

**Compared systems**:
- Centralized AlexNet (算法: `standalone`, proportion=1.0, 1 "virtual client" 合并全部 domain 数据)
- FedBN baseline 作为 FL 下限参照
- FDSE 已知 3-seed AVG Best 作为 SOTA 参照

**Metrics**: AVG Best, AVG Last, per-domain Best

**Setup details**:
- Backbone: AlexNet (同 FL 实验), Linear(1024 → num_classes) 分类头
- R=200, E=5 (PACS/DomainNet) / E=1 (Office), B=50, LR=0.05, WD=1e-3
- Seeds: {2, 15, 333} (3 seeds, 跟 FL 对齐)
- Optimizer: SGD 0.9 momentum, lr_decay 0.9998

**Success criterion**:
- Office ≥ 93 → **Go** 全量, +2pp target 可行
- Office 90-92 → **Marginal Go**, 降 target 到 +1pp
- Office < 90 → **换 backbone 或 venue**

**Failure interpretation**:
- Centralized 比 FDSE 还低 → AlexNet capacity 就是瓶颈, method novelty 救不了
- 建议方案: (a) 换 ResNet-18 重跑所有 baseline (b) 冲 WACV 不冲 CVPR

**Priority**: **MUST-RUN**

---

### Block B: CC-Bank Only (fixed α=0.5, no trajectory)

**Claim tested**: C2 — class-conditional style augmentation 是不是历史 AdaIN 第 5 次失败?

**Why this block exists**: 历史上 EXP-040/059/061/047 四次 AdaIN 都挂了. 必须先单独测 CC-Bank 避免跟 trajectory / α 耦合.

**Dataset / split**:
- **Office-Caltech10** (最关键, EXP-061 style sharing 套件在这崩过)
- PACS (EXP-059 stylehead AdaIN -2.54%)

**Compared systems**:
| System | AdaIN | Bank 结构 | α |
|:-:|:-:|:-:|:-:|
| Base FedBN | ❌ | — | — |
| CC-Bank fixed α=0.3 | ✅ | (class, client) 2D | 0.3 |
| **CC-Bank fixed α=0.5** | ✅ | (class, client) 2D | **0.5** |
| CC-Bank fixed α=0.7 | ✅ | (class, client) 2D | 0.7 |
| Domain-level bank (FedFA-like) ref | ✅ | client 1D | 0.5 |

**Metrics**: 3-seed mean AVG Best / AVG Last, per-domain Best

**Setup details**:
- 算法: `fedbn_ccbank.py` (继承 fedbn.py, 加 ~80 行)
- AdaIN 位置: encoder 最后 pooled feature (1024d), 不在 BN 层本身
- Bank 内容: 每 round client 上传 per-class batch μ/σ (n_c ≥ 8 才上传, 否则 skip)
- 采样: 训练时以 p=0.5 概率做 AdaIN, 从 other_client 的 same-class bank 采
- R=200, E=1 (Office) / E=5 (PACS), Seeds {2,15,333}, LR=0.05

**Success criterion** (Office 3-seed mean AVG Best):
- **> 89.2**: CC-Bank 真 work, 保留 → 加 trajectory + α 冲全量
- **88.0 - 89.2**: marginal, 需要 EMA smoothing + min sample threshold 才能救
- **< 88.0**: **kill CC-Bank**, 转 logit-level 路线

**Failure interpretation**:
- Office < 88.0: AdaIN 在 feature space 不适用 AlexNet capacity
- 可能原因: class-conditional 加剧了小样本 μ/σ 估计误差
- Plan B: logit-level (class-conditional classifier bank, FedDr+ 变体)

**Priority**: **MUST-RUN**

---

### Block C: Trajectory Prediction vs No-Prediction

**Claim tested**: C3 — 一阶 Taylor 预测 `p̂ = p + η·v` 真的比 `p̂ = p` 显著好?

**Why this block exists**: CVPR reviewer 说 FL E=5 noise 下 velocity estimate 方差 >> mean, 一阶预测等于放大噪声. 必须单独验证 trajectory 真贡献.

**Dataset**: **PACS** (4 client × 7 class, E=5, trajectory 最容易测出差异)

**Compared systems**:
| η 值 | 含义 | prototype smoothing |
|:-:|:-:|:-:|
| 0 | no prediction (退化成普通 prototype alignment) | ❌ |
| 0.3 | 保守预测 | ❌ |
| **0.5** | 默认 | ❌ |
| 0.8 | 激进预测 | ❌ |
| 0.5 + EMA(β=0.9) | 预测 + EMA 平滑 | ✅ |

**Metrics**: 3-seed mean AVG Best / AVG Last

**Setup details**:
- 算法: `fedbn_trajectory.py` (继承 fedbn.py, 加 ~60 行)
- Prototype: 每 class 全局均值 = client class mean 的加权平均
- 维护 `p_c^t` (position) + `v_c^t = p_c^t - p_c^{t-1}`
- 预测: `p̂_c^{t+1} = p_c^t + η · v_c^t`
- Client loss: `L = L_CE + 0.5 · (1 - cos(h, sg(p̂_{y})))` (stop_grad 必须)
- R=200, E=5, Seeds {2,15,333}
- 不加 CC-Bank, 不加 learnable α, 不加 curvature 降权

**Success criterion** (PACS 3-seed mean, η=0.5 vs η=0):
- **Δ > 0.5pp**: trajectory 有真实贡献, 保留
- **Δ 0.2 - 0.5pp**: marginal, 简化为 EMA prototype 可能更稳
- **Δ < 0.2pp**: **kill trajectory**, 退化为普通 prototype alignment

**Failure interpretation**:
- 如果 η=0 已经 > FedBN baseline 很多, 说明增益在 alignment 本身不在预测
- 如果 η=0.5 反而 < η=0, velocity 估计真的是噪声主导
- Plan: 把 trajectory component 整个砍掉, 保留 simple alignment

**Priority**: **MUST-RUN**

---

## 2. Compute Budget

### 每 run 时间估计 (基于历史数据)

| Task | E | 每 run 时间 | 单卡并行 (24GB) |
|:---:|:---:|:---:|:---:|
| Office-Caltech10 | 1 | ~50min - 1h | 5-7 runs |
| PACS_c4 | 5 | ~7-8h | 3-4 runs |
| DomainNet_c6 | 5 | ~4h | 5-7 runs |
| Office Centralized | 1 | ~45min (单 client 但数据全) | 5-7 runs |
| PACS Centralized | 5 | ~5-6h (单 client) | 3-4 runs |

### Sanity A (Centralized) runs

| 数据集 | Seeds | Runs | 单 run | 3-并行 wall |
|:---:|:---:|:---:|:---:|:---:|
| Office | 2, 15, 333 | 3 | 45min | 45min |
| PACS | 2, 15, 333 | 3 | 5h | 5h |
| DomainNet | 2, 15, 333 | 3 (可选) | 4h | 4h |

**子总**: 3-6 runs, wall 5h (Office + PACS 必要, DomainNet 可选)

### Sanity B (CC-Bank) runs

| 数据集 | α configs | Seeds | Runs | 单 run | 3-并行 wall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Office | α ∈ {0.3, 0.5, 0.7} | {2,15,333} | 9 | 1h | 3h (3 并行 3 轮) |
| PACS | α = 0.5 only (验证不退化) | {2,15,333} | 3 | 7h | 7h (3 并行) |

**子总**: 12 runs, wall ~10h (Office 3h + PACS 7h, 可并行)

### Sanity C (Trajectory) runs

| 数据集 | η configs | Seeds | Runs | 单 run | 3-并行 wall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PACS | η ∈ {0, 0.5} (主对照) | {2,15,333} | 6 | 7h | 14h (2 批 × 3 并行) |
| PACS | η=0.5 + EMA (可选) | {2} | 1 | 7h | 7h |

**子总**: 6-7 runs, wall 14h

### 总预算

| 必要 (MUST) | wall (3-并行) | 组件 |
|:---:|:---:|:---:|
| Sanity A Office + PACS | 6h | 3 + 3 runs |
| Sanity B Office + PACS | 10h | 9 + 3 runs |
| Sanity C PACS η∈{0,0.5} | 14h | 6 runs |
| **TOTAL** | **~30h wall** | **24-27 runs** |

**GPU·hour 估算**: ~100-120 GPU·h (24 runs × 平均 4-5h/run)

---

## 3. 对照矩阵 (一张表看全)

| Run ID | Block | Task | Algo | Seed | Key Param | Config File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| R001-003 | A | Office | standalone | {2,15,333} | proportion=1.0, 1 client | centralized_office.yml |
| R004-006 | A | PACS | standalone | {2,15,333} | proportion=1.0, 1 client | centralized_pacs.yml |
| R007-009 | A | DomainNet (可选) | standalone | {2,15,333} | proportion=1.0 | centralized_dn.yml (待写) |
| R010-018 | B | Office | fedbn_ccbank | 3seeds × 3α | α∈{0.3, 0.5, 0.7} | ccbank_office_a{0.3,0.5,0.7}.yml |
| R019-021 | B | PACS | fedbn_ccbank | {2,15,333} | α=0.5 | ccbank_pacs_a0.5.yml |
| R022-024 | C | PACS | fedbn_trajectory | {2,15,333} | η=0 | traj_pacs_eta0.yml |
| R025-027 | C | PACS | fedbn_trajectory | {2,15,333} | η=0.5 | traj_pacs_eta0.5.yml |
| (可选) R028 | C | PACS | fedbn_trajectory | 2 | η=0.5 + EMA | traj_pacs_eta0.5_ema.yml |

**计数**: Must-run 21 runs (A + B + C minimum), Nice-to-have 6 runs

---

## 4. 执行顺序 (Run Order + Milestones)

### M0: GPU 资源释放 (0.5 天)

**Goal**: 等现有 runs 完成腾 GPU

**Runs**: 无新 runs, 等
- seetacloud2 EXP-115 DomainNet 12 runs 完成 (R180+, 预计 2h)
- lab-lry EXP-116 lo=0 × 6 + EXP-117 orth_only DomainNet × 3 + EXP-118 fullBN × 3 完成

**Decision gate**: GPU 完全空闲

**Cost**: 0 (纯等待)

**Risk**: 这些 runs 失败需要重跑 → 确认都是健康的

---

### M1: Sanity A Office (最快验证 capacity) — 2h

**Goal**: C1 验证, Office capacity 有没有 +2pp 空间?

**Runs**: R001-003 (Office centralized × 3 seeds)

**Decision gate**: Office 3-seed mean ≥ 93 (Go) / 90-92 (Marginal) / < 90 (换 backbone)

**Cost**: 3 runs × 45min = 2.25h wall (3-并行)

**Risk**: standalone.py 可能有 bug (CLAUDE.md 记载 PACS standalone 有 standalone bug)
- **Mitigation**: 先跑 Office smoke test (R=20) 确认能跑, 再扩 R=200

---

### M2: Sanity B Office (CC-Bank 生死) — 3h

**Goal**: C2 验证, AdaIN 是不是第 5 次失败?

**Runs**: R010-018 (3 α × 3 seeds on Office)

**Decision gate**:
- Office 3-seed mean > 89.2: **CC-Bank 生** → M3/M4 继续
- Office 3-seed mean 88.0-89.2: 不确定 → 决定要不要加正则救
- Office 3-seed mean < 88.0: **CC-Bank 死** → 转 logit-level

**Cost**: 9 runs × 1h = 3h wall (3-并行)

**Risk**:
- AdaIN 第 5 次挂 — concede 并转 logit-level
- Small-class bank noise 污染训练 — 加 min sample threshold (n≥8) mitigate

---

### M3: Sanity A PACS + Sanity C PACS (trajectory 生死) — 14h

**Goal**: C1 PACS 验证 + C3 验证

**Runs**:
- R004-006 (PACS centralized × 3 seeds)
- R022-027 (PACS trajectory × 6 runs)

**可并行**: 是, GPU 够用的话 9 runs 并行

**Decision gate**:
- PACS centralized: capacity check
- Trajectory η=0.5 vs η=0: Δ > 0.5 (生) / < 0.2 (死)

**Cost**: 9 runs × 7h = 14h wall (3-并行两波)

**Risk**:
- PACS E=5 耗时长, 如果 Sanity B 先死掉, 可以提前 cancel

---

### M4: Sanity B PACS (验证 Office 结论能迁移到 PACS) — 7h

**Goal**: CC-Bank 在 PACS 是否也不退化? 历史 EXP-059 PACS -2.54% 的阴影能破吗?

**Runs**: R019-021 (3 seeds PACS α=0.5)

**Decision gate**:
- PACS 3-seed mean ≥ FedBN 79.01: 保留
- PACS 3-seed mean < 78.5: CC-Bank 在 PACS 也挂

**Cost**: 3 runs × 7h = 7h wall (3-并行)

**Risk**: 如果 M2 已死, M4 可以 skip

---

### M5: Decision Meeting (写 review report)

**Goal**: 基于 M1-M4 结果决定最终方案

**Output**: 更新 `FedPTR_FINAL_PROPOSAL.md` → V2 瘦身版 OR 换方向

**Cost**: 0.5 天阅读+决策

---

## 5. Go/No-Go 决策表 (全局)

| 结果组合 | 决策 | 下一步方案 |
|:---:|:---:|:---:|
| A pass, B pass, C pass | ✅ All go | 写全量 FedPTR (3 组件 + α 正则) |
| A pass, B pass, **C fail** | ⚠️ 砍 trajectory | 瘦身: CC-Bank + Learnable α (2 组件) |
| A pass, **B fail**, C pass | ⚠️ 砍 CC-Bank | logit-level 替代方案 + trajectory |
| A pass, **B fail**, **C fail** | ❌ 大换血 | 转 FedAdvStyle (对抗风格) 或 FedGLA (条件 MI) |
| **A fail** (Office < 90) | ❌ 换 backbone | ResNet-18 重跑 baseline, 降 target 或换 venue |
| A 边缘 (90-92) | 🟡 降 target | Target +1pp, 冲 WACV/BMVC |

---

## 6. 代码改动估计

### Sanity A: standalone.py 复用 (~10 行新增)
- 主要是 config 文件
- 需要 "centralized" task 设置 (1 virtual client 合并 data) 或用 `proportion=1.0` 模拟
- 行数: ~10 行 (config × 2-3 个)

### Sanity B: `fedbn_ccbank.py` (~100 行新增)

```python
# 继承 fedbn.py
class Server:
    - __init__: 新增 style_bank = defaultdict(dict)  # bank[class][client_id] = (μ, σ)
    - pack: 下发 style_bank
    - iterate: 聚合时更新 bank (从 client 上传收集)

class Client:
    - unpack: 加载 bank
    - train: 训练时 AdaIN 增强
        - 前向 h = encoder(x), 计算 batch μ_self, σ_self
        - 从 bank[y][other_client] 采 (μ', σ')
        - h_aug = σ' · (h - μ_self) / σ_self + μ'
        - h_final = α * h_aug + (1-α) * h
        - CE(classifier(h_final), y)
    - upload: 上传 class-conditional batch μ/σ
```

- 行数: ~100 行 (算法) + 3-6 个 config (~60 行) = **160 行**
- 单测: ~10 个 tests

### Sanity C: `fedbn_trajectory.py` (~80 行新增)

```python
class Server:
    - 新增: self.prototype_history = dict (维护 p_c^t, p_c^{t-1})
    - aggregate_prototype: 加权平均 class prototypes
    - predict_next: p_hat = p + η · v
    - pack: 下发 p_hat

class Client:
    - 上传: per-class local prototype
    - train loss: + 0.5 · (1 - cos(h, sg(p_hat[y])))
```

- 行数: ~80 行 (算法) + 5-6 个 config (~60 行) = **140 行**
- 单测: ~8 个 tests

**总代码**: ~310 行, 1 天能写完 + 测试

---

## 7. 启动命令清单

### Sanity A: Centralized AlexNet

```bash
# Office
CUDA_VISIBLE_DEVICES=1 python run_single.py \
  --task office_caltech10_c4 --algorithm standalone \
  --config ./config/office/centralized_office.yml \
  --logger PerRunLogger --seed 2 --gpu 0 &
# 重复 seed=15, 333

# PACS
CUDA_VISIBLE_DEVICES=1 python run_single.py \
  --task PACS_c4 --algorithm standalone \
  --config ./config/pacs/centralized_pacs.yml \
  --logger PerRunLogger --seed 2 --gpu 0 &
```

**需要新写**:
- `FDSE_CVPR25/config/office/centralized_office.yml`
- `FDSE_CVPR25/config/pacs/centralized_pacs.yml`
- (可能) 确认 standalone.py 在 Office/PACS 上能跑

### Sanity B: CC-Bank

```bash
# Office α=0.5 × 3 seeds
for seed in 2 15 333; do
  CUDA_VISIBLE_DEVICES=1 python run_single.py \
    --task office_caltech10_c4 --algorithm fedbn_ccbank \
    --config ./config/office/ccbank_office_a0.5.yml \
    --logger PerRunLogger --seed $seed --gpu 0 &
done

# 类似 α=0.3 / α=0.7
```

**需要新写**:
- `FDSE_CVPR25/algorithm/fedbn_ccbank.py` (~100 行)
- 6 个 config (3 α × 2 dataset)
- 10 个单元测试

### Sanity C: Trajectory

```bash
# η=0 × 3 seeds
for seed in 2 15 333; do
  CUDA_VISIBLE_DEVICES=1 python run_single.py \
    --task PACS_c4 --algorithm fedbn_trajectory \
    --config ./config/pacs/traj_pacs_eta0.yml \
    --logger PerRunLogger --seed $seed --gpu 0 &
done

# η=0.5 × 3 seeds
```

**需要新写**:
- `FDSE_CVPR25/algorithm/fedbn_trajectory.py` (~80 行)
- 2 个 config (η=0, η=0.5)
- 8 个单元测试

---

## 8. 数据收集 + 回填流程

### 每 run 完成后
1. JSON record 自动写入 `task/{task}/record/{algo}_*.json`
2. 用 Python 读取 `mean_local_test_accuracy` 算 max (Best) 和 last (Last)
3. 回填到本 NOTE.md 的"结果"章节

### 3-seed 齐了后
1. 算 mean ± std
2. 对比 Go/No-Go 判据
3. 做决策 → 写到 M5 Decision Meeting

### 代码模板
```python
# 收集结果
import json, glob
files = glob.glob('task/{task}/record/{algo}*S{seed}*.json')
for f in files:
    d = json.load(open(f))
    acc = d['mean_local_test_accuracy']
    best = max(acc); last = acc[-1]
    print(f'best={best*100:.2f} last={last*100:.2f}')
```

---

## 9. 潜在风险 + 缓解

| 风险 | 概率 | 影响 | 缓解 |
|:---:|:---:|:---:|:---:|
| standalone.py 在 Office/PACS 有未发现 bug | 中 | 高 (Sanity A 废) | R=20 smoke 先试, 再 scale 到 R=200 |
| CC-Bank 的 per-class batch μ/σ 在 class 样本 < 8 时崩 | 高 | 中 | min sample threshold: n_c ≥ 8 才收 bank |
| Trajectory velocity 数值不稳 (prototype 跳跃) | 高 | 中 | warmup: R < 10 不做 prediction (η=0 等效) |
| GPU 资源不够同时跑 3 个 sanity | 中 | 中 | 串行跑 (M1 → M2 → M3/M4) |
| 在 lab-lry 上 standalone 模式要特殊 task 定义 | 中 | 低 | 用 `proportion=1.0` 模拟 centralized 或写新 `central_office_1c` task |
| Sanity B PACS 第 5 次 AdaIN 挂 (-2.54% 重演) | 中 | 低 (只是确认 kill) | 这是"预期失败", 数据本身就是结论 |

---

## 10. 代码实现优先级 (给 implementation phase 用)

1. **First**: 写 `fedbn_ccbank.py` (100 行) + 2-3 个 Office config — Sanity B 最快出结果
2. **Second**: 写 centralized config (复用 standalone.py) — Sanity A 最快
3. **Third**: 写 `fedbn_trajectory.py` (80 行) + 2 个 PACS config — Sanity C 最慢 (PACS E=5)
4. **所有代码完成**后再统一部署, 不要边写边跑 (避免资源切换)

---

## 11. Final Checklist (实验设计质量检验)

- [x] 每个 sanity 对应明确 claim (C1/C2/C3)
- [x] 每个 sanity 有明确 Go/No-Go 判据
- [x] 每个 sanity 有 fallback 方案 (failure interpretation)
- [x] 3 seeds 保证统计效度
- [x] 资源预算合理 (30h wall, 120 GPU·h)
- [x] 对照组覆盖到基线 (FedBN, FedFA-like, etc.)
- [x] 代码量估计 (~310 行)
- [x] 实验启动命令具体化
- [x] 潜在 bug 风险 + 缓解 措施

---

## 12. 下一步 (本 NOTE 之后)

### 按 experiment-plan skill 的产出规范

- **文档**: 本 NOTE + `refine-logs/2026-04-22_FedPTR_refined/FedPTR_FINAL_PROPOSAL.md` (已有)
- **Tracker**: 需要一份 `EXPERIMENT_TRACKER.md` (可选, NOTE 足够覆盖)
- **代码实现**: 按优先级 CC-Bank → Centralized → Trajectory

### 等用户 Gate

- [ ] 批准开始实现代码 (CC-Bank 最高优先)
- [ ] 批准清理 lab-lry 现有 runs 腾 GPU 给 sanity

---

## 📎 相关文件

- **方案原文**: `refine-logs/2026-04-22_FedPTR_refined/FedPTR_FINAL_PROPOSAL.md`
- **大白话版**: `obsidian_exprtiment_results/知识笔记/大白话_7个新方案brainstorm_2026-04-22.md`
- **CVPR review**: (待存) `refine-logs/2026-04-22_FedPTR_refined/REVIEW_ROUND_1.md`
- **上游实验基线**:
  - FedBN: 79.01 (PACS) / 88.68 (Office) / 72.08 (DomainNet)
  - FDSE: 79.91 / 90.58 / 72.21
  - orth_uc1 (EXP-109/110/115): 80.64 / 89.17 / 72.49

---

*生成日期: 2026-04-22*
*基于 research-pipeline 的 experiment-plan Phase 5: 实验计划细化*
