# Round-2 Proposal: FedDSA-VIB (A) + FedDSA-VSC (B) 并行实验

**时间**: 2026-04-22
**Round**: 2 (基于 Round-1 Codex RETHINK pivot)
**方案**: A 和 B 两个独立实验并行,共享诊断 + 实验矩阵

---

## 1. 背景 (Context Recap)

### 过去失败
- **EXP-108 CDANN**: probe_sty_class MLP-256 = 0.96 (全破,class 被灌入 z_sty)
- **EXP-109 orth_only**: linear=0.24, MLP-256=**0.81** (线性 OK 非线性破)
- **EXP-111 lo=3 强正交**: MLP-64=0.20 (最佳), MLP-256=**0.71** (仍破)
- **Round 0 Swap 方案**: Codex RETHINK 6.2/10 — L_swap 是 L_aug 重新包装, von Kügelgen/MUNIT 理论错用

### Round-1 Codex Pivot
放弃 swap,转向 **Moyer 2018 minimal sufficient invariance + prototype-centered**:
- VIB (信息瓶颈) 压缩 z_sem
- KL prior 围绕 **semantic_prototype** (保留原型叙事)
- 可选 SupCon 升级 InfoNCE

---

## 2. 两个方案定义

### 方案 A: FedDSA-VIB (最小改动)
在原 FedDSA 基础上**只加 VIB**:

```
原 L = L_CE + L_aug + L_orth + L_HSIC + L_InfoNCE

新 L_A = L_CE 
       + λ_aug · L_aug (h-space, 保留)
       + λ_IB · L_VIB                           ★ 新
       + λ_orth · L_orth
       + λ_hsic · L_HSIC
       + λ_sem · L_InfoNCE                      (保留,原 prototype pull)
```

**VIB 实现**:
```python
# encoder 输出 (μ_sem, log_σ²_sem) 而非 z_sem
μ_sem, log_var_sem = semantic_head_stochastic(h)

# Reparameterize (训练时)
ε = torch.randn_like(μ_sem)
z_sem = μ_sem + torch.exp(0.5 * log_var_sem) * ε

# KL to class-conditioned prior (围绕语义原型)
prior_μ = semantic_prototype[y]        # 本类原型向量
prior_var = σ_prior² (固定 e.g. 1.0)
L_VIB = KL(N(μ_sem, σ²_sem) || N(prior_μ, σ_prior²))   # closed-form
```

### 方案 B: FedDSA-VSC (VIB + SupCon)
在方案 A 基础上**把 InfoNCE 换成 SupCon**:

```
新 L_B = L_CE 
       + λ_aug · L_aug (h-space)
       + λ_IB · L_VIB                           ★ 同 A
       + λ_orth · L_orth
       + λ_hsic · L_HSIC
       + λ_supcon · L_SupCon                    ★ 新 (替换 InfoNCE)
```

**SupCon 实现** (Khosla et al. NeurIPS 2020):
```python
# 多正样本对比: batch 内所有同类样本都是正例
L_SupCon = -sum_{i,p∈P(i)} log(exp(z_i·z_p/τ) / sum_{a} exp(z_i·z_a/τ))
         其中 P(i) = {j : y_j = y_i, j ≠ i}
```

### 为什么 A 和 B 都做?
- A 告诉我们: **VIB 单独够不够**?
- B 告诉我们: **SupCon 作为 prototype pull 升级有无增益**?
- 对比 = ablation (VIB alone vs VIB+SupCon upgrade)

---

## 3. 共享诊断指标 (50+)

### 3.1 原 FedDSA 已有 (不列出,DiagnosticLogger 21 个)

### 3.2 新增 — 训练时每 5 轮记录

#### VIB 诊断 (A/B 共用)
| 指标 | 含义 | 健康范围 | 报警 |
|------|-----|:-------:|:----:|
| `KL_mean` | 平均 KL(q\|\|prior) | 0.5-2.0 | > 5 或 < 0.05 |
| `sigma_sem_mean` | q 的平均 σ | 0.1-1.0 | → 0 或 > 10 |
| `sigma_sem_max` | σ 最大值 | < 2 | > 10 (数值崩) |
| `rate_R` | ELBO rate term | 稳定 | 爆发 |
| `distortion_D` | CE loss (用 z_sem 不是 μ_sem) | 单调降 | 反弹 |
| `z_sem_to_prior_cos` | μ_sem 与 prior μ cos | ↑ 到 > 0.7 | < 0.3 |
| `prior_var_eff` | σ_prior² 实际起作用大小 | 固定配置 | — |

#### SupCon 诊断 (仅 B)
| 指标 | 含义 | 健康范围 |
|------|-----|:-------:|
| `pos_sim_mean` | 同类样本 cos | ↑ > 0.7 |
| `neg_sim_mean` | 异类样本 cos | ↓ < 0.2 |
| `alignment_Wang_Isola` | E[\|\|z_i - z_j\|\|²] 同类 | ↓ |
| `uniformity_Wang_Isola` | log E[exp(-2·\|\|z_i-z_j\|\|²)] 全对 | 稳定 |
| `n_positive_avg` | batch 平均正样本数 | > 5 |

#### Prototype 诊断 (A/B 共用)
| 指标 | 含义 | 健康范围 |
|------|-----|:-------:|
| `prototype_drift_L2` | 每 round 全局 proto 位移 L2 | 衰减 |
| `intra_class_z_to_proto_cos` | z_sem 到本类 proto cos | ↑ > 0.8 |
| `inter_class_proto_min_cos` | 不同类 proto 最小 cos | < 0.3 (分开) |
| `proto_active_classes` | 有效 proto 数 | = 7 (PACS) |

#### Proxy Probe (训练时) — ⭐ 最关键
每 10 轮,用当前 (frozen) encoder 在 held-out set 上跑 probe:

| Probe | 用什么 probe | 健康预期 |
|-------|-------------|:-------:|
| `proxy_probe_sty_class_linear` | sklearn LogReg | ↓ 到 < 0.3 |
| `proxy_probe_sty_class_MLP64` | MLP h=64 | ↓ 到 < 0.4 |
| `proxy_probe_sty_class_MLP256` | MLP h=256 | ↓ 到 < 0.5 ★ |
| `proxy_probe_sem_class_linear` | LogReg | ≈ train acc |
| `proxy_probe_sem_domain_linear` | LogReg | < 0.5 |
| `proxy_probe_sty_domain_linear` | LogReg | > 0.8 (z_sty 有料) |

#### 梯度诊断 (A/B 共用)
| 指标 | 含义 | 报警 |
|------|-----|:----:|
| `grad_cos(CE, VIB)` | 梯度夹角 | 长期负 → 打架 |
| `grad_cos(CE, SupCon)` | B 专用 | 长期负 |
| `grad_norm_CE` / `grad_norm_VIB` / `grad_norm_SupCon` | 相对量级 | 某项 ≫ 其他 |

#### 每 100 轮输出 Health Snapshot
```
=== Round 100 Health ===
Task:     CE=0.42 acc(val)=0.78 proto_active=7/7
VIB:      KL=0.87 σ̄=0.34 μ_to_prior=0.81 ✓
SupCon:   pos=0.73 neg=0.15 align=0.45 ✓  [B only]
Geom:     cos(sem,sty)=0.02 z_sty_norm=0.45 ✓
Proxy probes (sty→class):
          linear=0.27 ↓✓  MLP-64=0.48 ↓✓  MLP-256=0.62 ↓✓
Proxy probes (sem→class):
          linear=0.81 (ref)
Proxy probes (sty→domain):
          linear=0.76 ✓ (sty is not dead)
Grads:    cos(CE,VIB)=+0.12 ✓ (not fighting)
Verdict:  ON TRACK
```

---

## 4. Post-hoc Evaluation (训练完)

### 4.1 主结果表 (PACS + Office × 3-seed mean)
与原 FedDSA 表对齐: AVG Best / ALL Best / per-domain.

### 4.2 Moyer Probe Sweep (FL 首次)
frozen encoder → 对每个 dataset × method × seed:
- 0 层 (linear LogReg)
- 1 层 MLP hidden=64
- 2 层 MLP hidden=64,64
- 3 层 MLP hidden=64,64,64

每层测:
- `probe_sty_class`
- `probe_sem_class`
- `probe_sty_domain`
- `probe_sem_domain`

+ control task selectivity (Hewitt-Liang 2019)

---

## 5. 实验矩阵

### 5.1 方法清单
| ID | 方法 | 复现状态 |
|----|------|:------:|
| M0 | orth_only (EXP-109/110) | ✅ 已跑 |
| M1 | CDANN (EXP-108) | ✅ 已跑 |
| M2 | FDSE (CVPR 2025) | ❌ 需新跑 (有源码) |
| M3 | FediOS (ML 2025) | ❌ 需新跑 (arxiv 2311.18559 有代码) |
| M4 | FedDSA-VIB (方案 A) | ❌ 待实现 |
| M5 | FedDSA-VSC (方案 B) | ❌ 待实现 |

### 5.2 Run 矩阵
- Datasets: **PACS_c4** + **office_caltech10_c4**
- Seeds: **{2, 15, 333}** (与 EXP-108/109/110/111 对齐)
- Method: M2/M3/M4/M5 (M0/M1 已有)
- 总 new runs: **4 methods × 2 datasets × 3 seeds = 24**

### 5.3 优先级
**Priority 1 (必做)**: M4 + M5 in PACS + Office × 3-seed = **12 runs**
**Priority 2 (论文要)**: M2 + M3 baseline = **12 runs**
(如 FDSE/FediOS 源码接入有问题,可用论文 reported 数字作对照)

### 5.4 GPU 预算估算
- R200 single run ≈ 3-4h on 4090 (PACS) / 2h (Office)
- 并发 4 runs 同卡 (前次经验) → wall ≈ 5-6h per batch
- 24 runs / 4 并发 = 6 batch → **~30-40h** (1-2 day)
- Priority 1 only: **~12-15h**

---

## 6. 预期结果

| Metric | orth_only | CDANN | FedDSA-VIB (pred) | FedDSA-VSC (pred) |
|--------|:---------:|:-----:|:-----------------:|:-----------------:|
| linear probe (sty→class) | 0.24 | 0.96 | 0.10-0.18 | **0.08-0.15** |
| MLP-64 probe | 0.69 | 0.96 | 0.35-0.45 | 0.30-0.40 |
| MLP-256 probe | 0.71 | 0.96 | 0.45-0.55 | **0.40-0.50** |
| 3-layer probe (Moyer) | ~0.75 | ~0.90 | 0.50-0.60 | **0.45-0.55** |
| PACS AVG Best | 80.64 | 80.08 | 80.5-81.5 | **81.0-82.0** |
| Office AVG Best | 89.09 | 89.54 | 89.0-89.5 | **89.3-89.8** |

**预期差异 (B - A)**: SupCon 对多正样本场景稳定性更好,PACS 上预期 +0.5pp acc / probe 更稳。

---

## 7. 单元测试设计 (实现前必须过)

### test_vib.py
```
1. VIB forward: 输入 h, 输出 μ, σ, z_sem 形状正确
2. 数值稳定: σ_sem 不会负 (用 exp(0.5 log_var) 而非直接 σ)
3. Reparameterize 梯度流: ∂L/∂μ 和 ∂L/∂log_var 非零
4. KL closed-form 正确: 对比手工计算一个小例子
5. Deterministic mode (eval): 不 sample,直接用 μ
6. Prior μ 从 prototype 取得正确: 给定 label 取对应 proto
```

### test_supcon.py
```
1. SupCon forward: 输入 [B, d] z_sem + [B] y → scalar loss
2. 单类 batch (只有 1 类): 退化为 0 或 skip
3. 每类都有正样本: n_pos per sample 正确统计
4. 温度 τ 效应: τ↓ → loss 变陡
5. 梯度流: ∂L/∂z_sem 非零
6. 对 InfoNCE 的 degenerate 一致性: 如果每类只 1 正样本,SupCon 退化为 InfoNCE
```

### test_prototype_prior.py
```
1. prototype 初始化: 全 None 时返回 default prior
2. prototype 更新: FedAvg 聚合逻辑
3. μ_prior 采样: 输入 y=[0,1,2],取对应 prototype 正确
4. 未收敛 class (n < threshold): fallback to default
5. Drift metric: L2(proto_new, proto_old) 正确
```

### test_diagnostic.py
```
1. Proxy probe 在 dummy encoder 上能跑通
2. Health snapshot 格式正确 (无 KeyError)
3. grad_cos 计算: ∠(g1, g2) 在 [-1, 1]
4. 所有 metric 都能被 serialize 到 JSON
```

### test_integration_smoke.py
```
1. Full pipeline 1 round 能跑通 (dummy data 8 samples)
2. Loss.backward() 不报 error
3. Param grad 非零 (所有可训参数)
4. 前 5 个 step 所有诊断 metric 有合理值
```

---

## 8. 实现阶段 (Phase 5c-d)

### 8.1 代码改动清单
- `algorithm/feddsa_vib.py` (方案 A 新文件)
- `algorithm/feddsa_vsc.py` (方案 B 新文件) 
- `algorithm/common/vib.py` (共享 VIB 模块)
- `algorithm/common/supcon.py` (SupCon loss)
- `algorithm/common/prototype_prior.py` (prototype-centered prior)
- `diagnostics.py` (扩展 metrics)
- `scripts/run_capacity_probes.py` (加 Moyer 0/1/2/3 层 + selectivity)
- `config/pacs/feddsa_vib.yml`, `feddsa_vsc.yml`
- `config/office/feddsa_vib.yml`, `feddsa_vsc.yml`
- `tests/test_*.py` (单元测试)

### 8.2 依赖
- 复用原 FedDSA algorithm / Server 逻辑
- 复用原 `_style_augment` (h-space AdaIN)
- 复用原 prototype 聚合

---

## 9. Self-review 挑毛病 (Phase 5b,下面详细展开)

已在下一节展开。

---

## 10. 部署前 gate

在部署到服务器前,必须:
1. ✅ Self-review 通过 (见 §11)
2. ✅ Codex round-2 review ≥ 7.5
3. ✅ 单元测试 all pass
4. ✅ 本地 smoke test (8 样本 1 round) 无 error
5. ✅ 用户最终确认

任一 gate 不过 → 不部署。

---

## 11. Self-review (Phase 5b) 

下一段写。
