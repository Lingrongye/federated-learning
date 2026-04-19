# Experiment Plan — FedDSA-SGPA (Fixed ETF + SGPA Inference)

**Problem**: 在跨域联邦学习 (feature-skew FL, PACS/Office) 上,Plan A orth_only 已达饱和 (Office AVG 82.55, PACS 82.31),如何再拿 ≥1% 且**可解释**?
**Method Thesis**: **Fixed Simplex ETF classifier + pooled-whitening Mahalanobis z_sty SGPA inference** — 一句话:训练端用固定几何分类头让特征提前对齐单纯形顶点 (Neural Collapse 加速),推理端用解耦出的风格做 reliability gate 指导 prototype correction。
**Date**: 2026-04-19
**Status**: EXP-096 smoke test 已过 (R50 Office AVG 84.98%, Neural Collapse 5 条证据齐全), **必须 multi-seed + control 验证后才能信**

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|----------------------------|---------------|
| **C1 (Primary)**: Fixed ETF training beats Plan A Linear training on feature-skew FL | 论文 dominant contribution 之一,如果不成立整个方案崩 | SGPA R200 Office AVG ≥ 84% (3-seed mean), 且 > Plan A 82.55% (同 seed 对比 Δ ≥ +1.5%) | B1, B2 |
| **C2 (Primary)**: Fixed ETF 本身 (而非副作用) 是 gain 来源 | 排除"pooled whitening 副作用 / class_centers 累积 / 别的辅助" 导致 gain 的可能 | Linear+whitening R200 Office AVG ≤ 83% (≤ Plan A + 0.5%), 和 ETF R200 差 ≥ 1% | B2 |
| **C3 (Supporting)**: SGPA 推理 (双 gate + proto) 免费 proto_vs_etf_gain > 0 | 论文 真正 dominant contribution (ETF 只是 supporting stabilizer) | SGPA 推理后 proto_acc - etf_acc ≥ +0.5% on EXP-096 checkpoint | B3 |
| **C4 (Anti-claim to rule out)**: ~~gain 只是 seed=2 运气~~ | EXP-075/078d/095 历史教训, 单 seed 必不可信 | 3-seed mean ≥ Plan A 对照 3-seed mean, std ≤ 1.5% | B1, B2 |

---

## Paper Storyline

### Main paper must prove
1. **Fixed ETF 替换 Linear 训练端有效** (C1): Office 3-seed R200 SGPA vs Plan A 统计显著
2. **ETF 本身是 gain 来源** (C2): Linear+whitening 控制组证明不是副作用
3. **SGPA 推理端有增量** (C3): proto_vs_etf_gain > 0

### Appendix can support
- Neural Collapse 诊断时间序列 (etf_align / inter_cls_sim / client_center_var / param_drift 演进曲线)
- PACS 验证 (如果时间允许,EXP-098)
- τ_etf sensitivity sweep (0.05 / 0.1 / 0.5)

### Experiments intentionally cut
- ❌ DomainNet (训练成本过高,不在 scope)
- ❌ 3+ class skew variants (我们 focus feature skew, 不 tackle label skew)
- ❌ Foundation model backbone (proposal 主线 scratch CNN)

---

## Experiment Blocks

### Block B1: SGPA Office R200 3-seed 主结果 (MUST-RUN)

- **Claim tested**: C1 (ETF 训练端有效) + C4 (不是 seed=2 运气)
- **Why this block exists**: EXP-096 smoke 只跑 seed=2 R50, 可能单 seed 运气 (历史教训: EXP-075 peak 81.7% 后崩 51.2%)
- **Dataset / split / task**: Office-Caltech10, 4 clients (Amazon/Webcam/DSLR/Caltech), 10 classes, feature skew
- **Compared systems**:
  - **Plan A orth_only** (EXP-083 基线, 3-seed mean AVG 82.55)
  - **SGPA (ours, use_etf=1)** — 主方法
- **Metrics**:
  - 主要: AVG Best / AVG Last / ALL Best / ALL Last (3-seed mean)
  - 次要: 3-seed std, drop (Best - Last), 各 domain per-client accuracy
  - 诊断 (Layer 1/2): etf_align / inter_cls_sim / client_center_var / param_drift R200 trajectories
- **Setup details**:
  - Backbone: AlexNet encoder + 双 128d 头 (与 EXP-083/084 一致)
  - Seeds: {2, 15, 333} (对齐 EXP-083/084/096)
  - R200 × E=1, LR=0.05, decay=0.9998, λ_orth=1.0, τ_etf=0.1
  - warmup_r=10, eps_sigma=1e-3, min_clients_whiten=2
  - **diag=1** (记录 Layer 1+2 全量诊断)
- **Success criterion**: SGPA 3-seed mean AVG Best ≥ 84% 且 ≥ Plan A 82.55% + 1.5% (同 seed 对比)
- **Failure interpretation**:
  - 若 3-seed mean 跌到 82-83% → "smoke test R50 84.98% 是 seed=2 运气", 方案不成立
  - 若某 seed 崩 (drop > 5%) → ETF 不稳, 可能需要 warmup / τ 调整
- **Table / figure target**: 主表 Table 1 Office, 诊断曲线 Fig 2
- **Priority**: **MUST-RUN**

### Block B2: Linear 对照 Office R200 3-seed (MUST-RUN)

- **Claim tested**: C2 (ETF 本身 vs 副作用)
- **Why this block exists**: C1 即使成立, 也要排除 gain 来自 pooled whitening / class_centers / 别的辅助
- **Dataset / split / task**: 同 B1
- **Compared systems**:
  - **Linear+whitening (use_etf=0)** — 对照: 保留 SGPA 所有基础设施 (whitening / style stats / diag), 唯一变量是 classifier 换回 `nn.Linear(128, 10)`
- **Metrics**: 同 B1
- **Setup details**: 同 B1, 但 `use_etf=0` → `self.head = nn.Linear(128, 10)`; Linear 参加 FedAvg 聚合
- **Success criterion (for C2)**: Linear+whitening 3-seed mean AVG Best ≤ 83% 且比 SGPA 至少低 1% (证明 ETF 本身贡献)
- **Failure interpretation**:
  - 若 Linear+whitening ≈ SGPA → ETF 没用, gain 来自别处 (whitening 副作用?), 方案要重思考
  - 若 Linear+whitening ≈ Plan A 82.55% (没显著变化) → ETF 是真贡献,✓
- **Table / figure target**: 主表 Table 1 控制列
- **Priority**: **MUST-RUN**

### Block B3: SGPA 推理独立 script (MUST-RUN, 零 GPU 成本)

- **Claim tested**: C3 (SGPA 推理端 proto_vs_etf_gain > 0)
- **Why this block exists**: flgo 默认 test 走 `model.forward()` = ETF argmax, 绕过 `test_with_sgpa`. 要验证论文真正的 dominant contribution (双 gate + proto 校正), 必须独立 script
- **Dataset / split / task**: 同 B1, 但使用 EXP-096 已有的 R50 checkpoint (无需重训)
- **Compared systems**:
  - ETF argmax (baseline, smoke test 已测)
  - SGPA 双 gate + top-m proto (主方法)
  - 消融: entropy-only gate / dist-only gate / no-gate T3A 式
- **Metrics** (Layer 3 全量 13 指标):
  - 主要: `proto_acc / etf_acc / proto_vs_etf_gain`
  - 次要: `reliable_rate / fallback_rate / entropy_rate / dist_rate`
  - 诊断: `dist_distribution / whitening_reduction / sigma_cond / proto_fill / proto_etf_offset`
- **Setup details**:
  - 独立 script `FDSE_CVPR25/scripts/run_sgpa_inference.py`
  - 加载 EXP-096 smoke test R50 checkpoint (test set already routed)
  - 对每个 client test set 跑 `test_with_sgpa`, 收 Layer 3 diag
  - 0 GPU-hours (CPU 足够)
- **Success criterion**: proto_vs_etf_gain mean ≥ +0.5% across 4 clients
- **Failure interpretation**:
  - 若 gain ≈ 0 → SGPA 推理无效, ETF argmax 已足够, 论文叙事要弱化 SGPA 强化 ETF
  - 若 gain < 0 → SGPA 拖后腿 (proto 被脏样本污染), 需调 τ_H/τ_S/m_top
- **Table / figure target**: 主表 Table 2 SGPA ablation
- **Priority**: **MUST-RUN** (零成本, 必做)

### Block B4: PACS 3-seed R200 (NICE-TO-HAVE, 等 GPU)

- **Claim tested**: C1 + C2 在 4-outlier PACS (更难) 场景
- **Why this block exists**: Office 是 single-outlier (DSLR 少样本), PACS 是 4-outlier (全域), 更 stress test. 理论上 K=7 单纯形下界 -1/6=-0.167 比 Office -0.111 更宽, ETF 收益可能更大
- **Dataset / split / task**: PACS_c4, 4 clients (Photo/Art/Cartoon/Sketch), 7 classes
- **Compared systems**: SGPA + Linear 对照 (同 B1/B2)
- **Setup details**: E=5 (PACS 惯例), 其他同 B1
- **Success criterion**: SGPA 3-seed mean AVG Best ≥ 82.5% (Plan A 82.31% + 0.5%)
- **Priority**: **NICE-TO-HAVE** (等 SCPR v2 释放 GPU 后再部署 EXP-098)
- **Table / figure target**: 主表 Table 1 PACS

### Block B5: Neural Collapse 诊断时间序列 (APPENDIX)

- **Claim tested**: 机制解释 — 为什么 ETF > Linear
- **Why this block exists**: 不是新 claim, 是对 C1/C2 的 root-cause 说明. 用 EXP-096/097 的 Layer 1/2 jsonl 直接画曲线
- **Metrics**: etf_align / inter_cls_sim / client_center_var / param_drift vs round
- **Setup**: offline 分析 script 读 jsonl 画图
- **Priority**: **NICE-TO-HAVE** (等 EXP-097 数据齐了画)
- **Table / figure target**: Fig 2 Neural Collapse evolution

### Block B6: τ_etf sensitivity (CUT, 除非 reviewer 问)

- 不影响 main paper 核心 claim, 作为 Appendix 选项保留

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0** Sanity | 代码改动 (use_etf flag) 通过单测 + codex review | N/A | 105 单测全绿 + codex READY/MINOR | 0 GPU-h | 低 |
| **M1** 并行部署 6 runs | B1+B2 Office × 6 runs (SGPA×3 + Linear×3) | R001-R006 | 全部跑完 R200 无 crash | ~3-4 GPU-h (6 并行 1.5-2h 实际 wall) | 中 (Linear 可能也 85%, C2 失败) |
| **M2** EXP-099 SGPA 推理 | B3 (0 GPU, 立即可做) | R007 (CPU) | proto_vs_etf_gain 可报 | 0 GPU-h | 低 |
| **M3** Decision | 根据 M1+M2 verdict 决定是否扩 PACS | N/A | C1+C2+C3 至少 2/3 支持 | N/A | — |
| **M4** PACS (条件性) | B4 PACS × 6 runs | R008-R013 | 等 SCPR v2 结束 (~1h) | ~25 GPU-h (AlexNet E=5) | 中 (4-outlier 更难) |
| **M5** Polish | B5 诊断曲线 + 回填所有 NOTE | N/A | — | 0 GPU-h | 低 |

### 关键决策门

- **M1 后**:
  - C1 SGPA ≥ 84% ✓ + C2 Linear ≤ 83% ✓ → 方案成立, 进 M2/M4
  - C1 ✓ + C2 ✗ (Linear 也 85%) → ETF 没贡献, kill SGPA, 重思考
  - C1 ✗ (SGPA ≤ 83%) → smoke 84.98% 是 seed 运气, kill 全部, 回归 Plan A + SAS
- **M2 后**:
  - C3 proto_vs_etf_gain ≥ 0.5% → 论文 dominant claim 成立
  - C3 < 0 → SGPA 推理反而 hurt, 论文要弱化 SGPA 强化 ETF-only

---

## Compute and Data Budget

- **Total estimated GPU-hours**: ~30 (M1 4h + M4 25h + buffer 1h)
- **并行策略**: seetacloud2 24GB 4090 单卡 + CPU 8 core
  - Office R200 × 6 并行: 显存 6×1.5 = 9GB, CPU 可能瓶颈但可接受 (EXP-095 经验 15 并行 OK)
  - 当前 SCPR v2 占 4GB, 与我们 6 runs 兼容
- **Data preparation**: 无新增 (Office/PACS task 已就绪)
- **Biggest bottleneck**: CPU 数据加载在 6 并行时变慢 (单 run R200 ~1h → 6 并行 ~2h wall)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| 新代码 use_etf 路径未测试, deploy 时崩溃 | M0 必须 105 单测 + codex review + 语法检查, 新增 Linear 路径专项单测 |
| 某 seed 训练崩溃 NaN (EXP-078d 教训) | 每 5 min log check, 若 NaN 立即 kill + 诊断 |
| 6 runs 并行 CPU 瓶颈导致远超 1h | 先部署 2-3 runs 验证速度, 再扩到 6 |
| Linear 对照也 85% → C2 失败 | 预期结果,写 NOTE 时诚实报 verdict, 方案回归 "orth_only + whitening" (不需要 ETF) |
| EXP-096 smoke 结果 84.98% 不可复现 (版本差异) | 新代码的 diag 路径可能轻微影响 RNG 顺序, 但 v2/v3 AVG Best 84.47% vs 84.98% 差 0.5% 是可接受非确定性 |
| SCPR v2 PACS 还在跑, 占 CPU | 先看 SCPR v2 剩余进度再决定 Office 并行度 |

---

## Final Checklist

- [ ] Main paper tables 被 B1+B2+B3 覆盖 (Office 主表 + SGPA ablation 表)
- [ ] Novelty 被 B2 Linear 对照隔离
- [ ] Simplicity 被 "use_etf flag 单一变量改动" 体现
- [ ] Frontier contribution (Neural Collapse + TTA + disentanglement) 被 B3 + B5 证明
- [ ] Nice-to-have (B4 PACS, B5 诊断曲线) 清楚分离
- [ ] 代码改动在 deploy 前通过 codex review (MAJOR 及以上全修)
- [ ] 每个 EXP 目录有 NOTE.md (EXP-095 格式)
- [ ] 实验结果按 EXP-095 格式回填 (claim → 结果表 → Δ 行 → verdict)
