# Experiment Plan — SCPR(Self-Masked Style-Weighted Multi-Positive InfoNCE)

**Problem**:跨域联邦学习 FedDSA Share 章节从未落地,M3 / SAS 各自不普适
**Method Thesis**:M3 等权 multi-positive InfoNCE → 按客户端风格相似度加权 + self-mask
**Date**:2026-04-19
**Based on**:`FINAL_PROPOSAL.md`(Refine READY 9.1/10)

---

## Claim Map

| Claim | 关键性 | 最小说服证据 | 关联 Blocks |
|-------|--------|-------------|------------|
| **C-A** SCPR > M3 uniform on PACS 全 outlier | 主(novelty isolation) | 同一代码路径、同 seed 下 SCPR(τ=0.3)AVG Best 3-seed mean ≥ M3 + 0.5% | B1, B2 |
| **C-B** SCPR ≥ SAS on Office 单 outlier 且普适 PACS | 主(anchor fix) | 同 seed 下 SCPR AVG Best ≥ SAS,且 PACS 未退化 | B1 |
| **C-C** 风格加权机制**在训练中真的被激活** | 机制诊断 | Spearman ρ(iso_k, gain_k) > 0(非 tautological)+ τ 敏感曲线呈 U 形 | B3, B5 |
| **Anti-claim 排除 1** 增益不是来自"更多参数" | 零新 trainable 自然排除 | 方法已 "0 new trainable, 1 new hyperparam",无需额外 ablation | — |
| **Anti-claim 排除 2** SCPR 不会在 PACS 崩(SAS 的坑) | 内置安全 | uniform-w 数学等价 M3(下界)+ A.3 ≥ A.2 | B1 |

---

## Paper Storyline

- **主 paper 必须证明**:
  1. Claim A(PACS,SCPR > M3)= 主表 1 一行
  2. Claim B(Office,SCPR ≥ SAS)= 主表 2 一行 + 普适性小结
  3. Claim C(τ 敏感性 + 机制激活 ρ)= 机制 figure
- **附录可支持**:
  - SCPR + SAS composability on Office
  - τ 扫描完整曲线
  - Attention heatmap per-client
- **故意砍掉**(不做实验):
  - SCPR + SAS 在 PACS(SAS 在 PACS 失效,无意义)
  - `ρ(w, -style_dist)` tautological 诊断(R4 reviewer 否掉)
  - grad_CE vs grad_InfoNCE cos_sim 监控(无关主 claim)
  - 其他 τ_nce / λ_sem 超参扫(Plan A 默认已最优)

---

## Experiment Blocks

### Block 1 — Main Anchor Result(Claim A + B)

- **Claim tested**:A + B
- **Why this block exists**:证明 SCPR 在两个数据集上同时超越最强 in-house baseline
- **Datasets**:
  - PACS(4 domains, 7 classes, AlexNet-from-scratch)
  - Office-Caltech10(4 domains, 10 classes, ResNet-18)
- **Compared systems**(按**同一代码路径、同 seed** 原则):
  - **A.1**:FedDSA orth_only(scpr=0, sas=0)→ Plan A baseline
  - **A.2**:FedDSA + M3 multi-pos(scpr=1, sas=0)→ uniform attention 下界
  - **A.3**:FedDSA + SCPR(scpr=2, scpr_tau=0.3, sas=0)→ 主方法
  - **B.1**:FedDSA orth_only(scpr=0, sas=0)= PACS A.1
  - **B.2**:FedDSA + SAS(scpr=0, sas=1, sas_tau=0.3)→ 参数路由基线
  - **B.3**:FedDSA + SCPR(scpr=2, scpr_tau=0.3, sas=0)→ 主方法
- **Metrics**(必收):
  - ALL Best / ALL Last(sample-weighted)
  - AVG Best / AVG Last(client-equal mean)
  - Per-domain Best / Last(Art / Cartoon / Photo / Sketch;Amazon / Caltech / DSLR / Webcam)
  - Best@Round(记录 best 出现在第几轮)
  - 3-seed mean ± std
- **Setup**:
  - PACS:R=200, E=5, B=50, LR=0.05, seeds={2, 15, 333}
  - Office:R=200, E=1, B=50, LR=0.05, seeds={2, 15, 333}
  - Plan A 基础超参不变(lo=1.0, lh=0.1, ls=1.0, tau=0.3, warmup 来自 Plan A 的配置)
  - FedBN 保留,正交解耦保留
- **Success criterion**:
  - Claim A:PACS A.3 AVG Best 3-seed mean ≥ 81.5% **且** A.3 ≥ A.2 + 0.5%
  - Claim B:Office B.3 AVG Best 3-seed mean ≥ 90.5% **且** B.3 ≥ B.2
- **Failure interpretation**:
  - A.3 < A.2 → style weighting 在 PACS 无用,回到 M3 作为论文 fallback 主张(缩为 "在 Office 上的贡献")
  - A.3 < 81.5% 但 ≥ A.2 → 机制方向对,但 gain 不足,需要加更深度的风格信号(z_sty 双层代替单层均值)
  - B.3 < B.2 → 参数路由仍优于原型路由(反常,需要深入查 attention entropy 和 outlier 加权分布)
- **Paper target**:
  - PACS 主表(Table 1)含 FedAvg / FedProx / FedBN / FedProto / FPL / FedPLVM / FDSE / Plan A / M3 / **SCPR**
  - Office 主表(Table 2)含同样 baseline 清单 + SAS / **SCPR**
- **Priority**:**MUST-RUN**

---

### Block 2 — Novelty Isolation(SCPR vs M3)

- **Claim tested**:Claim A 的严格 novelty 隔离
- **Why this block exists**:**这是 Claim A 的核心**,必须在完全相同代码路径、相同 seed 下对比 scpr=1(uniform)vs scpr=2(style-weighted)。避免跨文件、跨架构的污染对比(EXP-072 的 M3 是 feddsa_adaptive 的 128d z_sem,不可直接引用)
- **Dataset**:PACS
- **Compared**:
  - A.2 scpr=1(uniform multi-pos)vs A.3 scpr=2(style-weighted,τ=0.3)
  - **关键**:同 seed、同 config、唯一差异 = 权重算法
- **Metric**:AVG Best 3-seed mean 差异 Δ、per-seed Δ(验证不是某个 seed 的运气)
- **Success criterion**:Δ = A.3 − A.2 ≥ +0.5%(3-seed mean)
- **Failure interpretation**:style weighting 本身无价值,SCPR 的任何增益都来自"M3 架构 + Plan A 配置"的复现
- **Paper target**:Ablation Table,一行两列(scpr=1 / scpr=2)
- **Priority**:**MUST-RUN**(嵌在 Block 1 的 A.2/A.3 里)

---

### Block 3 — Simplicity Check(SCPR with/without Self-Mask)

- **Claim tested**:self-mask 作为默认设计(R1 reviewer 要求)而非可选项
- **Why this block exists**:Reviewer 可能质问"自掩码真的必要吗";需要一个显式 ablation 堵嘴
- **Dataset**:PACS(代价最低,小规模验证)
- **Compared**:
  - A.3(scpr=2, τ=0.3, **self-mask=ON**,默认)
  - A.3-nosm(scpr=2, τ=0.3, **self-mask=OFF**,即 attention 包含自己)
- **Metric**:AVG Best 3-seed mean
- **Success criterion**:self-mask=OFF 应该**下降**(验证 Share 会退化为 local-only 的假设)
- **Setup**:3 seeds,共 3 runs
- **Priority**:**NICE-TO-HAVE**(作为论文 Ablation Table 的一行;若时间紧可只跑 1 seed 作 qualitative)

---

### Block 4 — Frontier Necessity Check(**故意不做**)

方案明确声明不使用 LLM/VLM/Diffusion/RL。按 skill 规范,intentionally non-frontier 场景应**显式说明跳过**,不强加一个 block。

论文 Related Work 一段话解释:
> We intentionally stay within classical FL prototype alignment. Attention-based retrieval is sufficient as a modern primitive; introducing VLM/CLIP would violate the fair comparison with FedProto/FPL/FDSE baselines and change the research question from "can Share be fixed cleanly" to "does pretraining mask the mechanism."

- **Priority**:**CUT**(不跑 run,只写一段话)

---

### Block 5 — Failure Analysis & Mechanism Diagnosis(Claim C)

- **Claim tested**:Claim C — 风格加权机制在训练中真的被激活
- **Why this block exists**:反驳"SCPR 就是 M3 的 noisy 版本";提供 non-tautological 证据
- **Datasets**:PACS(主诊断)
- **Runs**:
  - τ 扫:scpr=2, scpr_tau ∈ {0.1, 0.3, 1.0, 3.0} × 3 seeds = 12 runs(τ=0.3 复用 A.3)
  - 实际新增 runs:τ ∈ {0.1, 1.0, 3.0} × 3 seeds = **9 runs**
- **核心指标**:
  - **Spearman ρ(iso_k, gain_k)**:
    - `iso_k = 1 − mean_{j ≠ k} cos(s_k, s_j)` 客户端 k 的风格孤立度(post-training)
    - `gain_k = acc_k^{SCPR τ=0.3} − acc_k^{M3 scpr=1}` per-client accuracy 差
    - 预期:**ρ > 0**(outlier 客户端从 SCPR 获益更多)
    - 非 tautological:gain_k 是 downstream accuracy,不是 cos 的函数
  - **H(w_k) attention entropy** 曲线(每轮):
    - 预期:中间 τ 下 H 稳定在 [0.3·log(K−1), 0.85·log(K−1)]
  - **Per-client attention heatmap**:每个 (k, j) 对的 w_{k→j} 平均值
- **Success criterion**:
  - ρ(iso_k, gain_k) > 0(在 4 客户端上用 4 个 (iso, gain) 点算 Spearman,虽然 N 小但方向性足够)
  - τ 扫曲线呈 U 形 / 单峰:τ=0.1 崩(过硬),τ=3.0 退化(接近 uniform),τ ∈ [0.3, 1.0] 最优
- **Failure interpretation**:
  - ρ ≈ 0:机制未激活,`s_k` 区分度不足 → fallback 降维 key
  - τ 扫无 U 形:风格 weighting 本身不敏感 → SCPR 可能只是"M3 + 噪声"的等价
- **Paper target**:
  - Figure 1:Attention heatmap + ρ 相关分析散点图
  - Figure 2:τ 扫曲线(x=τ, y=AVG Best 3-seed mean)
- **Priority**:**MUST-RUN**

---

### Appendix A — SCPR + SAS Composability(非主 claim)

- **Claim tested**:原型层路由(SCPR)与参数层路由(SAS)是否叠加有效
- **Runs**:Office,scpr=2 + sas=1 + τ=0.3 × 3 seeds = **3 runs**
- **Compared**:vs B.2(仅 SAS)/ B.3(仅 SCPR)
- **Metric**:AVG Best 3-seed mean
- **Expected**:B.4 ≥ max(B.2, B.3),Δ ≤ 1%(如果 Δ 大,说明两机制有强叠加;若 ≤ 0,说明两者互斥)
- **Priority**:**NICE-TO-HAVE**(附录 Table)

---

## Run Order and Milestones

| Milestone | Goal | Runs(新增) | Decision Gate | Cost(GPU·h) | Risk |
|-----------|------|------|---------------|------|------|
| **M0 Sanity** | 代码可跑、infra 不坏 | PACS scpr=2 τ=0.3 s=2 × R=20(20 轮快查) | 前 20 轮无 NaN、loss 单调下降、attention entropy 合理 | 0.5 | 启动失败/NaN |
| **M1 Baseline 补齐** | A.2 / B.1 / B.2 在新代码路径下复现 | PACS scpr=1 × 3 seeds<br>Office scpr=0 × 3 seeds(仅确认 Plan A 数据可用,能复用则跳过)<br>Office sas=1 × 3 seeds(复用 EXP-084 数据,不重跑) | A.2 3-seed mean ≥ 78.5%(M3 在 1024d z_sem 下的预期) | 6 | M3 在新架构下数字比 81.91% 低很多(EXP-072 是 128d) |
| **M2 主方法** | Claim A + B 主结果 | **PACS scpr=2 τ=0.3 × 3 seeds**<br>**Office scpr=2 τ=0.3 × 3 seeds** | PACS A.3 ≥ 81.5% **且** A.3 ≥ A.2 + 0.5% | 12 | SCPR < M3(方法证伪,需降级方案) |
| **M3 机制** | Claim C + τ 扫 | PACS scpr=2 τ ∈ {0.1, 1.0, 3.0} × 3 seeds = 9 runs | τ 扫出现 U 形;ρ > 0 | 18 | τ=0.1 全 NaN |
| **M4 附录** | SCPR+SAS composability | Office scpr=2 + sas=1 × 3 seeds | Δ vs max(B.2, B.3) 任意方向清晰即可 | 6 | 与 M2 Office 冲突(共享 GPU) |
| **M5 Polish** | 离线诊断 + attention 可视化 | 无新 run;跑 analyze 脚本 | ρ 算出来 > 0 | 0 | — |

**总 GPU·h ≈ 42.5**(低于原预算 60,因为复用了 Plan A / SAS 现有数据)。

---

## 精确 Config 清单(algo_para 顺序)

> `feddsa_scheduled.py` 当前 algo_para 顺序:`[lo, lh, ls, tau, sdn, pd, sm, bp, bw, cr, gli, lm, al, se, sas, sas_tau]`
>
> **Phase 2-A 需新增**:`scpr`(第 17 位)、`scpr_tau`(第 18 位)
> **Phase 2-A 要兼容**:现有所有 config 不受影响(scpr 默认 0 → 走原 orth_only 路径)

### PACS configs(AlexNet)

**共享 base**:`lo=1.0, lh=0.1, ls=1.0, tau=0.3, sdn=5, pd=128, sm=0, bp=60, bw=30, cr=80, gli=0, lm=1.0, al=0.25, se=1, sas=0, sas_tau=0.3`

| Config file | scpr | scpr_tau | 用途 |
|-------------|------|----------|------|
| `pacs/feddsa_plan_a.yml`(**已有** EXP-080) | 0 | 0.3 | A.1 baseline |
| `pacs/feddsa_scpr_uniform.yml` | 1 | 0.3 | A.2(M3 in-codepath) |
| `pacs/feddsa_scpr_tau03.yml` | 2 | 0.3 | A.3 主方法 |
| `pacs/feddsa_scpr_tau01.yml` | 2 | 0.1 | Block 5 τ 扫 |
| `pacs/feddsa_scpr_tau10.yml` | 2 | 1.0 | Block 5 τ 扫 |
| `pacs/feddsa_scpr_tau30.yml` | 2 | 3.0 | Block 5 τ 扫 |
| `pacs/feddsa_scpr_nosm.yml` | 2* | 0.3 | Block 3(需实现 nosm flag, MINOR) |

### Office configs(ResNet-18)

**共享 base**:`lo=1.0, lh=0.1, ls=1.0, tau=0.3, sm=0, sas=0, sas_tau=0.3`(其他同上)

| Config file | scpr | sas | 用途 |
|-------------|------|-----|------|
| `office/feddsa_plan_a.yml`(**已有** EXP-083) | 0 | 0 | B.1 baseline |
| `office/feddsa_sas_tau03.yml`(**已有** EXP-084) | 0 | 1 | B.2 SAS baseline |
| `office/feddsa_scpr_tau03.yml` | 2 | 0 | B.3 主方法 |
| `office/feddsa_scpr_sas.yml` | 2 | 1 | Appendix composability |

---

## 启动命令模板(对齐 CLAUDE.md 17.4.2)

```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
EXP_DIR=../../experiments/ablation/EXP-095_scpr

# 确保输出目录
mkdir -p $EXP_DIR/results $EXP_DIR/logs

# 示例:PACS SCPR τ=0.3 s=2
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_scheduled --gpu 0 \
  --config ./config/pacs/feddsa_scpr_tau03.yml --logger PerRunLogger --seed 2 \
  > $EXP_DIR/terminal_pacs_scpr_tau03_s2.log 2>&1 &
```

### 最大并行策略(seetacloud2 单卡 24GB)

**显存预估**:
- PACS AlexNet E=5:~4-6GB/run → 并行 **3-4 runs**
- Office ResNet-18 E=1:~10-12GB/run → 并行 **2 runs**

**Wall-clock 估算**(按 EXP-083/084 经验):
- PACS 单 run R=200:~1.5h
- Office 单 run R=200:~1.5h

**执行批次**(单卡,顺序):
1. **批 1**(PACS 主 A.2 + A.3):6 runs 并行 3 → 2 批 × 1.5h = 3h
2. **批 2**(Office 主 B.3):3 runs 并行 2 → 2 批 × 1.5h = 3h
3. **批 3**(PACS τ 扫):9 runs 并行 3 → 3 批 × 1.5h = 4.5h
4. **批 4**(Appendix SCPR+SAS):3 runs 并行 2 → 2 批 × 1.5h = 3h

**总 wall-clock ≈ 13-15h**(连续跑 1 天内完成)

### 若 seetacloud 恢复可并行

- seetacloud → PACS 所有 runs(15 runs)
- seetacloud2 → Office 所有 runs(6 runs)
- 总 wall-clock 压到 **7-8h**(约半天)

---

## collect_results.py 调用

```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
python collect_results.py --exp EXP-095 --task PACS_c4 --algorithm feddsa_scheduled --seed 2
python collect_results.py --exp EXP-095 --task PACS_c4 --algorithm feddsa_scheduled --seed 15
python collect_results.py --exp EXP-095 --task PACS_c4 --algorithm feddsa_scheduled --seed 333
# Office 同理
```

---

## NOTE.md 模板骨架

```markdown
# EXP-095: SCPR — Self-Masked Style-Weighted Multi-Positive InfoNCE

**日期**: 2026-04-19 启动
**算法**: feddsa_scheduled (scpr=2, sas=0)
**服务器**: seetacloud2
**状态**: 🔄 运行中

## 变体通俗解释

> 把 M3 "所有同类域原型等权拉近"改成"按风格相似度加权拉近",自己不算。
> 数学上 τ→∞ 等价 M3;τ=0.3 让风格近的域原型权重高(清晰信号),风格远的权重低(压噪声)。

## 技术细节
- SCPR loss = self-masked style-weighted SupCon multi-positive InfoNCE
- Style key: s_k = L2-normalized mean of z_sty on client k's train set
- Per-class renormalize: w_{k→j}^c 只在有 p_c^j 的 client 上归一化
- Gradient detach: p_c^j 和 s_j 都 .detach() (不回传 gradient 到 bank)

## 配置
| 参数 | 值 |
|------|-----|
| scpr | 2 (style-weighted) |
| scpr_tau | 0.3 |
| sas | 0 (SCPR-only) |
| sm | 0 (orth_only) |
| lo/lh/ls | 1.0/0.1/1.0 (Plan A) |
| tau_nce | 0.3 |
| LR | 0.05 |
| R | 200 |
| seeds | 2, 15, 333 |

## 🏆 结果槽位(待回填)

### Claim A: PACS 全 outlier, SCPR > M3 uniform

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Art | Cartoon | Photo | Sketch |
|------|------|---------|---------|---------|---------|-----|---------|-------|--------|
| **A.1 orth_only** (baseline, EXP-080) | 2   | — | — | — | — | — | — | — | — |
|                                         | 15  | — | — | — | — | — | — | — | — |
|                                         | 333 | — | — | — | — | — | — | — | — |
|                                         | **mean** | — | — | — | — | — | — | — | — |
| **A.2 M3 uniform** (scpr=1)            | 2   | — | — | — | — | — | — | — | — |
|                                         | 15  | — | — | — | — | — | — | — | — |
|                                         | 333 | — | — | — | — | — | — | — | — |
|                                         | **mean** | — | — | — | — | — | — | — | — |
| **A.3 SCPR τ=0.3** (scpr=2, OURS)      | 2   | — | — | — | — | — | — | — | — |
|                                         | 15  | — | — | — | — | — | — | — | — |
|                                         | 333 | — | — | — | — | — | — | — | — |
|                                         | **mean** | — | — | — | — | — | — | — | — |
| **Δ A.3 − A.2**                         | —   | — | — | — | — | — | — | — | — |
| **Δ A.3 − A.1**                         | —   | — | — | — | — | — | — | — | — |

### Claim B: Office 单 outlier, SCPR ≥ SAS

| 配置 | seed | ALL Best | ALL Last | AVG Best | AVG Last | Caltech | Amazon | DSLR | Webcam |
|------|------|---------|---------|---------|---------|---------|--------|------|--------|
| **B.1 orth_only** (EXP-083)            | 2/15/333 | — | — | — | — | — | — | — | — |
| **B.2 SAS τ=0.3** (EXP-084)            | 2/15/333 | — | — | — | — | — | — | — | — |
| **B.3 SCPR τ=0.3** (OURS)              | 2/15/333 | — | — | — | — | — | — | — | — |
| **Δ B.3 − B.2**                         | —   | — | — | — | — | — | — | — | — |

### Claim C: τ 敏感性 + outlier-ness 机制诊断

| τ_SCPR | PACS AVG Best 3-seed mean | H(w) 训练末均值 | Spearman ρ(iso_k, gain_k) |
|--------|---------------------------|----------------|--------------------------|
| 0.1 | — | — | — |
| 0.3 | — (= A.3) | — | — |
| 1.0 | — | — | — |
| 3.0 | — | — | — |

### 附录: SCPR + SAS Composability (Office)

| 配置 | AVG Best 3-seed mean | Δ vs SAS only | Δ vs SCPR only |
|------|---------------------|---------------|----------------|
| B.2 SAS only | — | — | — |
| B.3 SCPR only | — | — | — |
| SCPR + SAS | — | — | — |

## 结论

(实验完成后填)
```

---

## Fallback 策略

| 情况 | 信号 | Fallback 动作 |
|------|------|-------------|
| **启动失败**(部署 1 分钟内 log 无输出 / 进程不存在) | `ps aux \| grep run_single` 无结果 | 检查 config 语法、ast.parse 错误、algo_para 顺序 |
| **立即 OOM**(部署 5 分钟内 CUDA OOM) | log 有 "CUDA out of memory" | 减少并行度 3→2 或 2→1,重启 |
| **NaN loss**(前 20 轮内) | log 有 "loss=nan" 或 "inf" | 先 grep 定位 round,若 scpr_tau=0.1 → 升到 0.3;若 scpr_tau=0.3 → 检查 style_bank 初始化 |
| **Attention uniform 塌缩**(H(w_k) ≈ log(K−1) 持续 20+ 轮) | 跑完后离线算 H(w) 曲线 | 降 s_k 维度 / 换 attention key(仍无新参数)/ 调 τ 到 0.1 |
| **PACS A.3 < A.2**(3-seed mean) | collect 后发现 | **立即停手,降级方案**:论文主 claim 只保留 Office(B.3 ≥ B.2),PACS 放 negative result 附录 |
| **Office B.3 < B.2** | collect 后发现 | 检查 attention entropy 是否极低(τ 过硬);尝试 τ=1.0 替代 0.3 |

---

## 监控节奏

| 时刻 | 动作 | 触发条件 |
|------|------|--------|
| **T+1min** | SSH 查 ps aux \| grep run_single(进程数 = 并行度);nvidia-smi 显存 ≥ 4GB | 进程数不对或显存接近 0 → 启动失败,看 log |
| **T+5min** | tail log 前 10 轮,检查有无 "Round 1 / Round 2 / ..." 正常递增,无 NaN | 无 round 输出 → infra 问题;NaN → fallback 处理 |
| **T+1h** 起 | 每小时:`ls task/*/log/` 最新 log 末行,抽查 3 个 run 的 round | 某 run round 数长时间不变 → 进程卡死/OOM;log 有 "Round 200" → 完成 |
| **完成后** | collect_results.py 提 JSON,回填 NOTE.md | — |

---

## Compute and Data Budget

- **总 GPU·h**:~42.5(新增 runs)
- **服务器**:seetacloud2 主力(确认 0% util 空闲);seetacloud 恢复后可分担 PACS 部分
- **数据**:PACS + Office-Caltech10 均已就位(FDSE_CVPR25/task/)
- **最大 bottleneck**:seetacloud2 单卡时,总 wall-clock ~13-15h(约一个工作日)

---

## Risks and Mitigations

| Risk | 概率 | 影响 | Mitigation |
|------|------|------|-----------|
| **M3 在 1024d z_sem 下表现不如 EXP-072 的 128d 81.91%** | 高 | A.2 可能只有 78-80%,导致 A.3 ≥ A.2 的门槛下降 | 不 cite EXP-072 的 81.91%;只在新代码路径里定义 A.2 基线 |
| **SCPR < M3** | 中 | Claim A 证伪 | 提前设计好降级叙事(论文主 claim 变成 Office 专属) |
| **Office SCPR < SAS** | 中 | Claim B 证伪 | 查 attention entropy;若 τ 过硬,尝试 τ=1.0;若仍不行,论文降级到"普适性而非性能优势" |
| **Attention uniform 塌缩(4 clients 下 s_k 区分度不足)** | 中 | Claim C 证伪 | 用 style_head 投影到低维(仍无新参数) |
| **seetacloud 长期不恢复** | 中 | wall-clock ×2 | 只用 seetacloud2,顺序跑,1 天之内仍可完成 |
| **首次启动 bug**(algo_para 顺序错) | 低 | 部署失败 | 1-分钟监控立即发现;ast.parse + 单测在部署前拦截 |
| **git 同步失败** | 低 | 服务器跑旧代码 | 部署前强制 `git pull --no-rebase` 确认 Fast-forward |

---

## Final Checklist(执行前验收)

- [ ] 主 paper Table 1(PACS)含 FedAvg / FedBN / FedProto / FPL / FDSE / Plan A / M3 / **SCPR** 一列
- [ ] 主 paper Table 2(Office)含同样 baseline + SAS / **SCPR** 一列
- [ ] Novelty 隔离:Block 2 = Block 1 里的 A.2/A.3 直接对比
- [ ] 简洁性证明:Block 3(self-mask ablation)+ 方法已 0 新 trainable 自证
- [ ] Frontier 明确不用:Related Work 一段话说明
- [ ] Figure 1(attention heatmap + ρ 散点图)+ Figure 2(τ 扫曲线)
- [ ] 附录 composability 一行小表
- [ ] must-run(M0-M3)和 nice-to-have(M4 附录 + Block 3)分清
- [ ] seed={2, 15, 333} 在所有新 runs 中严格对齐
- [ ] 对照行数据齐全(SSH 从 seetacloud2 task record 提 A.1/B.1/B.2 per-seed JSON)
- [ ] 全指标槽位(ALL B/L + AVG B/L + per-domain)已在 NOTE.md 模板里

---

## Next Immediate Steps

1. Phase 1-B:SSH 到 seetacloud2 提取 Plan A (EXP-083) 和 SAS (EXP-084) 的 per-seed JSON,准备对照行
2. Phase 2-A:动手改 `feddsa_scheduled.py`,新增 `scpr`、`scpr_tau` 超参和 SCPR loss 分支
3. Phase 2-B/C/D/E:自审 → codex 审 → 修 issue → 单测
4. Phase 2-F/G:ast.parse + 单测 ALL PASS → 写 config YAML
5. Phase 3:创建 EXP-095 目录、git 同步、部署并行启动
6. Phase 4/5:1-分钟初检 → 每小时监控 → collect → 回填 NOTE.md
