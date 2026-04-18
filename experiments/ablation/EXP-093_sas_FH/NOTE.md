# EXP-093 | sas-FH (Style-Conditioned Classifier Routing) Triage

## 基本信息
- **日期**: 2026-04-18 启动
- **算法**: feddsa_scheduled (sm=0, orth_only, sas 扩展到 0..4)
- **服务器**: SC2 GPU 0（单卡串行）
- **状态**: 🔄 triage 启动中

## 动机

4 轮 GPT-5.4 research-refine 评审（8.3 → 9.0 → 9.1 → 9.4 **READY**）得出方案：把 Plan A 的 sas 从半头（仅 `semantic_head`）扩展到整 head 链（+ `classifier head`），零新参数、零新损失、零通信量增加。核心 claim：**style similarity is the natural routing signal for classifier-boundary sharing**；通过 A2 vs C2 counterfactual 证明因果性。

## 配置

| 编号 | sas | sem_head | classifier | 说明 |
|------|-----|----------|-----------|------|
| B0 | 0 | FedAvg | FedAvg | EXP-083 已有 3-seed |
| B1 | 1 | sas | FedAvg 按样本加权 | EXP-084 Plan A 已有 3-seed |
| **A2** | **2** | **sas** | **sas 风格条件化** | **OURS (新)** |
| **C2** | **3** | **sas** | **uniform-avg** | **核心 counterfactual** |
| C1 | 4 | sas | fully local | 次级消融 |

共同超参：LR=0.05, R200, E=1, B=50, decay=0.9998, sas_tau=0.3, L_orth=1.0, sm=0。

## Triage 协议

**Step 1：单 seed 2h 快速判**
- Office s=2 × {A2, C1, C2} → 串行 6h
- 关注 R50 / R100 / R150 的 Caltech AVG

**Step 2：决策规则（无 margin 豁免）**
- 若 `AVG_Best(A2) − max(AVG_Best(B1), AVG_Best(C2)) ≥ 0.5`：**A2 和 C2 都升级到 3-seed**
- 若 margin `< 0.5`：**停止**（主论点被伪证）

**Step 3：Mechanism 诊断（零训练）**
加载 A2 best-round checkpoint，跑 `fedavg_head.pt` vs `head_i` 的 swap diagnostic，看 Caltech 是否显著受益。

## 代码改动

- 文件：`FDSE_CVPR25/algorithm/feddsa_scheduled.py`
- 新增逻辑：`sas` flag 从 {0,1} 扩展到 {0,1,2,3,4}
- 关键代码（server pack() 内）：
  ```python
  # A2 (sas=2)
  sas_keys = [k for k in global_dict if k.startswith('semantic_head.')
                                      or k.startswith('head.')]
  per_param_i[k] = Σ_j w_{ij} · client_state[j][k]

  # C2 (sas=3): 
  per_param_i[k] = (1/|S|) · Σ_{j ∈ S} client_state[j][k]

  # C1 (sas=4): 
  per_param_i[k] = client_state[i][k]  # local only
  ```
- 新增：`fedavg_head.pt` 同轮快照保存（供 Claim 2 swap 用）
- 单元测试：`FDSE_CVPR25/tests/test_sas_fh.py` 12/12 通过

## 预期结果（决策点）

**A2 vs baseline (EXP-084)**：
- Caltech Best ≥ 77.0（+2% vs 75.0）
- AVG Best ≥ 90.5（超 FDSE 90.58）

**A2 vs C2（核心 novelty 测试）**：
- `AVG_Best(A2) − AVG_Best(C2) ≥ 0.5%` triage 即可通过
- 最终 3-seed：`A2 − C2 ≥ 1%` + per-seed 一致性

**A2 vs C1**：
- A2 > C1 → 证明分类器应该共享（哪怕 uniform share 也好过完全不 share）

## Triage 实时监控

| Round | 预期 Caltech AVG | A2 判读 |
|-------|-----------------|---------|
| R50 | 60-65（早期不稳） | 训练崩了（< 50）就 kill |
| R100 | 72-75 | < 72 → A2 退化 |
| R150 | 76-79（peak 期） | < 74 → thesis 危险 |
| R200 | 76-78（LR decay 后降） | Best 一般在 R120-R170 |

## 结果（待填）

### A2 (sas=2) single seed s=2
| Round | ALL Best | ALL Last | AVG Best | AVG Last | Caltech Best/Last |
|-------|---------|---------|---------|---------|---|
| R200 | - | - | - | - | - |

### C2 (sas=3) single seed s=2  
| Round | ALL Best | ALL Last | AVG Best | AVG Last | Caltech Best/Last |
|-------|---------|---------|---------|---------|---|
| R200 | - | - | - | - | - |

### C1 (sas=4) single seed s=2
| Round | ALL Best | ALL Last | AVG Best | AVG Last | Caltech Best/Last |
|-------|---------|---------|---------|---------|---|
| R200 | - | - | - | - | - |

### 对照（已有数据）

| 方法 | seeds | AVG Best | AVG Last | Caltech Best/Last |
|------|-------|----------|----------|-------------------|
| baseline (sas=0) | 3 | 88.61 | 87.30 | 72.6/70.2 |
| Plan A (sas=1) | 3 | 89.82 | 88.28 | 75.0/73.8 |
| FDSE | 3 | 90.58 | 89.22 | 78.9/77.7 |

## 决策树

- `A2 AVG Best ≥ 90.5 + A2 − C2 ≥ 0.5` → **扩 3-seed {15, 333}**（+8h），paper 核心结果
- `A2 AVG Best ∈ [89.82, 90.5)` → 边际提升，看 Caltech 改善幅度决定
- `A2 AVG Best < 89.82` → **停止**，用 Claim 2 swap 诊断 classifier 是否 bottleneck

## 下一步

- [ ] codex review 最终通过
- [ ] git commit + push
- [ ] SC2 git pull + clashctl 代理
- [ ] nohup 启动 A2/C1/C2 × s=2
- [ ] 每 30min 监控进度
