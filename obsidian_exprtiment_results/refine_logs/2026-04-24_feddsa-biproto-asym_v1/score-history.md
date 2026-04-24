# Score Evolution — FedDSA-BiProto

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1     | 8 | 6 | 6 | 8 | 5 | 6 | 6 | **6.5** | REVISE |
| 2     | 9 | 8 | 7 | 9 | 7 | 6 | 7 | **7.8** | REVISE |
| 3     | 9 | 8 | 8 | 9 | 8 | 8 | 7 | **8.25** | REVISE |
| 4     | 9.0 | 9.0 | 8.5 | 9.0 | 8.5 | 9.0 | 8.0 | **8.75** | REVISE |

**Trajectory**: 6.5 → 7.8 → 8.25 → 8.75 (**+2.25 累计**, 距 READY bar 9.0 还差 0.25)

## Dimension-level Movement

| Dimension | Start | End | Total Δ | Driver |
|---|:-:|:-:|:-:|---|
| Problem Fidelity | 8 | 9.0 | +1.0 | anchor 保持, R1 略扣, R2+ 见顶 |
| Method Specificity | 6 | 9.0 | +3.0 | gradient-flow table + EMA 公式 + ST 形式 |
| Contribution Quality | 6 | 8.5 | +2.5 | L_sem_proto 删除 + Pc 降级 + "to our knowledge" |
| Frontier Leverage | 8 | 9.0 | +1.0 | appropriate conservatism 接受 |
| Feasibility | 5 | 8.5 | +3.5 | stage-gated 预算 + present-classes-only 简化 |
| Validation Focus | 6 | 9.0 | +3.0 | C0 fix + −Pd MANDATORY + 3-suite visual |
| Venue Readiness | 6 | 8.0 | +2.0 | sharper 叙事, 但 D=K=4 empirical 天花板 |

## Stop Reason

**用户选择在 R4 (8.75) 停止, 理由 reviewer-confirmed**: "Remaining gap is not a design flaw; it is an empirical question (whether Pd shows real value beyond local domain-axis surrogate on D=K=4). Further method refinement has diminishing returns — only actual −Pd ablation can close the gap."

方案已达 near-READY, 适合进入 /experiment-plan + 代码实现阶段.
