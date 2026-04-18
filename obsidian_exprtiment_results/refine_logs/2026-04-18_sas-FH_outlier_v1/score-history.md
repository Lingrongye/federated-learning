# 得分演化 — Outlier v1 (sas-FH)

| 轮次 | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|-------|-----|-----|-----|-----|------|-----|-----|---------|---------|
| 1     | 9   | 9   | 8   | 8   | 9    | 6   | 6   | **8.3** | REVISE  |
| 2     | 10  | 9   | 9   | 9   | 9    | 8   | 8   | **9.0** | REVISE  |
| 3     | 10  | 10  | 9   | 9   | 9    | 8   | 8   | **9.1** | REVISE  |
| 4     | 10  | 10  | 9   | 9   | 9    | 9   | 9   | **9.4** | **READY** |

## 权重
PF 15% · MS 25% · CQ 25% · FL 15% · Feas 10% · VF 5% · VR 5%

## 演化
- **R1 (8.3)**：验证缺少 counterfactual 分离；主线过于"scope 微调"。
- **R2 (9.0)**：加入 C1/C2 + swap 诊断；把主线收紧为"routing signal"；把 PACS 降级。
- **R3 (9.1)**：精确的更新规则；同 round swap；Claim 3 矩阵；`S` 参与集写明。
- **R4 (9.4) READY**：C2 无条件 3-seed；统计围绕 effect size + per-seed 一致性；C1 次要；热力图折叠进证据段落。
