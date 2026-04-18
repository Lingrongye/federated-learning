# 精炼报告 — sas-FH outlier 补救

**问题**：FedDSA Plan A 的 sas 只个性化了 `semantic_head`，`classifier head` 仍然走 FedAvg → PACS art 65.69%、Office Caltech 77.68%。错误分析显示 conf ≥ 0.85 的 art 误分类共 70 例，集中在有机类别的混淆上。

**初始方案**：5 个候选（head sas / 多质心 / focal / BN sas / SupCon）。选定 head sas 作为主导方案。

**日期**：2026-04-18
**轮次**：4 / 5
**最终得分**：9.4 / 10
**最终裁决**：✅ **READY**

## 问题锚点（逐字）

- 结论：FedDSA Plan A 在 outlier 域失败的原因是 classifier 被全局 FedAvg。
- 瓶颈：共享的 classifier 平面稀释了 outlier 的决策边界。
- 非目标：不改 backbone、不新增可训练模块、不用推理技巧、不做 per-sample sas。
- 约束：3-seed {2,15,333} × R200 × E=1；SC2 单 GPU 2-4h/run；代码 ≤ 50 行；可与 EXP-083/084/086 对比。
- 成功标准：Caltech AVG Best +2%；其他域下降 < 1%；Office AVG Best ≥ FDSE 90.58。

## 输出文件
- 最终方案：`FINAL_PROPOSAL.md`
- 评审总结：`REVIEW_SUMMARY.md`
- 得分历史：`score-history.md`
- 逐轮：`round-{0,1,2,3}-{initial-proposal,review,refinement}.md`
- 评审原始日志：`round-{1,2,3,4}-review-raw.txt`

## 得分演化

| 轮次 | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|-------|-----|-----|-----|-----|------|-----|-----|---------|---------|
| 1     | 9   | 9   | 8   | 8   | 9    | 6   | 6   | **8.3** | REVISE  |
| 2     | 10  | 9   | 9   | 9   | 9    | 8   | 8   | **9.0** | REVISE  |
| 3     | 10  | 10  | 9   | 9   | 9    | 8   | 8   | **9.1** | REVISE  |
| 4     | 10  | 10  | 9   | 9   | 9    | 9   | 9   | **9.4** | **READY** |

## 逐轮评审记录

| 轮次 | 评审主要关切 | 改动了什么 | 结果 |
|-------|-------------------------|------------------|--------|
| 1 | **CRITICAL VF**：缺乏 counterfactual 把"风格条件化"与"通用 classifier 个性化"分离。**IMPORTANT VR**：故事读起来像 scope 微调。 | 加入 C1（完全 local）+ C2（uniform-avg）counterfactual；加入 Claim 2 classifier-swap 诊断；把主线收紧为"风格相似度是 routing 信号"；移除 Top-2 后备；把 PACS 降为负控 | 已解决 |
| 2 | **MINOR MS**：C2 更新规则有歧义。**MINOR VF**：C2 仅 1-seed。Claim 2 存在 same-round 混淆。 | 精确化 A2/C2/C1 的聚合规则；预先约定 margin 阈值的升级规则；同 round 保存 global_head 快照；加入风格相似度矩阵作为 Claim 3 | 部分解决 — margin 豁免仍存在 |
| 3 | **VF 阻断**：C2 在"strong margin"分支下仍可能只停留在 1-seed → 终稿验证不充分 | 取消 margin 豁免：只要 A2 升级 C2 就无条件一同升级到 3-seed；统计围绕 effect size + per-seed 一致性（n=3 的 t-test 注意事项）；S_t 参与集写明；热力图预测改为定性；Claim 3 降级为"机理证据"补充材料 | 已解决 |
| 4 | 无阻断性问题 | 执行提醒：在 pack() 中保存 global_head artifact、叙事围绕 A2 vs C2、对"confirmed"一词保持措辞谨慎 | READY |

## 最终方案快照（3-5 bullets）

- **方法**：sas-FH — 将 FedDSA Plan A 的 style-aware aggregation（sas）从只覆盖 `semantic_head` 扩展到整条 head 链 `{sem_head + classifier}`。零新增可训练参数，约 30 行代码。
- **Meta 主线**：联邦 DG 中的 classifier 聚合是一个 **routing 问题**，而不是 scope 问题；**风格相似度是决定哪些客户端应当共享分类边界的自然 routing 信号**。
- **主实验**：Office-Caltech10 c4、3-seed。决定性对比 **A2 > C2**（A2 = 风格条件化的 classifier 共享；C2 = uniform 的 per-client 聚合）分离出风格条件化的因果性 — 而不仅是"per-client classifier 有用"（后者 FedROD 已涵盖）。
- **机理证据**：(i) 个性化 head 与 FedAvg head 之间的同 round checkpoint swap（Claim 2）直接检验"特征对，边界错"；(ii) 风格相似度矩阵 + routing 权重热力图（机理证据）表明 routing 信号可解释。
- **范围边界**：PACS 作为附录负控 — 4-domain 互为 outlier 的分布没有 majority 参照，routing 无用武之地。Office 的"多数相似 + 少数 outlier"分布才是机理发挥之处。

## 方法演化要点

1. **最重要的简化**：移除"Top-2 多质心后备"与"encoder 最后一个 block 的消融"（R1）让论文专注于 routing claim。
2. **最重要的机理升级**：把"粒度扫"换成"匹配的 counterfactual C2"（R1→R2）。这把论文从"我们把 sas 扩展到更多层"提升到"我们证明风格是 classifier 共享的因果信号"。概念上的提升大于代码上的。
3. **最重要的验证升级**：无条件把 C2 升级到 3-seed（R3→R4）。去掉了"strong-margin 漏洞"，否则论文的核心新颖性 test 会留在验证不足的状态。

## 推回 / Drift 日志

| 轮次 | 评审建议 | 作者回应 | 结果 |
|-------|---------------|-----------------|---------|
| 1 | "删掉 A3（encoder last block）；削弱了'最小够用'的故事" | 同意，已删除 | 接受 |
| 1 | "PACS 应作为负控，而非并列 claim" | 同意，降级到附录 | 接受 |
| 1 | "删掉 Top-2 多质心后备" | 同意，分离为 future work 方向 | 接受 |
| 2 | "C2 更新规则有歧义" | 操作性地指定为 `per_param_i[k] = (1/\|S\|) · Σ head_j`，加入 sas=3 config flag | 接受 |
| 3 | "终稿中 C2 必须 3-seed，不受 margin 影响" | 同意，完全移除 margin 豁免 | 接受 |
| 3 | "热力图预测应为定性" | 同意，移除了像"> 0.7"这样的数值阈值 | 接受 |
| 4 | "在 pack() 中显式保存 global_head artifact" | 记为实现要求；标准 FedAvg pipeline 中已隐含 | 接受 |

**零 drift 接受。** 锚点在全部 4 轮中逐字保留。

## 剩余弱点（坦白）

1. **新颖性较窄**：scope 扩展 + 条件化选择。在评审 blind 阶段 CQ 上限可能是 9/10 而非 10/10。通过 meta-framing（routing 问题）缓解。
2. **n=3 统计强度**：FDG 惯例但本质较弱。Paired t-test 只作补充；主报告 = effect size + per-seed 一致性。
3. **Benchmark 特异性**：机理在"多数相似 + 少数 outlier"分布（Office）上有效。PACS 的互为 outlier 情形为 null。论文必须把这 framing 为**范围边界**，而非 limitation。
4. **被称为"只是把 key 塞进 sas dict"的风险**：通过 Claim 2（直接机理证据）与机理证据热力图（routing 权重可视化）缓解。

## 评审原始回复

<details>
<summary>第 1 轮 — Overall 8.3，REVISE</summary>

见 `round-1-review-raw.txt`。Session ID：`019d9ebd-235a-7ae3-9dbc-739cc40b9995`。

关键结论：
> 方案锚定良好且异常可实现，但尚未达到顶会门槛。主要问题不是方法 drift，而是当前验证计划没有干净地分离所 claim 的新颖性。

</details>

<details>
<summary>第 2 轮 — Overall 9.0，REVISE</summary>

见 `round-2-review-raw.txt`（同一 thread）。

关键结论：
> 修订版实质上更加锐利。主导贡献现在更清晰……方法更简单、没有过度构建。没有 READY 的唯一原因是：核心新颖性现在几乎完全依赖 A2 > C2，因此该对比需要被规范化，若结果有希望则需要足够的统计强度来验证。

</details>

<details>
<summary>第 3 轮 — Overall 9.1，REVISE</summary>

见 `round-3-review-raw.txt`。

关键结论：
> 唯一阻断性问题是证据层面：一旦论文做出因果 routing 的 claim，关键基线 C2 就必须被提升为终表的完整条目。修掉之后就非常接近 READY。

</details>

<details>
<summary>第 4 轮 — Overall 9.4，READY ✅</summary>

见 `round-4-review-raw.txt`。

关键结论：
> 之前的阻断性问题已修复。锚点得以保留、贡献聚焦、方法极简，验证在顶会早期阶段 proposal 中已经足够匹配 claim。
> **Verdict：READY**

</details>

## 下一步

- ✅ READY → 进入 `/experiment-plan` 做完整实验路线图
- 或直接实现 sas-FH（在 `feddsa_scheduled.py` 中约 40 行）+ 单元测试 + 在 Office 上做单 seed s=2 的 triage 运行（2h）
- 如果 triage 通过决策规则（A2 − max(B1, C2) ≥ 0.5）：强制把 A2 + C2 升级到 3-seed
- Claim 2 swap 诊断仅用 checkpoint（无额外训练）
- PACS 附录只在 Office 成功时才做
