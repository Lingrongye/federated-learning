# 评审总结 — sas-FH outlier 补救

**问题**：FedDSA Plan A 的 sas 只个性化了 `semantic_head`，`classifier head` 仍然走 FedAvg → 在风格异构下 outlier 域（PACS art 65.69%、Office Caltech 77.68%）被严重拖累。

**初始方案**：5 个候选 — Top-1 head sas / Top-2 多质心 / Top-3 focal / Top-4 BN sas / Top-5 SupCon。选定 Top-1 作为主导方案。

**日期**：2026-04-18
**轮次**：4 / 5
**最终得分**：9.4 / 10
**最终裁决**：✅ **READY**

## 问题锚点（不可变，逐轮原样保留）

- **瓶颈**：在风格异构下，共享 classifier 稀释了 outlier 域的决策边界
- **约束**：不改 backbone、不新增可训练模块、不用推理技巧、不做 per-sample sas；3-seed {2,15,333} × R200 × E=1；代码 ≤ 50 行；可与 EXP-083/084/086 对比
- **成功标准**：Caltech AVG Best +2；其他域下降 < 1%；Office AVG ≥ FDSE 90.58

## 逐轮解决日志

| 轮次 | 评审主要关切 | 改动了什么 | 是否解决？ | 剩余风险 |
|-------|------------------------|--------------|---------|----------------|
| 1 (8.3) | VF/VR 偏低：缺乏 counterfactual 来把"风格条件化"与"通用 classifier 个性化"分离开；主线读起来像范围微调；Top-2 后备方案和 PACS 并列 claim 分散了焦点 | 加入 C1（local）+ C2（uniform）counterfactual；加入 Claim 2 swap 诊断；把主线收紧为"routing signal"；把 PACS 降为负控；移除 Top-2 | 是 | C2 规则是否无歧义、统计强度 |
| 2 (9.0) | MS：C2 更新规则有歧义；VF：C2 仅 1-seed；"同 round" swap 存在混淆 | 精确化 C2 聚合规则 `= (1/\|S\|) · Σ head_j`；预先约定升级规则；同 round 保存 global_head 快照；加入风格相似度矩阵 | 基本解决 | C2 尚未无条件做 3-seed |
| 3 (9.1) | VF：C2 在 strong-margin 分支下仍可能只停留在 1-seed → 终稿验证不充分 | 取消 margin 豁免：只要 A2 升级 C2 就一同升级到 3-seed；统计围绕 effect size + per-seed 一致性；A2/C2 使用相同 S_t；热力图改为定性；Claim 3 降级为"机理证据" | 是 | — |
| 4 (9.4) | 无阻断性问题 | 仅剩执行提醒（在 pack() 中保存 global_head artifact、叙事围绕 A2 vs C2、措辞谨慎） | — | READY |

## 总体演化

- **方法更具体**：从"把 sas 扩展到 head" → 精确的 5 值 config 表 + 3 条显式聚合规则 + 参与集规范。
- **主导贡献更聚焦**：从"整个 head 的个性化" → "**classifier 聚合是 routing 问题；风格相似度提供 routing 信号**"。
- **去掉不必要的复杂度**：encoder 最后一个 block 的消融（R1）、多质心后备（R1）、量化 matrix 预测（R3）、PACS 并列 claim（R1 → 移到附录）。
- **现代技术杠杆**：有意保守 — 评审确认 FM 时代的原语会改变论文性质，而不是解决锚定的瓶颈。
- **避免 drift**：锚点在全部 4 轮中逐字保留；评审确认全程无 drift。

## 最终状态

- **锚点状态**：保留（4 轮均已确认）
- **聚焦状态**：紧凑 — 单一主导贡献；C1 / PACS / 热力图均降级为支撑证据
- **现代化状态**：适度前沿感知（不强塞 FM 时代扩展）
- **最终方法最强的部分**：
  1. 零新增可训练参数（纯聚合改动）
  2. 干净的 counterfactual C2 分离了因果性（"是风格条件化，还是单纯 per-client classifier？"）
  3. 同 round swap 诊断提供直接的机理证据
  4. 被 framing 为 routing 问题，而非 scope 扩展 → 将论文提升到 meta-framing 层级
- **剩余弱点**：
  - 新颖性较窄（scope 扩展 + 条件化选择）。评审 CQ 给 9/10 "defensible" 但未给 10。
  - n=3 本身统计强度有限（FDG 惯例；paired t-test 仅作补充）
  - PACS 作为边界条件意味着论文的增益依赖特定 benchmark（Office 的 outlier-heavy 分布）
