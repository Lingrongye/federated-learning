- `InfoNCE` 把 `z_sem` 往 same-class global prototype 拉，本质上是一个强 compression force。PACS 里它能压掉 domain variation；Office 里它可能会把有用的细微 domain-conditioned modes 一起压掉。
- `aux_w` 只升不降，而 `loss_aug` 没有任何类似权重，这是一个不对称训练设计。训练早期，augmentation branch 已经在大力改 feature distribution，但 semantic alignment 还没稳定；训练后期，augmentation 仍然持续施压，可能在 low-style-gap 数据上持续注入噪声。我认为这很可能是 Office 失败的一部分。
- `BN running stats private, affine aggregated` 不是小 quirk，而是一个统计假设不一致：`gamma/beta` 是在各自 local stats 下学出来的，汇总后不一定匹配任何 client。Office 这种 subtle shift 场景比 PACS 更容易受这类 mismatch 影响。

我的总体判断：  
`FedDSA` 作为一个 **heuristic system** 是可以工作的；但作为一个 **disentangle-style-and-share-style** 的方法，当前设计并不自洽。

**Q2 Why FDSE Wins On Office**

先说一个关键点：从 [FDSE paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf) 的 ablation 看，Office 上不是只有 `personalization` 在起作用。它们 Table 3 里，Office AVG 从 base `86.13` 到：
- `B+C`: `88.54`
- `A+C`: `89.25`
- `A+B`: `90.38`
- `A+B+C`: `91.58`

这里 `A=consensus maximization`，`B=similarity-aware personalization`，`C=consistency regularization`。直接读数上：
- `consensus maximization` 的贡献大于 `similarity-aware personalization`
- 最大的单项增益其实来自把 aggregation 和 regularization 一起做对
- 你把问题理解成“FDSE 靠 personalization 赢”，这不完整

为什么这在 Office 上特别有效：

- Office 的主要矛盾不是“style diversity 不够”，而是 **subtle client drift + update conflict**。各 domain 都是 real photos，差异小，但小差异足以让 `FedAvg` 在 shared parameters 上产生 destructive averaging。
- `consensus maximization` 在这种场景下比 style sharing 更对症。因为 shared semantic signal 很强，真正需要的是避免 aggregation 把它弄坏。
- `similarity-aware personalization` 让 residual client bias 留在 DSE 分支，而不是强行共享。Office 的 domain shift 不是 sketch/photo 那种需要大规模 style extrapolation 的 shift，而是 webcam noise、background habit、camera pipeline 这种细颗粒差异，personalization 更适配。
- `layer-wise decomposition` 很重要，因为 subtle shift 往往分布在多层，不会只集中在一个 final feature `h` 上。你的 FedDSA 把所有 style handling 压到单个 late representation，这很粗糙。
- `consistency regularization` 对 Office 这类近域更有效，因为 local/global BN stats 本来就接近，拉齐是合理的；对 PACS 这种大域差，拉太狠反而可能伤害。

能不能吸收 FDSE 而不丢掉你自己的 identity：

- 可以。把 `FedAvg` 换成 conflict-aware aggregation，只用于 `encoder + semantic_head + classifier`，不动 `style_head private` 和 `style bank`，FedDSA 还是 FedDSA。
- 也可以把 `similarity-aware` 用在 **style dispatch / bank selection** 上，而不是直接去做完整的 FDSE layer decomposition。这样保住 “decouple + share + align” 的主线。
- 但如果你什么都不改 aggregation，只继续在 Office 上拼 augmentation，我判断成功率不高。

**Q3 Style Asset vs Style Erasure**

这是你现在最有价值的 insight，但它目前还是一个 observation，不是 claim。

一个更严谨的说法是：

- 当 `style` 近似是 label-preserving nuisance，且 target domain 的 style support 不在 source support 内，`style sharing` 相当于一种 `support expansion / vicinal risk minimization`，是有利的。
- 当 source domains 已经都是相近的 real-photo，style gap 很小，或者 shared style stats 混入 semantic/class information，`style sharing` 就不再扩 support，而是在制造 off-manifold perturbation。这时更好的策略是 `style erasure / personalization / conflict-aware sharing`。
- 用形式化一点的话说，如果你把样本写成 `(c, s)`，其中 `c=content`, `s=style`，那么 style sharing 有效需要至少近似满足：
  `p(y | c, s) ≈ p(y | c)`  
  且 target 的 `p_t(s | c)` 能被 source 的 style mixing 合理覆盖。  
  PACS 大致接近这个条件；Office 很可能不接近，至少你现在用 `h`-stats 的 mixing 不接近。

这对 FedDSA 的根本含义是：

- 你的方法不该声称“sharing style is generally better than erasing style”
- 更合理的主张是：**style sharing is conditionally beneficial under large, label-preserving style heterogeneity**
- PACS 支持这个条件；Office 反例说明它不是 universal claim

这不是坏消息。坏消息是如果你继续写成 universal SOTA method，审稿人会直接打掉。好消息是这里确实有一条更真实的 paper story。

**Q4 Related Work: What They Actually Show**

先说结论：我能直接核实的 style-sharing papers，**没有一个我确认报告了 Office-Caltech10**；它们主流用的是 `OfficeHome`、`PACS`、`VLCS`、`IWildCam`。所以你现在的负例数据集和相关工作常用 benchmark 之间有一点错位。

- **CCST (WACV 2023)**  
  在 [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Chen_Federated_Domain_Generalization_for_Image_Recognition_via_Cross-Client_Style_Transfer_WACV_2023_paper.pdf) 里，`OfficeHome` 平均是 `63.56`，相对 `MixStyle 62.49`、`FedDG 62.77` 只高不到 `1%`。作者还直接写了：因为 `domain style discrepancy` 小，所以 DG methods 在 `OfficeHome` 上的提升“小于 1%”。这几乎就是你 Office 现象的前人版本。  
  它的 gating 方式不是 learned gate，而是：
  `single image style` vs `overall domain style` 二选一，加上 augmentation level `K` 控制强度。  
  经验上，`overall domain style` 更稳，`single image style` 更随机。
- **FedCCRL (arXiv 2024)**  
  [paper](https://arxiv.org/pdf/2410.11267) 报告 `OfficeHome` Avg `68.31`，优于 `CCST 66.42`。但更值得你注意的是 per-domain pattern：相对 `CCST`，`Art +1.88`，`Clipart +5.19`，而 `Product +0.24`，`Real +0.23`。  
  这说明它在 stylized domains 上增益明显，在 real-photo-like domains 上几乎没 gain。  
  它不是纯 style sharing；它把 `cross-client statistics transfer` 和 `domain-invariant feature perturbation + representation/prediction alignment` 绑在一起。  
  它的“gating”更像是弱化 raw sharing：只上传一个比例 `r` 的 stats，并用 dual alignment 稳住表示。
- **StyleDDG (arXiv 2025/2026)**  
  [paper](https://arxiv.org/pdf/2504.06235) 没有 `OfficeHome`，但用了 `PACS` 和 `VLCS`。作者明确说 `VLCS` 的 gains 比 `PACS` 小，因为 `style gap` 更小。  
  这是与你观察最一致的外部证据之一：style-sharing 在 small-gap real-image benchmarks 上收益显著变弱。  
  它没有显式 gate，但受 neighbor topology 限制，本身不是全局无条件 all-to-all sharing。
- **FISC / PARDON (arXiv 2024 / ICDCS 2025)**  
  我在当前环境里没能拿到可检索的 paper PDF 表格，因此 **无法严谨核实它们的 exact OfficeHome numbers**。我能核实的是它们的 [repo](https://github.com/judydnguyen/PARDON-FedDG) 和 metadata/abstract 明确写了 `PACS + Office-Home + IWildCam`，并且 repo acknowledgement 里出现了 `FINCH`，method 描述是 `abstracted / interpolative local styles + contrastive learning`。  
  这至少说明两件事：  
  1. 它们不是 raw one-style-per-client 的 all-to-all sharing，而是做了 style abstraction / interpolation  
  2. 它们已经在往“clustered / abstracted style bank”方向走，这正是你该吸收的点  
  但 exact OfficeHome per-domain behavior，我这里不能负责地硬编。

FedDSA 能从这些工作学什么：

- 不要做 **unconditional raw style sharing**
- style sharing 要么有 `strength control`，要么有 `abstraction / clustering`，要么要配强的 `alignment`
- 如果 benchmark 是 real-photo-like，增益通常会明显缩小，甚至接近 0；这不是你独有的问题

**Q5 Ranked Improvements**

我按“1 周内可做 + 解决 Office 特异性失败 + 保留 decouple/share/align identity”的标准排序。

1. **Distance-gated style dispatch**
- 把固定 `5 external styles + Beta(0.1,0.1)` 改成基于 inter-client style distance 的 gate。
- 最简单做法：`d_ij = ||μ_i-μ_j||_2 + ||logσ_i-logσ_j||_2`。只有 `d_ij > δ` 才允许 client `j` 的 style 发给 `i`；同时把 mixing strength 设成随 `d_ij` 单调变化，而不是固定 extreme mixing。
- 动机：PACS 上大 gap 保留 transfer；Office 上小 gap 自动降到接近 no-transfer。
- 这是我认为成功率最高的改动。

2. **Replace FedAvg with consensus-aware aggregation on shared semantic parameters**
- 只对 `encoder + semantic_head + sem_classifier` 引入 FDSE 式 `consensus maximization` aggregation；`style_head` 继续 private。
- 动机：Office 的核心更像 aggregation conflict，不像 style scarcity。你不需要变成 FDSE，只需要把 shared semantic path 的 aggregation 做对。
- 这会直接补你现在方法在 low-gap datasets 上最弱的一环。

3. **Clustered or class-conditional style bank**
- 不要“一 client 一 style”。把 server 端收到的 `(μ,σ)` 做 `FINCH / k-means` clustering，或者更直接一点，按 class 维护 class-conditional style centroids。
- 动机：你当前 bank 太粗，会把 class mixture 当成 style。Office 尤其容易被这种 semantic contamination 伤到。
- 这是从 `FISC/PARDON` 路线最值得抄的东西。

4. **Give `L_aug` the same schedule, and decay it**
- 现在 `loss_aug` 从头到尾都强，且无 decay。我会改成：
  `L = L_task + w_aug(r) * L_aug + aux_w(r) * (λ_orth L_orth + λ_sem L_InfoNCE)`  
  其中 `w_aug` 先 warmup，再在后 1/3 training cosine decay。
- 动机：augmentation 早期扩 support，后期不该继续主导 boundary refinement。Office 的 synthetic perturbation 很可能是 late-stage damage。

5. **Tether `z_sty` to the actual shared style asset**
- 加一个小 `MLP g(z_sty)`，回归当前 sample 的 stop-gradient style stats，或者对 augmented sample 回归 transferred stats：
  `L_style = ||g(z_sty)-sg(s)||²`
- 动机：把 `decouple` 和 `share` 真正接上。否则 `style_head` 永远只是一个 story component。
- 这条不是最容易立竿见影，但它最能修正你方法的 conceptual hole。

我不建议你继续把时间花在更多 `loss variants / VAE / asymmetry` 上。你的问题已经不是二阶细节，而是 regime mismatch 和 method coupling mismatch。

**Q6 Claim Pivot**

四个选项里，最弱的是：

- `Stability claim`：不够。`3 seeds`、单 dataset、小 std，顶会不会买。
- `Budget efficiency claim`：只有在你做了严格 matched compute curve 之后才有戏；否则很容易被说是在 cherry-pick rounds。

最可 defend 的是：

- **`Problem-focused claim`**：style sharing 在 FedDG 中是 regime-dependent 的，large-gap stylized domains 受益，small-gap real-photo domains 受损；FedDSA 揭示了这个边界条件。

但我建议你把它再升级一点：

- **更好的 pivot**：  
  `Unconditional style sharing is not universally beneficial in FedDG; its utility is governed by inter-domain style gap. We characterize this regime and propose an adaptive style-sharing variant.`

为什么这比单纯 case study 更强：

- 你不只是报告“PACS 赢、Office 输”
- 你给出一个 measurable moderator：`style gap`
- 你再给一个 adaptive variant：`distance-gated bank`
- 这样 paper 从“negative result with one method”变成“new empirical principle + simple fix”

如果你 1 周内做不出 Office 非负迁移，我建议不要再写“beat FDSE”。那样是自杀式 framing。  
如果你能做出：
- PACS 继续赢
- Office 至少接近或不输
- DomainNet 再拿到正向结果  
那就还有主会希望。  
如果做不到，就转 problem-focused claim，甚至考虑 workshop / second-tier。

**Q7 Mock NeurIPS Review**

**Summary**
- 本文提出 `FedDSA`，通过 dual-head representation split、feature-statistics-based style sharing 和 prototype alignment 处理 cross-domain federated learning。方法在 PACS 上优于 FDSE，但在 Office-Caltech10 上全面落后。

**Strengths**
- 方法动机直观，`decouple + share + align` 的 high-level narrative 清楚。
- PACS 上对 `Sketch/Art` 的 gains 不是噪声，说明方法确实抓到了 large style gap 场景的一部分规律。
- 发现了一个有研究价值的 empirical pattern：style-sharing methods 可能是 regime-dependent。
- 系统简单，通信上相对轻量。

**Weaknesses**
- 主要 conceptual flaw：`style_head` 不参与 style sharing / augmentation，导致“decoupling”与“sharing”脱节，style disentanglement claim 缺乏支撑。
- `cos²` orthogonality 不是有效的 independence objective，不能证明 semantic/style decoupling。
- style bank 过于粗糙，一 client 一组 final-feature stats 很可能混入 semantic/class composition。
- 在 final feature `h` 上直接做 `AdaIN`，容易破坏 discriminative semantics，尤其在 low-style-gap real-photo datasets 上。
- aggregation 仍然是 `FedAvg`，没有应对 Office 这类 subtle client conflict；与 FDSE 的正面比较因此在机制上不公平。
- training objective 不对称：`L_aug` 无 schedule/decay，而 auxiliary losses 只有 warmup。
- 实验覆盖不够强。当前主叙事是“对标 FDSE”，但你已经在一个核心 benchmark 上全输。只用两个 dataset、`3 seeds`，说服力不足。

**Score**
- `4/10`，`Weak Reject`

**Confidence**
- `4/5`

**Biggest single thing that would move it toward Accept**
- 把方法从 **unconditional raw style sharing** 改成 **style-gap-aware adaptive sharing**，并证明它在 `PACS + Office + DomainNet` 上至少不再出现明显负迁移；同时把 `z_sty` 和 shared style asset 真正绑定起来。没有这一步，这篇文章最核心的 story 是不闭合的。

**最后一句直话**

如果你目标真的是 `CVPR/ICCV/NeurIPS/ICML`，我不建议继续把时间花在“再试几个 loss/architecture trick”。  
你现在需要的是 **one decisive redesign**：`adaptive style sharing`, not more tuning.

**Sources**
- [FDSE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf)
- [CCST, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Chen_Federated_Domain_Generalization_for_Image_Recognition_via_Cross-Client_Style_Transfer_WACV_2023_paper.pdf)
- [FedCCRL, arXiv 2024](https://arxiv.org/pdf/2410.11267)
- [StyleDDG, arXiv 2025/2026](https://arxiv.org/pdf/2504.06235)
- [PARDON repo](https://github.com/judydnguyen/PARDON-FedDG)

我没能在当前环境里直接核实 `FISC/PARDON` 论文正文中的 exact OfficeHome table values；我能确认的是其 abstract/repo/metadata 显示它们评估了 `Office-Home`，并采用了 `abstracted/interpolative style` 路线，这支持“不要 raw all-to-all sharing”的判断，但不足以让我负责地给出精确数值。
tokens used
133,601
**结论先行**

以 top-tier 标准看，FedDSA 现在的问题不是“还差一点调参”，而是有一个 structural mismatch：你声称在 decouple `style/semantic`，但真正被 `share` 和用于 augmentation 的不是 `z_sty`，而是 final backbone feature `h` 的 statistics。于是方法里真正起作用的部分，更像是 `final-feature style perturbation + prototype alignment + weak orth regularization`，而不是一个自洽的 `style-semantic disentanglement` 框架。这个设计在 PACS 这种 large style gap 上可能有效，但在 Office 这种 low-style-gap real-photo 场景上，很容易变成 off-manifold noise injection。

**Q1 Design Flaws**

- `dual-head shared encoder` 不是不能用，但就你现在的 objective 而言，它不构成可信的 disentanglement。`z_sem` 有 task loss，`z_sty` 没有任何直接的 style supervision，也不参与 augmentation path。这样 `style_head` 更像一个被 `L_orth` 推出来的 residual sink，而不是真正的 style extractor。
- 你说的 `CRITICAL BUG`，我认为是 **conceptual bug，不是 implementation bug**。`AdaIN` 用的是 `h` 的 `(mu, sigma)`，不是 `z_sty`。这意味着 `decouple` 和 `share` 两部分在因果上是断开的。`style_head` 只通过 `orth loss` 间接影响 encoder，因此“Decouple-Share-Align”这三个词在当前方法里并没有形成闭环。
- 所以，`z_sty` 目前最合理的解释不是 “style representation”，而是 **anti-semantic anchor / nuisance sponge**。这不是没价值，但它和你论文想卖的 story 不一样。审稿人会抓这一点。
- `L_orth = cos²(z_sem, z_sty)` 是很弱的 decoupling objective。Orthogonality 不等于 independence。高维里两个向量完全可以 cosine 接近 0，但仍然携带高度重叠的信息；信息只要经过旋转、分块、非线性重编码，就能泄漏。你自己也已经意识到 HSIC 是 optional，这更说明 disentanglement claim 站不稳。
- 更糟的是，`h` 本身已经是 late feature，通常混了大量 semantic content。你直接在 `h` 上做 `AdaIN`，实际上在 perturb 的不只是 style，也包括 class-relevant evidence。PACS 上这种粗暴 perturb 可能还能帮你扩展 support；Office 上更可能是在破坏 object cues。
- `global style bank` 每 client 只上传一组 1024-d `(mu, sigma)`，这个粒度过粗。对于多类、多模态 client，这个 bank entry 会混进 class composition、foreground occupancy、background ratio、viewpoint 等信息，不是纯 style。Office 这种 white-background product photo 特别容易中这个问题。
- `InfoNCE` 把 `z_sem` 往 same-class global prototype 拉，本质上是一个强 compression force。PACS 里它能压掉 domain variation；Office 里它可能会把有用的细微 domain-conditioned modes 一起压掉。
- `aux_w` 只升不降，而 `loss_aug` 没有任何类似权重，这是一个不对称训练设计。训练早期，augmentation branch 已经在大力改 feature distribution，但 semantic alignment 还没稳定；训练后期，augmentation 仍然持续施压，可能在 low-style-gap 数据上持续注入噪声。我认为这很可能是 Office 失败的一部分。
- `BN running stats private, affine aggregated` 不是小 quirk，而是一个统计假设不一致：`gamma/beta` 是在各自 local stats 下学出来的，汇总后不一定匹配任何 client。Office 这种 subtle shift 场景比 PACS 更容易受这类 mismatch 影响。

我的总体判断：  
`FedDSA` 作为一个 **heuristic system** 是可以工作的；但作为一个 **disentangle-style-and-share-style** 的方法，当前设计并不自洽。

**Q2 Why FDSE Wins On Office**

先说一个关键点：从 [FDSE paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf) 的 ablation 看，Office 上不是只有 `personalization` 在起作用。它们 Table 3 里，Office AVG 从 base `86.13` 到：
- `B+C`: `88.54`
- `A+C`: `89.25`
- `A+B`: `90.38`
- `A+B+C`: `91.58`

这里 `A=consensus maximization`，`B=similarity-aware personalization`，`C=consistency regularization`。直接读数上：
- `consensus maximization` 的贡献大于 `similarity-aware personalization`
- 最大的单项增益其实来自把 aggregation 和 regularization 一起做对
- 你把问题理解成“FDSE 靠 personalization 赢”，这不完整

为什么这在 Office 上特别有效：

- Office 的主要矛盾不是“style diversity 不够”，而是 **subtle client drift + update conflict**。各 domain 都是 real photos，差异小，但小差异足以让 `FedAvg` 在 shared parameters 上产生 destructive averaging。
- `consensus maximization` 在这种场景下比 style sharing 更对症。因为 shared semantic signal 很强，真正需要的是避免 aggregation 把它弄坏。
- `similarity-aware personalization` 让 residual client bias 留在 DSE 分支，而不是强行共享。Office 的 domain shift 不是 sketch/photo 那种需要大规模 style extrapolation 的 shift，而是 webcam noise、background habit、camera pipeline 这种细颗粒差异，personalization 更适配。
- `layer-wise decomposition` 很重要，因为 subtle shift 往往分布在多层，不会只集中在一个 final feature `h` 上。你的 FedDSA 把所有 style handling 压到单个 late representation，这很粗糙。
- `consistency regularization` 对 Office 这类近域更有效，因为 local/global BN stats 本来就接近，拉齐是合理的；对 PACS 这种大域差，拉太狠反而可能伤害。

能不能吸收 FDSE 而不丢掉你自己的 identity：

- 可以。把 `FedAvg` 换成 conflict-aware aggregation，只用于 `encoder + semantic_head + classifier`，不动 `style_head private` 和 `style bank`，FedDSA 还是 FedDSA。
- 也可以把 `similarity-aware` 用在 **style dispatch / bank selection** 上，而不是直接去做完整的 FDSE layer decomposition。这样保住 “decouple + share + align” 的主线。
- 但如果你什么都不改 aggregation，只继续在 Office 上拼 augmentation，我判断成功率不高。

**Q3 Style Asset vs Style Erasure**

这是你现在最有价值的 insight，但它目前还是一个 observation，不是 claim。

一个更严谨的说法是：

- 当 `style` 近似是 label-preserving nuisance，且 target domain 的 style support 不在 source support 内，`style sharing` 相当于一种 `support expansion / vicinal risk minimization`，是有利的。
- 当 source domains 已经都是相近的 real-photo，style gap 很小，或者 shared style stats 混入 semantic/class information，`style sharing` 就不再扩 support，而是在制造 off-manifold perturbation。这时更好的策略是 `style erasure / personalization / conflict-aware sharing`。
- 用形式化一点的话说，如果你把样本写成 `(c, s)`，其中 `c=content`, `s=style`，那么 style sharing 有效需要至少近似满足：
  `p(y | c, s) ≈ p(y | c)`  
  且 target 的 `p_t(s | c)` 能被 source 的 style mixing 合理覆盖。  
  PACS 大致接近这个条件；Office 很可能不接近，至少你现在用 `h`-stats 的 mixing 不接近。

这对 FedDSA 的根本含义是：

- 你的方法不该声称“sharing style is generally better than erasing style”
- 更合理的主张是：**style sharing is conditionally beneficial under large, label-preserving style heterogeneity**
- PACS 支持这个条件；Office 反例说明它不是 universal claim

这不是坏消息。坏消息是如果你继续写成 universal SOTA method，审稿人会直接打掉。好消息是这里确实有一条更真实的 paper story。

**Q4 Related Work: What They Actually Show**

先说结论：我能直接核实的 style-sharing papers，**没有一个我确认报告了 Office-Caltech10**；它们主流用的是 `OfficeHome`、`PACS`、`VLCS`、`IWildCam`。所以你现在的负例数据集和相关工作常用 benchmark 之间有一点错位。

- **CCST (WACV 2023)**  
  在 [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Chen_Federated_Domain_Generalization_for_Image_Recognition_via_Cross-Client_Style_Transfer_WACV_2023_paper.pdf) 里，`OfficeHome` 平均是 `63.56`，相对 `MixStyle 62.49`、`FedDG 62.77` 只高不到 `1%`。作者还直接写了：因为 `domain style discrepancy` 小，所以 DG methods 在 `OfficeHome` 上的提升“小于 1%”。这几乎就是你 Office 现象的前人版本。  
  它的 gating 方式不是 learned gate，而是：
  `single image style` vs `overall domain style` 二选一，加上 augmentation level `K` 控制强度。  
  经验上，`overall domain style` 更稳，`single image style` 更随机。
- **FedCCRL (arXiv 2024)**  
  [paper](https://arxiv.org/pdf/2410.11267) 报告 `OfficeHome` Avg `68.31`，优于 `CCST 66.42`。但更值得你注意的是 per-domain pattern：相对 `CCST`，`Art +1.88`，`Clipart +5.19`，而 `Product +0.24`，`Real +0.23`。  
  这说明它在 stylized domains 上增益明显，在 real-photo-like domains 上几乎没 gain。  
  它不是纯 style sharing；它把 `cross-client statistics transfer` 和 `domain-invariant feature perturbation + representation/prediction alignment` 绑在一起。  
  它的“gating”更像是弱化 raw sharing：只上传一个比例 `r` 的 stats，并用 dual alignment 稳住表示。
- **StyleDDG (arXiv 2025/2026)**  
  [paper](https://arxiv.org/pdf/2504.06235) 没有 `OfficeHome`，但用了 `PACS` 和 `VLCS`。作者明确说 `VLCS` 的 gains 比 `PACS` 小，因为 `style gap` 更小。  
  这是与你观察最一致的外部证据之一：style-sharing 在 small-gap real-image benchmarks 上收益显著变弱。  
  它没有显式 gate，但受 neighbor topology 限制，本身不是全局无条件 all-to-all sharing。
- **FISC / PARDON (arXiv 2024 / ICDCS 2025)**  
  我在当前环境里没能拿到可检索的 paper PDF 表格，因此 **无法严谨核实它们的 exact OfficeHome numbers**。我能核实的是它们的 [repo](https://github.com/judydnguyen/PARDON-FedDG) 和 metadata/abstract 明确写了 `PACS + Office-Home + IWildCam`，并且 repo acknowledgement 里出现了 `FINCH`，method 描述是 `abstracted / interpolative local styles + contrastive learning`。  
  这至少说明两件事：  
  1. 它们不是 raw one-style-per-client 的 all-to-all sharing，而是做了 style abstraction / interpolation  
  2. 它们已经在往“clustered / abstracted style bank”方向走，这正是你该吸收的点  
  但 exact OfficeHome per-domain behavior，我这里不能负责地硬编。

FedDSA 能从这些工作学什么：

- 不要做 **unconditional raw style sharing**
- style sharing 要么有 `strength control`，要么有 `abstraction / clustering`，要么要配强的 `alignment`
- 如果 benchmark 是 real-photo-like，增益通常会明显缩小，甚至接近 0；这不是你独有的问题

**Q5 Ranked Improvements**

我按“1 周内可做 + 解决 Office 特异性失败 + 保留 decouple/share/align identity”的标准排序。

1. **Distance-gated style dispatch**
- 把固定 `5 external styles + Beta(0.1,0.1)` 改成基于 inter-client style distance 的 gate。
- 最简单做法：`d_ij = ||μ_i-μ_j||_2 + ||logσ_i-logσ_j||_2`。只有 `d_ij > δ` 才允许 client `j` 的 style 发给 `i`；同时把 mixing strength 设成随 `d_ij` 单调变化，而不是固定 extreme mixing。
- 动机：PACS 上大 gap 保留 transfer；Office 上小 gap 自动降到接近 no-transfer。
- 这是我认为成功率最高的改动。

2. **Replace FedAvg with consensus-aware aggregation on shared semantic parameters**
- 只对 `encoder + semantic_head + sem_classifier` 引入 FDSE 式 `consensus maximization` aggregation；`style_head` 继续 private。
- 动机：Office 的核心更像 aggregation conflict，不像 style scarcity。你不需要变成 FDSE，只需要把 shared semantic path 的 aggregation 做对。
- 这会直接补你现在方法在 low-gap datasets 上最弱的一环。

3. **Clustered or class-conditional style bank**
- 不要“一 client 一 style”。把 server 端收到的 `(μ,σ)` 做 `FINCH / k-means` clustering，或者更直接一点，按 class 维护 class-conditional style centroids。
- 动机：你当前 bank 太粗，会把 class mixture 当成 style。Office 尤其容易被这种 semantic contamination 伤到。
- 这是从 `FISC/PARDON` 路线最值得抄的东西。

4. **Give `L_aug` the same schedule, and decay it**
- 现在 `loss_aug` 从头到尾都强，且无 decay。我会改成：
  `L = L_task + w_aug(r) * L_aug + aux_w(r) * (λ_orth L_orth + λ_sem L_InfoNCE)`  
  其中 `w_aug` 先 warmup，再在后 1/3 training cosine decay。
- 动机：augmentation 早期扩 support，后期不该继续主导 boundary refinement。Office 的 synthetic perturbation 很可能是 late-stage damage。

5. **Tether `z_sty` to the actual shared style asset**
- 加一个小 `MLP g(z_sty)`，回归当前 sample 的 stop-gradient style stats，或者对 augmented sample 回归 transferred stats：
  `L_style = ||g(z_sty)-sg(s)||²`
- 动机：把 `decouple` 和 `share` 真正接上。否则 `style_head` 永远只是一个 story component。
- 这条不是最容易立竿见影，但它最能修正你方法的 conceptual hole。

我不建议你继续把时间花在更多 `loss variants / VAE / asymmetry` 上。你的问题已经不是二阶细节，而是 regime mismatch 和 method coupling mismatch。

**Q6 Claim Pivot**

四个选项里，最弱的是：

- `Stability claim`：不够。`3 seeds`、单 dataset、小 std，顶会不会买。
- `Budget efficiency claim`：只有在你做了严格 matched compute curve 之后才有戏；否则很容易被说是在 cherry-pick rounds。

最可 defend 的是：

- **`Problem-focused claim`**：style sharing 在 FedDG 中是 regime-dependent 的，large-gap stylized domains 受益，small-gap real-photo domains 受损；FedDSA 揭示了这个边界条件。

但我建议你把它再升级一点：

- **更好的 pivot**：  
  `Unconditional style sharing is not universally beneficial in FedDG; its utility is governed by inter-domain style gap. We characterize this regime and propose an adaptive style-sharing variant.`

为什么这比单纯 case study 更强：

- 你不只是报告“PACS 赢、Office 输”
- 你给出一个 measurable moderator：`style gap`
- 你再给一个 adaptive variant：`distance-gated bank`
- 这样 paper 从“negative result with one method”变成“new empirical principle + simple fix”

如果你 1 周内做不出 Office 非负迁移，我建议不要再写“beat FDSE”。那样是自杀式 framing。  
如果你能做出：
- PACS 继续赢
- Office 至少接近或不输
- DomainNet 再拿到正向结果  
那就还有主会希望。  
如果做不到，就转 problem-focused claim，甚至考虑 workshop / second-tier。

**Q7 Mock NeurIPS Review**

**Summary**
- 本文提出 `FedDSA`，通过 dual-head representation split、feature-statistics-based style sharing 和 prototype alignment 处理 cross-domain federated learning。方法在 PACS 上优于 FDSE，但在 Office-Caltech10 上全面落后。

**Strengths**
- 方法动机直观，`decouple + share + align` 的 high-level narrative 清楚。
- PACS 上对 `Sketch/Art` 的 gains 不是噪声，说明方法确实抓到了 large style gap 场景的一部分规律。
- 发现了一个有研究价值的 empirical pattern：style-sharing methods 可能是 regime-dependent。
- 系统简单，通信上相对轻量。

**Weaknesses**
- 主要 conceptual flaw：`style_head` 不参与 style sharing / augmentation，导致“decoupling”与“sharing”脱节，style disentanglement claim 缺乏支撑。
- `cos²` orthogonality 不是有效的 independence objective，不能证明 semantic/style decoupling。
- style bank 过于粗糙，一 client 一组 final-feature stats 很可能混入 semantic/class composition。
- 在 final feature `h` 上直接做 `AdaIN`，容易破坏 discriminative semantics，尤其在 low-style-gap real-photo datasets 上。
- aggregation 仍然是 `FedAvg`，没有应对 Office 这类 subtle client conflict；与 FDSE 的正面比较因此在机制上不公平。
- training objective 不对称：`L_aug` 无 schedule/decay，而 auxiliary losses 只有 warmup。
- 实验覆盖不够强。当前主叙事是“对标 FDSE”，但你已经在一个核心 benchmark 上全输。只用两个 dataset、`3 seeds`，说服力不足。

**Score**
- `4/10`，`Weak Reject`

**Confidence**
- `4/5`

**Biggest single thing that would move it toward Accept**
- 把方法从 **unconditional raw style sharing** 改成 **style-gap-aware adaptive sharing**，并证明它在 `PACS + Office + DomainNet` 上至少不再出现明显负迁移；同时把 `z_sty` 和 shared style asset 真正绑定起来。没有这一步，这篇文章最核心的 story 是不闭合的。

**最后一句直话**

如果你目标真的是 `CVPR/ICCV/NeurIPS/ICML`，我不建议继续把时间花在“再试几个 loss/architecture trick”。  
你现在需要的是 **one decisive redesign**：`adaptive style sharing`, not more tuning.

**Sources**
- [FDSE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf)
- [CCST, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Chen_Federated_Domain_Generalization_for_Image_Recognition_via_Cross-Client_Style_Transfer_WACV_2023_paper.pdf)
- [FedCCRL, arXiv 2024](https://arxiv.org/pdf/2410.11267)
- [StyleDDG, arXiv 2025/2026](https://arxiv.org/pdf/2504.06235)
- [PARDON repo](https://github.com/judydnguyen/PARDON-FedDG)

我没能在当前环境里直接核实 `FISC/PARDON` 论文正文中的 exact OfficeHome table values；我能确认的是其 abstract/repo/metadata 显示它们评估了 `Office-Home`，并采用了 `abstracted/interpolative style` 路线，这支持“不要 raw all-to-all sharing”的判断，但不足以让我负责地给出精确数值。
