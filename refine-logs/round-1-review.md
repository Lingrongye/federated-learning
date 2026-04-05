# Round 1 Review (GPT-5.4 xhigh)

## Scores
| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8/10 |
| Method Specificity | 6/10 |
| Contribution Quality | 6/10 |
| Frontier Leverage | 5/10 |
| Feasibility | 7/10 |
| Validation Focus | 7/10 |
| Venue Readiness | 6/10 |
| **Overall** | **6.3/10** |

## Verdict: REVISE

## Key Issues
1. **Method Specificity (6)**: z_aug = z_sem + λ*MixStyle太模糊；应明确在哪做增强、InfoNCE正负样本怎么构造、训练loop
2. **Contribution Quality (6)**: 模块太多(解耦+HSIC+对比+风格库+调度)；应精简为一个优雅机制
3. **Frontier Leverage (5)**: 纯ResNet-18在2026显得过时；应至少加一个DINO/CLIP变体实验
4. **Venue Readiness (6)**: 新颖性声称脆弱；2x2矩阵看起来被刻意设计

## Simplification Suggestions
- 去掉need-aware dispatch（除非消融证明大幅提升）
- 去掉L_sem_con如果简单原型拉力够用
- 单个per-domain均值太粗糙，不配叫"bank"
- 核心消融应是2×2: share/no-share × decouple/no-decouple

## Modernization Suggestions
- 加一个frozen DINO/CLIP骨干变体
- 如果只用ResNet-18，明确定位为mechanism-first而非SOTA-architecture-first

## Drift Warning
- 从"修复模糊原型"漂移到"通用风格FedDG"
- 如果收益来自增强而非更干净的原型，锚点丢失
- 向量加法注入可能被认为是注入噪声而非真正的风格迁移
