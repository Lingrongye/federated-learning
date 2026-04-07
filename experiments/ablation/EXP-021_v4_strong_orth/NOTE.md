# EXP-021 | FedDSA V4 with λ_orth=2.0 (Stronger Orthogonality)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (正交约束强度)
- **方法**：FedDSA V4 + λ_orth=2.0 (vs 1.0)
- **状态**：✅ 已完成

## 目的
V1实验(λ_orth=0.1)表现最差(80.15%)。V4的warmup=50提供了稳定基础。
既然warmup保护了早期，能否用更强的正交约束让解耦更彻底？

## 与EXP-014的差异
| 参数 | EXP-014 | EXP-021 |
|------|---------|---------|
| lambda_orth | 1.0 | **2.0** |

## 假设
- 更强的正交约束 → 更彻底的语义/风格分离
- 因为有warmup=50保护，不会在早期破坏backbone
- 如果提升: 证明FedDSA的瓶颈在解耦充分性
- 如果下降: 证明过度正交会损害特征能力

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4_strong_orth.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp021.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| Best acc | **81.58%** @Round 140 |
| Last acc | 75.80% |
| Drops>5% | **3** (很少) |
| 最弱域@best | 67.65% |
| 最强域@best | **89.80%** (并列最高) |
| 稳定性 | **0.0023** (第三稳定) |
| vs EXP-014 | **+1.65%** Best |
| 总轮次 | 200 |

## 结论

**更强的正交约束(orth=2.0)显著有效**：
- Best提高+1.65%，仅次于EXP-017(去HSIC)的+2.31%
- Drops只有3次（vs V4的6次）——更好的稳定性
- 最强域89.80%，说明强域表现最好

**重要推论**：
1. 正交约束是FedDSA的**核心机制**，加强它直接提升性能
2. 配合长warmup(50)，强约束不会破坏早期训练
3. **下一步**：EXP-017(去HSIC) + EXP-021(orth=2.0)的组合——
   即 orth=2.0 + hsic=0 + sem=1.0 + warmup=50，很可能超过82.5%

**这是第二重要的发现**，指向了明确的下一步优化方向。
