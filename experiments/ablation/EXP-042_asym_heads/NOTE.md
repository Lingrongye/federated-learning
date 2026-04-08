# EXP-042 | Asymmetric Heads + Residual + L2-norm

## 基本信息
- **方法**：semantic_head深(3层+残差)，style_head浅(1层+L2norm)
- **算法**：feddsa_asym
- **状态**：⏳ 待执行

## 目的
原版两个head对称(都2层)，但任务难度不同：
- 语义映射难 → 需要深层
- 风格只是统计量 → 浅层即可
- L2归一化风格 → 更稳定

## 架构改动
```python
# Semantic: 3层 + 残差
sem = relu(bn(linear1(h))) → relu(bn(linear2())) → linear3()
z_sem = sem + 0.1 * residual_proj(h)

# Style: 1层 + L2归一化
z_sty = F.normalize(style_head(h), dim=1)
```

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_asym --gpu 0 \
    --config ./config/pacs/feddsa_exp042.yml \
    --logger PerRunLogger --seed 2 > /tmp/exp042.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|

## 结论
