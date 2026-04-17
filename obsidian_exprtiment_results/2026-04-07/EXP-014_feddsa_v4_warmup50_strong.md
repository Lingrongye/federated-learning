# EXP-014 | FedDSA V4: Long Warmup + Strong Weights

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (基于V3验证的最优配置)
- **方法**：FedDSA V4: warmup=50 + 原版强权重 (1.0/0.1/1.0)
- **模型**：AlexNet from scratch
- **框架**：FDSE (flgo)
- **状态**：✅ 已完成

## 目的
EXP-013(V3 长warmup)显示warmup=50能显著提高稳定性(drops从14降到3)，best达81.68%接近FDSE 82.16%。
V4结合V3的长warmup + 原版的强权重，验证是否能超越FDSE。

## 与之前实验的差异
| 参数 | 原版(EXP-006) | V3(EXP-013) | **V4(本实验)** |
|------|--------------|-------------|---------------|
| lambda_orth | 1.0 | 0.5 | **1.0** |
| lambda_hsic | 0.1 | 0.05 | **0.1** |
| lambda_sem | 1.0 | 0.5 | **1.0** |
| warmup_rounds | 10 | 50 | **50** |
| num_rounds | 500 | 200 | **200** |

## 假设
V3用了较弱权重还能达到81.68%。V4恢复强权重，理论上：
- 收敛到的Best更高（强权重→更强约束）
- 稳定性仍由warmup保证（drops应仍较少）
- 期望Best ≥ 82%

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_v4.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp014.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| Best acc | **79.93%** @Round 48 |
| Last acc | 77.31% |
| Drops>5% | 6 |
| 最弱域@best | 66.67% |
| 最强域@best | 87.76% |
| 稳定性(last20 std) | 0.0030 |
| 与FDSE差距 | **-2.23%** |
| 总轮次 | 200 |

## 结论

**假设未得到验证：V4 (warmup=50 + 强权重) Best仅79.93%，低于V3(warmup=50+弱权重)的81.68%。**

核心发现：
1. **强权重反而不好**——V3(弱权重 0.5/0.05/0.5) Best=81.68% > V4(强权重 1.0/0.1/1.0) Best=79.93%
2. Drops=6(vs原版14)，稳定性有提升但Best下降
3. 这提示：HSIC可能是罪魁祸首，强λ_hsic=0.1让训练难以收敛
4. **验证方式**：EXP-017 (V4去HSIC) Best=82.24% **超越FDSE！** 证实了HSIC有害

V4不是最优，EXP-017才是。
