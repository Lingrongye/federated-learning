# EXP-071 | Domain-Aware Prototype Alignment

## 基本信息
- **目的**: 用域感知多正例 InfoNCE 替代全局均值原型对齐，解决"语义稀释"
- **算法**: feddsa_domain_aware
- **灵感来源**: FedDAP (CVPR 2026) — 域感知原型概念
- **状态**: ⏳ 待执行

## 核心改动

### 问题
当前 FedDSA 的 InfoNCE 将各域同类原型 **平均** 后对齐:
```
G_proto["dog"] = avg(Photo-dog, Art-dog, Sketch-dog, Cartoon-dog)
```
Photo-dog 被迫拉向混有 Sketch/Art 信息的平均原型 → **语义稀释**

### 解决方案: 域感知多正例对比
1. **Server** 存储 per-(class, client_id) 原型，不再只存全局均值
2. **Client** 使用 SupCon-style 多正例 InfoNCE:
   - **正例**: 所有域的同类原型（含自己域）
   - **负例**: 所有域的异类原型
   - 每个正例独立计算，再平均

### 数学形式
```
L_DA = -1/|P(c)| * Σ_{p ∈ P(c)} [sim(z, p)/τ - log Σ_j exp(sim(z, j)/τ)]

P(c) = {P[c, d] | d ∈ all_domains}  # 同类跨域原型集合
```

### vs 原版 InfoNCE
| | 原版 | Domain-Aware |
|---|---|---|
| 原型数 | C 个 (每类1个均值) | C×D 个 (每类每域1个) |
| 正例 | 1个均值原型 | D个域原型 |
| 稀释 | 强 (平均合并) | 无 (保留域结构) |
| 跨域迁移 | 通过共享均值 | 通过多正例对比 |

## 预期效果
- **Office +2-4%**: Photo/Amazon 不再被其他域稀释
- **PACS +0.5-1%**: Sketch/Art 保持自己的原型空间
- 总体: 缩小与 FDSE 的差距

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
EXP_DIR=../../experiments/ablation/EXP-071_domain_aware_protos

mkdir -p $EXP_DIR/results $EXP_DIR/logs

# PACS seed=2
nohup $PY run_single.py --task PACS_c4 --algorithm feddsa_domain_aware --gpu 0 \
  --config ./config/pacs/feddsa_domain_aware.yml --logger PerRunLogger --seed 2 \
  > $EXP_DIR/pacs_s2.log 2>&1 &

# Office seed=2
nohup $PY run_single.py --task office_caltech10_c4 --algorithm feddsa_domain_aware --gpu 0 \
  --config ./config/office/feddsa_domain_aware.yml --logger PerRunLogger --seed 2 \
  > $EXP_DIR/office_s2.log 2>&1 &
```

## 对照
| 数据集 | FedDSA baseline (3-seed) | FDSE (3-seed) | EXP-071 Target |
|---|---|---|---|
| PACS | 80.93 ± 0.30 | 80.36 ± 1.67 | ≥ 81.5 |
| Office | 89.13 ± 2.42 | 90.58 ± 2.22 | ≥ 90.5 |

## 结果
| 数据集 | ALL Best | AVG Best | AVG Last | Gap vs baseline |
|---|---|---|---|---|
| PACS s2 | | | | |
| Office s2 | | | | |

## 结论
