# EXP-065 | DomainNet Regime Test

## 目的
验证 GPT-5.4 review 的核心诊断:**"style sharing is regime-dependent"**

基于 PACS vs Office 的对比发现:
- **PACS** (high style gap: sketch, art_painting, cartoon, photo) → FedDSA 赢
- **Office** (low style gap: 全 real photos) → FedDSA 输

**DomainNet 是理想验证**:6 个域覆盖 stylized + real 的完整光谱
| Domain | 风格类型 |
|---|---|
| sketch | 极高风格差 (黑白线稿) |
| quickdraw | 极高风格差 (简笔画) |
| clipart | 高风格差 (illustration) |
| painting | 中风格差 (艺术绘画) |
| infograph | 中风格差 (信息图) |
| real | 低风格差 (真实照片) |

## 假设

**H3 (regime-dependent)**:
- FedDSA 在 sketch/quickdraw/clipart 上赢 FDSE (high gap domains)
- FedDSA 在 real 上输或持平 (low gap domain)
- FedDSA 的平均性能接近 FDSE

如果 H3 成立,我们就有足够证据支持 GPT-5.4 建议的 pivot claim:
> "Unconditional style sharing is not universally beneficial in FedDG; its utility is governed by inter-domain style gap."

## 设置
- Dataset: DomainNet (10 classes subset, 6 clients)
- Backbone: AlexNet
- R=200, E=1, B=50, LR=0.05
- 3 seeds (2, 15, 333)

## 运行命令
```bash
# FedDSA
nohup python run_single.py --task domainnet_c6 --algorithm feddsa --gpu 0 \
  --config ./config/domainnet/feddsa_domainnet.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp065_feddsa_s2.out 2>&1 &

# FDSE
nohup python run_single.py --task domainnet_c6 --algorithm fdse --gpu 0 \
  --config ./config/domainnet/fdse_domainnet.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp065_fdse_s2.out 2>&1 &
```

## 结果

### AVG Best (3-seed mean)
| Method | s2 | s15 | s333 | Mean |
|---|---|---|---|---|
| FedDSA | | | | |
| FDSE | | | | |

### Per-domain Delta (FedDSA − FDSE)

| Domain | FedDSA | FDSE | Delta | Predicted regime |
|---|---|---|---|---|
| sketch | | | | FedDSA wins (high gap) |
| quickdraw | | | | FedDSA wins (high gap) |
| clipart | | | | FedDSA wins (high gap) |
| painting | | | | Tie |
| infograph | | | | Tie |
| real | | | | FDSE wins (low gap) |

## 决策树
- **H3 成立** (stylized wins, real loses) → pivot to regime-dependent claim,写 analysis paper
- **H3 部分成立** (win stylized but worse overall) → 需要 layer-wise decomposition 或更深层改动
- **H3 不成立** (all domains lose) → FedDSA 本质问题,考虑彻底 pivot

## 参考:FDSE 论文 DomainNet R500 结果
| Method | ALL | AVG |
|---|---|---|
| FedAvg | 69.17 | 67.53 |
| FedBN | 74.75 | 72.25 |
| Ditto | 75.18 | 72.82 |
| FDSE | 76.77 | 74.50 |

## 结论
