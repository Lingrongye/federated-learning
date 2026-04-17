# EXP-025 | FedDSA without InfoNCE (Core 3 losses only)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (loss简化)
- **方法**：只保留CE + CE_aug + Orth (删InfoNCE + HSIC)
- **算法**：feddsa (原版，config改动)
- **状态**：⏳ 待执行

## 目的
验证InfoNCE的必要性。假设：CE_aug已经隐式做了原型对齐，InfoNCE可能冗余。
如果Best不降 → 可以简化方法学，论文故事更干净
如果Best降 → 证明InfoNCE必要，需要EXP-028/029/030解决冲突

## 与EXP-017的差异
| 参数 | EXP-017 | EXP-025 |
|------|---------|---------|
| lambda_sem (InfoNCE) | 1.0 | **0.0** |

## Loss结构
```
EXP-017:  CE + CE_aug + λ_orth*Orth + λ_sem*InfoNCE   (4个loss)
EXP-025:  CE + CE_aug + λ_orth*Orth                    (3个loss)
```

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa --gpu 0 \
    --config ./config/pacs/feddsa_exp025.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp025.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| vs EXP-017 | |

## 结论
