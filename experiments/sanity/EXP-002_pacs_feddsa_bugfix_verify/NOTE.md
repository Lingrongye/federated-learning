# EXP-002 | FedDSA Bug Fix Verify (PACS, 50轮, pretrained)

## 基本信息
- **日期**：2026-04-05
- **类型**：sanity
- **数据集**：PACS（4域x7类，每域=1客户端）
- **方法**：FedDSA（bug修复后）
- **状态**：⏳ 待执行

## 目的
验证 EXP-001 发现的 4 个 bug 修复后效果：
1. 风格仓库应增长到 4 条（每个域1条）
2. train_loss 应走 semantic_head 正确路径
3. 辅助损失 warmup 应避免早期准确率暴跌
4. 预训练 backbone 应大幅提升准确率

## 与 EXP-001 的差异
| 参数 | EXP-001 | EXP-002 | 原因 |
|------|---------|---------|------|
| pretrained | false | **true** | PACS小数据集需要预训练 |
| lr | 0.01 | **0.005** | 预训练用更小学习率避免破坏特征 |
| global_rounds | 10 | **50** | 观察收敛趋势 |
| warmup_rounds | 5 | **10** | 辅助损失warmup需要更多轮数 |
| eval_gap | 2 | **5** | 减少评估开销 |
| 风格仓库逻辑 | cosine去重(bug) | **per-client slot** | 修复 |
| train_metrics | 旧head(bug) | **semantic_head路径** | 修复 |
| 辅助损失 | 全量(bug) | **线性warmup** | 修复 |

## 运行命令
```bash
cd /home/lry/code/federated-learning/PFLlib/system && \
CUDA_VISIBLE_DEVICES=1 /home/lry/conda/envs/pfllib/bin/python main.py \
    -data PACS -m ResNet18 -algo FedDSA -pt \
    -ncl 7 -nc 4 -gr 50 -ls 5 -lr 0.005 -lbs 32 \
    -eg 5 -t 1 \
    -lo 1.0 -lh 0.1 -ls2 1.0 -tau 0.1 \
    -wr 10 -sdn 5 -sdd 0.95 -sbm 50 \
    -did 1 \
    -edir /home/lry/code/federated-learning/experiments/sanity/EXP-002_pacs_feddsa_bugfix_verify
```

## 验证要点
- [ ] Style bank size 在前几轮增长到 4
- [ ] Train loss 数值合理（不再是未训练head的loss）
- [ ] Round 0-10 准确率不暴跌（warmup生效）
- [ ] 最终准确率显著高于 EXP-001 的 21.2%

## 结果

| 指标 | 值 |
|------|---|
| 平均准确率 | |
| art_painting | |
| cartoon | |
| photo | |
| sketch | |
| 域间标准差 | |
| 最佳轮次 | |

## 结论

## 问题与备注
