# EXP-002 | FedDSA Bug Fix Verify (PACS, 50轮, pretrained)

## 基本信息
- **日期**：2026-04-05
- **类型**：sanity
- **数据集**：PACS（4域x7类，每域=1客户端）
- **方法**：FedDSA（bug修复后）
- **状态**：✅ 已完成

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
- [x] Style bank size 在前几轮增长到 4 ✅ Round 1即增长到4
- [x] Train loss 数值合理 ✅ 从1.96降至~0，走semantic_head正确路径
- [x] Round 0-10 准确率不暴跌 ✅ Round 0: 14.8% → Round 5: 92.5%，无暴跌
- [x] 最终准确率显著高于 EXP-001 的 21.2% ✅ 95.8% vs 21.2%

## 结果

| 指标 | 值 |
|------|---|
| Best平均准确率 | **95.80%** |
| Final平均准确率 | 95.76% |
| art_painting | 93.16% |
| cartoon | 96.76% |
| photo | 97.85% |
| sketch | 95.73% |
| 域间标准差 | 0.0173 |
| 最佳轮次 | eval #5 (Round 25) |
| 训练耗时 | 3294s (~65s/round) |

### 收敛曲线

| Round | Avg Acc | Train Loss | Style Bank |
|-------|---------|------------|------------|
| 0 | 14.8% | 1.959 | 0 |
| 5 | 92.5% | 0.019 | 4 |
| 10 | 89.6% | 0.122 | 4 |
| 15 | 94.8% | 0.000 | 4 |
| 20 | 94.6% | 0.000 | 4 |
| 25 | **95.8%** | 0.000 | 4 |
| 30 | 95.8% | 0.000 | 4 |
| 35 | 95.7% | 0.000 | 4 |
| 40 | 95.8% | 0.000 | 4 |
| 45 | 95.6% | 0.000 | 4 |
| 50 | 95.8% | 0.000 | 4 |

## 结论

4个bug修复全部验证通过：
1. **风格仓库**: 稳定4条（EXP-001仅1条）
2. **Train loss**: 从1.96正常下降到~0（EXP-001报的是错误路径的loss）
3. **Warmup生效**: Round 0→5平滑上升，无EXP-001的暴跌现象
4. **预训练backbone**: 准确率95.8%（EXP-001仅21.2%）

**注意**：95.8%的绝对值不能直接与论文对比（预训练+个性化评估），需要跑同配置基线(FedAvg/FedBN/FedProto)才能衡量FedDSA的真实提升。Train loss趋近0暗示可能过拟合。

## 问题与备注
- Round 10出现准确率回落(92.5%→89.6%)，随后恢复——可能是风格增强开始(warmup_rounds=10)时的短暂扰动
- Train loss从Round 15后一直为0.0000，存在过拟合风险
- 需要跑基线实验(EXP-003~005)才能评估FedDSA的实际贡献
