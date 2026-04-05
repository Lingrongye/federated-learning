# EXP-001 | FedDSA Sanity Check (PACS, 10轮)

## 基本信息
- **日期**：2026-04-05
- **类型**：sanity
- **数据集**：PACS（4域×7类，每域=1客户端）
- **方法**：FedDSA（Decouple-Share-Align）
- **状态**：✅ 已完成（发现多个bug）

## 目的
验证 FedDSA 代码能正常运行：损失下降、无报错、准确率超过随机水平（>14%）、风格仓库能正常增长。

## 与默认配置的差异
- global_rounds: 200 → 10，原因：仅快速验证，不需要完整训练
- warmup_rounds: 10 → 5，原因：缩短warmup以便在10轮内观察风格增强效果
- times: 3 → 1，原因：sanity check只需1次运行

## 运行
- **命令**：
```bash
cd /home/lry/code/federated-learning/PFLlib/system && /home/lry/conda/envs/pfllib/bin/python main.py \
    -data PACS -m ResNet18 -algo FedDSA \
    -ncl 7 -nc 4 -gr 10 -ls 5 -lr 0.01 -lbs 32 \
    -eg 2 -t 1 \
    -lo 1.0 -lh 0.1 -ls2 1.0 -tau 0.1 \
    -wr 5 -sdn 5 -sdd 0.95 -sbm 50 \
    -did 1 \
    -edir /home/lry/code/federated-learning/experiments/sanity/EXP-001_pacs_feddsa_sanity
```
- **GPU**：GPU 1（RTX 3090）
- **Seeds**：1
- **开始时间**：
- **结束时间**：
- **耗时**：

## 结果

| 指标 | 值 |
|------|---|
| 平均准确率 | 21.2% |
| art_painting | 14.3% |
| cartoon | 28.5% |
| photo | 20.1% |
| sketch | 21.0% |
| 域间标准差 | 0.051 |
| 最佳轮次 | eval #5 (round 10) |
| 训练耗时 | 1085s (~97s/round) |

> 结果自动生成到 results/metrics.json

## 结论

Sanity check通过（无报错、loss下降、准确率超过随机14.3%），但发现以下**4个bug**需修复后才能进入正式实验：

1. **[CRITICAL] 风格仓库始终只有1条**：cosine去重阈值0.95太严格，共享backbone下不同域的μ向量相似度天然>0.95。应改为按client_id管理slot。
2. **[HIGH] train_loss报的是错误路径**：基类`train_metrics()`走`model(x)`即base→head路径，但FedDSA的head从未训练。需重写`train_metrics()`。
3. **[HIGH] 未使用预训练backbone**：PACS只有~10K图片，从头训练ResNet-18非常困难。竞争方法(FDSE/FedBN)都用pretrained。
4. **[MEDIUM] 辅助损失在早期过于激进**：Round 0特征是随机的就施加正交+HSIC+InfoNCE，Round 2准确率从20%暴跌到10%。需要辅助损失warmup。

修复已提交，下次实验(EXP-002)使用修复后代码+预训练backbone。

## 问题与备注
- 两张GPU使用率较高（GPU0 ~20GB/24GB，GPU1 ~18GB/24GB），选择GPU 1
- 已修复 main.py 中 `-tau` 参数重复定义的 argparse 冲突
- PACS 原始数据来自 flgo 包自带数据（符号链接到 PFLlib/dataset/PACS/rawdata/PACS）
- GPU内存: 643MB allocated on cuda:0（模型较小，内存不是瓶颈）
