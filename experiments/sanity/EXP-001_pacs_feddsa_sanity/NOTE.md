# EXP-001 | FedDSA Sanity Check (PACS, 10轮)

## 基本信息
- **日期**：2026-04-05
- **类型**：sanity
- **数据集**：PACS（4域×7类，每域=1客户端）
- **方法**：FedDSA（Decouple-Share-Align）
- **状态**：⏳ 待执行

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
| 平均准确率 | |
| art_painting | |
| cartoon | |
| photo | |
| sketch | |
| 域间标准差 | |
| 最佳轮次 | |

> 结果自动生成到 results/metrics.json

## 结论（必填！写发现和判断，不是只贴数字）

## 问题与备注
- 两张GPU使用率较高（GPU0 ~20GB/24GB，GPU1 ~18GB/24GB），选择GPU 1
- 已修复 main.py 中 `-tau` 参数重复定义的 argparse 冲突
- PACS 原始数据来自 flgo 包自带数据（符号链接到 PFLlib/dataset/PACS/rawdata/PACS）
