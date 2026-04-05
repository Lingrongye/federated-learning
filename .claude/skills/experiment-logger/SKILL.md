---
name: experiment-logger
description: "科研实验记录管理技能。每次运行实验、分析结果、复现基线时必须使用此技能。当用户提到'跑实验'、'运行实验'、'复现'、'基线对比'、'消融实验'、'sanity check'、'实验结果'等关键词时自动触发。核心要求：任何实验执行前必须先在实验日志中创建记录条目，执行后必须回填结果和结论。"
---

# 科研实验记录管理技能

## 概述

本技能用于规范化管理科研项目中的所有实验记录。核心理念：**半年后你回看实验记录，能在5分钟内找到任意一次实验的设置、结果和结论。**

每次执行实验时，必须严格遵循"记录先行"原则——先写日志再跑实验，跑完后回填结果。这不是可选的，是强制要求。

---

## 目录结构

首次使用时，检查并创建以下目录结构：

```
experiments/
├── configs/                    # 实验配置文件（yaml）
├── logs/                       # 训练日志（自动生成）
├── results/                    # 结果汇总
│   ├── main_table.csv          # 主表
│   ├── ablation_table.csv      # 消融表
│   └── figures/                # 图表
├── EXPERIMENT_LOG.md           # ★ 实验日志（最重要！）
└── EXPERIMENT_TRACKER.md       # 实验状态追踪表
```

如果目录不存在，静默创建。如果 `EXPERIMENT_LOG.md` 不存在，用下面的模板初始化。

---

## ★ 核心规则：实验日志（EXPERIMENT_LOG.md）

这是整个技能中最重要的文件。**任何实验执行前，必须先在此文件中追加一条新记录。**

### 日志条目模板

每条记录必须包含以下字段：

```markdown
## EXP-{编号} | {日期} | {简短标题}
- **目的**：为什么要跑这个实验？要验证什么假设？
- **类型**：sanity check / 基线复现 / 主实验 / 消融实验 / 参数搜索 / 诊断实验
- **配置文件**：configs/{文件名}.yaml（完整参数见此文件）
- **运行命令**：`完整的命令行`
- **数据集**：{数据集名称}
- **方法**：{算法名称}
- **与默认配置的差异**：{只写这次改了什么，完整参数在yaml里}
- **GPU**：{GPU编号或型号}
- **随机种子**：{seed值}
- **开始时间**：{时间}
- **状态**：🔄 运行中 / ✅ 完成 / ❌ 失败 / ⏸️ 中断

### 结果（实验完成后填写）
- **结束时间**：
- **耗时**：
- **核心指标**：

| 指标 | 值 |
|------|---|
| 平均准确率 | |
| 各域准确率 | |
| 域间标准差 | |
| 最佳轮次 | |

### 结论（必填！不是数字，是你的发现和判断）
- 

### 问题与备注
- 
```

### 编号规则

- 编号格式：EXP-001, EXP-002, ... 递增，永远不重复
- 读取当前 EXPERIMENT_LOG.md，找到最后一个编号，+1 作为新编号
- 如果文件为空，从 EXP-001 开始

### 执行流程

```
用户说"跑实验" / "运行xxx" / "复现yyy"
    │
    ▼
第1步：读取 EXPERIMENT_LOG.md，获取最新编号
    │
    ▼
第2步：在文件末尾追加新条目（填写目的/配置/命令，状态设为🔄）
    │
    ▼
第3步：创建或确认配置文件 configs/{名称}.yaml
    │
    ▼
第4步：执行实验
    │
    ▼
第5步：实验完成后，回填结果、结论、耗时，状态改为✅或❌
    │
    ▼
第6步：更新 EXPERIMENT_TRACKER.md 对应行
    │
    ▼
第7步：如果是主表/消融实验，更新 results/main_table.csv 或 ablation_table.csv
```

---

## 配置文件管理

每次实验必须有一个对应的 yaml 配置文件，保存在 `experiments/configs/` 下。配置文件是完整参数的唯一真实来源，日志里只记与默认值的差异。

### 配置文件模板

```yaml
# experiments/configs/{数据集}_{方法}_{版本}.yaml
experiment:
  id: "EXP-{编号}"
  name: "{描述性名称}"
  date: "{YYYY-MM-DD}"
  purpose: "{一句话目的}"

dataset:
  name: "{数据集名}"
  num_clients: 
  split: "{划分方式}"

model:
  backbone: 
  pretrained: 
  feat_dim: 
  proj_dim: 

training:
  global_rounds: 
  local_epochs: 
  batch_size: 
  lr: 
  optimizer: 
  momentum: 
  weight_decay: 
  seeds: []

method:
  name: "{算法名}"
  # 算法特有参数全部写在这里

notes: ""
```

### 命名规则

```
{数据集}_{方法}_{用途}_{版本}.yaml

示例：
pacs_feddsa_main_v1.yaml          # PACS上FedDSA主实验第1版
pacs_fedavg_baseline.yaml         # PACS上FedAvg基线
pacs_feddsa_ablation_no_share.yaml # 消融：关闭风格共享
digit5_feddsa_sanity.yaml         # Digit-5上sanity check
```

**已有配置文件跑过实验后，永远不修改。新实验写新配置文件。**

---

## 实验状态追踪表（EXPERIMENT_TRACKER.md）

用于快速查看所有实验的状态全景。

### 模板

```markdown
# 实验状态追踪

| 编号 | 日期 | 类型 | 方法 | 数据集 | Seeds | 状态 | 核心结果 | 配置文件 |
|------|------|------|------|--------|-------|------|---------|---------|
```

每次在 EXPERIMENT_LOG.md 新增条目时，同步在此表追加一行。实验完成后回填核心结果和状态。

---

## 结果汇总表（results/main_table.csv）

用于直接生成论文主表，每跑完一组实验填入对应行。

### CSV格式

```csv
method,dataset,seed,domain_1,domain_2,domain_3,domain_4,avg,std,config,exp_id,date
```

domain列名根据数据集调整（PACS: art_painting/cartoon/photo/sketch）。

### 汇总脚本

```python
import pandas as pd
df = pd.read_csv('experiments/results/main_table.csv')
summary = df.groupby(['method','dataset'])['avg'].agg(['mean','std']).round(2)
print(summary.to_markdown())
```

---

## 失败实验的记录

失败的实验同样重要，必须记录。除了标准字段外，额外填写：

```markdown
### 失败原因
- {具体错误信息或现象}

### 解决方案
- {怎么修复的，或还未解决}

### 教训
- {下次怎么避免}
```

记录失败实验可以避免重复踩坑，也是科研诚信的一部分。

---

## Git集成

每次重要实验完成后，建议提交一次：

```bash
git add experiments/EXPERIMENT_LOG.md experiments/configs/ experiments/results/
git commit -m "实验记录: EXP-{编号} {简短描述}"
```

代码变更和实验记录分开提交，方便回溯。

---

## 检查清单

每次实验前自查：

- [ ] EXPERIMENT_LOG.md 中已创建新条目（状态🔄）
- [ ] 配置文件已保存到 configs/
- [ ] 运行命令已记录
- [ ] 当前代码已 git commit（代码版本可追溯）

每次实验后自查：

- [ ] 结果已回填到日志条目
- [ ] **结论已写**（不是数字，是发现和判断）
- [ ] 状态已更新（✅ 或 ❌）
- [ ] EXPERIMENT_TRACKER.md 已更新
- [ ] 如果是主表实验，main_table.csv 已更新

---

## 常见错误提醒

| 错误 | 后果 | 避免方式 |
|------|------|---------|
| 只记命令不建配置文件 | 无法复现 | 每次实验必须有yaml |
| 只记成功不记失败 | 重复踩坑 | 失败实验也写进日志 |
| 改了代码不记版本 | 不知道结果对应哪版代码 | 重要修改先git commit |
| 只贴数字不写结论 | 写论文时忘了发现 | 结论字段强制要求 |
| 多个实验覆盖同一日志 | 结果丢失 | 每次新建条目，永不修改旧条目 |
| 跑完实验忘记回填 | 日志形同虚设 | 完成后立即回填，不要"等会儿再写" |
