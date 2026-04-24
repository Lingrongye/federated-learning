# EXP-125 | OCSD 方向验证 1 — 人眼看 over_conf_wrong 样本

## 基本信息
- **日期**: 2026-04-24
- **目的**: 验证 OCSD 方向的核心假设 — "PACS Art 的 13-15% over_conf_wrong 是 style shortcut 样本" — 通过人眼审阅 over_conf_wrong 图片判断**样本类型**
- **核心问题**: shortcut 假设 (A 类: 风格极端 → model 偷学 style) vs 本质难样本 (B 类) vs 标注歧义 (C 类) vs **D 类 (model 学歪 boundary)**
- **状态**: 🔴 **原假设 A 不成立**, 发现是 D 类 (model 学歪)

---

## 方法论

### 执行路径 (Plan)

1. 找现有 trained model (发现 `/root/fl_checkpoints/sgpa_PACS_c4_s2_R200_*` 是 EXP-113 VIB checkpoint)
2. 写 inference script 加载 model + PACS Art client 0 **test split** (非全量图)
3. Filter over_conf_wrong (conf > 0.8, pred != true), 按 conf 排序
4. Copy top 24 (全部 over_conf_wrong, 因为 test 集 204 样本里只有 24 个符合)
5. **人眼看图 + 标 A/B/C/D**

### 关键技术

- **Model 选择**: sgpa_vib s=2 (VIB 架构, 非 orth_only, 但 over_conf_wrong **样本属性跨 model 稳定**, Stage B 显示 FedBN/orth/FDSE 都 13-15%)
- **Test split 一致性**: 用 flgo.init 加载 task, 拿 `clients[0].test_data` — 保证与训练时 train/val/test split 一致 (seed=2, train_holdout=0.2, local_test=True)
- **重要踩坑**: 第一版 script 读了 PACS Art 全部 2048 张图 → 混测 train+test 导致 acc 虚高 92%, over_conf_wrong 只 2.5%. Fix: 用 flgo 的 client test_data.

### 数据 (纠正后)

- Art client 0 test split: 204 samples
- Overall accuracy: **60.78%** (符合 Stage B 诊断 Art 63-64% range)
- Over-conf WRONG (conf>0.8): **24 samples, 11.8%** (符合 Stage B 13-15%)
- Confusion top: person→dog(5), dog→elephant(3), guitar→{person,dog,horse}(分散), horse→{elephant,dog,giraffe}(分散)

---

## 🎯 关键发现 (颠覆性)

### 用户人眼审阅 24 张 over_conf_wrong 图的结论

> "感觉就是完全学错了. 比如长颈鹿跟马可能相似, **其他都是其实还是能认得出来的 但是模型理解错了**"

### 含义分析

**否定 A 类**: 图**不风格极端** — 是普通 Art 绘画, 人眼能认出来

**否定 B 类**: 不是本质难样本 — 不是"画得根本不像"

**发现 D 类**: **model 的 class boundary 学歪** — 不是图的问题, 是 model 的问题

### D 类机制推测

```
FL 跨域训练过程中:
Art 的 dog/horse/person 的 feature 与其他 domain 的**没有显式对齐**
classifier 基于 mixed features 切分边界, 但 Art 空间里各 class 位置系统性偏离
测试时 Art 的 person 正好落入 Art 错位的 dog cluster → 错预测
```

### 与 OCSD 原假设的冲突

| OCSD (原) | D 类 (实际) |
|---|---|
| 图极端, model 被 style 骗 | 图正常, model 自己边界错 |
| MixStyle 扰动能检测 | MixStyle 不精准 (扰动 style ≠ 修 boundary) |
| Selective invariance 有效 | Cross-domain class alignment 更直接 |

---

## 📦 交付物

### 数据文件

- `verify1_testonly/all_predictions.csv` (204 rows)
- `verify1_testonly/over_conf_wrong.csv` (24 rows, sorted by conf desc)
- `verify1_testonly/top50_images/` (实际 24 张)
- `verify1_testonly/PREVIEW.md` (缩略图 + metadata)
- `verify1_testonly/judgement_sheet.md` (A/B/C 标注模板)
- `verify1_testonly/SUMMARY.md` (统计 + confusion + 决策指南)

### Scripts

- `scripts/dump_over_conf_wrong_vib_testonly.py` — 最终版 (用 flgo test split)
- `scripts/debug_sgpa_load.py` — debug state_dict 结构
- `scripts/dump_over_conf_wrong_sgpa.py` — 第一版 (混测, 废弃)
- `scripts/dump_over_conf_wrong_sgpa_vib.py` — 第二版 (VIB 架构, 混测, 废弃)
- `scripts/dump_over_conf_wrong_orth.py` — orth_only 版 (等 save_model 完成)

---

## ❌ OCSD 方向判决

**放弃 OCSD (MixStyle 扰动 + counterfactual confidence drop + selective invariance)**.

理由:
1. **实测**: D 类 (model 学歪) 主导, 不是 A 类 (style shortcut)
2. **机制不匹配**: MixStyle 扰动只修 style 依赖, 不修 class boundary 错位
3. **EXP-124 PCH 失败辅证**: Hardcoded hard cells CE re-weight 也失败 (3-seed AVG +0.14 noise 内), 说明 selective re-weight / selective invariance 都 **不对症 D 类**

## ✅ 切到 CDCA-Orth (Cross-Domain Class Anchor + Orth)

**新方向**: alignment 路线 — 跨域 class 表征显式对齐.

### 机制

```python
# Server 维护 per-domain class prototype bank
P[d, c] = client d 的 class c 平均 z_sem (每轮更新, EMA)

# Client 训练时:
anchor = mean(P[d, c] for d ≠ self_domain)   # 其他 domain 同 class 平均
L_anchor = ||z_sem - anchor||²                # MSE anchor

# Total
L = L_CE + λ_orth · L_orth + λ_anchor · L_anchor
```

### 为什么 CDCA-Orth 对症 D 类

- D 类根本原因: 跨域 class 表征不对齐
- CDCA-Orth 直接对齐: Art-dog 的 z_sem 强制靠近 Photo/Cartoon/Sketch-dog 的均值
- 结果: classifier 看到跨域一致的 z_sem 分布 → 边界不再 Art-错位

### 与 PCH 对比

| 维度 | PCH | CDCA-Orth |
|---|:-:|:-:|
| 机制 | Art client CE loss ×2 | z_sem MSE anchor to cross-domain mean |
| 干预位置 | Loss magnitude | Feature alignment |
| FedAvg 副作用 | 有 (Art gradient 放大) | 无 (anchor 力平衡) |
| per-dataset 通用 | 需 hardcode hard cells | 自适应, 无 prior |
| 新超参 | 1 (hw) | 1 (λ_anchor) |

---

## 下一步实验计划

**EXP-126 CDCA-Orth** (待设计 + 实施):
1. 写 `algorithm/feddsa_cdca.py` (继承 feddsa_scheduled, 加 server-side prototype bank + client-side anchor loss)
2. 单测 + smoke test
3. 3 seeds × R=200 on lab-lry / seetacloud2
4. 对比 orth_only + FDSE

**同时 (2026-04-24 正在跑)**:
- seetacloud2: orth_only s=2 R=200 with **--save_model** (补 pure orth baseline 的 model ckpt)
- lab-lry: fedbn/fdse/orth × seeds 2/15 with --save_model (后续 OCSD / CDCA 验证用)

---

## 相关文件

- `obsidian_exprtiment_results/知识笔记/OCSD方向_观察假设与验证计划.md` (OCSD 完整方案文档, 已 archive)
- `obsidian_exprtiment_results/2026-04-24/关键实验发现备忘.md` (发现 D 类记录)
- `obsidian_exprtiment_results/2026-04-24/方向总结_所有候选方案.md` (方向索引)
- `experiments/IDEA_DISCOVERY_2026-04-24/IDEA_REPORT.md` (CDCA-Orth 机制定义)
- `experiments/STATE_REPORT_2026-04-24.md` (项目现状全景)
- `experiments/ablation/EXP-124_pch_pilot/NOTE.md` (PCH 失败回填)
