# EXP-089 | Lab-lry 扩展实验（seed 补齐 + 方案 A 加强）

## 基本信息
- **日期**: 2026-04-18
- **服务器**: Lab-lry GPU 1
- **状态**: 🔄 运行中

## 动机

SC2 主力跑 EXP-082-088，但 3 个 PACS Local runs 因 standalone 算法与 PACS task 不兼容而崩溃（c.model is None），资源释放。

Lab-lry GPU 1 有 ~20GB 空闲（GPU 0 被别人 100% 占用），部署 3 个有用实验：

1. **PACS 方案 A s=42**：EXP-086 目前 3-seed {2,15,333} 正在跑，补 s=42 做 4-seed mean（加强证据）
2. **Office 方案 A s=42**：EXP-084 已有 {2,15,333} 3-seed，补 s=42 做 4-seed 对齐 "今日 seed 集"
3. **PACS orth_only LR=0.05 s=4388**：当前 PACS orth_only LR=0.05 只有 3-seed {2,15,333}，补 s=4388 向 FDSE 官方 5-seed {2,15,333,4388,967} 靠拢

## 配置

| # | Task | Config | Seed | 目的 |
|---|------|--------|------|------|
| 1 | PACS | feddsa_orth_lr05_sas.yml | 42 | 方案 A 4-seed {2,15,333,42} |
| 2 | Office | feddsa_orth_lr05_sas.yml | 42 | 方案 A 4-seed {2,15,333,42} |
| 3 | PACS | feddsa_orth_lr05_save.yml | 4388 | 向 5-seed 对齐 FDSE |

**⚠️ Lab-lry 注意事项**：
- GPU 0 被 wjc 100% 占用，只能用 GPU 1
- GPU 1 之前被外部 SIGKILL（其他用户抢），有死亡风险
- 如果 run 3h+ 不更新 → 已挂 → SC2 补跑

## ETA
- PACS E=5 R200 × 2 runs ≈ 6h
- Office E=1 R200 × 1 run ≈ 2h

## 结果（待填）

### PACS 方案 A s=42

| R | ALL Best | ALL Last | AVG Best | AVG Last |
|---|---------|---------|---------|---------|
| - | - | - | - | - |

### Office 方案 A s=42

| R | ALL Best | ALL Last | AVG Best | AVG Last |
|---|---------|---------|---------|---------|
| - | - | - | - | - |

### PACS orth_only LR=0.05 s=4388

| R | ALL Best | ALL Last | AVG Best | AVG Last |
|---|---------|---------|---------|---------|
| - | - | - | - | - |

## 合并后预期

### PACS 方案 A 4-seed {2, 15, 333, 42}
- 如果方案 A 对 PACS 也有效（Art 提升）→ 4-seed mean AVG Best ≥ orth_only baseline 81.4%
- s=42 通常较稳定，应该是正贡献

### Office 方案 A 4-seed
- 当前 3-seed mean AVG Best 89.82
- s=42 加入可能把 mean 拉到 ~89.5-90.0

### PACS orth_only 5-seed
- 当前 3-seed mean 80.41
- s=4388 在 FDSE 论文里是 AVG Last 68.78（崩），可能我们 orth_only 也会崩
- 即使崩，5-seed mean 更贴近 "真实期望"
