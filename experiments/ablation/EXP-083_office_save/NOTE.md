# EXP-083 | Office orth_only LR=0.05 + 保存 best checkpoint (方案 B 基础)

## 基本信息
- **日期**: 2026-04-17 启动 / 2026-04-18 结果
- **算法**: feddsa_scheduled (mode=0, 纯 orth_only)
- **服务器**: Lab-lry GPU 1
- **状态**: 🔄 运行中
- **关联**: 方案 B (Prototype-based classifier) 的训练基础

## 动机

之前 Office orth_only LR=0.05 虽然达到 AVG Best 89.44（低 FDSE 90.58 -1.14%），但：
1. 没有保存模型 checkpoint → 无法做原型推理 ablation
2. 无法做误分类可视化（Caltech 是瓶颈，acc 74.1% vs FDSE 77.7%）

**本实验 = orth_only LR=0.05 严格复跑 + 打开 se=1 保存 best 轮次模型**，供：
- 方案 B：用 class prototype 做 test-time 推理 vs FC head 推理 (ablation)
- 误分类可视化：看 Caltech 上误分类图片

## 配置

| 参数 | 值 |
|------|---|
| seed | 2, 15, 333 (对齐 FDSE EXP-051) |
| LR | 0.05 |
| R | 200 |
| E | 1 (Office) |
| sm | 0 (orth_only) |
| se | 1 (保存 best checkpoint) |

## 预期结果

与之前 EXP-080 Office LR=0.05 结果相同（不应有差异，只是开了 se flag）：

| 指标 | 预期 | FDSE 基线 (EXP-051) |
|------|------|---------------------|
| ALL Best 3-seed | ≈ 83.87 | 86.38 |
| ALL Last 3-seed | ≈ 83.34 | 85.05 |
| AVG Best 3-seed | ≈ 89.44 | 90.58 |
| AVG Last 3-seed | ≈ 88.71 | 89.22 |

+ 每 seed 保存 best round 的 global + per-client checkpoints 到 `~/fl_checkpoints/`

## 部署

| 服务器 | GPU | seeds |
|--------|-----|-------|
| Lab-lry | GPU 1 | 2, 15, 333 |

## 下一步（训练完成后）

1. scp 所有 checkpoints 到本地 / 分析服务器
2. 运行 `eval_fl_errors.py --ckpt_dir ~/fl_checkpoints/<tag>` 做误分类可视化
3. **方案 B 原型推理 ablation**：加载 checkpoint 的 `semantic_head` → 计算每类原型 → argmin 距离作为 prediction → 对比 FC head prediction
4. 如果原型推理在 Caltech 上明显更好 → 方案 B 有效

## 结果（待填）

### Office 3-seed (2/15/333) R200

| seed | ALL Best | ALL Last | AVG Best | AVG Last | best@R | 崩溃? |
|------|---------|---------|---------|---------|-------|-------|
| 2 | - | - | - | - | - | - |
| 15 | - | - | - | - | - | - |
| 333 | - | - | - | - | - | - |
| **mean** | - | - | - | - | - | - |

### 方案 B 原型推理 ablation（待 eval script 跑完）

| 推理方式 | Caltech | Amazon | DSLR | Webcam | mean |
|---------|---------|---------|------|--------|------|
| FC head (orth_only 训练默认) | - | - | - | - | - |
| Prototype (argmin dist) | - | - | - | - | - |
| Δ | - | - | - | - | - |
