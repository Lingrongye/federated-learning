# EXP-081 | FDSE 基线补 s=42 — 同 seed 严格对比

## 基本信息
- **日期**: 2026-04-16 启动 / 2026-04-17 完成
- **算法**: fdse (原版)
- **服务器**: SC2 GPU 0
- **状态**: ✅ R200 完成

## 动机

EXP-076/078 的 orth_only / mse_alpha 用 seeds {2, 333, 42}，但 FDSE 基线（EXP-049 PACS 5-seed / EXP-051 Office 3-seed）没有 s=42。补齐 s=42 让两边可以做 3-seed {2, 333, 42} 的严格同 seed mean 对比。

## 结果（R200 完整，ALL/AVG × Best/Last）

### PACS FDSE s=42

| R | ALL Best | ALL Last | AVG Best | AVG Last | drop AVG |
|---|---------|---------|---------|---------|---------|
| 201 | 81.44 | 78.33 | **79.75** | **76.35** | 3.40 |

### Office FDSE s=42

| R | ALL Best | ALL Last | AVG Best | AVG Last | drop AVG |
|---|---------|---------|---------|---------|---------|
| 201 | 82.94 | 81.35 | **89.16** | **87.63** | 1.53 |

## 合并后严格 3-seed {2, 333, 42} 对比

### PACS R200

| 方法 | ALL Best | ALL Last | AVG Best | AVG Last |
|------|---------|---------|---------|---------|
| FDSE (EXP-049+081 s=2/333/42) | 81.77 | 79.40 | **80.16** | **77.45** |
| orth_only LR=0.1 (EXP-076) | 83.45 | 76.49 | 81.69 | 73.87 |
| ΔvsFDSE | +1.68 | **-2.91** | +1.53 | **-3.58** ❌ |

→ **PACS 同 seed {2,333,42}**：orth_only LR=0.1 Best 略赢但 Last 输 FDSE -3.58%

### Office R200

| 方法 | ALL Best | ALL Last | AVG Best | AVG Last |
|------|---------|---------|---------|---------|
| FDSE (EXP-051+081 s=2/333/42) | 85.05 | 83.86 | **89.89** | **88.44** |
| orth_only LR=0.1 (EXP-076) | 83.74 | 82.55 | 89.41 | 88.47 |
| ΔvsFDSE | -1.31 | -1.31 | -0.48 | **+0.03** ≈ |

→ **Office 同 seed {2,333,42}**：orth_only LR=0.1 AVG Last 打平 FDSE（+0.03）

## FDSE 5-seed 完整数据（EXP-049 + EXP-081，参考用）

PACS R200 5-seed {2, 15, 333, 4388, 967}：ALL 81.78/76.46, AVG **80.24/75.57**（s=4388 Last 暴跌到 68.78 拉垮 mean）

加上 s=42 后 6-seed mean：
- AVG Best = (80.81+79.00+79.93+80.98+80.49+79.75)/6 = **80.16**
- AVG Last = (78.09+76.64+77.92+68.78+76.40+76.35)/6 = **75.70**

## 关键发现

1. **FDSE s=42 AVG Best 79.75 Last 76.35** — 接近其他 seed 均值，不是异常 seed
2. **PACS {2,333,42} orth_only LR=0.1 vs FDSE**：Best +1.53% 但 Last -3.58% 严重落后（LR 本身是 orth_only 的问题）
3. **Office {2,333,42} orth_only LR=0.1 vs FDSE**：AVG Last 几乎打平 (+0.03)
4. **这个 EXP 的价值**：为论文 main table 提供同 seed 可比的 FDSE 基线数据

## 下一步

- 结合 EXP-080 LR=0.05 → PACS orth_only + LR=0.05 才是真正超 FDSE 的方案
- 论文 main table 应报告：
  - {2, 15, 333} 3-seed：orth_only LR=0.05 超 FDSE（4 指标全超）
  - {2, 333, 42} 3-seed：orth_only LR=0.1 Best +1.53% Last -3.58%
  - 5-seed {2, 15, 333, 4388, 967}：若补完 orth_only LR=0.05 5-seed，预期仍超 FDSE
