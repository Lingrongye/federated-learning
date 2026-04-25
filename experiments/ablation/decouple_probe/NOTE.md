# Decouple Probe — 可视化双头在 trunk 层面的共享程度

**日期**: 2026-04-24
**目的**: 验证 Codex R2 的 "shared-trunk 诅咒" 假说,看 L_orth 是否真的让两路在 weight/channel 层面独立

## 方法

对 3 个 R200 ckpt,提取 `semantic_head.0.weight` 和 `style_head.0.weight`([128, 1024]),算:
- `corr_abs_mean` = `|W_sem·W_sty^T|.abs().mean()` — 行向量 cos 的绝对均值
- `channel_usage_pearson` = `pearson(|W_sem|.mean(0), |W_sty|.mean(0))` — 两路对每个 trunk channel 的依赖是否同向
- `heavy_overlap` = top-25% sem channel 与 top-25% sty channel 的重合数(随机 = 64/256)

脚本: `FDSE_CVPR25/scripts/visualize_decouple.py`

## Ckpt 选取

| label | path | config |
|---|---|---|
| PACS_s2_vib | `/root/fl_checkpoints/sgpa_PACS_c4_s2_R200_1776795770/global_model.pt` | EXP-113 VIB (uw=1, uc=1, lo=1) |
| PACS_s15_vib | `sgpa_PACS_c4_s15_R200_1776795835` | 同上 |
| Office_s15_vib | `sgpa_office_caltech10_c4_s15_R200_1776796210` | plain arch (非 VIB 的 EXP-113 orth_uc1) |

## 结果

| label | corr avg | corr max | usage pearson | heavy-overlap |
|---|:-:|:-:|:-:|:-:|
| PACS s=2 | 0.0254 | 0.119 | +0.040 | 64/256 (25%) |
| PACS s=15 | 0.0238 | 0.121 | −0.011 | 61/256 (24%) |
| Office s=15 | 0.0247 | 0.120 | −0.016 | 56/256 (22%) |
| random baseline | — | — | 0 | 64/256 (25%) |

figs: `experiments/ablation/decouple_probe/figs/decouple_*.png`

## 核心发现

**三个数据集结果完全一致,是结构性结论**:

1. **L_orth 在权重层面做得近乎完美**: corr avg 0.025 → 128 sem 行与 128 sty 行几乎两两正交
2. **两路对 trunk channel 的使用已经独立**: pearson ≈ 0, heavy-overlap ≈ random 25% → **两路挑的是不同的 channel 子集**
3. **但 sem 整体 |weight| > sty 整体 |weight|** → L_CE 只训 sem 所以 sem 的 magnitude 被放大

## 颠覆原假说

**原假说(Codex R2 描述)**: 两路共享 trunk channel → probe 读出 z_sty class = "两路读同一批 channel 导致 class 重复"

**实际**: 两路 channel 使用完全独立,但 probe 仍读出 z_sty class 0.81 (EXP-111)

**正确机制**:
- L_CE 让 trunk 的**每个 channel 都含 class 信息**(pooled 所有维度都被 CE 梯度影响过)
- 无论 z_sty 挑哪批 channel,**每个 channel 都能读出 class**
- **病不在 head 层的共享,而在 trunk 本身的 class-wide dispersion**

## 对三个选项的判决

| 选项 | 原判 | 诊断后新判 | 理由 |
|---|:-:|:-:|---|
| A 权重行正交 | 冗余,可能无效 | **完全无效** | corr 已 0.025,物理下限附近 |
| B channel mask 互斥 | pooling 抹平,mask 难学 | **完全无效** | channel 使用已经自然独立了 |
| C spatial mask (F2DC) | 侵入大但真 F2DC | **仍是唯一 head 路线** | 换维度(空间而非 channel),空间上 class 不是全局弥散 |

## 新的行动路径

1. **弃用 A 和 B** — 它们解决的问题不存在
2. **要做 F2DC 必须上 C(spatial mask)** — 但要 ~200 行 encoder 重写
3. **绕开 head 战场** — 跨域 class prototype anchor:
   - Client 上传 per-class mean(z_sem) → server 聚合 global proto
   - Client loss += `λ·||z_sem - global_proto[y]||²`
   - 让同 class 跨 domain 的 z_sem 对齐,不管 z_sem 读哪些 channel

## 后续建议实验

- **EXP-125** (提议): 跨域 class prototype anchor. 先在 Office 跑 3 seeds, Office baseline -1.49pp 差距如能填上则 ship
- 可选: 做更细的 trunk channel probe — 给每个 channel 单独训 linear classifier 看能不能读 class(验证"每个 channel 都含 class"这个假说)
