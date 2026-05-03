# FDSE 复现严格 review (我自己 review, codex 独立 review 等待中)

**date**: 2026-05-04
**问题**: F²DC framework 下 FDSE PACS s=15 R18 peak 53.40 后 collapse 到 R99 33.58 (drift -19.82pp)
**背景**: FDSE [CVPR'25] 是 paper baseline, 训崩影响 main figure 视觉效果. 我们怀疑复现代码有 bug.

---

## 文件比较

| 文件 | 路径 | 行数 |
|---|---|:--:|
| 我们 | `F2DC/models/fdse.py` | 321 |
| 原版 | `FDSE_CVPR25/algorithm/fdse.py` | 428 |
| paper | `papers/Wang_Federated_Learning_with_Domain_Shift_Eraser_CVPR_2025_paper.pdf` | — |
| paper text | `papers/FDSE_extracted.txt` | — |
| 我们 backbone | `F2DC/backbone/ResNet_FDSE.py` | (DSEConv 照抄 FDSE_CVPR25) |

## 🐛 critical bug (按 priority)

### 🔴 Bug 1: PACS lmbd 配置错 (50× 差异)

**我们配置**: `F2DC/utils/best_args.py` PACS 的 fdse:
```python
'fdse': {
    'local_lr': 0.01,
    'local_batch_size': 64,
    'lmbd': 0.01,           # ← default
    'fdse_tau': 0.5,
    'fdse_beta': 0.1,       # ← default
}
```

**原版 PACS 配置** (来自 `FDSE_CVPR25/task/PACS_c4/log/*.log` 文件名 `algopara_0.50.50.001`):
- **lmbd = 0.5** (我们 0.01 — **50× 错**)
- tau = 0.5 ✓
- **beta = 0.001** (我们 0.1 — **100× 错**)

**FDSE_CVPR25/algorithm/fdse.py** line 96:
```python
self.init_algo_para({'lmbd': 0.01, 'tau':0.5, 'beta':0.1,})
```
这是 default, 但 PACS / Office 实际跑实验时 override 成 dataset-specific config.

**Dataset-specific 实测配置**:
| Dataset | lmbd | tau | beta | LR |
|---|:--:|:--:|:--:|:--:|
| **PACS** | **0.5** | 0.5 | **0.001** | 5e-2 (我们用 1e-2) |
| Office-Caltech10 | 0.05 | 0.5 | 0.05 | 1e-1 (我们用 1e-2) |
| DomainNet | 0.05 | 0.5 | 0.05 | 5e-2 |

**修复**: 改 best_args.py per-dataset:
```python
'fl_pacs': {
  'fdse': {'local_lr': 0.01, 'local_batch_size': 64, 'lmbd': 0.5, 'fdse_tau': 0.5, 'fdse_beta': 0.001},
}
'fl_officecaltech': {
  'fdse': {'local_lr': 0.01, 'local_batch_size': 64, 'lmbd': 0.05, 'fdse_tau': 0.5, 'fdse_beta': 0.05},
}
```

(注: LR 我们继续用 0.01 跟其他 baseline 统一 — 主表协议要求, 不算 bug. 只改 lmbd/beta.)

### 🔴 Bug 2: EPS 数值改写 (改了 KL 语义)

**原 FDSE** line 261-262:
```python
loss_mean += w*((g.pow(2) - fn.pow(2))/(2*vn)).mean()
loss_var  += w*0.5*((torch.log(vn/(v+1e-8)) + v/(vn+1e-8)).mean())
```
分母直接 `vn` 跟 `v+1e-8`/`vn+1e-8`.

**我们 F²DC** line 290-300:
```python
EPS = 1e-2
vn = vn_list[li].clamp(min=EPS)
v_safe = v.clamp(min=EPS)
loss_mean = loss_mean + w * ((g.pow(2) - fn.pow(2)) / (2 * vn)).mean()
loss_var = loss_var + w * 0.5 * ((torch.log(vn / v_safe) + v_safe / vn).mean())
```
我们 clamp vn / v 到 1e-2. **改了 KL 散度数值** — 当 vn < 1e-2 时, 我们的 loss 比原版小 (1/0.01=100 vs 原版 1/0.001=1000).

**理由**: 我们注释说 "实测 batch 2 loss=322k → batch 3 -inf → batch 4 nan" — 是数值稳定 fix.

**问题**: 这个 fix 把 lmbd=0.5 PACS 训练时的 L_reg 大小**人为压制**, 可能让 L_reg 失效, 导致 model 后期 collapse.

**修复**: 改用 1e-8 epsilon (跟原版一致), 加 finite-loss 检查替代 clamp:
```python
loss_mean = w * ((g.pow(2) - fn.pow(2)) / (2*vn + 1e-8)).mean()
loss_var = w * 0.5 * ((torch.log(vn/(v+1e-8) + 1e-8) + v/(vn+1e-8)).mean())
if not torch.isfinite(loss_mean + loss_var):
    continue  # skip this batch L_reg (don't degrade EMA)
```

### 🟡 Bug 3: clip_grad 缺失

**原 FDSE** line 266:
```python
if self.clip_grad > 0:
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
```

**我们** 直接 `loss.backward(); optimizer.step()` 没 clip.

**潜在影响**: 当 lmbd=0.5 大 reg 时, gradient 容易爆炸, 导致 collapse.

**修复**: 加 `torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)` 在 backward 后.

### 🟡 Bug 4: per-layer EMA 修改 (我们 fix vs 原版 latent bug)

**原 FDSE** line 244-245:
```python
fn = None
vn = None
for iter in range(self.num_steps):
    ...
    for g, w, f, v, ln in zip(global_means, weights, feature_maps, global_vars, layers):
        fn = (1. - ln.momentum) * fn + ln.momentum * mf if fn is not None else mf
```
**单一 fn/vn 跨 layer iteration**. AlexNetEncoder 各 layer channel 数不同 (64/192/384/256/256/1024/1024), broadcast 必崩.

**我们 fix** line 270-271:
```python
fn_list = [None] * len(layers)
vn_list = [None] * len(layers)
```
per-layer dict 维护独立 EMA.

**疑点**: 如果原版必崩, FDSE paper 怎么训练出来的? 可能性:
1. 原版 lmbd 大时 R0/R1 跳过 L_reg (`self.server.current_round > 1`), R2 才进入 — 第 1 次 zip 遍历时 fn=None, 第 1 layer 设 mf, 第 2 layer broadcast 错 → RuntimeError → 整个 train 崩
2. **更可能**: 原版作者实际跑实验时**lmbd=0** 或者**没用 L_reg** (lmbd 命名暗示这是 regularization weight), 论文 ablation 才用 L_reg. paper 主表数字是 lmbd=0 跑的 fedavg + DSEConv backbone, 没 L_reg
3. 或者 fn/vn 在某 layer dim 一致 (e.g., 所有 layer 都在 1024 dim) — 但 AlexNetEncoder 不是

**修复**: 我们 fix 是必要 (没 fix 必崩), 但应该**核查 paper main 数字是否 lmbd=0** (如是, 我们 lmbd=0.5 跟 paper 不一致).

### 🟢 Bug 5 (minor): server.iterate 已经实现 OK

我们 `_fdse_aggregate` (line 133-199) 实现了:
- shared_keys (含 dfe/head/stem/shortcut): QP 优化 lambda
- personalized_keys (dse_conv 主体): cosine softmax
- local_keys (dse_bn.running_): mean 聚合 (FedBN 原则)

跟原版 Server.iterate 一致 ✓

### 🟢 backbone DSEConv/DSELinear 实现 OK

`F2DC/backbone/ResNet_FDSE.py` 的 DSEConv 跟 `FDSE_CVPR25/algorithm/fdse.py` line 14-50 一致, dfe_conv + dse_bn + dfe_bn + dfe_bias + dse_conv + relu/leakyrelu 全有.

---

## 总修复计划

按 priority 优先修 1 跟 2:

### Phase 1: 配置修复 (改 best_args.py)
- PACS: lmbd=0.5, beta=0.001
- Office: lmbd=0.05, beta=0.05
- Digits: 保留 default 0.01/0.1 (没 dataset-specific 数据)

### Phase 2: 代码修复 (改 fdse.py)
- 改 EPS 1e-2 → 1e-8 + finite-loss skip
- 加 clip_grad_norm_

### Phase 3: 验证
- sub1 上重跑 PACS lmbd=0.5 s=15 (单 seed 验证), 看 R100 collapse 是否 fix
- 如 fix → 主跑 PACS s={15, 333} + Office s={15, 333} 4 runs

## 预期结果

如果 bug fix 对, FDSE PACS R100 应该:
- 不出现 R18 peak + R99 collapse 模式
- best 接近 70+ (跟 FDSE paper PACS 数字 ~80 还有差距, 因 LR/E/R 协议不同)
- 至少跟 F2DC vanilla / FedAvg 在 R100 时持平 (~65-70)
