# SGPA 诊断监控设计 — 让实验结果可以反推根因

> 2026-04-19 为 FedDSA-SGPA 实验设计的可观测框架。21 个中间指标 + 失败决策树,代码 53/53 测试通过。
>
> **代码**: `FDSE_CVPR25/diagnostics/sgpa_diagnostic_logger.py`
> **测试**: `FDSE_CVPR25/tests/test_sgpa_diagnostic_logger.py` (53 passed, 4.6s)

## 一、为什么要做这件事

过去 85+ 实验的共同失败模式:**只记录 accuracy,不记录过程**。
- InfoNCE 崩了? 不知道是哪一轮开始 cos_grad 穿零的 (直到 EXP-075 才发现)
- SCPR 失效? 不知道是 style attention 在哪一层坍塌 (事后推测 K=4 注意力不足)
- ETF 可能会伤训练? SGPA 可能永远不激活? 没有监控就只能再试一次

**本设计的目标**: SGPA 实验跑完后,**光看 diag 日志就能判断失败根因**,不再需要重跑。

## 二、三层 21 个指标总览

```
Layer 1 训练端 (6 个) ── 特征几何 + 梯度健康
Layer 2 聚合端 (2 个) ── 跨 client 漂移度
Layer 3 推理/SGPA (13 个) ── gate 行为 + 原型质量 + 预测一致性
```

每个指标定义: **公式 → 正常范围 → 异常信号 → 根因**。

### Layer 1: 训练端 (6)

| # | 指标 | 公式 | 正常范围 | 异常 → 根因 |
|---|------|------|---------|------------|
| 1 | **正交性** | `(norm(z_sem)·norm(z_sty))².mean()` | [0.02, 0.15] | >0.3 → L_orth 被 CE 压制 |
| 2 | **ETF 对齐度** | 对每类 `cos(z_sem_c_mean, M[:,c])` 取均值 | R0~0 → R200 接近 1 | 不升 → 特征没学会对齐顶点 |
| 3 | **类内聚合** | 同类 `pairwise cos` 均值 | 升到 >0.6 | 不升 → backbone 没学到语义 |
| 4 | **类间分离** | 类 centers `pairwise cos` 均值 | 降到 <0.2 | 不降 → 混淆严重 |
| 5 | **梯度夹角** | `cos(∇L_CE, ∇L_orth)` (对某一共享层) | >0 | <0 → 两 loss 打架 (InfoNCE 崩的死因) |
| 6 | **梯度强度比** | `‖∇L_CE‖ / ‖∇L_orth‖` | [1, 10] | >100 → L_orth 被压;<0.1 → L_orth 过强 |

### Layer 2: 聚合端 (2)

| # | 指标 | 公式 | 正常 | 异常 → 根因 |
|---|------|------|------|------------|
| 7 | **client 类中心方差** | 各 client 的 `z_sem_c_mean` stack 后 `var(dim=0).mean()` | 随训练下降 | 不降 → 各 client 语义没对齐 |
| 8 | **参数漂移** | `mean_k ‖θ_k - θ_global‖_2` | 随训练下降 | 稳定不降 → 还在漂移 |

**Plan A baseline 对比**: Plan A 也要跑一次上述 Layer 1+2 指标,两个数据集对比。**ETF 版本的指标 7/8 应该明显比 Plan A 版本小 (这是 ETF 消除漂移的证据)**。

### Layer 3: 推理/SGPA (13)

| # | 指标 | 公式 | 正常 | 异常 → 根因 |
|---|------|------|------|------------|
| 9 | **reliable rate** | `(H<τ_H & dist<τ_S).mean()` | [0.3, 0.7] | =0 gate 太紧;=1 gate 没过滤 |
| 10 | **entropy gate rate** | `(H<τ_H).mean()` | [0.3, 0.9] | 异常则 τ_H calibrate 坏 |
| 11 | **dist gate rate** | `(dist<τ_S).mean()` | [0.3, 0.9] | 异常则 τ_S calibrate 坏 |
| 12 | **dist_min 分布** | {min, p10, p50, p90, max} | 双峰或长尾 | 单峰无 gap → 白化失效 |
| 13 | **白化 scatter reduction** | 白化前后 cross-client `d.std(-1).mean()` 比率 | <0.5 | ≈1 → 白化没起作用 |
| 14 | **Σ 条件数** | `cond(Σ_global)` | <1e4 | >1e6 → 病态,增大 ε |
| 15 | **proto bank 填充** | `{c: len(supports[c])}` | 每类 ≥10 | 某类 0 → 该类在 test 分布稀疏 |
| 16 | **proto 对 ETF 偏移** | `1 - cos(proto[c], M[:,c])` 每类 | 渐变 | 不偏 → proto 没做校正 (SGPA 没作用) |
| 17 | **fallback rate** | `(1 - activated).mean()` | <0.3 | >0.7 → SGPA 几乎没接管 |
| 18 | **proto_pred accuracy** | `(pred_proto == y).mean()` | 应 ≥ etf_acc | 比 etf 差 → SGPA 拖后腿 |
| 19 | **etf_pred accuracy** | `(pred_etf == y).mean()` | 基线参考 | — |
| 20 | **预测一致率** | `(pred_proto == pred_etf).mean()` | 0.6~0.95 | 过低 → proto/etf 严重分歧,需排查 |
| 21 | **proto vs etf gain** | `proto_acc - etf_acc` | >0 | ≤0 → **核心 bug,SGPA 无效** |

## 三、失败决策树 (跑完就能 debug)

### 分支 1: SGPA 没提分 (proto_vs_etf_gain ≤ 0)

```
proto_vs_etf_gain ≤ 0
├─ reliable_rate < 0.1 → gate 太紧
│   └─ 调 τ_S quantile (目前 p30, 试 p50) / τ_H (目前 p50, 试 p70)
│
├─ reliable_rate > 0.95 → gate 没作用
│   ├─ 检查 dist_min 分布: 若无 gap,白化失效 → 查指标 13/14
│   └─ τ_H/τ_S 过大 → calibrate quantile 降 (p10/p20)
│
├─ proto_pred acc < etf_pred acc (很差)
│   ├─ 可靠样本伪标签错误率高 → 查 Layer 1:
│   │   ├─ intra_cls_sim 低? → backbone 语义差
│   │   └─ etf_align 低? → ETF 几何没对准
│   └─ top-m 太大导致污染 → 减小 m_top (35→20)
│
├─ proto vs ETF 偏移 (指标 16) ≈ 0 → proto 没校正
│   ├─ reliable 样本全是同类 (proto 退化到 ETF vertex) → 检查 proto_fill 分布
│   └─ 可能需要更多 test batch 才能 fill 起来 → 延长 warmup
│
└─ 某类 proto 永远不激活 (指标 15 某类 = 0)
    ├─ test batch 中该类样本少 → 不可避免,依赖 ETF fallback
    └─ warmup 5 batch 不够 → 延长到 10
```

### 分支 2: ETF 伤训练 (指标 7/8 没比 Plan A 好)

```
ETF 训练指标没比 Plan A 好 OR 精度下降
├─ etf_align (指标 2) 不升
│   ├─ τ_etf 太大,logits 无区分度 → 调小 τ_etf (0.1 → 0.05)
│   └─ ETF β 系数错 → 查构造代码 assert
│
├─ intra_cls_sim (指标 3) 不升 OR inter_cls_sim (指标 4) 不降
│   └─ L_orth 权重 λ_orth 过大,压 CE → 降 λ_orth (1.0 → 0.5)
│
├─ cos(∇CE, ∇orth) 变负 (指标 5)
│   └─ ETF 与 L_orth 在 z_sem 上冲突 → 加 warmup 让 CE 先跑 10 轮再启 L_orth
│
└─ client_center_var (指标 7) 不降
    └─ ETF 理论上消除漂移但实践无效 → 或者方案失败,或 seed 不够
```

### 分支 3: 推理阶段早期崩 (warmup 后前几 batch)

```
Warmup 后前 5~10 batch 精度骤降
├─ τ_H/τ_S calibrate 坏 (第一批次特殊)
│   → 把 warmup 从 5 延长到 10 batch,或用 running EMA 平滑
│
└─ proto 初始化差
    └─ 检查 cold-start: proto 应初始化为 ETF vertex 而非 zero
```

## 四、集成到 client/server 的 hook 点

### 训练端 (clientdsa.py)

```python
# 在 train() 循环里每 E=5 epoch 取最后一个 batch 做 snapshot
if self.diag_enabled and self.current_epoch % 5 == 0:
    # Layer 1 metrics (用最后一个 batch 的 z_sem, z_sty, labels)
    metrics = {}
    metrics['orth'] = DL.orthogonality(z_sem, z_sty)
    metrics['etf_align_mean'], _ = DL.etf_alignment(z_sem, labels, self.M, K)
    metrics['intra_cls'] = DL.intra_class_similarity(z_sem, labels, K)
    metrics['inter_cls'] = DL.inter_class_similarity(z_sem, labels, K)
    
    # 梯度指标: 用两次 backward 分别取 ∇L_CE 和 ∇L_orth
    # (实现需要 save_grad hook 或两次 loss.backward 再取)
    metrics['cos_grad'] = DL.gradient_cosine(grad_ce_flat, grad_orth_flat)
    metrics['grad_norm_ratio'] = DL.gradient_norm_ratio(grad_ce_flat, grad_orth_flat)
    
    self.diag_logger_train.record(round_id=self.current_round, metrics_dict=metrics)
```

### 聚合端 (serverdsa.py)

```python
# 在 aggregate_parameters() 里,聚合前记录
if self.diag_enabled:
    # 所有 client 的 z_sem_class_means 需要每轮上传 (轻量, 128×K floats per client)
    center_var = DL.client_center_variance(all_client_centers)
    param_drift = DL.param_drift(all_client_params, self.global_model.state_dict()['backbone.conv1.weight'])
    self.diag_logger_agg.record(round_id=self.current_round, metrics_dict={
        'client_center_var': center_var,
        'param_drift': param_drift,
    })
```

### 推理端 (clientdsa.test_with_sgpa)

```python
# 每个 test batch 都记录
with torch.no_grad():
    # ... 计算 z_sem, z_sty, logits_etf, H, dist_min, reliable, proto, pred ...
    
    metrics = {}
    metrics.update(DL.gate_rates(H, dist_min, self.tau_H, self.tau_S))
    metrics.update(DL.dist_distribution(dist_min))
    metrics.update(DL.whitening_reduction(z_sty, z_sty_white, mu_raw, mu_white))
    metrics['sigma_cond'] = DL.sigma_condition_number(self.Sigma_global)
    _, metrics['proto_fill_mean'] = DL.proto_fill(self.supports, K)
    metrics['proto_etf_offset_mean'], _ = DL.proto_etf_offset(self.proto, self.M, K)
    metrics['fallback_rate'] = DL.fallback_rate(activated)
    metrics.update(DL.prediction_accuracy(pred_proto, pred_etf, labels))  # labels=真实y
    
    self.diag_logger_test.record(round_id=self.current_round, metrics_dict=metrics)
```

## 五、离线分析: 9 张诊断图

跑完实验后,画 9 张图一眼看问题 (脚本模板):

```python
# plot_diagnostics.py (后续实现,这里只列要画的图)
# 1. 训练端: etf_align / intra_cls / inter_cls 3 曲线 vs round
# 2. 训练端: cos_grad / grad_norm_ratio vs round
# 3. 聚合端: client_center_var Plan A vs ETF 两条线
# 4. 推理端: reliable_rate / fallback_rate 每 batch
# 5. 推理端: dist_min 分布直方图 (每 client 一张)
# 6. 推理端: 白化前后 scatter 对比
# 7. 推理端: proto_fill 柱状图每类
# 8. 推理端: proto_etf_offset 热力图
# 9. 推理端: proto_acc vs etf_acc per domain
```

## 六、测试验证结果 ✅

**文件**: `FDSE_CVPR25/tests/test_sgpa_diagnostic_logger.py`
**运行**: `D:/anaconda/python.exe -m pytest FDSE_CVPR25/tests/test_sgpa_diagnostic_logger.py -v`

### 测试结果 (2026-04-19 跑通)
- **53 / 53 tests passed** (4.59 秒,零失败)
- 覆盖:
  - TestOrthogonality × 4 (identical, orthogonal, antiparallel, B=1 edge case)
  - TestETFAlignment × 4 (perfect, random, missing_classes, empty_batch)
  - TestIntraClassSimilarity × 3
  - TestInterClassSimilarity × 3
  - TestGradientCosine × 3 (parallel, antiparallel, orthogonal)
  - TestGradientNormRatio × 3 (equal, triple, zero_orth 不 inf)
  - TestClientCenterVariance × 3 (identical, nonzero, two_clients 精确数值验证)
  - TestParamDrift × 2
  - TestGateRates × 3
  - TestDistDistribution × 1
  - TestWhiteningReduction × 2
  - TestSigmaConditionNumber × 2 (identity, degenerate 1e-6)
  - TestProtoFill × 2
  - TestProtoETFOffset × 4 (aligned, opposite, none, zero_proto)
  - TestFallbackRate × 3
  - TestPredictionAccuracy × 3
  - TestRecordAndDump × 6 (含 tensor 序列化 / multi-client / 空 dump 不创建文件 / invalid stage assert)
  - TestEndToEnd × 2 (完整训练轮 workflow + 完整推理 batch workflow)

### 关键验证 (回答"代码真能用吗")
- ✅ **数学公式正确**: ETF vertex 对齐时 `etf_alignment = 1.0`;反向对齐时 `proto_etf_offset = 2.0`
- ✅ **Edge case 稳定**: B=1 时 orthogonality 不 NaN;全 reliable/none reliable 都返回合理值
- ✅ **数值稳定**: `zero_orth` 时 `grad_norm_ratio` 返回大数但不 `inf`
- ✅ **序列化**: tensor/numpy 自动转 JSON;multi-client 分文件
- ✅ **端到端**: 6 个 Layer 1 + 13 个 Layer 3 metric 全部通过完整 workflow

## 七、使用规范

1. **实验前**: 导入 `from diagnostics import SGPADiagnosticLogger as DL`,在 client/server init 时创建 3 个 logger (train/aggregate/test)
2. **实验中**: 按 §四 的 hook 点调用 `.record()`
3. **实验后**: `.dump()` flush 剩余 buffer,用 §五 脚本画 9 张图
4. **异常时**: 根据 §三 决策树找根因,不要直接重跑

## 八、下一步

- [ ] 集成到 `clientdsa.py` / `serverdsa.py` (等 SGPA 方案实际开发时一起加)
- [ ] 实现 §五 9 张诊断图的 matplotlib 脚本 (可以等第一次实验数据出来后再实现)
- [ ] 在 CLAUDE.md 里添加 "SGPA 实验必须开启诊断日志" 的硬性规则
