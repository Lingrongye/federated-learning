# EXP-096 — FedDSA-SGPA Smoke Test (Office R50)

**创建时间**: 2026-04-19 23:00
**目的**: 验证 **FedDSA-SGPA** 算法实现可跑通 (Fixed ETF classifier + pooled whitening + SGPA inference),CE loss 下降,不 NaN。不追求最终性能,纯功能验证。

## 一句话解释

**FedDSA-SGPA** 是 Plan A 的升级:
- **训练端**: 把 `sem_classifier` 从 `Linear(128, K)` 换成**固定 Simplex ETF buffer** (seeded,所有 client 一致,零可训参数,消除 FedAvg 漂移源)
- **推理端**: 每 test batch 通过**双 gate** (entropy + Mahalanobis-in-pooled-whitening-space-of-z_sty) 筛 reliable 样本 → top-m per-class proto bank → cos(z_sem, proto) 分类,未激活类用 ETF fallback
- 完全 backprop-free 推理,训练成本 = Plan A,通信增量 ~66KB/round

## 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| dataset | office_caltech10_c4 | 10 类,4 client (A/W/D/Caltech) |
| backbone | AlexNet + 双 128d 头 | 同 Plan A |
| algorithm | feddsa_sgpa | 新 variant |
| seeds | **2 only** | smoke test 只跑 1 seed |
| num_rounds | **50** | smoke test, 不是 R200 |
| num_epochs | 1 | Office E=1 惯例 |
| lr | 0.05 | Plan A 最优 |
| λ_orth | 1.0 | Plan A |
| τ_etf | 0.1 | Fixed ETF 温度 (refine 决议 β=1 合并到 τ) |
| warmup_r | 10 | whitening 聚合前不启用 SGPA |
| min_clients_whiten | 2 | 至少 2 client 才构造 pooled Σ |

## 成功标准 (smoke test)

- [ ] 训练跑完 R50 不 crash/NaN
- [ ] CE loss 持续下降 (R0 ~2.3 → R50 应 <1.0)
- [ ] Server 成功构造 pooled whitening (log 里看到 `source_mu_k` broadcast)
- [ ] test_with_sgpa 输出合理 accuracy (>50%,即 ETF 能分类)
- [ ] SGPA `reliable_rate` 不全 0 也不全 1 (warmup 后)

## 失败时降级

如果:
- CE 不下降 → 退回 feddsa.py orth_only 基线,检查 classify() 与 model.head(z_sem) 的差异
- Σ_global 奇异 (eigh 报警) → eps_sigma 加大到 1e-2
- SGPA 反而 hurt (sgpa_acc < etf_acc) → 先只看 etf_acc (= Plan A + ETF 的效果)

## 下一步

smoke test 过了后:
1. **EXP-097**: Office 3 seeds × R200 full run
2. **EXP-098**: PACS 3 seeds × R200 (在 SCPR v2 释放 GPU 后)
3. **EXP-099**: ETF-only 消融 (只训练端改,推理还用 ETF argmax,证 ETF 不伤)

## 参考

- FINAL_PROPOSAL: `obsidian_exprtiment_results/refine_logs/2026-04-19_feddsa-eta_v1/FINAL_PROPOSAL.md`
- 代码: `FDSE_CVPR25/algorithm/feddsa_sgpa.py` (608 行)
- 单测: `FDSE_CVPR25/tests/test_feddsa_sgpa.py` (26 tests pass)
- 知识笔记: `obsidian_exprtiment_results/知识笔记/FedDSA-SGPA方案_风格门控原型校正.md`

## 执行记录

```
# 部署命令 (seetacloud2, GPU 0)
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25
PY=/root/miniconda3/bin/python
EXP_DIR=../../experiments/ablation/EXP-096_sgpa_smoke
mkdir -p $EXP_DIR/results $EXP_DIR/logs

nohup $PY run_single.py --task office_caltech10_c4 --algorithm feddsa_sgpa --gpu 0 \
  --config ./config/office/feddsa_sgpa_smoke.yml --logger PerRunLogger --seed 2 \
  > $EXP_DIR/terminal_s2.log 2>&1 &
```

## 结果 (待回填)

| 指标 | 值 |
|------|-----|
| R50 ETF accuracy | 待填 |
| R50 SGPA accuracy | 待填 |
| SGPA reliable_rate | 待填 |
| proto_vs_etf_gain | 待填 |
| 训练耗时 | 待填 |
| 最终 verdict | 待填 |
