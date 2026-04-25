请用 GPT-5.4 严格审计以下 4 个文件,找出所有 bug / 设计缺陷 / 训练不稳定风险点。

# Context

联邦跨域学习项目 FedDSA-DualEnc 算法实现,框架是 flgo (FDSE_CVPR25 fork)。

- Backbone: AlexNet from scratch (1024-d pooled feature)
- 方法: 双 encoder (E_sem 512d FC + E_sty 16d VAE) + AdaIN-style modulation decoder + 跨 client 风格仓库 + cycle anatomy consistency
- 4-loss: L_CE + 0.001·L_rec + 1.0·L_saac + 0.01·L_dsct + KL_warmup·L_kl
- 聚合: encoder + E_sem + decoder + classifier 进 FedAvg, E_sty (mu_head + logvar_head) + BN running stats 严格本地
- BiProto 之前因 EMA Pd 自循环坍缩失败 (z_sty SVD ER = 2.73), 这次必须避免类似 mode collapse

# 核心要审查的文件 (绝对路径)

1. `D:\桌面文件\联邦学习\FDSE_CVPR25\algorithm\feddsa_dualenc.py` — 主算法 (Server + Client + 4 loss)
2. `D:\桌面文件\联邦学习\FDSE_CVPR25\scripts\dualenc_probes.py` — probe diagnostics
3. `D:\桌面文件\联邦学习\FDSE_CVPR25\scripts\dualenc_visualize.py` — 可视化 dump
4. `D:\桌面文件\联邦学习\FDSE_CVPR25\config\office\feddsa_dualenc.yml` — config

# 参考文件 (现有 baseline, 看新代码是否兼容)

`D:\桌面文件\联邦学习\FDSE_CVPR25\algorithm\feddsa.py` — orth_only 已验证版

# 单元测试已经过的项 (11/11 PASS)

- 模型 forward 形状 / AdaIN IN 数学 / VAE reparam 可微 / 4 loss 各自非零梯度 / saac GT detach 验证 / decoder 真用 z_sty / 聚合 key 分类正确 / normalize_target 边界 / kl warmup ramp / sample_swap 边界 / dsct InfoNCE smoke

# 请重点检查

## 架构正确性

1. `Server._init_agg_keys` 把 BN running_*/num_batches_tracked private, 把 BN affine (gamma/beta) shared 是 FedBN 半本地策略 — 这跟 feddsa.py 一致吗? 在 dual-encoder 上有没有问题?
2. `Server.pack` 的 style bank dispatch 逻辑: 排除当前 client 自己 + 当 bank 为空时 fallback 到全部. 这个 fallback 在 single-client edge case 下会泄漏自己 z_sty, 是否需要禁止?
3. `Server.iterate` 里 `for cid, samples in zip(self.received_clients, style_samples_list)` — 这个 zip 顺序是否保证一致?
4. `_aggregate_shared` 用 sample-count 加权, 但 BN affine (weight/bias) 在 shared_keys 里, 这个聚合是否会破坏 FedBN 本地化的初衷 (gamma/beta 应该跟 running stats 一起本地)?

## 训练稳定性

5. Cycle 二次 forward 的 GPU memory: encode + heads + decode + encode + heads, 是不是会让 batch=50 + AlexNet 在 24GB 4090 上 OOM? 需要 grad checkpointing 吗?
6. L_saac 用 F.l1_loss 默认 mean reduction, 在 sem_dim=512 上 mean 大概多大? 跟 lambda_saac=1.0 配合后是否会主导 CE (大约 1-2)?
7. L_kl 在 KL warmup 完后 = 0.01 * sum_over_16_dim_mean_over_batch, 这个量级是多少? 会不会 posterior collapse?
8. `_dsct_loss` 里把 batch 内除自己外都当正例, 会不会 batch 内有同 class 不同 view 但被当负例 (CDDSA 是否也这样)?
9. `_sample_swap` 里 `alpha = alpha / alpha.abs().sum(...)` 归一化让 sum |alpha| = 1, 但混合后 z_sty_swap 量级跟原 z_sty 量级是否一致? VAE 训练后 z_sty ~ N(0, 1), 混合后可能 var < 1 (cancellation), decoder 看到的分布会偏离训练分布吗?

## 代码质量 + 边界

10. `_compose_upload` sub-sample 用 `torch.randperm(N)[:200]`, 但每个 client epoch 之间没有 seed 控制, 跨 round 风格仓库样本会变化, 这是 desired 还是 bug?
11. `FedDSADualEncModel` 的 `forward(x)` 默认走 z_sem -> head, framework eval 时是否真的会调用这个 path? (orth_only 也这样, 应该 OK 但确认)
12. `_normalize_target` 的 [0,1] vs [-1,1] 自动判断, 用 `x.min() >= -0.5 and x.max() <= 1.5`, 边界合理吗? PACS 的 transform 是 PILToTensor 后除 255, 应该 [0,1]; 但如果某个 batch 全是黑色 (x_max < 0.5) 会怎样?
13. `Decoder.upsample` 最后一层 `nn.Tanh()` 输出 [-1, 1], 但如果原始数据是 [0, 1] 范围, L_rec target 必须是 x*2-1 (代码确实做了) — 验证一下计算, 没遗漏.
14. `Server.pack` 用 `copy.deepcopy(self.model)` 每轮每 client 一次, 4 client × R200 = 800 次 deepcopy, 全模型 ~30MB, 总计 24GB 内存压力. 是否需要复用?

## Novelty 防御

15. cycle GT 端 detach 是核心防 BiProto-style mode collapse 的设计. 但在 saac_active=False 时根本不进 cycle path, 早期 round 风格仓库还没建起来, 这段时间 z_sty 学什么? KL push toward N(0,I) + L_dsct + 重建 — 是否足够?
16. 风格仓库每个 client slot 在 client 训练完才更新, 所以 round t 的 client 看到的是 round t-1 的别 client 风格. 第一轮 (round 0) bank 完全是空, 直到至少有 1 个 client 训练完才有数据. saac_warmup_rounds=10 是否足够?

# 输出格式

按 CRITICAL / IMPORTANT / MINOR 三级输出每个发现, 给修复建议. 报告 < 1500 字.

如果发现 CRITICAL 级别的 bug 直接说明 — 我会立即修复, 不进入实验阶段.
