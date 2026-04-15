# 多原型联邦学习文献搜索报告 (2026-04-16)

> 搜索目的: 为 FedDSA 的 M3 域感知原型对齐模块寻找更好的改进方案
> 搜索范围: 2024-2025 多原型/域感知原型/原型压缩/对齐损失

---

## 一、已有本地论文 (papers/ 目录)

| 方法 | 会议 | 多原型策略 | 对齐损失 | 原型数量控制 | 源码 |
|---|---|---|---|---|---|
| FPL | CVPR 2023 | FINCH 聚类生成多原型 | 分层 InfoNCE + MSE | 自动(1-3/类) | `RethinkFL/models/fpl.py` |
| FedPLVM | NeurIPS 2024 | 双层 FINCH (本地→全局) | α-sparsity InfoNCE (sim^0.25) | 压缩到 ~1/5 | `FedPLVM/utils/update.py` |
| MP-FedCL | IoT Journal 2024 | k-means 多原型 | 有监督对比 | 固定 k | 无 |
| I2PFL | arXiv 2025 | 域内 MixUp + 域间重加权 | GPCL + APA | 无压缩 | 无 |
| FedProto | AAAI 2022 | 单原型(类均值) | MSE 硬对齐 | 每类 1 个 | `PFLlib/` |
| FedPall | ICCV 2025 | 原型混合特征 | InfoNCE + 对抗 | 每类 1 个 | 无 |
| FISC/PARDON | ICDCS 2025 | FINCH 风格聚类 + AdaIN | 三元组对比 | 自动 | `PARDON-FedDG/` |

### 关键源码分析

**FPL 的 FINCH 聚类** (`RethinkFL/models/fpl.py:52-88`):
- 收集各客户端同类原型 → FINCH(cosine距离) → 取最粗粒度聚类(c[:, -1])
- 每个聚类取均值作为聚类中心
- 典型压缩: 4域 → 1-3 个聚类原型/类

**FPL 的分层 InfoNCE** (`fpl.py:90-121`):
- 第一层: 样本 vs 同类聚类原型(多正样本 InfoNCE) → `xi_info_loss`
- 第二层: 样本 vs 类均值原型(MSE 回归) → `cu_info_loss`
- 总损失: `hierar_info_loss = xi_info_loss + cu_info_loss`

**FedPLVM 的 α-sparsity InfoNCE** (`FedPLVM/utils/update.py:101-134`):
- `logits = cosine_similarity.pow(alpha)` → alpha=0.25 使相似度分布更稀疏
- 效果: 增强类间分离力, 尤其在 FL 小数据集场景

**FedPLVM 的双层聚类** (`FedPLVM/utils/util.py:42-75`):
- 本地: 每客户端用 FINCH 聚类本地样本特征 → 本地多原型
- 全局: 服务器对收集的原型再次 FINCH → 全局压缩到 ~1/5
- 两种选项: `cluster_protos` (k-means) 或 `cluster_protos_finch`

---

## 二、新发现论文 (Web 搜索, 2024-2025)

### 2.1 FedDAP — 域感知原型学习 ★★★ (最直接竞争者)

- **来源**: [arXiv 2604.06795](https://arxiv.org/abs/2604.06795v1), 2025
- **核心**: 余弦加权域原型融合 + 双对齐策略(域内+跨域)
- **方法细节**:
  1. **余弦加权融合**: 
     - 一致性分数: `S_j = Σ_{k≠j} cos(p_j^(c,d), p_k^(c,d))`
     - 注意力权重: `α_j = exp(S_j/τ_agg) / Σ_l exp(S_l/τ_agg)`
     - 聚合: `P^(c,d) = Σ_j α_j · p_j^(c,d)`
  2. **域内对齐 (L_DPA)**: `L_DPA = Σ_c (1 - cos(z_i^c, P^(c,d_m)))`
  3. **跨域对比 (L_CPCL)**: 拉近同类异域原型, 推开异类原型
  4. **总损失**: `L = L_CE + λ₁·L_DPA + λ₂·L_CPCL`
- **结果**: PACS **84.63%**, DomainNet 65.20%, Office-10 72.53%
- **与我们的关系**: 
  - 思路几乎和 M3 一样(保留域级原型)
  - **优势**: 加了余弦加权 + 双对齐
  - **劣势**: 在混合特征空间操作(未解耦)
  - **关键差异**: 我们在解耦 z_sem 空间做, 原型更纯净

### 2.2 FedGMKD — 差异感知聚合 ★★☆

- **来源**: [NeurIPS 2024](https://openreview.net/forum?id=c3OZBJpN7M)
- **核心**: GMM 生成原型特征 + 差异感知聚合(DAT)
- **关键机制**: 
  - Cluster Knowledge Fusion (CKF): 用 GMM 生成原型
  - Discrepancy-Aware Technique: 按数据质量+数量加权客户端贡献
- **与我们的关系**: 原型质量加权思路可借鉴, 但目标不同(它处理 label skew)

### 2.3 FedTGP — 可训练全局原型 ★★☆

- **来源**: [arXiv 2401.03230](https://arxiv.org/abs/2401.03230), 2024
- **核心**: 全局原型不再是均值, 而是**可训练参数** + 自适应 margin 对比学习
- **关键创新**: 
  - 服务器端原型是 nn.Parameter, 通过对比学习优化
  - Adaptive-margin: 根据类间距离动态调整 margin
- **与我们的关系**: "可训练原型"概念有趣, 但增加了服务器端训练复杂度

### 2.4 FedRFQ — 原型冗余消除+质量过滤 ★☆☆

- **来源**: [arXiv 2401.07558](https://arxiv.org/html/2401.07558v1), 2024
- **核心**: SoftPool 减少冗余 + BFT 检测过滤低质量原型
- **方法**: L2 距离度量原型质量, Byzantine Fault Tolerance 排除异常
- **数据集**: MNIST/FEMNIST/CIFAR-10 (非域偏移场景)
- **与我们的关系**: 质量过滤机制可参考, 但场景不同

### 2.5 FedDPA — 动态原型对齐 ★☆☆

- **来源**: [MDPI Electronics 2025](https://www.mdpi.com/2079-9292/14/16/3286)
- **核心**: 类级原型 + 对比对齐(域内紧凑 + 域间分离)
- **与我们的关系**: 对齐策略可参考, 但未做特征解耦

### 2.6 FedSC — 语义感知协作 ★☆☆

- **来源**: [arXiv 2506.21012](https://arxiv.org/html/2506.21012), 2025
- **核心**: 关系原型 + 域间对比学习
- **关键**: 不只是点级原型, 还构建类间关系(图结构)
- **与我们的关系**: 关系级原型是更高阶的概念, 当前阶段过于复杂

### 2.7 FedCPD — 原型增强蒸馏 ★☆☆

- **来源**: [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0612.pdf)
- **核心**: 特征蒸馏防遗忘 + 原型增强类区分
- **与我们的关系**: 蒸馏思路可在未来考虑

### 2.8 MPFT — 多域原型微调 ★☆☆

- **来源**: [ICLR 2025](https://openreview.net/forum?id=3wEGdrV5Cb)
- **核心**: 预训练模型(CLIP) + 多域原型微调, 单轮通信
- **与我们的关系**: 预训练+原型是新范式, 但我们走 ResNet/AlexNet 路线

---

## 三、关键洞察与改进方向

### 3.1 趋势分析

| 趋势 | 代表方法 | 启发 |
|---|---|---|
| **不等权对待原型** | FedDAP(余弦加权), FedGMKD(差异感知) | 原型应按质量/一致性加权 |
| **双层对齐(域内+跨域)** | FedDAP(DPA+CPCL) | 比单一 SupCon 更精细 |
| **原型压缩聚类** | FPL/FedPLVM(FINCH), MP-FedCL(k-means) | 减少冗余原型 |
| **α-sparsity 增强** | FedPLVM | 增强类间分离 |
| **可训练原型** | FedTGP | 原型可学习, 但复杂度高 |
| **解耦特征空间** | FedDSA(我们), FedSeProto, FediOS | 在纯净空间做原型更好 |

### 3.2 对初始方案的修改建议

**原方案**: FINCH 聚类 + α-sparsity InfoNCE

**修改后**: 
1. ❌ **去掉 FINCH** — 4 个域原型太少, FINCH 不稳定
2. ✅ **加入余弦加权融合** (借鉴 FedDAP) — 不需要聚类, 直接加权
3. ✅ **双对齐损失** (借鉴 FedDAP) — 域内对齐 + 跨域对比
4. ✅ **保持 α-sparsity** (借鉴 FedPLVM) — 增强类间分离
5. ✅ **核心差异**: 在解耦 z_sem 空间做 (vs FedDAP 的混合空间)

### 3.3 2×2 定位矩阵(更新版)

|  | 等权原型 | 加权/聚类原型 |
|--|---------|-------------|
| **混合空间** | FedProto, M3(当前) | FedDAP, FPL, FedPLVM |
| **解耦空间** | — | **FedDSA-M4(我们,新)** |

**我们的 novelty**: 首次在解耦语义空间中做加权域感知原型对齐 + 双层对齐损失

---

## 四、参考文献

1. FPL - Huang et al. "Rethinking Federated Learning with Domain Shift: A Prototype View" CVPR 2023
2. FedPLVM - Wang et al. "Taming Cross-Domain Representation Variance" NeurIPS 2024
3. MP-FedCL - Qiao et al. "Multiprototype Federated Contrastive Learning" IoT Journal 2024
4. I2PFL - Le et al. "Intra- and Inter-Domain Prototypes" arXiv 2025
5. FedDAP - [arXiv 2604.06795](https://arxiv.org/abs/2604.06795v1) 2025
6. FedGMKD - [NeurIPS 2024](https://openreview.net/forum?id=c3OZBJpN7M)
7. FedTGP - [arXiv 2401.03230](https://arxiv.org/abs/2401.03230) 2024
8. FedRFQ - [arXiv 2401.07558](https://arxiv.org/html/2401.07558v1) 2024
9. FedDPA - [MDPI Electronics](https://www.mdpi.com/2079-9292/14/16/3286) 2025
10. FedSC - [arXiv 2506.21012](https://arxiv.org/html/2506.21012) 2025
11. FedCPD - [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0612.pdf)
12. MPFT - [ICLR 2025](https://openreview.net/forum?id=3wEGdrV5Cb)
