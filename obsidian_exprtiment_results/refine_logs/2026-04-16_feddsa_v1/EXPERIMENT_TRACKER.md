# Experiment Tracker

| Run ID | Milestone | Block | Purpose | System/Variant | Dataset | Seeds | Priority | Status | Notes |
|--------|-----------|-------|---------|----------------|---------|-------|----------|--------|-------|
| R001 | M0 | - | Sanity: 数据加载+单client过拟合 | Full method | PACS-Photo | 1 | MUST | TODO | 验证pipeline |
| R002 | M1 | B1 | 基线复现: FedProto | FedProto | PACS | 3 | MUST | TODO | PFLlib代码 |
| R003 | M1 | B1 | 基线复现: FedBN | FedBN | PACS | 3 | MUST | TODO | PFLlib代码 |
| R004 | M1 | B1 | 基线复现: FDSE | FDSE | PACS | 3 | MUST | TODO | FDSE_CVPR25代码 |
| R005 | M2 | B1 | Full method首次运行 | Ours (full) | PACS | 3 | MUST | TODO | 验证>FedProto |
| R006 | M3 | B2 | 核心消融: Decouple-only | decouple+no-share | PACS | 3 | MUST | TODO | |
| R007 | M3 | B2 | 核心消融: Share-only | no-decouple+share | PACS | 3 | MUST | TODO | 关键对照 |
| R008 | M3 | B2 | 核心消融: Neither | no-decouple+no-share | PACS | 3 | MUST | TODO | |
| R009 | M3 | B3 | 约束消融: orth-only | L_orth only | PACS | 3 | MUST | TODO | |
| R010 | M3 | B3 | 约束消融: HSIC-only | L_HSIC only | PACS | 3 | MUST | TODO | |
| R011 | M4 | B3 | 解耦诊断: 线性探针 | linear probes on z_sem/z_sty | PACS | 1 | MUST | TODO | 域/类预测 |
| R012 | M4 | B3 | 原型质量: 紧凑度 | compactness metrics | PACS | 1 | MUST | TODO | vs FedProto |
| R013 | M4 | B3 | 可视化: t-SNE | z_sem by class/domain | PACS | 1 | MUST | TODO | Figure 3 |
| R014 | M5 | B1 | 主表: FedAvg | FedAvg | PACS+DomainNet | 3 | MUST | TODO | |
| R015 | M5 | B1 | 主表: FedProx | FedProx | PACS+DomainNet | 3 | MUST | TODO | |
| R016 | M5 | B1 | 主表: FPL | FPL | PACS+DomainNet | 3 | MUST | TODO | RethinkFL代码 |
| R017 | M5 | B1 | 主表: FedPLVM | FedPLVM | PACS+DomainNet | 3 | MUST | TODO | FedPLVM代码 |
| R018 | M5 | B1 | 主表: FISC | FISC | PACS+DomainNet | 3 | MUST | TODO | 按论文复现 |
| R019 | M5 | B1 | 主表: Ours | Full method | PACS+DomainNet | 3 | MUST | TODO | |
| R020 | M5 | B1 | 主表: Ours | Full method | Digit-5+Office | 3 | MUST | TODO | 支撑数据集 |
| R021 | M6 | B4 | 增强对比: Entangled AdaIN | entangled AdaIN | PACS | 3 | MUST | TODO | |
| R022 | M6 | B4 | 增强对比: Entangled MixStyle | entangled MixStyle | PACS | 3 | MUST | TODO | |
| R023 | M6 | B4 | 增强对比: No augmentation | decouple+align only | PACS | 3 | MUST | TODO | |
| R024 | M6 | B5 | 开销分析 | all methods | PACS | 1 | NICE | TODO | Table 6 |
| R025 | M6 | B6a | 调度消融: need-aware | need-aware dispatch | PACS | 3 | NICE | TODO | appendix |
| R026 | M6 | B6b | 风格库K消融 | K={1,3,5} | PACS | 3 | NICE | TODO | appendix |
| R027 | M6 | B6c | 超参敏感性 | sweep λ_orth/λ_hsic/τ | PACS | 1 | NICE | TODO | appendix |
| R028 | M6 | B6d | DINOv2骨干 | frozen DINOv2-S | PACS | 3 | NICE | TODO | supplementary |
