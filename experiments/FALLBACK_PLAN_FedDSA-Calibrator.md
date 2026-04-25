# FedDSA-Calibrator —— 兜底方案

**创建日期**: 2026-04-24
**目的**: 作为所有探索失败的**兜底保底**。如果 PGA、spatial DFD、prototype anchor 都失败,这条方案是**最后能保证 Office/PACS 不退的最小侵入改动**。
**定位**: **不是主攻方向**,是保险。
**实施成本**: ~50 行代码,1 个新超参,4 个 GPU-hour × 3 seed = 12 GPU-hours。

---

## 一、核心动机(一句话)

> EXP-108/109/111/113 + decouple_probe 已证明:**z_sty 里有 class 信号(MLP-256 probe 0.81),但 L_orth 阻止不了**。我们不去"治" class mirror,而是**利用它**—— 加一个 calibrator 从 z_sty 里**显式提取**出 class 信号,作为 z_sem 的辅助补充,让最终分类器看到更完整的 class 信息。

和 F2DC 思想同源 ("calibrate rather than eliminate"),但**不需要 feature-map 级 mask**(F2DC 选项 C 代价 200 行),只在 pooled 双头的 128d 向量上加一个 MLP。

---

## 二、架构改动(对照图)

### 现状(feddsa_scheduled.py,sm=0 orth_only)

```
x
 ↓ AlexNet encoder
pooled [1024d]
 ↓ 双头
 ├── semantic_head (2 Linear + BN + ReLU)
 │      ↓
 │    z_sem [128d] ────→ sem_classifier ────→ logits
 │      ↑
 │    L_orth ←→ z_sty
 │      ↓
 └── style_head (2 Linear + BN + ReLU)
        ↓
      z_sty [128d]    (只被 L_orth 约束,最终不参与分类)
```

### Calibrator 改动(新增红色部分)

```
x
 ↓ AlexNet encoder
pooled [1024d]
 ↓ 双头
 ├── semantic_head → z_sem [128d] ─────────────────┐
 │     ↑                                            │
 │   L_orth ←→ z_sty                                │
 │     ↓                                            │
 └── style_head → z_sty [128d]                      │
                     ↓                              │
             ★ corrector (新 MLP 128→128→128)        │
                     ↓                              │
              z_sty_cal [128d] ─────────────────────┤
                     ↓                              ↓
            ★ aux_cls (新 Linear 128→num_classes)    │
                     ↓                              │
               L_aux (监督 calibrator 提 class)       │
                                                    ↓
                                 z_combined = z_sem + λ · z_sty_cal
                                                    ↓
                                 sem_classifier → logits
                                                    ↓
                                   L_CE (task loss)
```

---

## 三、新增模块代码(示意)

```python
# FDSE_CVPR25/algorithm/feddsa_calibrator.py  (新文件, 继承 feddsa_scheduled.py)

class FedDSACalibratorModel(FedDSAModel):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128, **kw):
        super().__init__(num_classes, feat_dim, proj_dim, **kw)
        self.corrector = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.aux_cls = nn.Linear(proj_dim, num_classes)

    def forward(self, x, lam=1.0):
        pooled = self.encode(x)
        z_sem = self.semantic_head(pooled)
        z_sty = self.style_head(pooled)

        # 残差 calibrator (F2DC 风格)
        z_sty_cal = z_sty + self.corrector(z_sty)

        # 最终分类输入: z_sem + λ · z_sty_cal
        z_combined = z_sem + lam * z_sty_cal
        logits = self.sem_classifier(z_combined)

        # 附加: aux_cls 监督
        aux_logits = self.aux_cls(z_sty_cal)

        return {
            'logits': logits, 'aux_logits': aux_logits,
            'z_sem': z_sem, 'z_sty': z_sty, 'z_sty_cal': z_sty_cal,
        }
```

---

## 四、Loss 组成

$$
\mathcal{L}_{\text{total}} = \underbrace{\text{CE}(\text{logits}, y)}_{\mathcal{L}_{CE}: \text{主任务}}
+ \underbrace{\lambda_{\text{orth}} \cdot \cos^2(z_{\text{sem}}, z_{\text{sty}})}_{\mathcal{L}_{\text{orth}}: \text{原有约束, 保留}}
+ \underbrace{\lambda_{\text{aux}} \cdot \text{CE}(\text{aux\_logits}, y)}_{\mathcal{L}_{\text{aux}}: \text{新增, 监督 calibrator}}
$$

**三件事同时跑**:
1. `L_CE`:逼 `z_sem + λ·z_sty_cal` 合起来能分类
2. `L_orth`:保留 cos(z_sem, z_sty)≈0 的方向正交
3. `L_aux`:**calibrator 必须真的从 z_sty 里提取出 class 信号**(这是关键)

**为什么 L_aux 是关键**:
- 如果没有 L_aux,corrector 可能学成恒等函数(z_sty_cal ≈ z_sty),对主任务没帮助
- 加了 L_aux,corrector 被强制"挖 z_sty 里的 class 维度",z_sty_cal 才有用

---

## 五、超参数 schedule(防止梯度冲突)

**新超参就 1 个**:`λ_aux` 和 `λ(z_sty_cal 的混合权重)`

**warmup schedule**(必须!防重演 InfoNCE 崩盘):

| Round | `λ` (z_sty_cal 权重) | `λ_aux` | 说明 |
|:-:|:-:|:-:|---|
| 0~20 | 0 | 0 | 纯 CE + L_orth,等 encoder 稳定 |
| 20~40 | 0 → 0.3 线性增 | 0 → 0.5 线性增 | 引入 calibrator |
| 40~200 | 0.3 恒定 | 0.5 恒定 | 稳态训练 |

**为什么 warmup**:
- 0~20 round 时 encoder 还在学 representation,过早上 L_aux 会让 z_sty 被污染
- 20 round 后基线 z_sem / z_sty 已经成型,再开 calibrator 才有意义

---

## 六、实施步骤(可直接交给人/agent 执行)

```markdown
Step 1: 创建 FDSE_CVPR25/algorithm/feddsa_calibrator.py
  - 继承 feddsa_scheduled.py 的 FedDSAModel
  - 加 self.corrector + self.aux_cls
  - 改 forward 返回 dict (见上)
  - 客户端 train_step 新增 L_aux + schedule

Step 2: 加 config
  - FDSE_CVPR25/config/office/feddsa_calibrator_r200.yml
  - FDSE_CVPR25/config/pacs/feddsa_calibrator_r200.yml
  - algo_para: [lam_aux=0.5, lam_mix=0.3, warmup_start=20, warmup_end=40]

Step 3: 代码验证 (必须, CLAUDE.md 强制)
  - python -c "import ast; ast.parse(open('feddsa_calibrator.py').read())"
  - 单元测试: 梯度流 + 边界情况
  - codex exec 代码审查

Step 4: 启动 Office 3-seed pilot
  - EXP-125_calibrator_office_r200/
  - seeds: 2, 15, 333
  - 预计 2.5h × 3 seed, 并发 ~3h wall

Step 5: 判决 (严格版)
  - 3-seed mean AVG Best > 90.58? ✓ → 继续 PACS
  - 3-seed mean AVG Best ≤ 89.5? ✗ → kill, 不在这上面浪费更多 GPU
  - 中间值 89.5~90.5 → 再跑 PACS 看综合
```

---

## 七、预期效果(严格定量)

### 最好情况(target)
| 数据集 | orth_only baseline | Calibrator 预期 | Δ |
|---|:-:|:-:|:-:|
| Office AVG Best | 89.09 | **91.0~91.5** | **+1.9~2.4 pp** |
| PACS AVG Best | 80.64 | **81.0~82.0** | **+0.4~1.4 pp** |

**逻辑**:Office 上 L_orth 推走的 class 残余更多(Office 风格差异小,class vs style 边界模糊),calibrator 能回收更多;PACS 上 orth_only 已经较强,calibrator 增量小。

### 保守情况
| 数据集 | orth_only | Calibrator | Δ |
|---|:-:|:-:|:-:|
| Office | 89.09 | **89.5~90.0** | **+0.4~0.9 pp** |
| PACS | 80.64 | **80.5~81.0** | **±0 ~ +0.4 pp** |

**这种情况也可以接受**:Office 没过 FDSE baseline 90.58,但**至少不退**,而且多了一个 "z_sty 被真正利用" 的 novelty 卖点。

### 失败情况(必须接受的可能性)
- 如果 **梯度冲突** → L_CE 和 L_aux 打架 → 可能出现 orth_only 80+ 但 calibrator 跌到 78-
- 如果 **calibrator 学成恒等** → 对主任务零贡献,accuracy 和 orth_only 持平
- 两种情况下都要立刻 kill,不继续调参

---

## 八、风险 + 预防

| 风险 | 机制 | 预防 |
|---|---|---|
| L_CE 和 L_aux 梯度冲突 | 两路都想主导 z_sty_cal,信号打架 | **Warmup schedule**(见第五节);λ_aux 从 0.1 起 |
| Calibrator 学成恒等 | 没 L_aux 逼它,corrector 变成 noop | L_aux 是硬性监督,aux_cls accuracy 会上报,<30% 立即 kill |
| z_sty_cal 污染 z_sem | z_sty_cal 强度 λ 太大,主任务被 z_sty 拖 | λ 上限 0.5,default 0.3 |
| Aux_cls 学到 trivial 解 | z_sty 已经有 class 信号,aux_cls 不用 calibrator 就能预测 | 对比消融: L_aux on z_sty (无 corrector) vs L_aux on z_sty_cal; 如果差不多, corrector 没价值 |

---

## 九、和已有方法的对比定位

| 方法 | 对 z_sty 的态度 | 代码改动 | 验证过 |
|---|---|:-:|:-:|
| FedBN | 不解耦 | 0 | ✓ |
| FedDSA orth_only | 推远 z_sty,丢弃 | 0 (baseline) | ✓ |
| FDSE | 擦除 domain 信息 | 大 (层分解) | ✓ |
| F2DC | 校准 domain feature (feature map 级) | 大 (~200 行) | 论文 |
| **FedDSA-Calibrator** (本方案) | **pooled 级 calibrator 提取 z_sty 的 class** | **~50 行** | **待验证** |

**Novelty 卖点**(如果真 work):
- "pooled-level calibration in federated double-head decoupling"
- 对比 F2DC:**不需要 spatial mask 架构重写**,代价 1/4
- 对比 CDANN:不对抗,纯校准

---

## 十、和主攻方向的关系

**这是兜底,不是主攻**。主攻顺序:

1. **首选**:跨域 class prototype anchor(不碰 z_sty,直接拉齐跨 domain z_sem) —— 最 clean
2. **次选**:spatial DFD(F2DC 选项 C) —— novelty 最高但成本大
3. **兜底(本方案)**:FedDSA-Calibrator —— 不创新但稳,保证不退

**决策树**:
```
首选 pilot → 过线 ? 
   ├── 是 → 直接用首选, 写论文
   └── 否 → 次选 pilot → 过线 ? 
             ├── 是 → 用次选
             └── 否 → 跑兜底 Calibrator (本方案) → 一般能让数据不退
```

---

## 十一、停止条件(明确)

开始跑 EXP-125_calibrator_office_r200 后:

**继续条件**:
- s=2 在 R40 时 AVG > 86% (warmup 结束后 accuracy 不塌)
- s=2 完整 200 round,AVG Best > 89.5

**立即 kill 条件**:
- R40 后 AVG 持续 < 85%(梯度冲突)
- aux_cls accuracy 始终 < 30%(calibrator 没学到东西)
- loss NaN / inf

---

## 十二、需要的上游决策

**在开始实施前,以下要先确认**:

1. ✅ Baseline: orth_only 3-seed 数据(Office 89.09, PACS 80.64 已确认)
2. ⚠️ 实施优先级: 本方案是兜底,**确认主攻路径失败后**再启动
3. ⚠️ GPU 资源: 3-seed × 3 GPU-hour = 12 GPU-hours on seetacloud2/lab-lry
4. ⚠️ 人员: 我能写实现,但需要用户拍板"何时启动"

---

## 十三、一句话总结

这份兜底方案做一件事:**在不改架构的前提下(pooled 双头保留),加一个 50 行的 MLP + 监督信号,把 z_sty 里已经证实存在的 class 残余信号"洗"出来,以期填上 Office -1.49 差距**。预期 0.4~2.4 pp 提升,失败情况下 accuracy 不退。

**这个方案随时可以实施,但建议先跑主攻路径**(跨域 prototype anchor),两者都失败再上这个。
