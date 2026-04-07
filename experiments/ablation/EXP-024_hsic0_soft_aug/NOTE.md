# EXP-024 | HSIC=0 + Softer Style Augmentation (Beta 0.5)

## 基本信息
- **日期**：2026-04-07
- **类型**：ablation (稳定性优化 - 减少增强噪声)
- **方法**：FedDSA-Stable 软化风格增强
- **算法**：feddsa_stable (新建)
- **状态**：⏳ 待执行

## 目的
原版FedDSA使用 Beta(0.1, 0.1) 采样 AdaIN 混合系数 alpha。
Beta(0.1, 0.1) 是U形分布，大多数采样值接近0或1（极端混合）。
这导致训练中 alpha 忽大忽小，是后期震荡的可能来源。

## Beta分布对比
```
Beta(0.1, 0.1): ∩∪ 形 (U-shaped)    90%的样本在 [0,0.1]∪[0.9,1]
Beta(0.5, 0.5): ∪ 浅U形             比较均匀
Beta(1.0, 1.0): — 均匀分布
```

## 算法改进
使 `style_alpha` 可配置，修改 `_style_augment()`：
```python
alpha = np.random.beta(self.style_alpha, self.style_alpha)
```

## 与EXP-017的差异
| 参数 | EXP-017 | EXP-024 |
|------|---------|---------|
| Algorithm | feddsa | **feddsa_stable** |
| style_alpha (Beta param) | 0.1 (hardcoded) | **0.5** (中度混合) |
其他相同

## 假设
- 风格混合更温和 → 训练信号更稳定
- 可能Best略降(因为增强效果弱)
- 但Last应该更好
- 目标: Last ≥ 79%

## 运行命令
```bash
cd /root/autodl-tmp/federated-learning/FDSE_CVPR25 && \
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_stable --gpu 0 \
    --config ./config/pacs/feddsa_exp024.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp024.out 2>&1 &
```

## 结果

| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| Best-Last gap | |
| Drops>5% | |
| vs EXP-017 | |

## 结论
