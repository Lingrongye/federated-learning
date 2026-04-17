# EXP-028b | Uncertainty Weighting with Clamp Fix

## 基本信息
- **日期**：2026-04-08
- **类型**：ablation (修复EXP-028)
- **方法**：feddsa_auto with log_sigma clamping
- **状态**：⏳ 待执行

## 目的
EXP-028 (Uncertainty Weighting) 训练崩溃了，准确率从72%掉到17%。
原因：log_sigma 学得太极端，某个loss权重失控。
修复：每轮 clamp log_sigma 到 [-2, 2] 防止极端值。

## 修复内容
```python
# Before (EXP-028):
self._log_sigma_orth_val = self.log_sigma_orth.detach().cpu().item()

# After (EXP-028b):
self._log_sigma_orth_val = float(self.log_sigma_orth.detach().cpu().clamp(-2, 2).item())
```
- Clamp 范围 [-2, 2] 对应权重范围 [exp(-2), exp(2)] ≈ [0.13, 7.4]
- 既允许自适应，又防止失控

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_auto --gpu 0 \
    --config ./config/pacs/feddsa_exp028.yml \
    --logger PerRunLogger --seed 2 \
    > /tmp/exp028b.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| 是否再次崩溃 | |

## 结论
