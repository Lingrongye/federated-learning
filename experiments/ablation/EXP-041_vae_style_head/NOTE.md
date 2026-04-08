# EXP-041 | VAE-style Probabilistic Style Head

## 基本信息
- **方法**：style_head输出(mu, log_var)，重参数采样 + KL loss
- **算法**：feddsa_vae
- **状态**：⏳ 待执行

## 目的
当前style head输出确定性向量。VAE式让它输出分布：
- 自然的风格建模
- KL正则化自带解耦效果
- 类似MUNIT的内容-风格分离

## 架构改动
```python
# 原版
z_sty = style_head(h)

# VAE版
mu_sty = style_mu_head(h)
logvar_sty = style_logvar_head(h)
z_sty = mu_sty + exp(0.5*logvar_sty) * eps  # 重参数化
loss_kl = -0.5 * mean(1 + logvar - mu² - exp(logvar))
```

λ_kl = 0.01 (轻微正则，避免posterior collapse)

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_vae --gpu 0 \
    --config ./config/pacs/feddsa_exp041.yml \
    --logger PerRunLogger --seed 2 > /tmp/exp041.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|

## 结论
