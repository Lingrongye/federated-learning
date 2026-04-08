# EXP-040 | Multi-Layer Style Injection

## 基本信息
- **方法**：AdaIN同时在mid层(fc1后)和final层(fc2后)应用
- **算法**：feddsa_multilayer
- **状态**：⏳ 待执行

## 目的
当前只在backbone最后做AdaIN。启发自StyleGAN: 不同层捕获不同抽象级别的风格。
多层注入 → 更彻底的风格适应。

## 架构改动
```
Encoder:
  x → conv_features → avgpool → fc1 → bn6 → relu → [mid, 1024-d]
                                                      ↓ AdaIN(style_mid)
                                   → fc2 → bn7 → relu → [final, 1024-d]
                                                           ↓ AdaIN(style_final)
  → semantic_head → head
```
每个客户端维护两个风格仓库：mid层统计量 + final层统计量

## 运行命令
```bash
nohup /root/miniconda3/bin/python run_single.py \
    --task PACS_c4 --algorithm feddsa_multilayer --gpu 0 \
    --config ./config/pacs/feddsa_exp040.yml \
    --logger PerRunLogger --seed 2 > /tmp/exp040.out 2>&1 &
```

## 结果
| 指标 | 值 |
|------|---|
| Best acc | |
| Last acc | |
| vs EXP-017 (82.24%) | |

## 结论
