# EXP-048 | FixBN:修复 BN Running Stats 聚合 Bug

## 基本信息
- **日期**:2026-04-09
- **类型**:critical bug fix
- **算法**:feddsa_fixbn
- **配置**:feddsa_exp048.yml (与 EXP-017 完全一致的 algo_para)
- **状态**:✅ 已完成 (多 seed 验证见 EXP-050)

## Bug 描述(已验证)

### 代码证据
`FDSE_CVPR25/algorithm/feddsa.py` L128-137:
```python
def _init_agg_keys(self):
    all_keys = list(self.model.state_dict().keys())
    self.private_keys = set()
    for k in all_keys:
        if 'style_head' in k:
            self.private_keys.add(k)
        elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
            self.private_keys.add(k)   # ← BUG:running stats 被标为 private
    self.shared_keys = [k for k in all_keys if k not in self.private_keys]
```

### 运行时验证
Python 测试脚本 `rr/verify_bn_bug.sh` 模拟 3 客户端训练 + server 聚合,结果:
```
Initial server bn1.running_mean[:5] = [0. 0. 0. 0. 0.]
After 3 clients training:
  Client 0: bn1.running_mean[:5] = [-0.010  0.006  0.004  0.021  0.020]
  Client 1: bn1.running_mean[:5] = [-0.960  0.002 -0.160  0.175  0.448]
  Client 2: bn1.running_mean[:5] = [-1.936 -0.265 -0.594  0.219  0.525]
After server aggregation:
  server bn1.running_mean[:5] = [0. 0. 0. 0. 0.]   ← 没变!
  server bn1.running_var[:5]  = [1. 1. 1. 1. 1.]   ← 没变!
```

### 影响
- 服务器 BN running stats 永远在 init (0, 1) 状态
- 测试时(global_test 用 server model)BN 变成"identity + 学得的仿射"
- 训练时用 batch stats 归一化 → 测试时没有 → train/test 分布不匹配
- **这是 FedDSA gap ~5% 而 FDSE gap ~2% 的潜在根因**

### FDSE 对照
FDSE (`fdse.py` L98) 把 `'dfe'` 前缀的所有 key(包括 dfe_bn.running_mean)加入 shared_weight_keys,L124-131 显式聚合 running_ 键。所以 FDSE 服务器 BN stats 正常更新。

## 修复方案(Option A)
`feddsa_fixbn.py` Server 覆盖 `_init_agg_keys`,只保留 `style_head` 为 private,BN 所有键走 FedAvg 聚合:
```python
def _init_agg_keys(self):
    all_keys = list(self.model.state_dict().keys())
    self.private_keys = set()
    for k in all_keys:
        if 'style_head' in k:
            self.private_keys.add(k)
    self.shared_keys = [k for k in all_keys if k not in self.private_keys]
```

Client.unpack() **不改**:仍然 skip running_ 键 → 客户端保留本地 FedBN stats 用于训练,与 FDSE 行为一致(server aggregated for test,client local for train)。

## 对比设计
| Exp | 算法 | 配置 | 预期 |
|---|---|---|---|
| EXP-017 | feddsa (buggy) | v4_no_hsic.yml | Best 82.24, Last 75.46, gap 6.78 |
| **EXP-048** | **feddsa_fixbn** | exp048.yml (相同 algo_para) | **Best ≥ 83?** **Gap < 3?** |

**只变 BN 聚合策略,其他一切相同** — 直接测 bug 的影响。

## 假设
- **Best 上升 1-2%**(因为测试时 BN 正确工作)
- **Last 上升 3-5%**(train/test 匹配,后期震荡消失)
- **Gap 从 6.78 降到 <3**(接近 FDSE 水平)

如果假设成立 → 这是 paper 的重大发现,bug 修复就是贡献之一。
如果假设不成立 → bug 影响小,当前结论仍有效。

## 运行命令
```bash
nohup python run_single.py --task PACS_c4 --algorithm feddsa_fixbn --gpu 0 \
  --config ./config/pacs/feddsa_exp048.yml --logger PerRunLogger --seed 2 \
  > /tmp/exp048.out 2>&1 &
```

## 结果
| 指标 | EXP-017 (buggy) | EXP-048 (fixed) | Delta |
|---|---|---|---|
| Best | 82.24 | 80.73 | **-1.51** |
| Last | 75.46 | 76.27 | **+0.81** |
| Gap | 6.78 | 4.46 | **-2.32 (改善 34%)** |

## 结论
- **假设部分成立**:Gap 确实从 6.78 缩到 4.46(改善 34%),但远没到"<3"
- **Best 反而下降 1.51%**:聚合 BN running stats 让 server model 测试更"正确"但偏向均值化,削弱了单 seed 峰值
- **Last 提升 0.81%**:train/test 匹配改善了后期稳定性
- **总结**:BN bug 的影响是"trade-off"而非单方面损失:
  - 旧版:broken BN 充当隐性正则化,Best 高但不稳定
  - 新版:BN 正常,Best 低但更稳定
- **不建议直接替换原版**:需要多 seed 验证(见 EXP-050),如果均值 > 原版 80.74 才考虑
