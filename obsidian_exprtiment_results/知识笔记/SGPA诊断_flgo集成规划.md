# SGPA 诊断框架 → flgo 集成规划

> 2026-04-19. 具体说明 `SGPADiagnosticLogger` 如何插到 `feddsa_scheduled.py` 的 flgo 框架里。
> 结论: **100% 可集成**,参考现有 `_grad_conflict_log` 模式即可。

## 一、flgo 框架现状(我读完代码的发现)

### Client 端 (`feddsa_scheduled.py:664-...`)
```python
class Client(flgo.algorithm.fedbase.BasicClient):
    def __init__(self, ...):
        # 已有 algo_para 配置读取 (e.g. schedule_mode, scpr_mode, ...)
        # 已有 self._grad_conflict_log = None 这种诊断槽
    
    def unpack(self, svr_pkg):  # 从 server 收 payload
        return (self.model, svr_pkg['global_protos'], svr_pkg['style_bank'], 
                svr_pkg['current_round'])
    
    def pack(self):  # 上传 server
        return {
            'model': ...,
            'protos': self._local_protos,
            'style_stats': self._local_style_stats,
            'grad_conflict': self._grad_conflict_log,    # ← 诊断字段已经存在
            # ↑ 这就是我们要跟进的 pattern
        }
    
    def train(self, model, ...):
        # 已经暴露了:
        #   - z_sem, z_sty (双头输出)
        #   - y (真实 label)
        #   - output = model.head(z_sem) (logits)
        #   - self.current_round
        #   - is_last_batch (已判断)
        # 已有:
        #   if should_log_grad and is_last_batch and loss_sem > 0:
        #       self._log_grad_conflict(...)  # ← 诊断 hook 已经存在
```

### Server 端
```python
class Server(flgo.algorithm.fedbase.BasicServer):
    def iterate(self):
        res = self.communicate(self.selected_clients)
        # res 已经包含所有 client 的 pack() 上传字段
        # res['grad_conflict']: list of per-client grad dicts  ← 诊断信号在这里汇总
```

**关键发现**: flgo 的 pack/unpack 机制天然支持 "每 client 额外上传诊断字段",server 天然能收齐。**诊断框架根本不需要改 flgo 的任何底层**,只是在 algorithm/ 里加 hook。

## 二、集成 hook 点具体清单

### Hook 1: Client.__init__ — 初始化 logger

```python
# feddsa_sgpa.py Client.__init__ (假设未来创建)
from FDSE_CVPR25.diagnostics import SGPADiagnosticLogger as DL

def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # ... 现有初始化 ...
    
    # 诊断 logger (新增)
    diag_dir = os.path.join(self.option.get('task_dir', '.'), 'diag_logs')
    self.dl_train = DL(client_id=self.id, stage='train', 
                       log_dir=diag_dir, dump_every_n=5)
    self.dl_test = DL(client_id=self.id, stage='test', 
                      log_dir=diag_dir, dump_every_n=1)
    self.diag_interval = kwargs.get('diag_interval', 5)  # 每 5 轮记录训练指标
```

### Hook 2: Client.train — Layer 1 训练端指标 (6 个)

```python
def train(self, model, *args, **kwargs):
    # ... 现有逻辑直到 last batch ...
    
    # === 新增: 诊断记录 ===
    should_diag = (self.current_round % self.diag_interval == 0 
                   and is_last_batch)
    
    if should_diag:
        with torch.no_grad():
            # 1-4: 特征几何 (复用现有 z_sem, z_sty, y, model.M)
            metrics = {
                'orth': DL.orthogonality(z_sem, z_sty),
                'etf_align_mean': DL.etf_alignment(
                    z_sem, y, model.M, self.num_classes)[0],
                'intra_cls': DL.intra_class_similarity(
                    z_sem, y, self.num_classes),
                'inter_cls': DL.inter_class_similarity(
                    z_sem, y, self.num_classes),
            }
            
            # 5-6: 梯度指标 (跟 _log_grad_conflict 一样的 pattern)
            # 分别对 loss_task 和 loss_orth 做 backward, 取对某共享层的梯度
            shared_layer = model.backbone.conv1  # 或最后一层共享 conv
            g_ce = _grad_of(shared_layer, loss_task)  # 需要辅助函数
            g_orth = _grad_of(shared_layer, loss_orth)
            metrics['cos_grad'] = DL.gradient_cosine(
                g_ce.flatten(), g_orth.flatten())
            metrics['grad_norm_ratio'] = DL.gradient_norm_ratio(
                g_ce.flatten(), g_orth.flatten())
        
        self.dl_train.record(
            round_id=self.current_round, metrics_dict=metrics)
```

### Hook 3: Client.pack — 上传聚合端原始量

```python
def pack(self):
    pkg = {
        'model': copy.deepcopy(self.model.to('cpu')),
        # ... 现有字段 ...
        'grad_conflict': self._grad_conflict_log,
        
        # 新增: 上传类中心供 server 算 client_center_variance
        'class_centers': self._local_class_centers,  # [K, d] tensor
    }
    return pkg
```

在 `train()` 结尾额外累积类中心:
```python
# train() 末尾, 在 return 之前
class_centers = torch.zeros(self.num_classes, z_sem.shape[-1])
for c in range(self.num_classes):
    mask = (accumulated_y == c)
    if mask.sum() > 0:
        class_centers[c] = accumulated_z_sem[mask].mean(0).cpu()
self._local_class_centers = class_centers
```

### Hook 4: Server.iterate — Layer 2 聚合端指标

```python
def iterate(self):
    res = self.communicate(self.selected_clients)
    
    # ... 现有 aggregation 逻辑 ...
    
    # === 新增: 聚合端诊断 ===
    if self.current_round % self.diag_interval == 0:
        all_centers = res.get('class_centers', [])  # list of [K, d]
        if all_centers:
            center_var = DL.client_center_variance(all_centers)
            
            # param drift: 算各 client 的 backbone 第一层参数到 global 的距离
            global_p = self.model.backbone.conv1.weight.detach().cpu().flatten()
            client_ps = [m.backbone.conv1.weight.detach().cpu().flatten() 
                         for m in res['model']]
            drift = DL.param_drift(client_ps, global_p)
            
            self.dl_agg.record(round_id=self.current_round, metrics_dict={
                'client_center_var': center_var,
                'param_drift': drift,
            })
```

### Hook 5: Client.test_with_sgpa — Layer 3 推理端指标 (13 个)

```python
@torch.no_grad()
def test_with_sgpa(self):
    self.model.eval()
    for batch_idx, (x, y) in enumerate(self.test_data):
        # ... 现有 SGPA 推理: z_sem, z_sty, logits_etf, H, dist_min, reliable,
        #     proto 更新, pred ...
        
        # === 新增: 完整 Layer 3 指标记录 ===
        metrics = {}
        metrics.update(DL.gate_rates(H, dist_min, self.tau_H, self.tau_S))
        metrics.update(DL.dist_distribution(dist_min))
        metrics.update(DL.whitening_reduction(
            z_sty, z_sty_white, self.source_mu_raw, self.source_mu_white))
        metrics['sigma_cond'] = DL.sigma_condition_number(self.Sigma_global)
        _, metrics['proto_fill_mean'] = DL.proto_fill(
            self.supports, self.num_classes)
        metrics['proto_etf_offset_mean'], _ = DL.proto_etf_offset(
            self.proto, self.M, self.num_classes)
        metrics['fallback_rate'] = DL.fallback_rate(activated)
        metrics.update(DL.prediction_accuracy(pred_proto, pred_etf, y))
        
        self.dl_test.record(round_id=self.current_round, 
                            metrics_dict=metrics)
```

## 三、Wire-up 复杂度评估

| 改动类型 | LoC | 风险 |
|---------|-----|------|
| import + __init__ 初始化 3 个 logger | ~8 行 | 低 |
| Client.train() 加 Layer 1 metrics | ~20 行 | 中 (需要辅助梯度函数) |
| Client.pack() 加 class_centers | ~5 行 | 低 |
| Server.iterate() 加 Layer 2 metrics | ~15 行 | 低 |
| Client.test_with_sgpa() 加 Layer 3 metrics | ~25 行 | 低 |
| **合计** | **~75 行** | 低 |

梯度辅助函数 (唯一稍复杂的):
```python
def _grad_of(module, loss):
    """取 loss 对 module 参数的梯度 (不清零其他梯度)."""
    grads = torch.autograd.grad(
        loss, module.parameters(), retain_graph=True, allow_unused=True)
    flat = torch.cat([g.flatten() for g in grads if g is not None])
    return flat
```

注意: `retain_graph=True` 确保下一次 backward 不崩,`allow_unused=True` 避免部分参数 orth 不 touch 导致 error。

## 四、已验证的兼容性检查清单

- [x] **DiagnosticLogger 53/53 单元测试通过** (独立于 flgo)
- [x] **flgo Client.train() 暴露 z_sem, z_sty, y, current_round** (`feddsa_scheduled.py:798-803`)
- [x] **pack/unpack 机制支持添加任意 payload** (现有 `grad_conflict` 先例)
- [x] **Server.iterate() 能 collect 所有 client 字段** (`res['grad_conflict']` 先例)
- [x] **jsonl 多 client 分文件写入** (tests/TestRecordAndDump.test_multi_client_separate_files)
- [x] **tensor/numpy 自动序列化** (tests/test_tensor_serialization)
- [ ] **梯度取法** — 需要 SGPA 实际实现时验证 `torch.autograd.grad(retain_graph=True)` 不冲突 `optimizer.zero_grad()`
- [ ] **大 batch 梯度 flatten 内存占用** — AlexNet conv1 weight ~ 23K,flatten 24K floats OK

## 五、集成时机

**现在不集成**。等到:
1. FedDSA-SGPA 实际实现 (clientdsa_sgpa.py / serverdsa_sgpa.py 新建) 时
2. 单元测试 ETF buffer / SGPA 推理正确性都通过后
3. 再同时 wire-up 5 个 hook

**原因**: 诊断代码和算法代码耦合紧(需要 z_sem/z_sty/loss_orth 这些变量),单独集成不省事;一起集成能一次 review 完整。

## 六、codex 审核结论(待填写)

codex 审核的 6 维评分 + MUST_FIX / SUGGESTIONS 回填到本节,审核完成后再更新。
