"""EXP-118 | 完整 FedBN (γ/β 也本地化) vs 半 FedBN 对照

基于 feddsa_scheduled, 仅覆盖 _init_agg_keys, 让整个 BN 层 (包含 weight/bias/running_*)
全部进 private_keys 不聚合. 其他所有行为 (模型/损失/聚合逻辑) 和 feddsa_scheduled 完全一致.

目的: 量化 "BN γ/β 参与 FedAvg" vs "BN 完全本地" 的 accuracy 差异.
- EXP-080 用 feddsa_scheduled (半 FedBN) Office 3-seed = 89.44
- 本实验 feddsa_scheduled_fullbn (完整 FedBN) Office 3-seed = ?
- Δ = γ/β 聚合的影响

预期: 如 |Δ| < 0.3pp -> 差异无实质, 两种写法都 OK
      如 Δ > 0.5pp  -> 完整 FedBN 更好, 应该改代码
      如 Δ < -0.5pp -> 半 FedBN 更好 (保留 γ/β 跨域信号), paper 叙事更丰富
"""
from algorithm.feddsa_scheduled import (
    Server as _Server,
    Client as _Client,
    FedDSAModel,
    init_global_module,  # ← flgo 需要这个 hook 来构造 Server.model (FedDSAModel)
    init_local_module,
    init_dataset,
)

__all__ = [
    "Server", "Client", "FedDSAModel",
    "init_global_module", "init_local_module", "init_dataset",
]


class Server(_Server):
    def _init_agg_keys(self):
        """完整 FedBN: 所有 bn.* key (含 weight/bias/running_*) 全部本地, 不聚合."""
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k:
                self.private_keys.add(k)
            elif 'bn' in k.lower():
                # ← 关键差异: 去掉 running_/num_batches_tracked 过滤,
                #             整层 BN (含 γ=weight, β=bias) 都本地
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]


class Client(_Client):
    """Client 逻辑跟 feddsa_scheduled.Client 完全一致, 不用改."""
    pass
