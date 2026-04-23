"""
Digits-5 任务的默认 Model wrapper.

这个文件把 config.py 里定义的 DigitModel 包装成 flgo 需要的 FModule.
FModule 是 flgo 框架的 Module 基类, 支持参数聚合(加减乘), 是 FedAvg 等算法的基础.

用法: flgo 在 TaskPipe 初始化时会调用 init_global_module(server_object), 为 server
和各 client 创建 Model 实例. 如果 init_global_module 写成 `pass` (历史坑 EXP-119),
会 silent 导致 NoneType 错误.

这个 default_model 用于**所有不需要特殊架构的算法**:
  - fedavg / fedbn / fedprox / moon / ditto / scaffold / feddyn / ...
FDSE 算法因为要把 conv 层换成 DSEConv, 需要单独写 fdse_model.py (同目录, 后续添加).
feddsa_sgpa 算法因为要接 encode 接口, 需要单独写 feddsa_sgpa_model.py (同上).
"""

from ..config import get_model
from flgo.utils.fmodule import FModule


class Model(FModule):
    """
    包装 config.DigitModel 成 flgo FModule.

    为什么不直接 `class Model(FModule, DigitModel)`:
      多继承会让 FModule 的参数算术 (add/sub/mul/...) 绕一层, 可能有 bug.
      子持有 + delegate forward 更稳.
    """

    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def init_local_module(object):
    """
    客户端本地模块初始化. flgo 允许给某些算法 (如 Ditto) 分配本地专属 model,
    这里所有算法本地不用单独 model, pass 就行 (global model 下发后覆盖).
    """
    pass


def init_global_module(object):
    """
    Server 端初始化全局 model.

    重要: 必须实际构造 Model 实例绑到 object.model 上, **不能只写 pass**.
    EXP-119 历史上踩过 silent kill bug: 如果这里 pass, 训练时 object.model = None,
    第一次 forward 报 NoneType AttributeError, 但 flgo 错误处理会吞掉直接挂死.
    """
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)
