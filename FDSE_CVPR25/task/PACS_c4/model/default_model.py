from ..config import get_model
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    # Server 端初始化全局 model. 若写 pass 会 silent 导致 server.model=None,
    # 下游 c.test(server.model, ...) 崩 NoneType get_device (历史坑, commit bbbcf97
    # 误改为 pass, 2026-04-23 恢复)
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)
