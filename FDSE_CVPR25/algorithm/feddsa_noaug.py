"""
FedDSA-NoAug: Disable style augmentation entirely (dispatch=None always).

Hypothesis test:
    If disabling style aug on Office IMPROVES performance over original FedDSA,
    it confirms that Beta(0.1, 0.1) AdaIN mixing is the source of negative transfer
    on low-style-gap datasets.

    If disabling HURTS performance, style aug is providing some value even if suboptimal.

Keep everything else: dual-head decouple + orth loss + InfoNCE align.
"""
import copy
from algorithm.feddsa import Server as BaseServer, Client as BaseClient


class Server(BaseServer):
    def pack(self, client_id, mtype=0):
        """Never dispatch styles — effectively disables style augmentation."""
        return {
            'model': copy.deepcopy(self.model),
            'global_protos': copy.deepcopy(self.global_semantic_protos),
            'style_bank': None,
            'current_round': self.current_round,
        }


class Client(BaseClient):
    pass


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    from algorithm.feddsa import init_global_module as base_init
    base_init(object)
