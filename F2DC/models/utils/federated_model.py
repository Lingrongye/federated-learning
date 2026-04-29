import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
import os


class FederatedModel(nn.Module):
    """
    Federated learning model.
    """
    NAME = None
    N_CLASS = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform
        self.num_samples = []
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.trainloaders = None
        self.testlodaers = None

        self.epoch_index = 0

        self.checkpoint_path = checkpoint_path() + self.args.dataset + '/' + self.args.structure + '/'
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list

        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            prev_net = prev_nets_list[net_id]
            prev_net.load_state_dict(net_para)

    def _compute_daa_freq(self, online_clients_len):
        """F2DC paper Eq (10)(11) — Domain-Aware Aggregation reweight.

        Eq (10):  d_k = sqrt(C/2) * |n_k/N - 1/Q|
                  其中 C = num_classes (paper 把 B_k^c 简化为 n_k/N const vec),
                       Q = num_domains (PACS/Office/Digits 都 = 4)
        Eq (11):  p_k = sigmoid(α·n_k/N - β·d_k) / Σ_j sigmoid(...)
                  default α=1.0, β=0.4 (paper Fig 7 推荐范围)

        Note: paper 的 B_k 在 label-balance 简化后只依赖 n_k/N, d_k 实际上
        只 reflect sample-size 偏离 1/Q 的程度. 不真正依赖 client 持有的 domain id.
        """
        n_arr = np.array(online_clients_len, dtype=np.float64)
        N = n_arr.sum()
        sample_share = n_arr / N
        Q = float(getattr(self.args, 'num_domains_q', 4))
        C = float(getattr(self.args, 'num_classes', 7))
        # Eq (10) 简化形式 d_k = sqrt(C/2) * |n_k/N - 1/Q|
        d = np.sqrt(C / 2.0) * np.abs(sample_share - 1.0 / Q)
        alpha = float(getattr(self.args, 'agg_a', 1.0))
        beta = float(getattr(self.args, 'agg_b', 0.4))
        # Eq (11) sigmoid + normalize
        score = alpha * sample_share - beta * d
        sig = 1.0 / (1.0 + np.exp(-score))
        freq = sig / sig.sum()
        # 暴露给 diagnostic hook (cold path 后处理用)
        self._last_daa_freq = freq.tolist()
        return freq

    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        nets_list = self.nets_list

        c_samples_list = self.num_samples

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        # === diagnostic: 在 sync 前算 client 跟 global 的 drift ===
        # 必须在 client.load_state_dict(global_w) 之前算, 否则 diff = 0
        try:
            import torch as _t
            old_global_keys = list(global_w.keys())
            critical_patterns = ['conv1.weight', 'layer1.0.conv1', 'layer4.0.conv1',
                                 'classifier.weight', 'linear.weight', 'cls.weight']
            actual_layers = [k for k in old_global_keys
                             if any(p in k for p in critical_patterns)]
            old_global_snapshot = {k: global_w[k].detach().clone().float()
                                   for k in actual_layers}
            layer_l2 = {}
            grad_l2 = {}
            for net_id in online_clients:
                client_sd = self.nets_list[net_id].state_dict()
                client_drift = {}
                sq_sum = 0.0
                for layer in actual_layers:
                    if layer in client_sd:
                        diff = client_sd[layer].detach().float() - old_global_snapshot[layer]
                        l2 = float(diff.norm().item())
                        client_drift[layer] = l2
                        sq_sum += l2 ** 2
                layer_l2[net_id] = client_drift
                grad_l2[net_id] = sq_sum ** 0.5
            self._last_layer_l2 = layer_l2
            self._last_grad_l2 = grad_l2
        except Exception as _e:
            # 任何错误都不要让 train 死
            self._last_layer_l2 = {}
            self._last_grad_l2 = {}

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            online_clients_all = np.sum(online_clients_len)
            if getattr(self.args, 'use_daa', False):
                freq = self._compute_daa_freq(online_clients_len)
            else:
                freq = online_clients_len / online_clients_all
        else:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]

        # NOTE (2026-04-29): 维持 F2DC release 默认行为 — state_dict 全 key 聚合.
        # 论文 Sec 4.3 文字描述 A_D/A_C/m 本地保留, 但 release 代码 + 论文报数字 76.47
        # 实际是按全聚合 + global_net eval 跑出来的 (跟我们当前流程一致).
        # 之前曾尝试加聚合白名单 (commit 0e5f1df fix #1), 但跟 global_net eval 不兼容
        # (global_net.dfd/dfc 永远是 random init, eval forward 走 garbage path), 已 revert.
        # class_proto buffer (persistent=False) 仍然不在 state_dict, 由 _sync_global_proto
        # 显式同步到 global_net.dfc_module (在 f2dc_pg.loc_update 末尾调用).
        first = True
        for index, net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[index]

        global_net.load_state_dict(global_w)


        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
            