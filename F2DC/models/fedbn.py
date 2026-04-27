import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np
from models.utils.federated_model import FederatedModel


def collect_bn_keys(net):
    """枚举 net 中所有 BN/GroupNorm/LayerNorm 类型 module 的 state_dict keys.
    必须基于 named_modules 类型, 不能用字符串 match —
    ResNet 的 nn.Sequential(Conv, BN) 里 BN 的 key 是 'shortcut.1.weight' 不含 'bn'.
    """
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.GroupNorm, nn.LayerNorm, nn.SyncBatchNorm)
    bn_keys = set()
    for name, mod in net.named_modules():
        if isinstance(mod, norm_types):
            for pname, _ in mod.named_parameters(recurse=False):
                bn_keys.add(f"{name}.{pname}")
            for bname, _ in mod.named_buffers(recurse=False):
                bn_keys.add(f"{name}.{bname}")
    return bn_keys


class FedBN(FederatedModel):
    NAME = 'fedbn'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedBN, self).__init__(nets_list, args, transform)
        self._bn_keys = None  # lazy init in ini() after global_net 创建

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
        self._bn_keys = collect_bn_keys(self.global_net)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        self.num_samples = []
        all_clients_loss = 0.0
        for i in online_clients:
            c_loss, c_samples = self._train_net(i, self.nets_list[i], priloader_list[i])
            all_clients_loss += c_loss
            self.num_samples.append(c_samples)

        self.aggregate_nets_skip_bn()

        all_c_avg_loss = all_clients_loss / len(online_clients)
        return round(all_c_avg_loss, 3)

    def _train_net(self, index, net, train_loader):
        try:
            n_avail = len(train_loader.sampler.indices) if hasattr(train_loader.sampler, "indices") else len(train_loader.dataset)
        except Exception:
            n_avail = 1
        if n_avail == 0:
            print(f"[skip] client {index} has empty dataloader")
            return 0.0, 0
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)

        global_loss = 0.0
        global_samples = 0
        num_c_samples = 0
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            epoch_loss = 0.0
            epoch_samples = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
                bs = labels.size(0)
                epoch_loss += loss.item() * bs
                epoch_samples += bs
            global_loss += epoch_loss
            global_samples += epoch_samples
            num_c_samples = epoch_samples

        global_avg_loss = global_loss / global_samples
        return round(global_avg_loss, 3), num_c_samples

    def _is_bn_key(self, key):
        if self._bn_keys is None:
            self._bn_keys = collect_bn_keys(self.global_net)
        return key in self._bn_keys

    def aggregate_nets_skip_bn(self):
        """FedBN 聚合: BN 参数不参与聚合, 留本地。
        只构造 non-BN 子 dict, 用 strict=False load — 完全不 touch BN buffer."""
        online_clients = self.online_clients

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[i] for i in online_clients]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            total = float(np.sum(online_clients_len))
            freq = [n / total for n in online_clients_len]
        else:
            parti_num = len(online_clients)
            freq = [1.0 / parti_num for _ in range(parti_num)]

        # 收集 non-BN keys 列表
        first_sd = self.nets_list[online_clients[0]].state_dict()
        non_bn_keys = [k for k in first_sd if not self._is_bn_key(k)]

        # 加权累加
        agg = {}
        for idx, net_id in enumerate(online_clients):
            sd = self.nets_list[net_id].state_dict()
            for k in non_bn_keys:
                v = sd[k].detach()
                if k not in agg:
                    agg[k] = v.clone().float() * freq[idx]
                else:
                    agg[k] = agg[k] + v.float() * freq[idx]

        # 把累加结果 cast 回原 dtype
        for k in agg:
            agg[k] = agg[k].to(first_sd[k].dtype)

        # 用 strict=False load — 只覆盖 non-BN, BN 完全不 touch
        self.global_net.load_state_dict(agg, strict=False)
        for net in self.nets_list:
            net.load_state_dict(agg, strict=False)
