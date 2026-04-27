import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np
from models.utils.federated_model import FederatedModel


def collect_bn_keys(net):
    """枚举 state_dict 中所有属于 BN/Norm 模块的 keys (含 alias).

    ResNet 用 _features = nn.Sequential([..., bn1, ...]) 时, state_dict 里同一 BN buffer
    会有两个 alias key: 'bn1.running_mean' 跟 '_features.1.running_mean'. 必须把两个都
    判为 BN, 否则 alias key 会被错聚合, load 时通过 alias 写回真 BN buffer.

    实现: 收集所有 BN module 的 buffer/parameter tensor 的 data_ptr (storage 标识),
    然后检查 state_dict 每个 key 的 tensor data_ptr 是否在该 set 里.
    """
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.GroupNorm, nn.LayerNorm, nn.SyncBatchNorm)
    bn_storage_ptrs = set()
    for _, mod in net.named_modules():
        if isinstance(mod, norm_types):
            for _, p in mod.named_parameters(recurse=False):
                bn_storage_ptrs.add(p.data_ptr())
            for _, b in mod.named_buffers(recurse=False):
                bn_storage_ptrs.add(b.data_ptr())

    bn_keys = set()
    sd = net.state_dict()
    for k, v in sd.items():
        if v.data_ptr() in bn_storage_ptrs:
            bn_keys.add(k)
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
        """FedBN 聚合:
        - non-BN 参数 (conv/classifier 等): client + global_net 都聚合 (FedAvg style)
        - BN 参数: client 留本地不聚合 (FedBN 核心) 但 global_net 用 client mean (供 eval)

        F2DC 框架 global_evaluate 用 model.global_net 评估, 如果 global_net 的 BN 永远是
        init 0, eval 时 BN 没归一化 → accuracy = 乱猜. 所以 global_net BN 必须取 client
        mean (FedAvg-style avg of client BN), 但不能 sync 回 client (违反 FedBN idea)."""
        online_clients = self.online_clients

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[i] for i in online_clients]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            total = float(np.sum(online_clients_len))
            freq = [n / total for n in online_clients_len]
        else:
            parti_num = len(online_clients)
            freq = [1.0 / parti_num for _ in range(parti_num)]

        first_sd = self.nets_list[online_clients[0]].state_dict()
        non_bn_keys = [k for k in first_sd if not self._is_bn_key(k)]
        bn_keys = [k for k in first_sd if self._is_bn_key(k)]

        # 加权累加 ALL keys (non-BN + BN), 但分开 dispatch
        agg_non_bn = {}
        agg_bn = {}
        for idx, net_id in enumerate(online_clients):
            sd = self.nets_list[net_id].state_dict()
            for k in non_bn_keys:
                v = sd[k].detach()
                if k not in agg_non_bn:
                    agg_non_bn[k] = v.clone().float() * freq[idx]
                else:
                    agg_non_bn[k] = agg_non_bn[k] + v.float() * freq[idx]
            for k in bn_keys:
                # num_batches_tracked 是 int, 不参与 mean (取最后一个 client 的)
                if 'num_batches_tracked' in k:
                    agg_bn[k] = sd[k].detach().clone()
                    continue
                v = sd[k].detach()
                if k not in agg_bn:
                    agg_bn[k] = v.clone().float() * freq[idx]
                else:
                    agg_bn[k] = agg_bn[k] + v.float() * freq[idx]

        # cast 回原 dtype
        for k in agg_non_bn:
            agg_non_bn[k] = agg_non_bn[k].to(first_sd[k].dtype)
        for k in agg_bn:
            agg_bn[k] = agg_bn[k].to(first_sd[k].dtype)

        # global_net: load BOTH non-BN + BN (供 eval 用)
        merged = {**agg_non_bn, **agg_bn}
        self.global_net.load_state_dict(merged, strict=False)

        # client nets: 只 load non-BN (BN 留本地)
        for net in self.nets_list:
            net.load_state_dict(agg_non_bn, strict=False)
