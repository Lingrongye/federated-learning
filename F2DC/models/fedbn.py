import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np
from models.utils.federated_model import FederatedModel


class FedBN(FederatedModel):
    NAME = 'fedbn'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedBN, self).__init__(nets_list, args, transform)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

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

    @staticmethod
    def _is_bn_key(key):
        kl = key.lower()
        if 'bn' in kl or 'batchnorm' in kl:
            return True
        if 'running_mean' in kl or 'running_var' in kl or 'num_batches_tracked' in kl:
            return True
        return False

    def aggregate_nets_skip_bn(self):
        global_net = self.global_net
        nets_list = self.nets_list
        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[i] for i in online_clients]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            total = np.sum(online_clients_len)
            freq = online_clients_len / total
        else:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]

        first = True
        for index, net_id in enumerate(online_clients):
            net_para = nets_list[net_id].state_dict()
            if first:
                first = False
                for key in net_para:
                    if self._is_bn_key(key):
                        continue
                    global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    if self._is_bn_key(key):
                        continue
                    global_w[key] += net_para[key] * freq[index]

        global_net.load_state_dict(global_w)

        for net in nets_list:
            local_w = net.state_dict()
            for key in global_w:
                if self._is_bn_key(key):
                    continue
                local_w[key] = global_w[key]
            net.load_state_dict(local_w)
