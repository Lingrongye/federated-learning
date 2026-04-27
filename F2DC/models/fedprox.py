import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from models.utils.federated_model import FederatedModel


class FedProx(FederatedModel):
    NAME = 'fedprox'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProx, self).__init__(nets_list, args, transform)
        self.mu = args.mu

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

        self.aggregate_nets(None)

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
        self.global_net = self.global_net.to(self.device)
        global_weight_collector = [p.detach().clone() for p in self.global_net.parameters()]

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
                lossCE = criterion(outputs, labels)

                fed_prox_reg = 0.0
                for p, g_p in zip(net.parameters(), global_weight_collector):
                    fed_prox_reg = fed_prox_reg + ((p - g_p) ** 2).sum()
                loss = lossCE + (self.mu / 2.0) * fed_prox_reg

                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f, Prox = %0.3f" % (index, lossCE, fed_prox_reg)
                optimizer.step()
                bs = labels.size(0)
                epoch_loss += lossCE.item() * bs
                epoch_samples += bs
            global_loss += epoch_loss
            global_samples += epoch_samples
            num_c_samples = epoch_samples

        global_avg_loss = global_loss / global_samples
        return round(global_avg_loss, 3), num_c_samples
