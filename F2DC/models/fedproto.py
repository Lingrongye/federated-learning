import torch.optim as optim
import torch.nn as nn
import torch
import copy
from tqdm import tqdm
from models.utils.federated_model import FederatedModel


def agg_func(protos):
    """Average per-class proto list (used inside one client's local protos)."""
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for p in proto_list:
                proto = proto + p.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0].data
    return protos


class FedProto(FederatedModel):
    NAME = 'fedproto'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedProto, self).__init__(nets_list, args, transform)
        self.mu = getattr(args, 'mu', 1.0)
        self.global_protos = {}
        self.local_protos = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self, local_protos_list):
        agg_protos_label = {}
        for idx in self.online_clients:
            if idx not in local_protos_list:
                continue
            for label, proto in local_protos_list[idx].items():
                agg_protos_label.setdefault(label, []).append(proto)
        out = {}
        for label, proto_list in agg_protos_label.items():
            if len(proto_list) > 1:
                p = 0 * proto_list[0]
                for x in proto_list:
                    p = p + x
                out[label] = p / len(proto_list)
            else:
                out[label] = proto_list[0]
        return out

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

        self.global_protos = self.proto_aggregation(self.local_protos)
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
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        loss_mse = nn.MSELoss()

        global_loss = 0.0
        global_samples = 0
        num_c_samples = 0
        iterator = tqdm(range(self.local_epoch))

        agg_protos_label = {}
        for it in iterator:
            epoch_loss = 0.0
            epoch_samples = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                lossCE = criterion(outputs, labels)

                f = net.features(images)

                if len(self.global_protos) == 0:
                    lossProto = 0 * lossCE
                else:
                    f_target = f.detach().clone()
                    has_proto = torch.zeros(labels.size(0), dtype=torch.bool, device=self.device)
                    for i in range(labels.size(0)):
                        lab = labels[i].item()
                        if lab in self.global_protos:
                            f_target[i] = self.global_protos[lab].to(self.device)
                            has_proto[i] = True
                    if has_proto.any():
                        lossProto = loss_mse(f[has_proto], f_target[has_proto])
                    else:
                        lossProto = 0 * lossCE

                loss = lossCE + self.mu * lossProto
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE=%0.3f Proto=%0.3f" % (index, lossCE.item(), float(lossProto))
                optimizer.step()

                bs = labels.size(0)
                epoch_loss += lossCE.item() * bs
                epoch_samples += bs

                # 累 last epoch 的 prototype
                if it == self.local_epoch - 1:
                    f_det = f.detach()
                    for i in range(labels.size(0)):
                        lab = labels[i].item()
                        agg_protos_label.setdefault(lab, []).append(f_det[i])

            global_loss += epoch_loss
            global_samples += epoch_samples
            num_c_samples = epoch_samples

        self.local_protos[index] = agg_func(agg_protos_label)
        global_avg_loss = global_loss / global_samples
        return round(global_avg_loss, 3), num_c_samples
