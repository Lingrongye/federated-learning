"""
FedDSA-MultiLayer: Apply AdaIN style augmentation at multiple layers (not just final).

Inspired by StyleGAN: lower layers = texture, higher layers = content.
By injecting external style at multiple depths, we get more thorough style adaptation.

Architecture changes:
- Encoder exposes intermediate features after fc1
- Style augmentation applied at BOTH fc1 output (1024-d) AND fc2 output (1024-d)
- Each layer uses its own style statistics (collected from local data)
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fuf

from algorithm.feddsa import Client as BaseClient


class MultiLayerEncoder(nn.Module):
    """AlexNet encoder that exposes mid-layer features for multi-layer style injection."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('bn2', nn.BatchNorm2d(192)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(384)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)

    def forward_to_mid(self, x):
        """Forward to mid-layer (after fc1+bn6+relu)."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        return x  # [B, 1024] mid features

    def forward_from_mid(self, mid):
        """Forward from mid to final."""
        x = self.bn7(self.fc2(mid))
        x = self.relu(x)
        return x  # [B, 1024] final features

    def forward(self, x):
        mid = self.forward_to_mid(x)
        return self.forward_from_mid(mid)


class FedDSAMultiLayerModel(fuf.FModule):
    def __init__(self, num_classes=7, feat_dim=1024, proj_dim=128):
        super().__init__()
        self.encoder = MultiLayerEncoder()
        self.semantic_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        z_sem = self.semantic_head(h)
        return self.head(z_sem)

    def encode_mid(self, x):
        return self.encoder.forward_to_mid(x)

    def encode_from_mid(self, mid):
        return self.encoder.forward_from_mid(mid)

    def encode(self, x):
        return self.encoder(x)

    def get_semantic(self, h):
        return self.semantic_head(h)

    def get_style(self, h):
        return self.style_head(h)


class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({
            'lambda_orth': 1.0,
            'lambda_hsic': 0.0,
            'lambda_sem': 1.0,
            'tau': 0.1,
            'warmup_rounds': 50,
            'style_dispatch_num': 5,
            'proj_dim': 128,
        })
        self.sample_option = 'full'
        # Two banks: mid-layer style + final-layer style
        self.style_bank_mid = {}
        self.style_bank_final = {}
        self.global_semantic_protos = {}
        self._init_agg_keys()

        for c in self.clients:
            c.lambda_orth = self.lambda_orth
            c.lambda_hsic = self.lambda_hsic
            c.lambda_sem = self.lambda_sem
            c.tau = self.tau
            c.warmup_rounds = self.warmup_rounds
            c.proj_dim = self.proj_dim

    def _init_agg_keys(self):
        all_keys = list(self.model.state_dict().keys())
        self.private_keys = set()
        for k in all_keys:
            if 'style_head' in k:
                self.private_keys.add(k)
            elif 'bn' in k.lower() and ('running_' in k or 'num_batches_tracked' in k):
                self.private_keys.add(k)
        self.shared_keys = [k for k in all_keys if k not in self.private_keys]

    def pack(self, client_id, mtype=0):
        dispatched_mid = None
        dispatched_final = None
        if len(self.style_bank_final) > 0 and self.current_round >= self.warmup_rounds:
            available_mid = {cid: s for cid, s in self.style_bank_mid.items() if cid != client_id}
            available_final = {cid: s for cid, s in self.style_bank_final.items() if cid != client_id}
            if len(available_final) == 0:
                available_mid = self.style_bank_mid
                available_final = self.style_bank_final
            n = min(self.style_dispatch_num, len(available_final))
            keys = list(available_final.keys())
            chosen = np.random.choice(keys, n, replace=False)
            dispatched_mid = [available_mid[k] for k in chosen if k in available_mid]
            dispatched_final = [available_final[k] for k in chosen]

        return {
            'model': copy.deepcopy(self.model),
            'global_protos': copy.deepcopy(self.global_semantic_protos),
            'style_bank_mid': dispatched_mid,
            'style_bank_final': dispatched_final,
            'current_round': self.current_round,
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)

        models = res['model']
        protos_list = res['protos']
        proto_counts_list = res['proto_counts']
        style_mid_list = res['style_mid']
        style_final_list = res['style_final']

        self._aggregate_shared(models)

        for cid, sm, sf in zip(self.received_clients, style_mid_list, style_final_list):
            if sm is not None:
                self.style_bank_mid[cid] = sm
            if sf is not None:
                self.style_bank_final[cid] = sf

        self._aggregate_protos(protos_list, proto_counts_list)

    def _aggregate_shared(self, models):
        if len(models) == 0:
            return
        weights = np.array([len(self.clients[cid].train_data) for cid in self.received_clients], dtype=float)
        weights /= weights.sum()
        global_dict = self.model.state_dict()
        model_dicts = [m.state_dict() for m in models]
        for k in self.shared_keys:
            if 'num_batches_tracked' in k:
                continue
            global_dict[k] = sum(w * md[k] for w, md in zip(weights, model_dicts))
        self.model.load_state_dict(global_dict, strict=False)

    def _aggregate_protos(self, protos_list, counts_list):
        agg = {}
        for protos, counts in zip(protos_list, counts_list):
            if protos is None: continue
            for c, proto in protos.items():
                cnt = counts.get(c, 1)
                if c not in agg:
                    agg[c] = (proto * cnt, cnt)
                else:
                    prev_sum, prev_cnt = agg[c]
                    agg[c] = (prev_sum + proto * cnt, prev_cnt + cnt)
        self.global_semantic_protos = {}
        for c, (s, n) in agg.items():
            self.global_semantic_protos[c] = s / n


class Client(BaseClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        self.loss_fn = nn.CrossEntropyLoss()
        self.current_round = 0
        self.local_style_bank_mid = None
        self.local_style_bank_final = None
        self.global_protos = None

    def reply(self, svr_pkg):
        model, global_protos, sb_mid, sb_final, current_round = self.unpack(svr_pkg)
        self.current_round = current_round
        self.global_protos = global_protos
        self.local_style_bank_mid = sb_mid
        self.local_style_bank_final = sb_final
        self.train(model)
        return self.pack()

    def unpack(self, svr_pkg):
        global_model = svr_pkg['model']
        if self.model is None:
            self.model = global_model
        else:
            new_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in new_dict.keys():
                if 'style_head' in key:
                    continue
                if 'bn' in key.lower() and ('running_' in key or 'num_batches_tracked' in key):
                    continue
                new_dict[key] = global_dict[key]
            self.model.load_state_dict(new_dict)
        return (
            self.model,
            svr_pkg['global_protos'],
            svr_pkg['style_bank_mid'],
            svr_pkg['style_bank_final'],
            svr_pkg['current_round'],
        )

    def pack(self):
        return {
            'model': copy.deepcopy(self.model.to('cpu')),
            'protos': self._local_protos,
            'proto_counts': self._local_proto_counts,
            'style_mid': self._local_style_mid,
            'style_final': self._local_style_final,
        }

    def _adain(self, h, mu_ext, sigma_ext):
        mu_ext = mu_ext.to(h.device)
        sigma_ext = sigma_ext.to(h.device)
        mu_local = h.mean(dim=0)
        sigma_local = h.std(dim=0, unbiased=False).clamp(min=1e-6)
        alpha = np.random.beta(0.1, 0.1)
        mu_mix = alpha * mu_local + (1 - alpha) * mu_ext
        sigma_mix = alpha * sigma_local + (1 - alpha) * sigma_ext
        h_norm = (h - mu_local) / sigma_local
        return h_norm * sigma_mix + mu_mix

    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        optimizer = self.calculator.get_optimizer(
            model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum
        )
        aux_w = min(1.0, self.current_round / max(self.warmup_rounds, 1))

        proto_sum = {}
        proto_count = {}
        mid_sum = None; mid_sq_sum = None; mid_n = 0
        final_sum = None; final_sq_sum = None; final_n = 0

        for step in range(self.num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            x, y = batch_data[0], batch_data[-1]

            model.zero_grad()

            # Forward through encoder, exposing mid features
            mid = model.encode_mid(x)
            h = model.encode_from_mid(mid)
            z_sem = model.get_semantic(h)
            z_sty = model.get_style(h)

            output = model.head(z_sem)
            loss_task = self.loss_fn(output, y)

            # Multi-layer style augmentation
            loss_aug = torch.tensor(0.0, device=x.device)
            if (self.local_style_bank_mid is not None and self.local_style_bank_final is not None
                and self.current_round >= self.warmup_rounds):
                # Pick same external style for both layers (consistent style)
                idx = np.random.randint(0, len(self.local_style_bank_final))
                mu_mid, sig_mid = self.local_style_bank_mid[idx]
                mu_fin, sig_fin = self.local_style_bank_final[idx]

                # Inject at mid layer
                mid_aug = self._adain(mid, mu_mid, sig_mid)
                # Continue forward
                h_from_aug_mid = model.encode_from_mid(mid_aug)
                # Inject at final layer too
                h_aug = self._adain(h_from_aug_mid, mu_fin, sig_fin)
                # Pass through semantic head
                z_sem_aug = model.get_semantic(h_aug)
                output_aug = model.head(z_sem_aug)
                loss_aug = self.loss_fn(output_aug, y)

            loss_orth, loss_hsic = self._decouple_loss(z_sem, z_sty)

            loss_sem = torch.tensor(0.0, device=x.device)
            if self.global_protos and len(self.global_protos) >= 2:
                loss_sem = self._infonce_loss(z_sem, y)

            loss = loss_task + loss_aug + \
                aux_w * self.lambda_orth * loss_orth + \
                aux_w * self.lambda_hsic * loss_hsic + \
                aux_w * self.lambda_sem * loss_sem

            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

            if step >= self.num_steps - len(self.train_data) // self.batch_size - 1:
                with torch.no_grad():
                    z_det = z_sem.detach().cpu()
                    h_det = h.detach()
                    mid_det = mid.detach()
                    for i, label in enumerate(y):
                        c = label.item()
                        if c not in proto_sum:
                            proto_sum[c] = z_det[i].clone()
                            proto_count[c] = 1
                        else:
                            proto_sum[c] += z_det[i]
                            proto_count[c] += 1

                    b = h_det.size(0)
                    # Mid stats
                    bm = mid_det.mean(dim=0).cpu()
                    bsq = (mid_det ** 2).mean(dim=0).cpu()
                    if mid_sum is None:
                        mid_sum = bm * b; mid_sq_sum = bsq * b; mid_n = b
                    else:
                        mid_sum += bm * b; mid_sq_sum += bsq * b; mid_n += b
                    # Final stats
                    bm = h_det.mean(dim=0).cpu()
                    bsq = (h_det ** 2).mean(dim=0).cpu()
                    if final_sum is None:
                        final_sum = bm * b; final_sq_sum = bsq * b; final_n = b
                    else:
                        final_sum += bm * b; final_sq_sum += bsq * b; final_n += b

        self._local_protos = {c: proto_sum[c] / proto_count[c] for c in proto_sum}
        self._local_proto_counts = proto_count

        if mid_n > 1:
            mu = mid_sum / mid_n
            var = mid_sq_sum / mid_n - mu ** 2
            self._local_style_mid = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_mid = None
        if final_n > 1:
            mu = final_sum / final_n
            var = final_sq_sum / final_n - mu ** 2
            self._local_style_final = (mu, var.clamp(min=1e-6).sqrt())
        else:
            self._local_style_final = None


model_map = {
    'PACS': lambda: FedDSAMultiLayerModel(num_classes=7, feat_dim=1024, proj_dim=128),
    'office': lambda: FedDSAMultiLayerModel(num_classes=10, feat_dim=1024, proj_dim=128),
}


def init_dataset(object): pass
def init_local_module(object): pass
def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map.get(dataset, lambda: FedDSAMultiLayerModel())().to(object.device)
