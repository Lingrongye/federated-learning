"""FDSE (CVPR 2025) ported from FDSE_CVPR25/algorithm/fdse.py to F2DC framework.

完整保留原 FDSE 算法的 3 个核心:
1. 层分解 backbone (DSEConv/DSELinear, 见 backbone/ResNet_FDSE.py)
2. Server 差异化聚合:
   - shared_keys (含 'dfe', 'head'): QP 优化的 lambda 加权 (FedAvg-like 但 lambda 由 cvxopt 解出)
   - personalized_keys (含 dse_conv 主体): 按 cosine similarity softmax 加权
   - local_keys (dse_bn.running_): 完全本地, 不聚合 (FedBN 原则)
3. Client 训练加 L_reg consistency loss (用 hook 抓 dfe_bn 输入特征, 跟 global running mean/var 对齐)

Paper: Federated Learning with Domain Shift Eraser (Wang et al., CVPR 2025)
源码: FDSE_CVPR25/algorithm/fdse.py (我们项目主线 baseline 的官方实现)
"""
import os
import sys
import copy
import math
import numpy as np
import cvxopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.args import *
from models.utils.federated_model import FederatedModel


def _modeldict_weighted_average(dicts, weights):
    """替代 flgo.utils.fmodule._modeldict_weighted_average. 同接口同语义."""
    keys = list(dicts[0].keys())
    out = {}
    for k in keys:
        stacked = torch.stack([d[k].float() for d in dicts])
        w = torch.tensor(weights, device=stacked.device).float()
        # broadcast w to stacked shape
        for _ in range(stacked.ndim - 1):
            w = w.unsqueeze(-1)
        out[k] = (stacked * w).sum(dim=0)
    return out


class FDSE(FederatedModel):
    NAME = 'fdse'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super().__init__(nets_list, args, transform)
        self.args = args
        # FDSE 原 algo_para defaults (FDSE_CVPR25/algorithm/fdse.py line 96)
        self.lmbd = getattr(args, 'lmbd', 0.01)
        self.tau = getattr(args, 'fdse_tau', 0.5)
        self.beta = getattr(args, 'fdse_beta', 0.1)
        # Round counter (FDSE 原版 server.current_round, 我们用 epoch_index 等价)
        self.current_round = 0

    def ini(self):
        # ini phase (类似 FDSE Server.initialize): 拆 keys 三类
        # shared: 'dfe' / 'head' / stem (conv1, bn1) / shortcut (QP 加权聚合)
        #         — stem 和 shortcut 是普通 nn.Conv2d/BN, 不做 DSE 分解, 全局共享
        # local:  'dse_bn.running_' (不聚合, FedBN 原则)
        # personalized: 其他 (主要是 dse_conv, cosine sim softmax 聚合)
        sample_state = self.nets_list[0].state_dict()
        shared_substrings = ['dfe', 'head', '.shortcut.']
        shared_prefixes = ['conv1.', 'bn1.']  # stem only (用 prefix 避免误匹配 layer1.0.conv1)
        local_names = ['dse_bn.running_']
        self.shared_keys = [
            k for k in sample_state
            if any(s in k for s in shared_substrings)
            or any(k.startswith(p) for p in shared_prefixes)
        ]
        self.local_keys = [k for k in sample_state if any(s in k for s in local_names)
                           and k not in self.shared_keys]
        self.personalized_keys = [k for k in sample_state
                                   if k not in self.shared_keys and k not in self.local_keys]
        print(f"[FDSE] keys: shared={len(self.shared_keys)} local={len(self.local_keys)} personalized={len(self.personalized_keys)}")
        # global_net = first client's deepcopy
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)
        # 每 client 的 personalized state (per FDSE Server.initialize line 105)
        per_state = {k: v for k, v in sample_state.items() if k in self.personalized_keys}
        self.client_states = [copy.deepcopy(per_state) for _ in self.nets_list]
        # 收集 dfe_bn 模块的 attribute path (用于 hook)
        self.bnlayers = self._collect_bnlayer_paths(self.nets_list[0])

    @staticmethod
    def _collect_bnlayer_paths(model):
        """跟 FDSE Client.initialize line 165-167 一致, 取 dfe_bn 的 attribute path."""
        paths = []
        seen = set()
        for k in model.state_dict().keys():
            if 'dfe_bn' in k:
                # k 例如 'layer1.0.conv1.dfe_bn.weight' → path = '.layer1[0].conv1.dfe_bn'
                parts = k.split('.')[:-1]  # 去掉末尾 weight/bias/running_xxx
                path = ''.join([f'[{p}]' if p.isdigit() else f'.{p}' for p in parts])
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
        return paths

    def loc_update(self, priloader_list):
        total = list(range(self.args.parti_num))
        online = self.random_state.choice(total, self.online_num, replace=False).tolist()
        self.online_clients = online
        self.num_samples = []
        all_loss = 0.0
        # 1. 客户端本地训练 (含 L_reg)
        client_models_after_train = {}
        for i in online:
            # 应用本 client 的 personalized state + 当前 global shared
            self._dispatch_to_client(i)
            c_loss, c_samples = self._train_net(i, self.nets_list[i], priloader_list[i])
            all_loss += c_loss
            self.num_samples.append(c_samples)
            client_models_after_train[i] = self.nets_list[i].state_dict()
        # 2. 服务器聚合 (FDSE 差异化 3 类 keys 不同策略)
        self._fdse_aggregate(client_models_after_train)
        # 3. 同步 global_net 到所有 client (但 personalized 用 client_states 覆盖)
        for cid in range(self.args.parti_num):
            self._dispatch_to_client(cid)
        self.current_round += 1
        return round(all_loss / max(len(online), 1), 3)

    def _dispatch_to_client(self, cid):
        """把 global shared + 本 client personalized 装到 nets_list[cid] (FDSE Server.pack line 117)."""
        client_dict = {k: v for k, v in self.global_net.state_dict().items() if k in self.shared_keys}
        client_dict.update(self.client_states[cid])
        # local_keys (dse_bn.running_) 保留 nets_list[cid] 自己的 (不覆盖)
        self.nets_list[cid].load_state_dict(client_dict, strict=False)

    def _fdse_aggregate(self, client_models_dict):
        """完全照搬 FDSE Server.iterate line 121-167."""
        online = self.online_clients
        mdicts = [client_models_dict[cid] for cid in online]
        # ---- Step 1: shared keys QP 优化加权 ----
        current_shared = {k: v for k, v in self.global_net.state_dict().items() if k in self.shared_keys}
        shared_dicts = [{k: v for k, v in md.items() if k in self.shared_keys} for md in mdicts]
        new_shared = {}
        for k in self.shared_keys:
            if 'running_' in k or 'num_batches_tracked' in k:
                # BN running stats: 简单 stack mean (跟 FDSE line 127-130 一致)
                k_vecs = [md[k].float() for md in shared_dicts]
                if 'num_batches_tracked' in k:
                    new_shared[k] = torch.stack(k_vecs).sum(dim=0).squeeze(0).long()
                else:
                    new_shared[k] = torch.stack(k_vecs).mean(dim=0).squeeze(0)
            else:
                # weights: QP 求 optimal lambda
                shape = shared_dicts[0][k].shape
                crt_vec_k = current_shared[k].reshape(-1).to(self.device).float()
                k_vecs = [md[k].reshape(-1).to(self.device).float() - crt_vec_k for md in shared_dicts]
                k_norms = [t.norm() for t in k_vecs]
                k_vecs_norm = [t / (tn + 1e-8) for t, tn in zip(k_vecs, k_norms)]
                # 抑制 cvxopt verbose
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    op_lambda = self._optim_lambda(k_vecs_norm)
                finally:
                    sys.stdout.close()
                    sys.stdout = old_stdout
                op_lambda_t = torch.tensor([e[0] for e in op_lambda], dtype=torch.float32, device=self.device)
                new_vec_k = (op_lambda_t.unsqueeze(0) @ torch.stack(k_vecs_norm))[0]
                new_shared[k] = (torch.stack(k_norms).mean() * new_vec_k + crt_vec_k).reshape(shape)
        self.global_net.load_state_dict(new_shared, strict=False)

        # ---- Step 2: personalized keys cosine similarity softmax 聚合 ----
        per_dicts = [{k: v for k, v in md.items() if k in self.personalized_keys} for md in mdicts]
        for k in self.personalized_keys:
            if 'num_batches_tracked' in k:
                continue
            # 计算 client 间 cosine similarity
            per_k_vecs = [md[k].reshape(-1).float().to(self.device) for md in per_dicts]
            per_k_vecs_norm = [v / (v.norm() + 1e-8) for v in per_k_vecs]
            stacked = torch.stack(per_k_vecs_norm)
            sims = stacked @ stacked.T  # [N, N]
            per_k_dicts = [{k: md[k]} for md in per_dicts]
            for cid_idx, cid in enumerate(online):
                weight_cid = F.softmax(sims[cid_idx] / self.tau, dim=0).cpu().tolist()
                agg = _modeldict_weighted_average(per_k_dicts, weight_cid)
                self.client_states[cid][k] = agg[k].cpu()

        # ---- Step 3: 把 local_keys (dse_bn.running_) 也聚合到 global_net (仅供 evaluate 用)
        # 原 FDSE 用 client local model evaluate, 所以不需要 global 的 BN running.
        # F2DC 框架 global_evaluate 用 model.global_net, global_net 的 dse_bn.running 永远不更新
        # 会导致 eval 时 BN 用 init 0/1 stats → acc 严重低估. 加这步: 简单 mean 聚合 BN running.
        # 注意: client 训练时仍用自己的 dse_bn (本地保留), 这步只为 eval 服务.
        if self.local_keys:
            global_state = self.global_net.state_dict()
            for k in self.local_keys:
                if 'num_batches_tracked' in k:
                    vals = [md[k] for md in mdicts]
                    global_state[k] = torch.stack(vals).sum(dim=0).long()
                else:
                    vals = [md[k].float() for md in mdicts]
                    global_state[k] = torch.stack(vals).mean(dim=0)
            self.global_net.load_state_dict(global_state, strict=False)

    def _optim_lambda(self, grads):
        """QP 求 optimal lambda (FDSE line 169-189)."""
        n = len(grads)
        Jt = np.array([g.cpu().detach().numpy() for g in grads])
        P = 2 * np.dot(Jt, Jt.T)
        q = np.array([[0] for _ in range(n)])
        A = np.ones(n).T
        b = np.array([1])
        lb = np.zeros(n)
        ub = np.ones(n)
        G = np.zeros((2 * n, n))
        for i in range(n):
            G[i][i] = -1
            G[n + i][i] = 1
        h = np.zeros((2 * n, 1))
        for i in range(n):
            h[i] = -lb[i]
            h[n + i] = ub[i]
        return self._quadprog(P, q, G, h, A, b)

    def _quadprog(self, P, q, G, h, A, b):
        P = cvxopt.matrix(P.tolist())
        q = cvxopt.matrix(q.tolist(), tc='d')
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist(), tc='d')
        sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
        return np.array(sol['x'])

    def _train_net(self, index, net, train_loader):
        """跟 FDSE Client.train line 173-203 一致 (hook + L_reg)."""
        # patch: skip empty client
        try:
            n_avail = len(train_loader.sampler.indices) if hasattr(train_loader.sampler, 'indices') \
                      else len(train_loader.dataset)
        except Exception:
            n_avail = 1
        if n_avail == 0:
            print(f"[skip] client {index} empty dataloader")
            return 0.0, 0
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # 拿到所有 dfe_bn 模块对象 (FDSE line 175-180)
        layers = []
        for ln in self.bnlayers:
            mod = eval('net' + ln)
            layers.append(mod)
        # global running stats (deepcopy detach)
        global_means = [l.running_mean.to(self.device).detach().clone() for l in layers]
        global_vars = [l.running_var.to(self.device).detach().clone() for l in layers]
        # exponential weight per layer (FDSE line 184-185)
        weights = np.exp(np.array([self.beta * i for i in range(len(layers))]))
        weights = weights / weights.sum()
        # hook 抓 dfe_bn 输入特征 (FDSE line 186-191)
        feature_maps = []
        def hook(module, inp, out):
            x = inp[0]
            feature_maps.append(x.mean(dim=0).mean(dim=-1).mean(dim=-1) if x.ndim > 3 else x.mean(dim=0))
        lhooks = [l.register_forward_hook(hook) for l in layers]

        global_loss = 0.0
        global_samples = 0
        fn = None
        vn = None

        iterator = tqdm(range(self.local_epoch))
        for epoch_iter in iterator:
            epoch_loss = 0.0
            epoch_samples = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                # L_reg consistency loss (FDSE line 198-211)
                if self.current_round > 1 and self.lmbd > 0.0 and len(feature_maps) == len(layers):
                    loss_mean = 0.0
                    loss_var = 0.0
                    for g, w, f, v, ln in zip(global_means, weights, feature_maps, global_vars, layers):
                        mf = f.mean(dim=0) if f.ndim > 1 else f
                        vf = f.var(dim=0) if f.ndim > 1 else torch.zeros_like(f)
                        fn = (1.0 - ln.momentum) * fn + ln.momentum * mf if fn is not None else mf
                        vn = (1.0 - ln.momentum) * vn + ln.momentum * vf if vn is not None else vf
                        loss_mean = loss_mean + w * ((g.pow(2) - fn.pow(2)) / (2 * vn + 1e-8)).mean()
                        loss_var = loss_var + w * 0.5 * ((torch.log((vn + 1e-8) / (v + 1e-8)) + v / (vn + 1e-8)).mean())
                    loss = loss + self.lmbd * (loss_mean + loss_var)
                loss.backward()
                optimizer.step()
                feature_maps.clear()
                if fn is not None: fn = fn.detach()
                if vn is not None: vn = vn.detach()
                bs = labels.size(0)
                epoch_loss += loss.item() * bs
                epoch_samples += bs
                iterator.desc = f"Local Pariticipant {index} loss = {loss.item():.3f}"
            if epoch_samples > 0:
                global_loss += epoch_loss
                global_samples += epoch_samples
        for lh in lhooks:
            lh.remove()
        return round(global_loss / max(global_samples, 1), 3), epoch_samples
