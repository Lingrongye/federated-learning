import os
import sys
import numpy as np
import copy
import flgo.algorithm.fedavg as fedavg
from collections import OrderedDict
import cvxopt
import torch
import torch.nn as nn
import flgo.utils.fmodule as fuf
import math


class DSEConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=0, use_relu=True, bias=True, use_dse_activate=False, use_dse_bn=True, shortcut=True):
        super(DSEConv, self).__init__()
        self.oup = oup
        self.use_relu = use_relu
        self.use_dse_activate = use_dse_activate
        self.use_dse_bn = use_dse_bn
        self.ratio = ratio
        self.shortcut = shortcut
        if shortcut:
            init_channels = math.ceil(oup / ratio) if ratio>1 else oup
            new_channels = init_channels * (ratio - 1) if ratio>1 else oup
        else:
            init_channels = math.ceil(oup / ratio) if ratio>1 else oup
            new_channels = oup
        self.dfe_conv = nn.Conv2d(inp, init_channels, kernel_size, stride, padding, bias=True)
        self.dse_bn = nn.BatchNorm2d(init_channels)
        self.dfe_bias = nn.Parameter(torch.zeros(oup)) if bias else None
        self.dse_conv = nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False)
        self.dfe_bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x1 = self.dfe_conv(x)
        if self.use_dse_bn: x1 = self.dse_bn(x1)
        if self.use_dse_activate: x1 = self.leakyrelu(x1)
        if self.shortcut:
            x2 = self.dse_conv(x1)
            x = torch.cat([x1, x2], dim=1) if self.ratio>1 else x2+x1
        else:
            x = self.dse_conv(x1)
        if self.dfe_bias is not None:
            x = x[:, :self.oup, :, :] + self.dfe_bias.expand(x.shape[0], self.dfe_bias.shape[0]).reshape(x.shape[0], self.dfe_bias.shape[0], 1, 1)
        x = self.dfe_bn(x)
        if self.use_relu: x = self.relu(x)
        return x

class DSELinear(nn.Module):
    def __init__(self, inp, oup, ratio=2, dw_size=1, use_relu=True, use_dse_activate=False, use_dse_bn=True, shortcut=True):
        super().__init__()
        # init_channels = math.ceil(oup / ratio) if ratio>1 else oup
        # new_channels = oup
        self.ratio = ratio
        self.shortcut = shortcut
        if shortcut:
            init_channels = math.ceil(oup / ratio) if ratio>1 else oup
            new_channels = init_channels * (ratio - 1) if ratio>1 else oup
        else:
            init_channels = math.ceil(oup / ratio) if ratio>1 else oup
            new_channels = oup
        self.use_relu = use_relu
        self.use_dse_activate = use_dse_activate
        self.use_dse_bn = use_dse_bn
        # self.Vshare = nn.Linear(inp, init_channels)
        self.dfe_conv = nn.Conv2d(inp, init_channels, 1, bias=True)
        self.dse_bn = nn.BatchNorm2d(init_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dse_conv = nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False)
        self.dfe_bias = nn.Parameter(torch.zeros(oup))
        self.dfe_bn = nn.BatchNorm1d(oup)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        # self.Up = nn.Linear(init_channels, new_channels)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x1 = self.dfe_conv(x)
        if self.use_dse_bn:x1 = self.dse_bn(x1)
        if self.use_dse_activate: x1 = self.leakyrelu(x1)
        if self.shortcut:
            x2 = self.dse_conv(x1)
            x = torch.cat([x1, x2], dim=1) if self.ratio>1 else x2+x1
        else:
            x = self.dse_conv(x1)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dfe_bn(x + self.dfe_bias)
        if self.use_relu: x = self.relu(x)
        return x

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        # initialize user embedding for all clients
        self.init_algo_para({'lmbd': 0.01, 'tau':0.5,'beta':0.1,})
        self.model = self.model.__class__()
        shared_names = ['dfe','head',]
        local_names = ['dse_bn.running_']
        self.shared_weight_keys = [k for k in self.model.state_dict() if any([(s in k) for s in shared_names])]
        self.local_keys = [k for k in self.model.state_dict() if
                           any([(s in k) for s in local_names]) and k not in self.shared_weight_keys]
        self.personalized_weight_keys = [k for k in self.model.state_dict() if
                                         k not in self.shared_weight_keys and k not in self.local_keys]
        per_state = {k: v for k, v in self.model.state_dict().items() if k in self.personalized_weight_keys}
        self.client_states = [copy.deepcopy(per_state) for _ in self.clients]
        client_weights = np.array([len(c.train_data) for c in self.clients])
        self.client_weights = torch.from_numpy(client_weights / client_weights.sum()).unsqueeze(-1).to(self.device)

    def pack(self, client_id, mtype=0, *args, **kwargs):
        client_dict = {k: v for k, v in self.model.state_dict().items() if k in self.shared_weight_keys}
        client_dict.update(self.client_states[client_id])
        return {'md': copy.deepcopy(client_dict)}

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        # self.model = self.aggregate(models)
        mdicts = [m.state_dict() for m in models]
        # agg shared dict
        current_shard_dict = {k: v for k, v in self.model.state_dict().items() if k in self.shared_weight_keys}
        shared_dicts = [{k: v for k, v in md.items() if k in self.shared_weight_keys} for md in mdicts]
        new_shared_dicts = {}
        for k in self.shared_weight_keys:
            if "running_" in k or 'num_batches_tracked' in k:
                k_vecs = [md[k] for md in shared_dicts]
                if 'num_batches_tracked' in k:
                    new_shared_dicts[k] = torch.stack(k_vecs).sum(dim=0).squeeze(0)
                else:
                    # new_shared_dicts[k] = (torch.stack(k_vecs)*self.client_weights).sum(dim=0).squeeze(0)
                    new_shared_dicts[k] = torch.stack(k_vecs).mean(dim=0).squeeze(0)
            else:
                shape = shared_dicts[0][k].shape
                crt_vec_k = current_shard_dict[k].reshape(-1).to(self.device)
                k_vecs = [md[k].reshape(-1) - crt_vec_k for md in shared_dicts]
                k_norms = [t.norm() for t in k_vecs]
                k_vecs = [t / (tn + 1e-8) for t, tn in zip(k_vecs, k_norms)]
                old_stdout = sys.stdout  # backup current stdout
                sys.stdout = open(os.devnull, "w")
                op_lambda_k = self.optim_lambda(k_vecs)
                sys.stdout = old_stdout
                op_lambda_k = torch.tensor([ele[0] for ele in op_lambda_k]).float().to(self.device)
                new_vec_k = (op_lambda_k.unsqueeze(0) @ torch.stack(k_vecs))[0]
                new_shared_dicts[k] = (torch.stack(k_norms).mean() * (new_vec_k) + crt_vec_k).reshape(shape)
        # aggregate grads
        self.model.load_state_dict(new_shared_dicts, strict=False)

        # agg personalized dict
        per_dicts = [{k: v for k, v in md.items() if k in self.personalized_weight_keys} for md in mdicts]
        agg_dicts = [{} for _ in mdicts]
        for k in self.personalized_weight_keys:
            if 'num_batches_tracked' in k: continue
            per_dict_k = [md[k].reshape(-1) for md in per_dicts]
            per_dict_k = [p / p.norm() for p in per_dict_k]
            all_vecs = torch.stack(per_dict_k)
            sims_k = all_vecs @ all_vecs.T
            per_dict_k = [{k: md[k]} for md in per_dicts]
            for cid, sim_cid, agg_dict in zip(self.selected_clients, sims_k, agg_dicts):
                weight_cid = torch.nn.Softmax()(sim_cid/self.tau)
                self.client_states[cid].update({k: fuf._modeldict_weighted_average(per_dict_k, weight_cid)[k]})
        return

    def optim_lambda(self, grads):
        # create H_m*m = 2J'J where J=[grad_i]_n*m
        n = len(grads)
        Jt = []
        for gi in grads:
            Jt.append((copy.deepcopy(gi).cpu()).numpy())
        Jt = np.array(Jt)
        # target function
        P = 2 * np.dot(Jt, Jt.T)

        q = np.array([[0] for i in range(n)])
        # equality constraint λ∈Δ
        A = np.ones(n).T
        b = np.array([1])
        # boundary
        lb = np.array([0. for i in range(n)])
        ub = np.array([1. for i in range(n)])
        G = np.zeros((2 * n, n))
        for i in range(n):
            G[i][i] = -1
            G[n + i][i] = 1
        h = np.zeros((2 * n, 1))
        for i in range(n):
            h[i] = -lb[i]
            h[n + i] = ub[i]
        res = self.quadprog(P, q, G, h, A, b)
        return res

    def quadprog(self, P, q, G, h, A, b):
        """
        Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
        Output: Numpy array of the solution
        """
        P = cvxopt.matrix(P.tolist())
        q = cvxopt.matrix(q.tolist(), tc='d')
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist(), tc='d')
        sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
        return np.array(sol['x'])

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.model = copy.deepcopy(self.server.model)
        self.bnlayers = list(dict.fromkeys(
            ["".join([f'[{m}]' if m.isdigit() else f'.{m}' for m in k.split('.')[:-1]]) for k in
             self.model.state_dict().keys() if 'dfe_bn' in k]))

    def unpack(self, received_pkg):
        self.model.load_state_dict(received_pkg['md'], strict=False)
        return self.model

    def pack(self, model, *args, **kwargs):
        return {
            "model": copy.deepcopy(model.to(self.server.device)),
        }
    #
    @fuf.with_multi_gpus
    def train(self, model, *args, **kwargs):
        model.train()
        eps = model.head.weight.shape[0]
        layers = []
        for ln in self.bnlayers:
            name = 'model' + ln
            l = eval(name)
            layers.append(l)
        # layers = [eval('model'+ln) for ln in self.bnlayers]
        global_means = copy.deepcopy([l.running_mean.to(self.device).detach() for l in layers])
        global_vars = copy.deepcopy([l.running_var.to(self.device).detach() for l in layers])
        weights = np.exp(np.array([self.beta*i for i in range(len(layers))]))
        weights/=weights.sum()
        feature_maps = []
        def hook(model, input, output):
            feature_maps.append(
                input[0].mean(dim=0).mean(dim=-1).mean(dim=-1) if len(input[0].shape) > 3 else input[0].mean(dim=0))
        lhooks = []
        for l in layers:
            lh = l.register_forward_hook(hook)
            lhooks.append(lh)
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        fn = None
        vn = None
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss_reg = 0.
            if self.server.current_round > 1 and self.lmbd>0.:
                loss_mean = 0.
                loss_var = 0.
                for g, w, f, v, ln in zip(global_means, weights, feature_maps, global_vars, layers):
                    mf = f.mean(dim=0)
                    vf = f.var(dim=0)
                    fn = (1. - ln.momentum) * fn + ln.momentum * mf if fn is not None else mf
                    vn = (1.- ln.momentum) * vn + ln.momentum * vf if vn is not None else vf
                    loss_mean += w*((g.pow(2) - fn.pow(2))/(2*vn)).mean()
                    loss_var += w*0.5*((torch.log(vn/(v+1e-8))+v/(vn+1e-8)).mean())
                loss_reg += (loss_mean + loss_var)
                loss += self.lmbd*loss_reg
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
            feature_maps.clear()
            if fn is not None: fn = fn.detach()
            if vn is not None: vn = vn.detach()
        for lh in lhooks:
            lh.remove()
        return


class AlexNetEncoder(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self):
        super(AlexNetEncoder, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', DSEConv(3, 64, kernel_size=11, stride=4, padding=2)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv2', DSEConv(64, 192, kernel_size=5, padding=2)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv3', DSEConv(192, 384, kernel_size=3)),
                ('conv4', DSEConv(384, 256, kernel_size=3)),
                ('conv5', DSEConv(256, 256, kernel_size=3)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = DSELinear(256 * 6 * 6, 1024)
        self.fc2 = DSELinear(1024, 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class DomainnetModel(fuf.FModule):
    def __init__(self):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.head = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

class PACSModel(fuf.FModule):
    def __init__(self):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.head = nn.Linear(1024, 7)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

class OfficeModel(fuf.FModule):
    def __init__(self):
        super().__init__()
        self.encoder = AlexNetEncoder()
        self.head = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


# ============================================================
# Digits-5 专用 backbone (FedBN 标准 5-layer CNN 的 DSE 版)
# ============================================================
# 为什么需要: FDSE 原 AlexNetEncoder 是 224×224 输入的 5 conv AlexNet, 不适合 digits
# 的 32×32 小图像. DigitModelEncoderDSE 是 FedBN paper DigitModel 的 DSE 版本,
# 每个 Conv/FC 都替换成 DSEConv/DSELinear, 保持 FDSE 的"每层做 DFE+DSE 分解"核心思想.
#
# 结构 (输入 3×32×32, 输出 512 维 feature, 和 DigitModel 一致去掉最后 fc3 分类头):
#   conv1  DSEConv(3, 64)   + MaxPool -> 64×16×16
#   conv2  DSEConv(64, 64)  + MaxPool -> 64×8×8
#   conv3  DSEConv(64, 128)           -> 128×8×8
#   flatten -> 8192
#   fc1   DSELinear(8192, 2048)
#   fc2   DSELinear(2048, 512)
# 注意: DSEConv 内部已经带 BN (dse_bn + dfe_bn), 所以不需要额外在这里加 BN.
class DigitModelEncoderDSE(nn.Module):
    """FedBN 标准 DigitModel 的 DSE 版 encoder (for digit5_c5 task).

    和原 DigitModel 的对应:
      原 DigitModel 每个 conv/fc 后面跟 nn.BatchNorm + ReLU,
      DSE 版把 Conv/Linear 换成 DSEConv/DSELinear (里面已包含 BN + ReLU 激活),
      所以不需要在外面再包 BN.
    """

    def __init__(self):
        super(DigitModelEncoderDSE, self).__init__()
        # 三个 conv 块, kernel 5×5 padding=2 (和原 DigitModel 一致, 保持 32→16→8 感受野)
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', DSEConv(3, 64, kernel_size=5, stride=1, padding=2)),
                ('maxpool1', nn.MaxPool2d(kernel_size=2)),       # 32 -> 16
                ('conv2', DSEConv(64, 64, kernel_size=5, stride=1, padding=2)),
                ('maxpool2', nn.MaxPool2d(kernel_size=2)),       # 16 -> 8
                ('conv3', DSEConv(64, 128, kernel_size=5, stride=1, padding=2)),
                # conv3 后无 pool, 保持 8×8
            ])
        )
        # 两个 fc 块
        self.fc1 = DSELinear(128 * 8 * 8, 2048)   # 8192 -> 2048
        self.fc2 = DSELinear(2048, 512)            # 2048 -> 512

    def forward(self, x):
        x = self.features(x)                       # (B, 128, 8, 8)
        x = torch.flatten(x, 1)                    # (B, 8192)
        x = self.fc1(x)                            # (B, 2048)
        x = self.fc2(x)                            # (B, 512)
        return x                                   # 输出 feature, 后接 head 做分类


class Digit5ModelDSE(fuf.FModule):
    """Digits-5 task 的完整 FDSE model (encoder + 线性分类 head)."""

    def __init__(self):
        super().__init__()
        self.encoder = DigitModelEncoderDSE()
        self.head = nn.Linear(512, 10)              # 512 feature -> 10 数字类

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


model_map = {
    'PACS': PACSModel,
    'office': OfficeModel,
    'domainnet': DomainnetModel,
    'digit5': Digit5ModelDSE,   # 新增: task='digit5_c5' → 走 DigitModel 的 DSE 版
}

data_map = {

}


def init_dataset(object):
    pass


def init_local_module(object):
    pass


def init_global_module(object):
    dataset = os.path.split(object.option['task'])[-1].split('_')[0]
    if 'Server' in object.__class__.__name__:
        object.model = model_map[dataset]().to(object.device)
