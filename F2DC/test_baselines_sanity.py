"""
Sanity test for fedbn / fedprox / fedproto:
- import 不报错
- ini() + 1 batch _train_net 能跑通 (forward + backward + optimizer.step)
- aggregate_nets / proto_aggregation 能跑
- BN 不被聚合 (FedBN 专项)
- prox term > 0 (FedProx 专项)
- prototype dict 非空 (FedProto 专项)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/datasets")
sys.path.append(os.getcwd() + "/backbone")
sys.path.append(os.getcwd() + "/models")

import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler


def make_args(model_name, mu=0.01):
    args = argparse.Namespace(
        device_id=0, communication_epoch=2, local_epoch=1, parti_num=4, seed=42,
        rand_dataset=False, model=model_name, structure='heterogeneity',
        dataset='fl_pacs', pri_aug='weak', online_ratio=1.0, learning_decay=False,
        averaing='weight', save=False, save_name='test',
        gum_tau=0.1, tem=0.06, agg_a=1.0, agg_b=0.4, lambda1=0.8, lambda2=1.0,
        mu=mu, temperature=0.5, local_lr=0.01, local_batch_size=8,
        lmbd=0.01, fdse_tau=0.5, fdse_beta=0.1,
        pg_proto_weight=0.3, pg_attn_temperature=0.3, pg_warmup_rounds=30,
        pg_ramp_rounds=20, pg_server_ema_beta=0.8, num_classes=7, ma_select='resnet',
        csv_log=False,
    )
    return args


def make_dummy_loader(n=16, n_class=7, img_hw=64):
    x = torch.randn(n, 3, img_hw, img_hw)
    y = torch.randint(0, n_class, (n,))
    ds = TensorDataset(x, y)
    sampler = SubsetRandomSampler(np.arange(n))  # 用 np.array, 真实 loader 也是 np
    return DataLoader(ds, batch_size=8, sampler=sampler)


def make_resnet10(n_class=7):
    from backbone.ResNet import resnet10
    return resnet10(n_class)


def test_model(model_name, n_class=7):
    print(f"\n========== TESTING {model_name} ==========")
    args = make_args(model_name)
    nets = [make_resnet10(n_class) for _ in range(args.parti_num)]

    if model_name == 'fedbn':
        from models.fedbn import FedBN as Cls
    elif model_name == 'fedprox':
        from models.fedprox import FedProx as Cls
    elif model_name == 'fedproto':
        from models.fedproto import FedProto as Cls
    else:
        raise ValueError(model_name)

    model = Cls(nets, args, transform=None)
    model.trainloaders = [make_dummy_loader(n=16, n_class=n_class) for _ in range(args.parti_num)]
    model.ini()
    print(f"  [OK] {model_name} ini() — global_net created, weights synced")

    # 模拟 loc_update 关键步骤 (跳过 random_state 选 client, 直接全 online)
    model.online_clients = list(range(args.parti_num))
    model.num_samples = []

    # ============ FedBN: 训练前后, 比较 BN 跟非 BN params ============
    if model_name == 'fedbn':
        # 记录训练前 client0 跟 global 的 BN running_mean (相同)
        sd0_before = {k: v.clone() for k, v in nets[0].state_dict().items()}
        for i in model.online_clients:
            cl, ns = model._train_net(i, nets[i], model.trainloaders[i])
            model.num_samples.append(ns)
        print(f"  [OK] {model_name} _train_net ran on {len(model.online_clients)} clients")

        # 收集 client0 训练后的 BN running stat (应该已经 drift)
        sd0_after_train = {k: v.clone() for k, v in nets[0].state_dict().items()}
        bn_keys = [k for k in sd0_after_train if model._is_bn_key(k)]
        nbn_keys = [k for k in sd0_after_train if not model._is_bn_key(k)]
        print(f"  BN keys ({len(bn_keys)}): e.g., {bn_keys[:3]}")
        print(f"  Non-BN keys ({len(nbn_keys)}): e.g., {nbn_keys[:2]}")
        assert len(bn_keys) > 0, "FedBN: 没有检测到 BN keys, _is_bn_key 可能写错"

        # 检查训练后 client BN stat 是否变了 (drift = 训练有效)
        bn_drift = 0
        for k in bn_keys:
            if not torch.equal(sd0_before[k], sd0_after_train[k]):
                bn_drift += 1
        print(f"  BN drift after train: {bn_drift}/{len(bn_keys)} keys changed (expected > 0)")

        # 聚合
        model.aggregate_nets_skip_bn()
        sd0_after_agg = nets[0].state_dict()

        # 关键 assertion 1: 聚合后 CLIENT BN 留本地 (跟 train 后一致)
        bn_preserved = 0
        for k in bn_keys:
            if 'num_batches_tracked' in k:
                continue
            if torch.equal(sd0_after_train[k], sd0_after_agg[k]):
                bn_preserved += 1
        print(f"  CLIENT BN preserved after agg: {bn_preserved}/{len([k for k in bn_keys if 'num_batches_tracked' not in k])} (expected = ALL)")
        assert bn_preserved == len([k for k in bn_keys if 'num_batches_tracked' not in k]), \
            "FedBN BUG: client BN 在聚合后被改了 (应该 skip)"

        # 关键 assertion 2: GLOBAL_NET 的 BN 应该是 client mean (供 eval 用), 不是 init 0
        gw = model.global_net.state_dict()
        bn1_global = gw['bn1.running_mean']
        bn1_clients_mean = sum(nets[i].state_dict()['bn1.running_mean'] for i in range(args.parti_num)) / args.parti_num
        global_bn_eq_mean = torch.allclose(bn1_global, bn1_clients_mean, atol=1e-5)
        print(f"  GLOBAL_NET BN ≈ mean of client BN: {global_bn_eq_mean} (expected True, 供 eval)")
        assert global_bn_eq_mean, "FedBN BUG: global_net BN 不是 client mean, eval 时会用未归一化 BN → acc 乱猜"

        # 非 BN 参数应该被聚合 (跟训练后不同, 因为是 mean of all clients)
        nbn_changed = 0
        for k in nbn_keys:
            if not torch.equal(sd0_after_train[k], sd0_after_agg[k]):
                nbn_changed += 1
        print(f"  Non-BN changed after agg: {nbn_changed}/{len(nbn_keys)} (expected > 0, 因为 FedAvg 聚合了)")
        assert nbn_changed > 0, "FedBN BUG: 非 BN 参数没被聚合"

    # ============ FedProx: prox term > 0 ============
    elif model_name == 'fedprox':
        # 直接调一次 _train_net 看不报错
        for i in model.online_clients:
            cl, ns = model._train_net(i, nets[i], model.trainloaders[i])
            model.num_samples.append(ns)
        print(f"  [OK] {model_name} _train_net ran on {len(model.online_clients)} clients")

        # Manual: 验证 prox term 不为 0
        net = nets[0]
        gw = [p.detach().clone() for p in model.global_net.parameters()]
        # 让 client weights 跟 global drift 一点 (用一个 SGD step)
        x = torch.randn(4, 3, 64, 64).to(model.device)
        y = torch.randint(0, n_class, (4,)).to(model.device)
        opt = optim.SGD(net.parameters(), lr=0.1)
        loss = nn.CrossEntropyLoss()(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        prox = sum(((p - g) ** 2).sum() for p, g in zip(net.parameters(), gw))
        print(f"  prox term after 1 SGD step = {prox.item():.6f} (expected > 0)")
        assert prox.item() > 0, "FedProx BUG: client 跟 global 已 drift, prox 不该是 0"

        model.aggregate_nets(None)
        print(f"  [OK] aggregate_nets")

    # ============ FedProto: 训完 prototype dict 非空, 聚合后 global_protos 非空 ============
    elif model_name == 'fedproto':
        for i in model.online_clients:
            cl, ns = model._train_net(i, nets[i], model.trainloaders[i])
            model.num_samples.append(ns)
        print(f"  [OK] {model_name} _train_net ran on {len(model.online_clients)} clients")

        # 检查 local_protos 已被填充
        for i in model.online_clients:
            assert i in model.local_protos, f"client {i} 没有 local proto"
            assert len(model.local_protos[i]) > 0, f"client {i} prototype dict 空"
        print(f"  local_protos: {len(model.local_protos)} clients, "
              f"first client has {len(list(model.local_protos.values())[0])} class protos")

        # 聚合
        model.global_protos = model.proto_aggregation(model.local_protos)
        assert len(model.global_protos) > 0, "FedProto BUG: global proto 聚合后空"
        proto_dim = list(model.global_protos.values())[0].shape
        print(f"  global_protos: {len(model.global_protos)} classes, dim={proto_dim}")

        # 第二轮 _train_net (现在 global_protos 非空, 触发 MSE loss path)
        for i in model.online_clients:
            cl, ns = model._train_net(i, nets[i], model.trainloaders[i])
        print(f"  [OK] 第二轮 train (with global_protos) 跑通")

        model.aggregate_nets(None)
        print(f"  [OK] aggregate_nets")

    print(f"  ✅ {model_name} sanity PASSED")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    for m in ['fedbn', 'fedprox', 'fedproto']:
        test_model(m, n_class=7)
    print("\n========================================")
    print("ALL 3 BASELINES SANITY PASSED ✅")
