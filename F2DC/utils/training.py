import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import CsvWriter
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from backbone.ResNet import resnet10, resnet12
import time

# 测试函数，model是当前联邦学习的算法对象 比如说f2dc, testdl 是一个传进来的list的dataloarder，每个domain一个
def global_evaluate(
    model: FederatedModel, test_dl: DataLoader, setting: str, name: str
) -> Tuple[list, list]:
    # 每个domain的准确率
    accs = []
    # 全局模型，server端上保存的，联邦学习有两种一种是client.model.nets_lists
    net = model.global_net
    status = net.training
    # 全局模型设置为评估模式
    net.eval()
    # 这里测试的是全局模型在每个domain上的表现
    for j, dl in enumerate(test_dl):
        # total总样本数，top1,预测第一名出线的准确率，top5 真实标签出现在前5个多准确率
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                # 将数据集加载到device上
                images, labels = images.to(model.device), labels.to(model.device)
                # 如果模型是f2dc，f2dc_pg，f2dc_pgv33, f2dc_pg_ml,则需要传 is_eval=True 走 deterministic gumbel
                if model.NAME in ("f2dc", "f2dc_pg", "f2dc_pgv33", "f2dc_pg_ml", "f2dc_dse", "f2dc_pg_lab"):
                    # patch (2026-04-29): 必须传 is_eval=True 走 deterministic gumbel
                    # (gumbel_sigmoid.py:17 在 is_eval=False 时随机采样, 同 model 同 input
                    #  两次 eval 输出不一致, max_abs_diff ~0.0004, 引入 acc 噪声)
                    outputs, _, _, _, _, _, _ = net(images, is_eval=True)
                else:
                    # 普通模型通常只返回logits
                    outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        accs.append(top1acc)
    net.train(status)
    return accs

# 计算原型，硬编码为10 
def get_prototypes(features, labels):
    centers = []
    for i in range(10):
        idx = labels == i
        class_feat = features[idx]
        # 计算每个类的平均特征
        center = np.mean(class_feat, axis=0)
        centers.append(center)
    # 转换为numpy数组
    centers = np.array(centers)
    return centers


def get_features(net, dataloader, device):
    # 设置为评估模式
    net.eval()
    # 特征和标签列表
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            # 将数据集加载到device上
            x, y = x.to(device), y.to(device)
            # 获取特征
            feat = net.features(x)  # 512
            features.append(feat.cpu())
            labels.append(y.cpu())
    # 将特征和标签拼接起来
    features = torch.cat(features, dim=0).numpy()
    # 将标签拼接起来
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels

# 主要是获取全局模型的特征和标签
def extract_features(model, dataloader):
    # 获取全局模型
    net = model.global_net
    # 设置为评估模式
    net.eval()
    # 特征和标签列表
    features, labels = [], []
    # 遍历数据集
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(model.device), y.to(model.device)
            feat = net.features(x)  # 512
            features.append(feat.cpu())
            labels.append(y.cpu())
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels


def train(
    model: FederatedModel, private_dataset: FederatedDataset, args: Namespace
) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)
    model.N_CLASS = private_dataset.N_CLASS
    # 同步 args.num_classes (默认 7 是 PACS 值, Office/Digits 是 10)
    # 之前 main_run.py 注册 --num_classes 默认 7, 但 dataset 知道真值, 这里覆盖一下
    # 让所有依赖 args.num_classes 的诊断/proto buffer 都拿对值
    args.num_classes = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)
    # 本实验都是固定客户端进行分配
    if args.rand_dataset:
        max_num = 10
        is_ok = False
        while not is_ok:
            if model.args.dataset == "fl_officecaltech":
                domains_list = ["caltech", "amazon", "webcam", "dslr"]
                selected_domain_list = np.random.choice(
                    domains_list,
                    size=args.parti_num - domains_len,
                    replace=True,
                    p=None,
                )
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == "fl_digits":
                domains_list = ["mnist", "usps", "svhn", "syn"]
                selected_domain_list = np.random.choice(
                    domains_list, size=args.parti_num, replace=True, p=None
                )
            elif model.args.dataset == "fl_pacs":
                domains_list = ["photo", "art", "cartoon", "sketch"]
                selected_domain_list = np.random.choice(
                    domains_list, size=args.parti_num, replace=True, p=None
                )

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        # patch: fixed allocation per dataset per F2DC paper Sec 5.1
        # PACS: photo:2, art:3, cartoon:2, sketch:3 (10 client)
        # Office: caltech:3, amazon:2, webcam:2, dslr:3 (10 client)
        # Digits: mnist:3, usps:6, svhn:6, syn:5 (20 client)
        if model.args.dataset == "fl_pacs":
            selected_domain_dict = {"photo": 2, "art": 3, "cartoon": 2, "sketch": 3}
        elif model.args.dataset == "fl_officecaltech":
            selected_domain_dict = {"caltech": 3, "amazon": 2, "webcam": 2, "dslr": 3}
        elif model.args.dataset == "fl_digits":
            selected_domain_dict = {"mnist": 3, "usps": 6, "svhn": 6, "syn": 5}
        else:
            raise ValueError(f"Unknown dataset {model.args.dataset}")
        selected_domain_list = []
        for k, n in selected_domain_dict.items():
            selected_domain_list.extend([k] * n)
        # 不 permute (保持论文确定性 fixed allocation, seed 只影响后续 partition / model init)
        selected_domain_list = np.array(selected_domain_list)

        result = Counter(selected_domain_list)

    # print(result)
    print(f"selected_domain_list for {args.parti_num} clients as:")
    print(selected_domain_list)
    # 创建每个clinent的dataloader
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(
        selected_domain_list
    )
    # 保存到dataloader里面
    model.trainloaders = pri_train_loaders
    model.testlodaers = test_loaders

    # 暴露给 diagnostic hook (per-client domain id, for cold path 按 domain 分组)
    model._selected_domain_list = list(selected_domain_list)

    # 创建一个初始化模型 把所有的client 网络初始化为统一的权重 
    if hasattr(model, "ini"):
        model.ini()

    # === diagnostic hook init ===
    from utils.diagnostic import init_diag_state, dump_round_metadata, \
        should_dump_heavy_best, dump_heavy_snapshot, dump_meta
    diag_state = init_diag_state(args)

    accs_dict = {}
    mean_accs_list = []
    all_l2_dis = []
    best_mean_acc = 0.0

    if model.args.dataset == "fl_officecaltech":
        all_dataset_names = ["caltech", "amazon", "webcam", "dslr"]
    elif model.args.dataset == "fl_digits":
        all_dataset_names = ["mnist", "usps", "svhn", "syn"]
    elif model.args.dataset == "fl_pacs":
        all_dataset_names = ["photo", "art", "cartoon", "sketch"]

    Epoch = args.communication_epoch
    all_epoch_loss = []

    for epoch_index in range(Epoch):
        # 先记录一下当前的epoch
        model.epoch_index = epoch_index

        start_time = time.time()
        if hasattr(model, "loc_update"):
            # 进行一次本地的训练
            epoch_loss = model.loc_update(pri_train_loaders)
            all_epoch_loss.append(epoch_loss)
        end_time = time.time()
        print(
            "The " + str(epoch_index) + " Communcation Time:",
            round(end_time - start_time, 3),
        )

        # all_dis = 0.0
        # 测试model.global_net,
        accs = global_evaluate(
            model, test_loaders, private_dataset.SETTING, private_dataset.NAME
        )
        # LAB hook: 把 per-domain test acc 喂给 LabState (供 ROI/waste 诊断)
        # 仅 f2dc_pg_lab 实现了 lab_record_test_acc; 其它 model 没这个 method 静默跳过.
        if hasattr(model, "lab_record_test_acc"):
            try:
                model.lab_record_test_acc(epoch_index + 1, accs, all_dataset_names)
            except Exception as _lab_err:
                print(f"[LAB acc hook ERR] {_lab_err}")
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        print(
            "The " + str(epoch_index) + " Communcation Accuracy:",
            str(mean_acc),
            "Method:",
            model.args.model,
            "epoch loss:",
            str(epoch_loss),
        )
        print(accs)
        print()
        # 每一轮先dump一下结果
        # === diagnostic hook: light dump per round + heavy on best/final ===
        try:
            dump_round_metadata(model, epoch_index + 1, accs, all_dataset_names, args)
            # 保存best数据
            if mean_acc > best_mean_acc + 1e-6:
                if should_dump_heavy_best(epoch_index + 1, mean_acc, args, diag_state):
                    dump_heavy_snapshot(model, test_loaders,
                                        f"best_R{epoch_index+1:03d}",
                                        epoch_index + 1, mean_acc, args, diag_state)
                best_mean_acc = mean_acc
            # 保存最后一轮的模型数据
            if epoch_index == Epoch - 1:
                dump_heavy_snapshot(model, test_loaders,
                                    f"final_R{epoch_index+1:03d}",
                                    epoch_index + 1, mean_acc, args, diag_state)
        except Exception as _diag_err:
            print(f"[diag hook ERR] {_diag_err}")

        if args.save:
            if args.save_name == "No":
                pth_name = args.model
            else:
                pth_name = args.save_name

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)

    # === diagnostic hook: 写 meta.json ===
    try:
        dump_meta(args, diag_state, total_rounds=Epoch, final_acc=mean_accs_list[-1])
    except Exception as _meta_err:
        print(f"[diag meta ERR] {_meta_err}")

    print("ALL Loss: ", all_epoch_loss)
    return mean_accs_list
