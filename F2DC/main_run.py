# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import os
import sys
import socket
import torch.multiprocessing
import numpy as np

# multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy("file_system")
import warnings

warnings.filterwarnings("ignore")

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + "/datasets")
sys.path.append(conf_path + "/backbone")
sys.path.append(conf_path + "/models")

from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_prive_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime


def parse_args():
    parser = ArgumentParser(description="F2DC", allow_abbrev=False)

    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument(
        "--communication_epoch",
        type=int,
        default=100,
        help="global communication epoch in FL",
    )
    parser.add_argument(
        "--local_epoch", type=int, default=10, help="local epoch for client"
    )
    parser.add_argument(
        "--parti_num", type=int, default=10, help="number for local clients"
    )

    parser.add_argument("--seed", type=int, default=55, help="random seed")
    parser.add_argument(
        "--rand_dataset", type=lambda x: x.lower()=="true", default=False,
        help="random domain allocation; default False = fixed per F2DC paper Sec 5.1"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="f2dc",
        help="method name",
        choices=get_all_models(),
    )
    parser.add_argument("--structure", type=str, default="heterogeneity")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fl_pacs",
        choices=DATASET_NAMES,
        help="multi-domain dataset for experiments",
    )

    parser.add_argument("--pri_aug", type=str, default="weak", help="data augmentation")
    parser.add_argument(
        "--online_ratio", type=float, default=1, help="ratio for online clients"
    )
    parser.add_argument(
        "--learning_decay", type=bool, default=False, help="learning rate decay"
    )
    parser.add_argument(
        "--averaing", type=str, default="weight", help="averaging strategy"
    )

    parser.add_argument("--save", type=bool, default=True, help="save model params")
    parser.add_argument("--save_name", type=str, default="save_no", help="save name")

    parser.add_argument(
        "--gum_tau", type=float, default=0.1, help="gumbel concrete distribution"
    )
    parser.add_argument("--tem", type=float, default=0.06, help="DFD sep temperature")
    parser.add_argument("--agg_a", type=float, default=1.0, help="domain-aware agg α (paper Eq 11)")
    parser.add_argument("--agg_b", type=float, default=0.4, help="domain-aware agg β (paper Eq 11)")
    parser.add_argument("--use_daa", type=lambda x: x.lower()=="true", default=False,
                        help="enable F2DC paper Domain-Aware Aggregation (Eq 10/11)")
    parser.add_argument("--num_domains_q", type=int, default=4,
                        help="domain 数 Q (PACS=4, Office=4, Digits=4)")
    # ===== Diagnostic hook =====
    parser.add_argument("--dump_diag", type=str, default=None,
                        help="diagnostic dump dir (None=disabled). 启用后每 round dump 元数据 + best/final dump heavy snapshot")
    parser.add_argument("--dump_warmup", type=int, default=30,
                        help="diag: heavy dump warmup rounds (R 之前不 dump heavy)")
    parser.add_argument("--dump_min_gain", type=float, default=1.0,
                        help="diag: heavy dump 最小 acc gain (跟上次 dump 的 best 比)")
    parser.add_argument("--dump_min_interval", type=int, default=5,
                        help="diag: heavy dump 最小 round 间隔")
    parser.add_argument(
        "--lambda1", type=float, default=0.8, help="params for DFD loss"
    )
    parser.add_argument(
        "--lambda2", type=float, default=1.0, help="params for DFC loss"
    )

    # ===== Baseline 共用超参 (FedProx / FedProto / MOON) =====
    parser.add_argument("--mu", type=float, default=1.0,
                        help="FedProx prox weight / FedProto MSE weight / MOON contrastive weight")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="MOON contrastive temperature")
    parser.add_argument("--local_lr", type=float, default=0.01, help="local SGD lr")
    parser.add_argument("--local_batch_size", type=int, default=64, help="local batch size")

    # ===== FDSE (CVPR 2025) 超参 =====
    parser.add_argument("--lmbd", type=float, default=0.01, help="FDSE: L_reg consistency loss weight")
    parser.add_argument("--fdse_tau", type=float, default=0.5, help="FDSE: cosine sim softmax temperature for personalized agg")
    parser.add_argument("--fdse_beta", type=float, default=0.1, help="FDSE: per-layer exp weight for L_reg")

    # ===== PG-DFC v3.2 超参 =====
    parser.add_argument("--pg_proto_weight", type=float, default=0.3,
                        help="PG-DFC: target proto_weight (after warmup+ramp)")
    parser.add_argument("--pg_attn_temperature", type=float, default=0.3,
                        help="PG-DFC: cosine attention temperature")
    parser.add_argument("--pg_warmup_rounds", type=int, default=30,
                        help="PG-DFC: warmup rounds (proto_weight=0)")
    parser.add_argument("--pg_ramp_rounds", type=int, default=20,
                        help="PG-DFC: ramp rounds (proto_weight 0 → target)")
    parser.add_argument("--pg_server_ema_beta", type=float, default=0.8,
                        help="PG-DFC: server EMA β for cross-round prototype smoothing")
    parser.add_argument("--num_classes", type=int, default=7,
                        help="num classes (for prototype dim)")

    # ===== PG-DFC Multi-Layer (f2dc_pg_ml) 超参 =====
    parser.add_argument("--ml_aux_alpha", type=float, default=0.1,
                        help="ML deep sup loss weight (aux3 CE), 0 = disable lite branch")
    parser.add_argument("--ml_lite_channel", type=int, default=32,
                        help="ML DFD/DFC lite 内部 channel 数 (vs PG-DFC 原 64)")
    parser.add_argument("--ml_lite_tau", type=float, default=0.1,
                        help="ML lite gumbel tau (跟 layer4 DFD 一致 0.1 / 旧默认 0.5 mask3 学不动 EXP-141 v2)")

    parser.add_argument("--ma_select", type=str, default="resnet", help="backbone")

    # CPU core intra-op parallelism
    torch.set_num_threads(8)
    add_management_args(parser)
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]

    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main_F2DC(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    # 获取数据集对象
    priv_dataset = get_prive_dataset(args)
    # 获取模型对象，每个client一个模型 默认resnet_dc for f2dc
    backbones_list = priv_dataset.get_backbone(args.parti_num, None, args.model)
    # default f2dc method
    # 创建算法对象
    model = get_model(backbones_list, args, priv_dataset.get_transform())
    # model里面包含了所有的client模型也就是nets_list，server模型是global_net，所有的超参数arg
    args.arch = model.nets_list[0].name
    # 打印日志
    print(
        "{}_{}_{}_{}_{}".format(
            args.model,
            args.parti_num,
            args.dataset,
            args.communication_epoch,
            args.local_epoch,
        )
    )
    setproctitle.setproctitle(
        "{}_{}_{}_{}_{}".format(
            args.model,
            args.parti_num,
            args.dataset,
            args.communication_epoch,
            args.local_epoch,
        )
    )
    # 训练
    domains_acc_list = train(model, priv_dataset, args)

    print(f"Accuracy List {args.model} ({args.dataset}):", domains_acc_list)


if __name__ == "__main__":
    main_F2DC()
