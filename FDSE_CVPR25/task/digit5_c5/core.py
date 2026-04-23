"""
Digits-5 task 的 core.py — TaskGenerator / TaskPipe / TaskCalculator.

关键和 domainnet_c6 / office_caltech10 不同的地方:
  - config.py 的 train_data 是 **list of single Dataset** (不是 list of ConcatDataset)
    每个元素已经只是 train 数据 (已经去掉 test).
  - config.py 的 test_data 是 **list of single Dataset** (每 client 独立 test, subsample 到 1860).

所以 load_data 时:
  - train = train_data[cid] (直接用)
  - test  = test_data[cid] (直接用, 不从 train split)
  - val   = 可选从 train 里 split (train_holdout), 或 None (省事)

这个结构和 domainnet_c6 的 "train_data = [ConcatDataset([tr, te])...]" 不同, domainnet 是把
train+test concat 再在 load_data 里 split 出来; 我们是**直接用 config 里 per-client 的 test**.
好处: test 永远用固定的 1860 per domain subset, 对齐 FedBN paper 标准评估.
"""

import os
import torch.utils.data

try:
    import ujson as json
except ImportError:
    import json

from .config import train_data        # list of 5 Dataset, 每 client 一个 train
try:
    from .config import test_data     # list of 5 Dataset, 每 client 一个 test (1860 subsample)
except ImportError:
    test_data = None
try:
    from .config import val_data      # 预期是 None (val 走 train_holdout 生成)
except ImportError:
    val_data = None

from .config import data_to_device, eval, compute_loss

import flgo.benchmark.base as fbb


class TaskGenerator(fbb.FromDatasetGenerator):
    """Generator 完全 follow domainnet_c6 / office_caltech10 的写法, 只传 data."""

    def __init__(self):
        super().__init__(
            benchmark=os.path.split(os.path.dirname(__file__))[-1],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )


class TaskPipe(fbb.FromDatasetPipe):
    """TaskPipe 负责把 config 的 train_data / test_data 分发给各 client.

    这里重写了 load_data 以适配我们的 "train/test 分开 list" 结构:
      - domainnet_c6 的 load_data 假设 train_data[i] 是 ConcatDataset([train, test])
        内部用 cdata.datasets 解包. 我们的 train_data[i] 是单 Dataset, 没 .datasets 属性.
      - 我们直接从 self.train_data[cid] / self.test_data[cid] 取, 每 client 独立 train/test.
    """

    def __init__(self, task_path):
        super().__init__(
            task_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def split_dataset(self, dataset, p=0.0):
        """从 dataset 中按比例 p 切出 holdout, 返回 (keep, holdout).

        p=0 直接返回 (dataset, None); 否则 random split.
        """
        if dataset is None:
            return None, None
        if p == 0:
            return dataset, None
        n_hold = int(len(dataset) * p)
        n_keep = len(dataset) - n_hold
        if n_hold == 0:
            return dataset, None
        elif n_keep == 0:
            return None, dataset
        return torch.utils.data.random_split(dataset, [n_keep, n_hold])

    def save_task(self, generator):
        """把 task 元数据写到 data.json. flgo 框架要求的标准流程."""
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names}
        for cid in range(len(client_names)):
            feddata[client_names[cid]] = {'data': generator.local_datas[cid]}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as f:
            json.dump(feddata, f)

    def load_data(self, running_time_option: dict) -> dict:
        """加载并分发数据给 server 和各 client.

        Returns:
            {
              'server': {'test': server_test, 'val': server_val},
              'Client{cid}': {'train': ..., 'val': ..., 'test': ...}
            }
        Server test/val: 合并所有 client 的 test_data (这里每 client 已独立 test,
                         合并后是 5 domain × 1860 = 9300 个样本)
        Client train: 来自 self.train_data[cid], 按 train_holdout 比例 split 出 val
        Client test:  来自 self.test_data[cid], 直接使用 (严格 FedBN 评估协议)
        """
        train_data_list = self.train_data    # list of 5 Dataset
        test_data_list = self.test_data       # list of 5 Dataset
        val_data_val = self.val_data          # 预期 None

        # --- Server 端数据 ---
        # Server 通常不直接训练/eval, 但 flgo 要求 server 能跑 global test.
        # 这里把所有 client 的 test 合并作为 server_test, 相当于 "global average test".
        if test_data_list is not None:
            server_test = torch.utils.data.ConcatDataset(test_data_list)
        else:
            server_test = None
        if val_data_val is not None:
            # 如果 config 有 val_data (我们这里是 None), 也合并
            if isinstance(val_data_val, list):
                server_val = torch.utils.data.ConcatDataset(val_data_val)
            else:
                server_val = val_data_val
        elif server_test is not None:
            # 没 val 时, 从 server_test 切 test_holdout 比例做 val (保持和 flgo 其他 task 行为一致)
            server_test, server_val = self.split_dataset(
                server_test, running_time_option.get('test_holdout', 0.0)
            )
        else:
            server_val = None

        task_data = {'server': {'test': server_test, 'val': server_val}}

        # --- 每 client 分发 train/val/test ---
        n_clients = len(train_data_list)
        for cid in range(n_clients):
            cdata_train_full = train_data_list[cid]
            cdata_test = test_data_list[cid] if test_data_list is not None else None

            # 从 train 切出 val (holdout 用于训练过程中的周期 eval)
            cdata_train, cdata_val = self.split_dataset(
                cdata_train_full,
                running_time_option.get('train_holdout', 0.0)
            )

            task_data[f'Client{cid}'] = {
                'train': cdata_train,
                'val': cdata_val,
                'test': cdata_test,
            }
        return task_data


class TaskCalculator(fbb.BasicTaskCalculator):
    """TaskCalculator 把 data 喂给 model, 调用 config 里的 loss/eval 函数.

    所有方法和 domainnet_c6 的 TaskCalculator 完全一致 (只是去掉了 MyDataloader / collate_fn
    这些可选 hook, 因为 config.py 里没定义自定义 loader).
    """

    def __init__(self, device, optimizer_name='sgd'):
        super().__init__(device, optimizer_name)
        self.DataLoader = torch.utils.data.DataLoader

    def to_device(self, data, *args, **kwargs):
        return data_to_device(data, self.device)

    def get_dataloader(self, dataset, batch_size=64, *args, **kwargs):
        return self.DataLoader(dataset, batch_size=batch_size, **kwargs)

    @torch.no_grad()
    def test(self, model, data, batch_size=64, num_workers=0, pin_memory=False, **kwargs):
        data_loader = self.get_dataloader(
            data, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        return eval(model, data_loader, self.device)

    def compute_loss(self, model, data, *args, **kwargs):
        return compute_loss(model, data, self.device)
