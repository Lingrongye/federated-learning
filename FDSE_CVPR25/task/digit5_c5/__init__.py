# Task package entry point. flgo 通过这个文件 discover task 的 default_model 和
# partitioner, 必须显式 import + re-export, 否则 flgo.init() 会报
# "Model cannot be None when there exists no default model".
#
# 格式 follow FDSE_CVPR25/task/PACS_c4/__init__.py / task/domainnet_c6/__init__.py

from .model import default_model
import flgo.benchmark.partition

default_model = default_model
# IIDPartitioner + num_clients=100 是占位, 实际 client 数由 config.py 的 train_data list
# 长度决定 (5 client = 5 domain); flgo 的 FromDatasetGenerator / FromDatasetPipe 会
# 优先使用 config.py 里的数据, 忽略这个 default_partitioner. 写在这里只是满足 flgo
# 对 task module attribute 的 discover 要求.
default_partitioner = flgo.benchmark.partition.IIDPartitioner
default_partition_para = {'num_clients': 5}
