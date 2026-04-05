import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        # 设置模型为训练模式
        self.model.train()
        
        start_time = time.time()
        # 记录训练开始时间

        max_local_epochs = self.local_epochs
        # 如果客户端训练慢，则随机选择1到max_local_epochs // 2之间的本地轮数
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # 训练
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    # 如果x是列表，则把x[0]转换为当前设备
                    x[0] = x[0].to(self.device)
                else:
                    # 如果x不是列表，则把x转换为当前设备
                    x = x.to(self.device)
                # 把y转换为当前设备
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                # 计算输出
                output = self.model(x)
                # 计算损失
                loss = self.loss(output, y)
                # 清空梯度
                self.optimizer.zero_grad()
                # 计算梯度
                loss.backward()
                # 更新参数
                self.optimizer.step()

        # self.model.cpu()

        # 如果使用学习率衰减，则更新学习率
        if self.learning_rate_decay:
            # 更新学习率
            self.learning_rate_scheduler.step()

        # 记录训练次数
        self.train_time_cost['num_rounds'] += 1
        # 记录训练时间
        self.train_time_cost['total_cost'] += time.time() - start_time
