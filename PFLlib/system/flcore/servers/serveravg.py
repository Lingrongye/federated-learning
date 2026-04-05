import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    # 训练主循环
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # 选择客户端
            self.selected_clients = self.select_clients()
            # 发送模型到客户端
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                # 评估全局模型
                self.evaluate()

            for client in self.selected_clients:
                # 客户端训练
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            # 接收客户端模型
            self.receive_models()
            # 评估DLG
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            # 聚合参数
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            # 打印时间成本
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        # 打印平均时间成本
        print("\nAverage time cost per round.")
        # 打印平均时间成本
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # 保存结果
        self.save_results()
        # 保存全局模型
        self.save_global_model()

        if self.num_new_clients > 0:
            # 评估新客户端
            self.eval_new_clients = True
            # 设置新客户端
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
