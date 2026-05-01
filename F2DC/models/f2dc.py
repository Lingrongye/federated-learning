import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated Learning F2DC')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

# 用于dfd的第二个loss
def get_pred(out, labels): 
    # 获取预测的top-1和top-2标签
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    # 如果top-1标签与真实标签相同，则使用top-2标签作为错误标签，否则使用top-1标签作为错误标签
    wrong_high_label = torch.where(pred == labels, second_pred, pred)
    return wrong_high_label


class F2DC(FederatedModel):
    NAME = 'f2dc'
    COMPATIBILITY = ['heterogeneity']

    def __init__(self, nets_list, args, transform):
        super(F2DC, self).__init__(nets_list, args, transform)
        self.args = args
        self.tem = self.args.tem

    def ini(self):
        # 用第0个client 的模型初始化为global model,把同一份权重复制给所有的client 
        self.global_net = copy.deepcopy(self.nets_list[0])
        # 然后把第0个模型的权重取出，并且循环复制给所有的client model
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):

        total_clients = list(range(self.args.parti_num))
        # 实际上online client都是所有的客户端
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        # 计算本轮的累积loss,每个client 的样本数 
        self.online_clients = online_clients
        all_clients_loss = 0.0
        self.num_samples = []
        # 对本轮的每一个client都进行一次本地训练 
        for i in online_clients:
            c_loss, c_samples = self._train_net(i, self.nets_list[i], priloader_list[i])
            all_clients_loss += c_loss

            self.num_samples.append(c_samples)
        # 调用聚合函数 把本轮训练后的client modes 聚合成一个globalnet 再同步回去
        self.aggregate_nets(None)
        # 计算平均的损失 
        all_c_avg_loss = all_clients_loss / len(online_clients)
        return round(all_c_avg_loss, 3)


    def _train_net(self, index, net, train_loader):
        # patch: skip clients with empty dataloader (happens when rand_dataset
        # over-allocates same domain to multiple clients; sum percent > 100%)
        try:
            n_avail = len(train_loader.sampler.indices) if hasattr(train_loader.sampler, "indices") else len(train_loader.dataset)
        except Exception:
            n_avail = 1
        if n_avail == 0:
            print(f"[skip] client {index} has empty dataloader")
            return 0.0, 0

        # 把clientmodel 放到gpu上面
        net = net.to(self.device)
        # 整体进行训练
        net.train()
        # sgd优化器，交叉shang损失
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        num_c_samples = 0
        # local epoch 一共是10
        iterator = tqdm(range(self.local_epoch))
        global_loss = 0.0
        global_samples = 0

        for iter in iterator:
            epoch_loss = 0.0
            epoch_samples = 0
            # 对每一个epoch的每一个batch
            for batch_idx, (images, labels) in enumerate(train_loader):

                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                # 经过f2dc后会返回这么多不同的输出，假设k是类别数
                # out          最终主分类输出，输出（b,k)就是分类的logits
                # feat         最终分类前的 feature (b,512)
                # ro_outputs   robust feature 的辅助分类输出(b,k)
                # re_outputs   non-robust feature 的辅助分类输出(b,k)
                # rec_outputs  reconstructed feature 的辅助分类输出
                # ro_flatten   robust feature 池化后的向量(b,512)
                # re_flatten   non-robust feature 池化后的向量

                out, feat, ro_outputs, re_outputs, rec_outputs, ro_flatten, re_flatten = net(images)
                outputs = out

                wrong_high_labels = get_pred(out, labels)

                DFD_dis1_loss = torch.tensor(0.).to(self.device)
                if not len(ro_outputs) == 0:
                    for ro_out in ro_outputs:
                        # 要求dfd输出的ro特征能正确分类
                        DFD_dis1_loss += 1.0 * criterion(ro_out, labels)
                    DFD_dis1_loss /= len(ro_outputs)
                DFD_dis2_loss = torch.tensor(0.).to(self.device)
                if not len(re_outputs) == 0:
                    for re_out in re_outputs:
                        # 要求no_ro feature 去预测一个错误但是搞置信的类别
                        DFD_dis2_loss += 1.0 * criterion(re_out, wrong_high_labels)
                    DFD_dis2_loss /= len(re_outputs)
                # patch: original code has shape mismatch (scalar += [B]) on new PyTorch.
                # Math is unchanged: log(exp(x)) == x, so this is just sum(cos/tem)/B = mean(cos)/tem.
                # 这个loss要求no_ro 跟ro要不相关
                l_cos = torch.cosine_similarity(ro_flatten, re_flatten, dim=1)
                
                DFD_sep_loss = (l_cos / self.tem).mean()
                
                DFD_loss = DFD_dis1_loss + DFD_dis2_loss + DFD_sep_loss

                DFC_loss = torch.tensor(0.).to(self.device)
                if not len(rec_outputs) == 0:
                    for rec_out in rec_outputs:
                        # 要求重建后的feature也能正常分类
                        DFC_loss += 1.0 * criterion(rec_out, labels)
                    DFC_loss /= len(rec_outputs)
                # 两个loss之间的权重
                loss_DC = self.args.lambda1 * DFD_loss + self.args.lambda2 * DFC_loss 
                # 最终的整个输出的分类loss
                loss_CE = criterion(outputs, labels)

                loss = loss_CE + loss_DC
                # 正常进行反向传播
                loss.backward()

                iterator.desc = "Local Pariticipant %d DC = %0.3f, CE = %0.3f" % (index, loss_DC, loss_CE)

                optimizer.step()
                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
            
            avg_epoch_loss = epoch_loss / epoch_samples
            global_loss += epoch_loss
            global_samples += epoch_samples
            num_c_samples = epoch_samples
        # 返回当前的client的本地训练的平均loss以及样本数
        global_avg_loss = global_loss / global_samples

        return round(global_avg_loss, 3), num_c_samples 