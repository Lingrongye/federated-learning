import torch
from torch import nn


class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super(GumbelSigmoid, self).__init__()
        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

# 输入x,x是当前这个feature_unit为当作是robust的概率
    def forward(self, x, is_eval=False):
        r = 1 - x
        # 加p_value防止变成log(0),把两个log概率变成了log的概率
        x = (x + self.p_value).log()
        r = (r + self.p_value).log()
        # 训练时加入随机噪声
        if not is_eval:
            x_N = torch.rand_like(x)
            r_N = torch.rand_like(r)
        else:
            x_N = 0.5 * torch.ones_like(x)
            r_N = 0.5 * torch.ones_like(r)
        
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x = x + x_N
        # tau是温度参数，tau越大输出更加的平滑，更像是一个软概率，而tau小就是更加的尖锐，更接近0/1
        x = x / (self.tau + self.p_value)
        r = r + r_N
        r = r / (self.tau + self.p_value)
        # 把robust跟no_robust的概率大两个score都拼接起来，然后做softmax,第0个属于robust
        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)
        # x = torch.where(x >= 0.5, torch.ones_like(x), torch.zeros_like(x))
        return x