import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, features_num, class_num):
        super().__init__()
        self.W = nn.Parameter(torch.randn(features_num, class_num))

    def forward(self, features, s=1, m=1):
        # size为(3,10)
        w_value = F.normalize(self.W, dim=0)
        # size为(n,3)
        x_value = F.normalize(features, dim=1)
        # 防止反向传播时导致梯度小时的问题所以除以10，另外使梯度保持在一个比较大的范围内，加速模型收敛,cosa值的范围大致在0.04到0.06之间
        cosa = torch.matmul(x_value, w_value) / 10
        # 求出的角度a大致在1.5，1.6附近，相当于做了一个归一化
        a = torch.acos(cosa)
        # s设置成1，相当于|x|*|w|即将特征值压缩在一个单位圆中，m为增加的角度值，乘以10是因为前面除以了10，这里要乘回来
        loss = torch.exp(s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * torch.cos(a) * 10), dim=1, keepdim=True)
                                                       - torch.exp(s * torch.cos(a) * 10) +
                                                       torch.exp(s * torch.cos(a + m) * 10))
        return torch.log(loss)
