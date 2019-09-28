import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, class_num, features_num):
        super().__init__()
        self.class_num = class_num
        self.center = nn.Parameter(torch.randn(class_num, features_num))

    def forward(self, features, labels):
        # 根据labels个数将center个数扩充成于lables一样多
        center_expand = self.center.index_select(dim=0, index=labels)
        # 获取每种类别的个数
        count = torch.histc(labels, bins=self.class_num, min=0, max=self.class_num - 1)
        # 扩充count个数用于对每个类别的欧氏距离总和做除法,因为为整数如果要做除法给转换成float类型
        count_expand = count.index_select(dim=0, index=labels).float()
        # 求每一个类的特征值与该类中心点做欧氏距离后的平均值，用于梯度下降
        loss = torch.sum(torch.sqrt(torch.sum(torch.pow(features - center_expand, 2), dim=1)) / count_expand)
        return loss


if __name__ == '__main__':
    # labels = torch.tensor([0, 2, 1, 1, 2, 0, 1, 2])
    # center = torch.tensor([[1, 1], [2, 2], [3, 3]])
    # print(center)
    # center_expand = center.index_select(dim=0, index=labels)
    # print(center_expand)
    center = torch.tensor([[1, 1], [2, 2], [3, 3]])
    features = torch.tensor([[2, 5], [4, 4], [1, 1]])
    a = torch.sum(torch.pow(features - center, 2), dim=1)
    print(a)
