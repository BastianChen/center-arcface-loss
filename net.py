import torch.nn as nn
from centerloss import CenterLoss
from arcfaceloss import ArcFaceLoss


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(11 * 11 * 64, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 3),
            nn.PReLU()
        )
        self.centerloss = CenterLoss(10, 3)
        self.arcfaceloss = ArcFaceLoss(3, 10)
        self.nllloss = nn.NLLLoss()
        self.crossentropyloss = nn.CrossEntropyLoss()

    def forward(self, data):
        y = self.conv(data)
        y = y.reshape(y.size(0), -1)
        features = self.linear(y)
        outputs = self.arcfaceloss(features)
        return features, outputs

    def getLoss(self, features, outputs, labels):
        # nllLoss = self.nllloss(outputs, labels)
        centerLoss = self.centerloss(features, labels)
        crossentropyloss = self.crossentropyloss(outputs, labels)
        # return nllLoss + 0.01 * centerLoss
        # return nllLoss
        return crossentropyloss + 0.01 * centerLoss
