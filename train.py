import torch
from torchvision import datasets, transforms
from net import Net
import os
from torch.utils.data import DataLoader
from drawpoints import DrawPoints


class Trainer:
    def __init__(self, net_path, pictures_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_path = net_path
        self.pictures_path = pictures_path
        self.net = Net().to(self.device)
        self.Draw = DrawPoints()
        self.tranforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.train_data = DataLoader(datasets.MNIST("datasets/", train=True, download=False, transform=self.tranforms),
                                     batch_size=2000, shuffle=True)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        if os.path.exists(self.net_path):
            self.net.load_state_dict(torch.load(self.net_path))

    def train(self):
        epoch = 1
        loss_new = 100
        while True:
            for i, (data, label) in enumerate(self.train_data):
                data = data.to(self.device)
                label = label.to(self.device)
                features, outputs = self.net(data)
                loss = self.net.getLoss(features, outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print("第{0}轮第{1}轮loss为{2}".format(epoch, i, loss.item()))
                    self.Draw.draw(features.cpu().detach().numpy(), label.cpu().detach().numpy(), epoch, i,
                                   self.pictures_path, loss.item())
                if loss.item() < loss_new:
                    loss_new = loss.item()
                    torch.save(self.net.state_dict(), self.net_path)
            epoch += 1
