import torch
import torch.nn as nn
from torchinfo import summary


class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__()
        #Branch1
        self.branch1 = nn.Sequential(
            nn.LazyConv2d(c1, kernel_size=1),
            nn.ReLU()
        )
        #Branch2
        self.branch2 = nn.Sequential(
            nn.LazyConv2d(c2[0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.LazyConv2d(c3[0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1),
            nn.LazyConv2d(c4, kernel_size=1)             
        )

    def forward(self, X):
        return torch.cat(self.branch1(X),
                         self.branch2(X),
                         self.branch3(X),
                         self.branch4(X),
                        dim = 1)


class GoogleNet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b3(self):
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b4(self):
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b5(self):
        return nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
            nn.LazyLinear(n_classes)
        )


if __name__=="__main__":
    model = GoogleNet(n_classes=10)
    summary(model, input_size=(1, 3, 32, 32))

