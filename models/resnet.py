import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary



class Residual(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(num_channels, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
        )

        if use_1x1conv:
            self.branch2 = nn.Sequential(
                nn.LazyConv2d(num_channels, kernel_size=1, stride=strides),
                nn.LazyBatchNorm2d()
            )
        else:
            self.branch2 = None

    def forward(self, X):
        Y = self.branch1(X)
        if self.branch2:
            X = self.branch2(X)
        output = Y + X

        return F.relu(output)


class ResNet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def block(self, num_residuals, num_channels,first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

    def __init__(self, num_classes):
        super().__init__()
        arch = ((2, 64), (2, 128), (2, 256), (2, 512))
        self.net = nn.Sequential(self.b1())
        for i , b in enumerate(arch):
            self.net.add_module(f"b{i+2}", self.block(*b, first_block=(i==0)))
        self.net.add_module("last", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))

    def forward(self, X):
        return self.net(X)



if __name__=="__main__":
    model = ResNet(num_classes=10)
    summary(model, input_size=(1, 3, 32, 32))


