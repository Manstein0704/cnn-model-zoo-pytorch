import torch
import torch.nn as nn
from torchinfo import summary

####################
#VGG Scratch
####################

def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)



class VGG(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        arch = ((2, 64), (2, 128), (2, 256), (3, 512), (3, 512))
        vgg_blks = []
        for num_convs, out_channels in arch:
            vgg_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *vgg_blks,
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )

    def forward(self, X):
        return self.net(X)


if __name__=="__main__":
    model = VGG(num_classes=10)
    summary(model, input_size=(1, 3, 32, 32))

