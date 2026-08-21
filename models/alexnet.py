import torch
import torch.nn as nn
from torchinfo import summary



####################
#AlexNet Scratch
####################

#※Adjust the first convolutional layer's kernel size and stride for CIFAR-10.

class AlexNet(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )

    def forward(self, X):
        return self.net(X)

    
if __name__=="__main__":
    model = AlexNet(num_classes=10)
    summary(model, input_size=(1, 3, 32, 32))

