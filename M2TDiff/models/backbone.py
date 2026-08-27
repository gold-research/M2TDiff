import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.out_channels = 2048
    def forward(self, x):
        return self.body(x)
