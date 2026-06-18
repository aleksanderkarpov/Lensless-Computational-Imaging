import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


def res_stack(channels):
    return nn.Sequential(*[ResBlock(channels) for _ in range(4)])


class DownDRUNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], (3, 3), 1, 1, bias=False)
        self.res_block1 = res_stack(channels[0])
        self.sconv1 = nn.Conv2d(channels[0], channels[1], (2, 2), 2, 0, bias=False)
        self.res_block2 = res_stack(channels[1])
        self.sconv2 = nn.Conv2d(channels[1], channels[2], (2, 2), 2, 0, bias=False)
        self.res_block3 = res_stack(channels[2])
        self.sconv3 = nn.Conv2d(channels[2], channels[3], (2, 2), 2, 0, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.sconv1(x)
        skip1 = x
        x = self.res_block2(x)
        x = self.sconv2(x)
        skip2 = x
        x = self.res_block3(x)
        x = self.sconv3(x)
        skip3 = x
        return x, [skip1, skip2, skip3]


class UpDRUNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = res_stack(channels[3])
        self.tconv1 = nn.ConvTranspose2d(channels[3], channels[2], (2, 2), 2, 0, bias=False)
        self.res_block1 = res_stack(channels[2])
        self.tconv2 = nn.ConvTranspose2d(channels[2], channels[1], (2, 2), 2, 0, bias=False)
        self.res_block2 = res_stack(channels[1])
        self.tconv3 = nn.ConvTranspose2d(channels[1], channels[0], (2, 2), 2, 0, bias=False)
        self.res_block3 = res_stack(channels[0])
        self.conv1 = nn.Conv2d(channels[0], 3, (3, 3), 1, 1, bias=False)

    def forward(self, x, skips):
        skip1, skip2, skip3 = skips
        x = self.body(x)
        x = x + skip3
        x = self.tconv1(x)
        x = x + skip2
        x = self.res_block1(x)
        x = self.tconv2(x)
        x = x + skip1
        x = self.res_block2(x)
        x = self.tconv3(x)
        x = self.res_block3(x)
        x = self.conv1(x)
        return x
    
class DRUnet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = DownDRUNet(channels)
        self.up = UpDRUNet(channels)

    def forward(self, x):
        x, skips = self.down(x)
        x = self.up(x, skips)
        return x
