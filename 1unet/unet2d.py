import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn

from src.backbone import *
from src.utils import *


class UNet2D(nn.Module):
    def __init__(self, in_ch=4, out_ch=2, degree=64):
        super(UNet2D, self).__init__()

        chs = []
        for i in range(5):
            chs.append((2 ** i) * degree)

        self.downLayer1 = ConvBlock2d(in_ch, chs[0])
        self.downLayer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[0], chs[1]))

        self.downLayer3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[1], chs[2]))

        self.downLayer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[2], chs[3]))

        self.bottomLayer = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        ConvBlock2d(chs[3], chs[4]))

        self.upLayer1 = UpBlock2d(chs[4], chs[3])
        self.upLayer2 = UpBlock2d(chs[3], chs[2])
        self.upLayer3 = UpBlock2d(chs[2], chs[1])
        self.upLayer4 = UpBlock2d(chs[1], chs[0])

        self.outLayer = nn.Conv2d(chs[0], out_ch, kernel_size=3, stride=1, padding=1)

        # # Params initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        """
        :param x:   4D Tensor    BatchSize * 4(modal) * W * H
        :return:    4D Tensor    BatchSize * 2        * W * H
        """
        x1 = self.downLayer1(x)     # degree(32)   * 16    * W    * H
        x2 = self.downLayer2(x1)    # degree(64)   * 16/2  * W/2  * H/2
        x3 = self.downLayer3(x2)    # degree(128)  * 16/4  * W/4  * H/4
        x4 = self.downLayer4(x3)    # degree(256)  * 16/8  * W/8  * H/8

        x5 = self.bottomLayer(x4)   # degree(512)  * 16/16 * W/16 * H/16

        x = self.upLayer1(x5, x4)   # degree(256)  * 16/8 * W/8 * H/8
        x = self.upLayer2(x, x3)    # degree(128)  * 16/4 * W/4 * H/4
        x = self.upLayer3(x, x2)    # degree(64)   * 16/2 * W/2 * H/2
        x = self.upLayer4(x, x1)    # degree(32)   * 16   * W   * H
        x = self.outLayer(x)        # out_ch(2 )   * 16   * W   * H
        return x


if __name__ == "__main__":
    net = UNet2D(4, 5, degree=64)
    print("total parameter:" + str(netSize(net)))  # 3453,0437
    # torch.save(net.state_dict(), 'model.pth')

    # degree = 128                      1,3806,6818

    batch_size = 4
    a = torch.randn(batch_size, 4, 192, 192)
    b = net(a)
    print(b.shape)

