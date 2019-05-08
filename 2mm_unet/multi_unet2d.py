import sys
sys.path.append("..")


from src.utils import *
from src.backbone import *

import torch
import torch.nn as nn

import math
import numpy as np


def croppCenter(tensorToCrop,finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(2)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(2,dtype=int)
    croppBorders[0] = int(diff[0]/2)
    croppBorders[1] = int(diff[1]/2)

    return tensorToCrop[:, :,
                        croppBorders[0]:croppBorders[0] + finalShape[2],
                        croppBorders[1]:croppBorders[1] + finalShape[3]]


class Multi_Unet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32):
        super(Multi_Unet, self).__init__()
        print('~' * 50)
        print(' ----- Creating MULTI_UNET  ...')
        print('~' * 50)

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1) Flair 1
        self.down_1_0 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_0 = maxpool()
        self.down_2_0 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_0 = maxpool()
        self.down_3_0 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_0 = maxpool()
        self.down_4_0 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2) T1
        self.down_1_1 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_1 = maxpool()
        self.down_2_1 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_1 = maxpool()
        self.down_3_1 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_1 = maxpool()
        self.down_4_1 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_1 = maxpool()

        # Encoder (Modality 3) T1c
        self.down_1_2 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_2 = maxpool()
        self.down_2_2 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_2 = maxpool()
        self.down_3_2 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_2 = maxpool()
        self.down_4_2 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_2 = maxpool()

        # Encoder (Modality 4) T2
        self.down_1_3 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_3 = maxpool()
        self.down_2_3 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_3 = maxpool()
        self.down_3_3 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_3 = maxpool()
        self.down_4_3 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_3 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = ConvBlock2d(self.out_dim * 60, self.out_dim * 16)

        # ~~~ Decoding Path ~~~~~~ #

        self.upLayer1 = UpBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer4 = UpBlock2d(self.out_dim * 2, self.out_dim * 1)

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        # # Params initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, input):

        # ############################# #
        # ~~~~~~ Encoding path ~~~~~~~  #
        i0 = input[:, 0:1, :, :]   # Batch Size * 1 * volume_size * height * width
        i1 = input[:, 1:2, :, :]
        i2 = input[:, 2:3, :, :]
        i3 = input[:, 3:4, :, :]

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)  # Batch Size * outdim * volume_size * height * width
        down_1_1 = self.down_1_1(i1)
        down_1_2 = self.down_1_2(i2)
        down_1_3 = self.down_1_3(i3)

        # -----  Second Level --------
        # Batch Size * (outdim * 4) * (volume_size/2) * (height/2) * (width/2)
        input_2nd_0 = torch.cat((self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3)), dim=1)

        input_2nd_1 = torch.cat((self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0)), dim=1)

        input_2nd_2 = torch.cat((self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1)), dim=1)

        input_2nd_3 = torch.cat((self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2)), dim=1)

        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        down_2_2 = self.down_2_2(input_2nd_2)
        down_2_3 = self.down_2_3(input_2nd_3)

        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_0(down_2_1)
        down_2_2m = self.pool_2_0(down_2_2)
        down_2_3m = self.pool_2_0(down_2_3)

        input_3rd_0 = torch.cat((down_2_0m, down_2_1m, down_2_2m, down_2_3m), dim=1)
        input_3rd_0 = torch.cat((input_3rd_0, croppCenter(input_2nd_0, input_3rd_0.shape)), dim=1)

        input_3rd_1 = torch.cat((down_2_1m, down_2_2m, down_2_3m, down_2_0m), dim=1)
        input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)

        input_3rd_2 = torch.cat((down_2_2m, down_2_3m, down_2_0m, down_2_1m), dim=1)
        input_3rd_2 = torch.cat((input_3rd_2, croppCenter(input_2nd_2, input_3rd_2.shape)), dim=1)

        input_3rd_3 = torch.cat((down_2_3m, down_2_0m, down_2_1m, down_2_2m), dim=1)
        input_3rd_3 = torch.cat((input_3rd_3, croppCenter(input_2nd_3, input_3rd_3.shape)), dim=1)

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        down_3_2 = self.down_3_2(input_3rd_2)
        down_3_3 = self.down_3_3(input_3rd_3)

        # -----  Fourth Level --------
        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)
        down_3_1m = self.pool_3_0(down_3_1)
        down_3_2m = self.pool_3_0(down_3_2)
        down_3_3m = self.pool_3_0(down_3_3)

        input_4th_0 = torch.cat((down_3_0m, down_3_1m, down_3_2m, down_3_3m), dim=1)
        input_4th_0 = torch.cat((input_4th_0, croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)

        input_4th_1 = torch.cat((down_3_1m, down_3_2m, down_3_3m, down_3_0m), dim=1)
        input_4th_1 = torch.cat((input_4th_1, croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)

        input_4th_2 = torch.cat((down_3_2m, down_3_3m, down_3_0m, down_3_1m), dim=1)
        input_4th_2 = torch.cat((input_4th_2, croppCenter(input_3rd_2, input_4th_2.shape)), dim=1)

        input_4th_3 = torch.cat((down_3_3m, down_3_0m, down_3_1m, down_3_2m), dim=1)
        input_4th_3 = torch.cat((input_4th_3, croppCenter(input_3rd_3, input_4th_3.shape)), dim=1)

        down_4_0 = self.down_4_0(input_4th_0)  # 8C
        down_4_1 = self.down_4_1(input_4th_1)
        down_4_2 = self.down_4_2(input_4th_2)
        down_4_3 = self.down_4_3(input_4th_3)

        # ----- Bridge -----
        # Max-pool
        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)
        down_4_2m = self.pool_4_0(down_4_2)
        down_4_3m = self.pool_4_0(down_4_3)

        inputBridge = torch.cat((down_4_0m, down_4_1m, down_4_2m, down_4_3m), dim=1)
        inputBridge = torch.cat((inputBridge, croppCenter(input_4th_0, inputBridge.shape)), dim=1)

        bridge = self.bridge(inputBridge)

        # ############################# #
        # ~~~~~~ Decoding path ~~~~~~~  #
        skip_1 = (down_4_0 + down_4_1 + down_4_2 + down_4_3) / 4.0
        skip_2 = (down_3_0 + down_3_1 + down_3_2 + down_3_3) / 4.0
        skip_3 = (down_2_0 + down_2_1 + down_2_2 + down_2_3) / 4.0
        skip_4 = (down_1_0 + down_1_1 + down_1_2 + down_1_3) / 4.0

        x = self.upLayer1(bridge, skip_1)
        x = self.upLayer2(x, skip_2)
        x = self.upLayer3(x, skip_3)
        x = self.upLayer4(x, skip_4)

        return self.out(x)


if __name__ == "__main__":
    batch_size = 2
    num_classes = 5  # one hot
    initial_kernels = 32

    net = Multi_Unet(1, num_classes, initial_kernels)
    print("total parameter:" + str(netSize(net)))  # 2860,0325
    # torch.save(net.state_dict(), 'model.pth')
    MRI = torch.randn(batch_size, 4, 64, 64)    # Batchsize, modal, hight,

    if torch.cuda.is_available():
        net = net.cuda()
        MRI = MRI.cuda()

    segmentation_prediction = net(MRI)
    print(segmentation_prediction.shape)

