import sys
sys.path.append("..")


from src.utils import *
from src.backbone import *

import torch
import torch.nn as nn

import math
import numpy as np


class MMUnet(nn.Module):
    """Multi-Modal-Unet"""
    def __init__(self, input_nc, output_nc=5, ngf=32):
        super(MMUnet, self).__init__()
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
        self.bridge = ConvBlock2d(self.out_dim * 60, self.out_dim * 16) #

        # ~~~ Decoding Path ~~~~~~ #

        self.upLayer1 = UpBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer4 = UpBlock2d(self.out_dim * 2, self.out_dim * 1)

    def forward(self, input):
        # ~~~~~~ Encoding path ~~~~~~~  #
        i0 = input[:, 0:1, :, :]   # bz * 1  * height * width
        i1 = input[:, 1:2, :, :]
        i2 = input[:, 2:3, :, :]
        i3 = input[:, 3:4, :, :]

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)  # bz * outdim * height * width
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

        bridge = self.bridge(inputBridge)       # bz * 512 * 15 * 15

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

        return x


class LSTM0(nn.Module):
    def __init__(self, in_c=5, ngf=32):
        super(LSTM0, self).__init__()
        self.conv_gx_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)

    def forward(self, xt):
        """
        :param xt:      bz * 5(num_class) * 240 * 240
        :return:
            hide_1:    bz * ngf(32) * 240 * 240
            cell_1:    bz * ngf(32) * 240 * 240
        """
        gx = self.conv_gx_lstm0(xt)
        ix = self.conv_ix_lstm0(xt)
        ox = self.conv_ox_lstm0(xt)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell_1 = torch.tanh(gx * ix)
        hide_1 = ox * cell_1
        return cell_1, hide_1


class LSTM(nn.Module):
    def __init__(self, in_c=5, ngf=32):
        super(LSTM, self).__init__()
        self.conv_ix_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

    def forward(self, xt, cell_t_1, hide_t_1):
        """
        :param xt:          bz * (5+32) * 240 * 240
        :param hide_t_1:    bz * ngf(32) * 240 * 240
        :param cell_t_1:    bz * ngf(32) * 240 * 240
        :return:
        """
        gx = self.conv_gx_lstm(xt)         # output: bz * ngf(32) * 240 * 240
        gh = self.conv_gh_lstm(hide_t_1)   # output: bz * ngf(32) * 240 * 240
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)          # output: bz * ngf(32) * 240 * 240
        oh = self.conv_oh_lstm(hide_t_1)    # output: bz * ngf(32) * 240 * 240
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        ih = self.conv_ih_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        fh = self.conv_fh_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        f_sum = fx + fh
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt        # bz * ngf(32) * 240 * 240
        hide_t = ot * torch.tanh(cell_t)            # bz * ngf(32) * 240 * 240

        return cell_t, hide_t


class LSTM_MMUnet(nn.Module):
    def __init__(self, input_nc=1, output_nc=5, ngf=32, temporal=3):
        super(LSTM_MMUnet, self).__init__()
        self.temporal = temporal
        self.mmunet = MMUnet(input_nc, output_nc, ngf)
        self.lstm0 = LSTM0(in_c=output_nc , ngf=ngf)
        self.lstm = LSTM(in_c=output_nc , ngf=ngf)

        self.mmout = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        :param x:  5D tensor    bz * temporal * 4 * 240 * 240
        :return:
        """
        output = []
        mm_output = []
        cell = None
        hide = None
        for t in range(self.temporal):
            im_t = x[:, t, :, :, :]                # bz * 4 * 240 * 240
            mm_last = self.mmunet(im_t)              # bz * 32 * 240 * 240
            out_t = self.mmout(mm_last)              # bz * 5 * 240 * 240
            mm_output.append(out_t)
            lstm_in = torch.cat((out_t, mm_last), dim=1) # bz * 37 * 240 * 240

            if t == 0:
                cell, hide = self.lstm0(lstm_in)   # bz * ngf(32) * 240 * 240
            else:
                cell, hide = self.lstm(lstm_in, cell, hide)

            out_t = self.out(hide)
            output.append(out_t)

        return torch.stack(mm_output, dim=1), torch.stack(output, dim=1)


if __name__ == "__main__":
    batch_size = 2
    num_classes = 5
    ngf = 32

    net = LSTM_MMUnet(1, num_classes, ngf=ngf, temporal=3)
    print("total parameter:" + str(netSize(net)))   # 2860,3315
    MRI = torch.randn(batch_size, 3, 4, 64, 64)    # bz * temporal * modal * W * H

    mmout, predict = net(MRI)
    print(mmout.shape)
    print(predict.shape)  # (2, 3, 5, 64, 64)


