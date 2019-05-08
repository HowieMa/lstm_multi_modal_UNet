# coding:utf-8

import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.misc

from src.utils import *

modals = ['flair', 't1', 't1c', 't2']


class Brats15DataLoader(Dataset):
    def __init__(self, data_dir, temporal=3, conf='../config/train15.conf', train=True):
        self.temporal = temporal
        self.is_train = train
        img_lists = []
        train_config = open(conf).readlines()
        for data in train_config:
            img_lists.append(os.path.join(data_dir, data.strip('\n')))

        print('~' * 50)
        print('******** Loading data from disk ********')
        self.data = []
        self.data_raw = []
        self.freq = np.zeros(5)  # for label0,1,2,3,4
        count = 0
        for subject in img_lists:
            count += 1
            if count % 10 == 0:
                print('loading subject %d' %count)

            volume, label = Brats15DataLoader.get_subject(subject)   # 4 * 155 * 240 * 240
            volume = norm_vol(volume)
            self.freq += Brats15DataLoader.get_freq(label)
            # ********** change data type from numpy to torch.Tensor **********
            volume = torch.from_numpy(volume).float()
            label = torch.from_numpy(label).long()

            self.data_raw.append([volume, label, subject])

        self.freq = self.freq / np.sum(self.freq)
        self.weight = np.median(self.freq) / self.freq

        print('********  Finish loading data  ********')
        print('********  Temporal length is  %d ********' % self.temporal)
        print('********  Weight for all classes  ********')
        print(self.weight)
        self.sample_random()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        [images_vol, labels_vol, names] = self.data[index]  # get whole data for one subject
        return images_vol, labels_vol, names

    def sample_random(self):
        self.data = []
        print("************** Resample data ***************************************")
        for (volume, label, subject) in self.data_raw:
            if self.is_train is True:  # for training data, randomly select volume with length of temporal
                images_vol, labels_vol, names = self.get_slices(volume, label, subject)
                for i in range(len(images_vol)):
                    self.data.append([images_vol[i], labels_vol[i], names[i]])
            else:
                volume = np.transpose(volume, (1, 0, 2, 3))
                self.data.append([volume, label, subject])
        if self.is_train is True:
            print('********  Total number of 3D volume is ' + str(len(self.data)))
        else:
            print('********  Total number of Subject is ' + str(len(self.data)))
        print('~' * 80)

    def get_slices(self, volume, label, subject):
        """
        For training, get volume randomly; For testing, get volume step by step
        :param volume:     4D Float Tensor         4 * 155 * 240 * 240
        :param label:   3D Float Tensor             155 * 240 * 240
        :param subject: string
        :return:
        images_vol     Float Tensor List     [temporal * 4 * 240 * 240] * times
        labels_vol     Float Tensor List     [temporal * 240 * 240] * times
        names_vol       list [[str1, str2, str_tmp], [str4, str5, str_tmp] ...   ]
        """
        images_vol = []
        labels_vol = []
        names_vol = []
        img_starts = []

        if self.is_train is True:  # for training, random select
            self.zero_vol = torch.zeros((4, 240, 240))  #
            no_zero_start = 0
            no_zero_end = 0
            for i in range(volume.shape[1]):           # 155
                if not (volume[:, i, :, :] == self.zero_vol).all():
                    no_zero_start = i
                    break
            for i in range(no_zero_start, volume.shape[1]):
                if (volume[:, i, :, :] == self.zero_vol).all():
                    no_zero_end = i
                    break
            times = int((no_zero_end - no_zero_start) / self.temporal)

            rand_start = no_zero_start + np.random.randint(-self.temporal/2 , self.temporal/2 + 1)
            for t in range(times):
                img_starts.append(rand_start + t * self.temporal)

        else:   # for test data
            times = volume.shape[1] / self.temporal  # 155 / 3 = 50
            for t in range(times):
                start = t * self.temporal
                img_starts.append(start)

        for start in img_starts:
            tmp_im = []
            tmp_lbl = []
            tmp_name = []  # list [str1, str2, str_tmp]
            for i in range(start, start + self.temporal):
                tmp_im.append(volume[:, i, :, :])       # 4 * 240 * 240
                tmp_lbl.append(label[i, :, :])          # 240 * 240
                tmp_name.append(subject + '=slice' + str(i))    #
            tmp_im = torch.stack(tmp_im)    # temporal * 4 * 240 * 240
            tmp_lbl = torch.stack(tmp_lbl)  # temporal * 240 * 240

            images_vol.append(tmp_im)
            labels_vol.append(tmp_lbl)
            names_vol.append(tmp_name)

        return images_vol, labels_vol, names_vol

    @staticmethod
    def get_subject(subject):
        """
        :param subject: absolute dir
        :return:
        volume  4D numpy    4 * 155 * 240 * 240
        label   4D numpy    155 * 240 * 240
        """
        # **************** get file ****************
        files = os.listdir(subject)  # [XXX.Flair, XXX.T1, XXX.T1c, XXX.T2, XXX.OT]
        multi_mode_dir = []
        label_dir = ""
        for f in files:
            if f == '.DS_Store':
                continue
            if 'Flair' in f or 'T1' in f or 'T2' in f:    # if is data
                multi_mode_dir.append(f)
            elif 'OT.' in f:        # if is label
                label_dir = f

        # ********** load 4 mode images **********
        multi_mode_imgs = []  # list size :4      item size: 155 * 240 * 240
        for mod_dir in multi_mode_dir:
            path = os.path.join(subject, mod_dir)  # absolute directory
            img = load_mha_as_array(path + '/' + mod_dir + '.mha')
            multi_mode_imgs.append(img)

        # ********** get label **********
        label_dir = os.path.join(subject, label_dir) + '/' + label_dir + '.mha'
        label = load_mha_as_array(label_dir)  #

        volume = np.asarray(multi_mode_imgs)
        return volume, label

    @staticmethod
    def get_freq(label):
        """
        :param label: numpy 155 * 240 * 240     val: 0,1,2,3,4
        :return:
        """
        class_count = np.zeros((5))
        for i in range(5):
            a = (label == i) + 0
            class_count[i] = np.sum(a)
        return class_count


# test case
if __name__ == "__main__":
    data_dir = '../data_sample/'
    conf = '../config/sample15.conf'
    temporal = 3
    brats15 = Brats15DataLoader(data_dir=data_dir, temporal=temporal, conf=conf,train=True)
    image3d, label3d, im_name = brats15[20]
    print('image size ......')
    print(image3d.shape)             # (3, 4, 240, 240)

    print('label size ......')
    print(label3d.shape)             # (3, 240, 240)
    print(im_name)

    for t in range(temporal):
        name = im_name[t].split('/')[-1]
        save_one_image_label(image3d[t, ...], label3d[t, ...], 'img5_seq/img_label_%s.jpg'%name)

    brats15 = Brats15DataLoader(data_dir=data_dir, temporal=temporal, conf=conf, train=False)
    image3d, label3d, im_name = brats15[0]
    print('image size ......')
    print(image3d.shape)             # (155, 4, 240, 240)
    print('label size ......')
    print(label3d.shape)             # (155, 240, 240)


