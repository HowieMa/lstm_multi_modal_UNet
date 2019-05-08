from unet2d import UNet2D
from data_loader.data_brats15 import Brats15DataLoader
from src.utils import *

from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ********** Hyper Parameter **********
data_dir = '/home/haoyum/download/BRATS2015_Training'
conf_test = '../config/test15.conf'
save_dir = 'test_res/'
saved_model_path = 'ckpt/best_epoch.pth'
batch_size = 32


# multi-GPU
cuda_available = torch.cuda.is_available()
device_ids = [0, 1, 2, 3]  # multi-GPU
torch.cuda.set_device(device_ids[0])


def to_var(tensor):
    return Variable(tensor.cuda() if cuda_available else tensor)


def evaluation(net, test_dataset, criterion, save_dir=None):
    """
    :param net:
    :param test_dataset:  data loader batch size = 1
    :param criterion:
    :param temporal:
    :return:
    """
    test_loss = []
    iou_5_class_all = []
    dice_whole_tumor = []

    with torch.no_grad():
        net.eval()
        for step, (images_vol, label_vol, subject) in enumerate(test_dataset):
            # images_vol     5D tensor     (bz, 155, 4, 240, 240)
            # label_vol      4D tensor     (bz, 155, 240, 240)
            subj_target = label_vol.long().squeeze()  # 3D tensor  155 * 240 * 240
            subj_predict = torch.zeros(label_vol.squeeze().shape)  # 3D tensor  155 * 240 * 240
            for t in range(155):  #
                image = to_var(images_vol[:, t, ...])   # 4D  bz(1) * 4 * 240 * 240
                label = to_var(label_vol[:, t, ...])    # 4D tensor   bz(1)  * 240 * 240
                predicts = net(image)  # 4D tensor   bz(1) * 5 * 240 * 240

                loss_valid = criterion(predicts, label.long())
                test_loss.append(float(loss_valid))

                # softmax and reverse
                predicts = one_hot_reverse(predicts)  # 3D long T     bz(1)* 240 * 240 (0-4)
                subj_predict[t, ...] = predicts.squeeze().long().data

            # calculate IoU
            subj_5class_iou = cal_subject_iou_5class(subj_target, subj_predict)  # list length 4
            subj_whole_tumor_dice = cal_subject_dice_whole_tumor(subj_target, subj_predict)  # label(1+2+3+4)

            iou_5_class_all.append(subj_5class_iou)
            dice_whole_tumor.append(subj_whole_tumor_dice)

            # ******************** save image **************************
            if save_dir is not None:
                hl, name = subject[0].split('/')[-2:]
                img_save_dir = save_dir + hl + '/' + name + '.nii.gz'
                save_array_as_mha(subj_predict, img_save_dir)

            # print('subject ...' + subject[0])
            # print(subj_5class_iou)
            # print(subj_whole_tumor_dice)

        print('Dice for whole tumor is ')
        average_iou_whole_tumor = sum(dice_whole_tumor) / (len(dice_whole_tumor) * 1.0)
        print(str(average_iou_whole_tumor))

        for i in range(5):
            iou_i = []
            for iou5 in iou_5_class_all:
                iou_i.append(iou5[i])
            average_iou_label_i = sum(iou_i) / (len(iou_i) * 1.0)
            print('Iou for label ' + str(i) + '   is    ' + str(average_iou_label_i))

    return average_iou_whole_tumor, test_loss


def load_model():
    net = UNet2D(in_ch=4, out_ch=5, degree=64)
    if cuda_available:
        net = net.cuda()
        net = nn.DataParallel(net, device_ids=device_ids)

    state_dict = torch.load(saved_model_path)
    net.load_state_dict(state_dict)
    return net


if __name__ == "__main__":

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir + 'HGG'):
        os.mkdir(save_dir + 'HGG')
    if not os.path.exists(save_dir + 'LGG'):
        os.mkdir(save_dir + 'LGG')

    net = load_model()

    print('test data ....')
    test_data = Brats15DataLoader(data_dir=data_dir, conf=conf_test, train=False)  # 30 subject, 4650 images
    test_dataset = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    weight = torch.from_numpy(test_data.weight).float()  # weight for all class
    weight = to_var(weight)
    criterion = nn.CrossEntropyLoss(weight=weight)

    evaluation(net, test_dataset, criterion, save_dir=save_dir)


