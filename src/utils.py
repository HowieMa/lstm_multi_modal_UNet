# coding:utf-8

import SimpleITK as sitk
import numpy as np
import scipy.misc
import torch
import os
import torch.nn.functional as F

"""
***********************  Data loader related   ***********************
"""


def load_mha_as_array(img_name):
    """
    get the numpy array of brain mha image
    :param img_name: absolute directory of 3D mha images
    :return:
        nda  type: numpy    size: 150 * 240 * 240
    """
    img = sitk.ReadImage(img_name)
    nda = sitk.GetArrayFromImage(img)
    return nda


def save_array_as_mha(volume, img_name):
    """
    save the numpy array of brain mha image
    :param img_name: absolute directory of 3D mha images
    """
    out = sitk.GetImageFromArray(volume)
    sitk.WriteImage(out, img_name)


def get_whole_tumor_labels(label):
    """
    whole tumor in patient data is label 1 + 2 + 3 + 4
    :param label:  numpy array      size : 155 * 240 * 240  value 0-4
    :return:
    label 1 * 155 * 240 * 240
    """
    label = (label > 0) + 0  # label 1,2,3,4
    return label


def get_tumor_core_labels(label):
    """
    tumor core in patient data is label 1 + 3 + 4
    :param label:  numpy array      size : 155 * 240 * 240  value 0-4
    :return:
    label 155 * 240 * 240
    """

    label = (label == 1) + (label == 3) + (label == 4) + 0  # label 1,3,4 = 1
    return label


def netSize(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


"""
***********************  Normalization   ***********************
"""


def normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]  # ignore background
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std

    # out_random = np.random.normal(0, 1, size=volume.shape)
    # out[volume == 0] = out_random[volume == 0]
    return out


def norm_vol(data):
    data = data.astype(np.float)
    index = data.nonzero()
    smax = np.max(data[index])
    smin = np.min(data[index])

    if smax - smin == 0:
        return data
    else:
        data[index] = (data[index] - smin * 1.0) / (smax - smin)
        return data


def norm(data):
    data = np.asarray(data)
    smax = np.max(data)
    smin = np.min(data)
    if smax - smin == 0:
        return data
    else:
        return (data - smin) / ((smax - smin) * 1.0)


def norm4(data):
    data = np.asarray(data)
    smax = 4.0
    smin = 0.0
    return (data - smin) / ((smax - smin) * 1.0)


"""
*********************** Save image related in training  ***********************
"""


def save_one_image_label(images, label, save_dir):
    """
    :param images:
    :param label:
    :param save_dir:
    :return:
    """
    output = np.zeros((240, 250 * 5))  # H, W
    for m in range(4):  # for each modal
        output[:, 250 * m: 250 * m + 240] = norm(images[m, :, :])
    output[:, 250 * 4: 250 * 4 + 240] = norm4(label)
    scipy.misc.imsave(save_dir, output)


def save_one_image_label_pre(images, label, predict, save_dir):
    """
    :param images:
    :param label:
    :param predict:
    :param save_dir:
    :return:
    """
    output = np.zeros((240, 250 * 6))  # H, W
    for m in range(4):  # for each modal
        output[:, 250 * m: 250 * m + 240] = norm(images[m, :, :])
    output[:, 250 * 4: 250 * 4 + 240] = norm4(predict)
    output[:, 250 * 5: 250 * 5 + 240] = norm4(label)
    scipy.misc.imsave(save_dir, output)


def save_train_vol_images(images, predicts, labels, index, epoch, save_dir='ckpt/'):
    """
    :param images:      5D float tensor     bz * temporal * 4(modal) * 240 * 240
    :param predicts:    4D long Tensor      bz * temporal * 240 * 240
    :param labels:      4D long Tensor      bz * temporal * 240 * 240
    :return:
    """
    images = np.asarray(images.cpu().data)
    predicts = np.asarray(predicts.cpu().data)
    labels = np.asarray(labels.cpu())

    if not os.path.exists(save_dir + 'epoch' + str(epoch)):
        os.mkdir(save_dir + 'epoch' + str(epoch))
    for b in range(images.shape[0]):  # for each batch
        for t in range(predicts.shape[1]):  # temporal
            name = index[t][b].split('/')[-1]
            save_one_image_label_pre(images[b,t,:,:,:], label=labels[b,t,:,:], predict=predicts[b,t,:,:],
                                 save_dir=save_dir + 'epoch' + str(epoch) + '/b_' +str(b) + name + '.jpg')


def save_train_images(images, predicts, labels, index, epoch, save_dir='ckpt/'):
    """
    :param images:      4D float tensor     bz * 4(modal)  * height * weight
    :param predicts:    3D Long tensor      bz * height * weight
    :param labels:      3D Long tensor      bz * height * weight
    :param index:       list                [str] * bz
    :return:
    """
    images = np.asarray(images.cpu().data)
    predicts = np.asarray(predicts.cpu().data)
    labels = np.asarray(labels.cpu())

    if not os.path.exists(save_dir + 'epoch' + str(epoch)):
        os.mkdir(save_dir + 'epoch' + str(epoch))
    for b in range(images.shape[0]):  # for each batch
        name = index[b].split('/')[-1]
        save_one_image_label_pre(images[b,:,:,:], labels[b,:,:], predicts[b,:,:],
                                 save_dir=save_dir + 'epoch' + str(epoch) + '/b_' +str(b) + name + '.jpg')


"""
*********************** Data transformation  ***********************
"""


def one_hot_reverse(predicts):
    predicts = F.softmax(predicts, dim=1)  # 4D float Tensor  bz * 5 * 240 * 240
    return torch.max(predicts, dim=1)[1]  # 3D long Tensor  bz * 240 * 240 (val 0-4)


def one_hot_reverse3d(predicts):
    predicts = F.softmax(predicts, dim=2)  # 5D float Tensor   bz * temporal * 5 * 240 * 240
    return torch.max(predicts, dim=2)[1]   # 4D long Tensor  bz*temporal* 240 * 240 (val 0-4)


"""
************************ Evaluation related ****************************
"""


def cal_iou(predict, target):
    """
    iou = |A ^ B| / |A v B| = |A^B| / (|A| + |B| -  |A^B|)
    :param predict: 1D Long array  bz  * height * weight
    :param target:  1D Long array  bz  * height * weight
    :return:
    """
    smooth = 0.0001
    intersection = float((target * predict).sum())
    union = float(predict.sum())+ float(target.sum()) - intersection
    return (intersection + smooth) / (union + smooth)


def cal_ious(predicts, target, num_class=5):
    """
    :param predicts:    3D tensor   bz * 240 * 240 (val 0-4)
    :param target:      3D tensor   bz * 240 * 240 (val 0-4)
    :return:
    """
    ious = []
    predicts = np.asarray(predicts.long().data)  # np.array bz * 240 * 240 (val 0-4)
    target = np.asarray(target.long().data)      # np.array bz * 240 * 240 (val 0-4)

    for b in range(predicts.shape[0]):     # for each batch
        im_target = target[b, :, :].flatten()
        im_predict = predicts[b, :, :].flatten()
        for i in range(1, num_class):    # for label i (1,2,3,4), ignore label 0
            tar = ((im_target == i) + 0)     # 2D Long Tensor, 240 * 240
            predict = ((im_predict == i) + 0)
            score = cal_iou(predict, tar)
            ious.append(score)
    return ious


def cal_ious3d(predicts, target, num_class=5):
    """
    :param predicts:    4D tensor   bz * temporal * 240 * 240 (val 0-4)
    :param target:      4D tensor   bz * temporal * 240 * 240 (val 0-4)
    :return:
    """
    ious = []           # len:  bz * temporal * class
    predicts = np.asarray(predicts.long().data)
    target = np.asarray(target.long().data)
    for b in range(predicts.shape[0]):          # for each batch
        for t in range(predicts.shape[1]):  # for each temporal
            im_target = target[b, t, :, :].flatten()        # one image 2D
            im_predicts = predicts[b, t, :, :].flatten()    # one image 2D
            for i in range(1, num_class):  # for label i (1,2,3,4), ignore label 0
                predict = ((im_predicts == i) + 0)  # 2D Long np.array 240 * 240
                tar = ((im_target == i) + 0)  # 2D Long np.array 240 * 240
                score = cal_iou(predict, tar)
                ious.append(score)
    return ious


def meanIoU(predicts, target, numclass=5):
    """
    :param predicts:    3D tensor   bz * 240 * 240 (val 0-4)
    :param target:      3D tensor   bz * 240 * 240 (val 0-4)
    :return:
    """
    if len(predicts.shape) == 3:
        ious = cal_ious(predicts, target, num_class=numclass)
    elif len(predicts.shape) == 4 :
        ious = cal_ious3d(predicts, target, num_class=numclass)
    return np.mean(ious)


def cal_subject_iou_5class(predicts, target):
    """
    :param predicts:    3D Tensor   155 * 240 * 240 (val 0-4)
    :param target:      3D Tensor   155 * 240 * 240 (val 0-4)
    :return:
    """
    ious = []           # len:  bz * temporal * class
    predicts = np.asarray(predicts.long()).flatten()
    target = np.asarray(target.long()).flatten()
    for i in range(5):  # for label i (0,1,2,3,4)
        predict = ((predicts == i) + 0)  # 2D Long np.array 240 * 240
        tar = ((target == i) + 0)  # 2D Long np.array 240 * 240
        score = cal_iou(predict, tar)
        ious.append(score)
    return ious


def cal_subject_dice_whole_tumor(predicts, target):
    """
    :param predicts:    3D Tensor   155 * 240 * 240 (val 0-4)
    :param target:      3D Tensor   155 * 240 * 240 (val 0-4)
    :return:
    """
    predicts = np.asarray(predicts.long()).flatten()  # 1d long (155 * 240 * 240)
    target = np.asarray(target.long()).flatten()      # 1d long (155 * 240 * 240)

    predict = ((predicts > 0) + 0)  # 1D Long np.array 240 * 240
    tar = ((target > 0) + 0)        # 1D Long np.array 240 * 240
    # score = cal_iou(predict, tar)
    score = dice(predict, tar)
    return score


def dice(predict, target):
    """
    dice = 2*|A^B| / (|A| + |B|)
    :param predict: 1D numpy
    :param target:  1D numpy
    :return:
    """
    smooth = 0.0001
    intersection = float((target * predict).sum())
    return (2.0 * intersection + smooth) / (float(predict.sum())
                                            + float(target.sum()) + smooth)


def sensitivity(predict, target):
    """
    :param predict: 1D numpy
    :param target:  1D numpy
    :return:
    """
    smooth = 0.0001
    intersection = float((target * predict).sum())
    return ( intersection + smooth) / (float(target.sum()) + smooth)


def PPV(predict, target):
    """
    :param predict: 1D numpy
    :param target:  1D numpy
    :return:
    """
    smooth = 0.0001
    intersection = float((target * predict).sum())
    return ( intersection + smooth) / (float(predict.sum()) + smooth)


def all_dice(predicts, label):
    """
    :param predicts: 3D tensor   bz * 240 * 240 (val 0-4)
    :param label:
    :return:
    """
    dices = []      # list 5
    for i in range(5):      # for class i (0,1,2,3,4)
        predict = ((predicts == i) + 0).long()  # 3D Long Tensor, bz * 240 * 240
        lbl = ((label == i) + 0).long()
        score = dice(predict, lbl)
        dices.append(score)
    return dices

