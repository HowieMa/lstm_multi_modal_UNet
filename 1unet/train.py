from unet2d import UNet2D
from src.utils import *
from data_loader.data_brats15 import Brats15DataLoader
from test import evaluation

from torch.utils.data import DataLoader
from torch.autograd import Variable


import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ********** Hyper Parameter **********
data_dir = '/home/haoyum/download/BRATS2015_Training'
conf_train = '../config/train15.conf'
conf_valid = '../config/valid15.conf'
save_dir = 'ckpt/'

learning_rate = 0.0001
batch_size = 32
epochs = 100

cuda_available = torch.cuda.is_available()
device_ids = [0, 1, 2, 3]       # multi-GPU
torch.cuda.set_device(device_ids[0])


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ******************** build model ********************
net = UNet2D(in_ch=4, out_ch=5, degree=64)  # multi-modal =4, out binary classification one-hot
if cuda_available:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)

# ******************** data preparation  ********************
print('train data ....')
train_data = Brats15DataLoader(data_dir=data_dir, conf=conf_train,train=True)  # 224 subject, 34720 images
print('valid data .....')
valid_data = Brats15DataLoader(data_dir=data_dir,  conf=conf_valid,train=False)   #

# data loader
train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_dataset = DataLoader(dataset=valid_data, batch_size=1, shuffle=True)


def to_var(tensor):
    return Variable(tensor.cuda() if cuda_available else tensor)


def run():
    score_max = -1.0
    best_epoch = 0
    weight = torch.from_numpy(train_data.weight).float()    # weight for all class
    weight = to_var(weight)                                 #

    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(weight=weight)

    for epoch in range(1, epochs + 1):
        print('epoch....................................' + str(epoch))
        train_loss = []
        # *************** train model ***************
        print('train ....')
        net.train()
        for step, (image, label, index) in enumerate(train_dataset):
            image = to_var(image)    # 4D tensor   bz * 4(modal) * 240 * 240
            label = to_var(label)    # 3D tensor   bz * 240 * 240 (value 0-4)

            optimizer.zero_grad()       #
            predicts = net(image)    # 4D tensor   bz * 5(class) * 240 * 240
            loss_train = criterion(predicts, label.long())
            train_loss.append(float(loss_train))
            loss_train.backward()
            optimizer.step()

            # ****** save sample image for each epoch ******
            if step % 200 == 0:
                print('..step ....%d' % step)
                print('....loss....%f' %loss_train)
                predicts = one_hot_reverse(predicts)  # 3D long Tensor  bz * 240 * 240 (val 0-4)
                save_train_images(image, predicts, label, index, epoch, save_dir=save_dir)

        # ***************** calculate valid loss *****************
        print('valid ....')
        current_score, valid_loss = evaluation(net, valid_dataset, criterion, save_dir=None)

        # **************** save loss for one batch ****************
        print('train_epoch_loss ' + str(sum(train_loss) / (len(train_loss) * 1.0)) )
        print('valid_epoch_loss ' + str(sum(valid_loss) / (len(valid_loss) * 1.0)) )

        # **************** save model ****************
        if current_score > score_max:
            best_epoch = epoch
            torch.save(net.state_dict(),
                       os.path.join(save_dir , 'best_epoch.pth'))
            score_max = current_score
        print('valid_meanIoU_max ' + str(score_max))
        print('Current Best epoch is %d' % best_epoch)

        if epoch == epochs:
            torch.save(net.state_dict(),
                       os.path.join(save_dir, 'final_epoch.pth'))

    print('Best epoch is %d' % best_epoch)
    print('done!')


if __name__ == '__main__':
    run()


