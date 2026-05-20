# 导入必要的库
# coding: utf-8
import torch
from torch.autograd import Variable
import sklearn.utils
from src.SPA.augmentation import *
import numpy as np
import cv2
from torch.utils.data import DataLoader, Subset

# 将数据转换为变量，如果CUDA可用，将其移动到GPU
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# 将数据转换为需要梯度的变量，如果CUDA可用，将其移动到GPU
def to_var_grad(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=True)

# 执行数据增强
def run_n_aug(x, y, n_aug, num_classes):
    # 数据增强选项，根据n_aug选择具体的增强方式
    if n_aug == 1:
        x = horizontal_flip(x)
    elif n_aug == 2:
        x = random_crop(x)
    elif n_aug == 3:
        x = random_transfer(x)
    elif n_aug == 4:
        x = random_rotation(x)
    elif n_aug == 5:
        x, y = mixup(image=x, label=y, num_classes=num_classes)
    elif n_aug == 6:
        x = cutout(x)
    elif n_aug == 7:
        x = random_erasing(x)
    elif n_aug == 8:
        x, y = ricap(image_batch=x, label_batch=y, num_classes=num_classes)
        x = to_var(x)
        y = to_var(y)
    elif n_aug == 12:
        x = horizontal_flip(x)
        x = random_crop(x)
    elif n_aug == 17:
        x = horizontal_flip(x)
        x = random_erasing(x)
    elif n_aug == 34:
        x = random_transfer(x)
        x = random_rotation(x)

    return x, y

# 自适应数据增强
def self_paced_augmentation(images, labels, flag_noise, index, n_aug, num_classes):
    x, y = run_n_aug(images, labels, n_aug, num_classes)  # 进行数据增强

    # 将数据转换为numpy数组
    images = np.array(images.data.cpu())
    labels = np.array(labels.data.cpu())
    x = np.array(x.data.cpu())
    y = np.array(y.data.cpu())

    # 根据噪声标志更新增强后的数据
    x = np.where(flag_noise[index].reshape(-1, 1, 1, 1) < 1, images, x)
    if labels.ndim > 1:
        y = np.where(flag_noise[index].reshape(-1, 1) < 1, labels, y)

    # 将数据转换回变量
    x = to_var(torch.from_numpy(x).float())
    y = to_var(torch.from_numpy(y))

    return x, y

# 根据损失值更新噪声标志
def flag_update(loss, judge_noise):
    flag_noise = np.where(loss < judge_noise, 0, 1)  # 如果损失小于阈值，标志为0，否则为1
    # flag_noise = np.where(loss > judge_noise, 0, 1)  # 反转标志更新逻辑
    return flag_noise

# 获取数据的索引
def collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    indices = [i for i in range(len(batch))]
    return torch.stack(data), torch.tensor(labels), torch.tensor(indices)

