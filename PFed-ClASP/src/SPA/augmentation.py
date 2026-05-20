# coding: utf-8
import cv2
# from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
import torch
from src.SPA.util import *
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import rotate

def random_noise(image):
    noise_scale = 0.001
    noise = np.random.randn(np.array(image.data).shape[0], np.array(image.data).shape[1], np.array(image.data).shape[2], np.array(image.data).shape[3])
    image = image + to_var(torch.from_numpy(noise_scale * noise).float())
    # util.save_images(image)
    return image

def horizontal_flip(image):
    n, _, _, _ = image.shape
    image2 = image.clone()

    rand = np.random.rand(n)
    reverse = torch.arange(image.shape[3] - 1, -1, -1)

    for i in range(n):
        if rand[i] < 0.5:
            image2[i] = image[i, :, :, reverse]

    # util.save_images(image2)

    return image2

def vertical_flip(image):
    n, _, _, _ = image.shape
    image2 = image.clone()

    rand = np.random.rand(n)
    reverse = torch.arange(image.shape[2] - 1, -1, -1)

    for i in range(n):
        if rand[i] < 0.5:
            image2[i] = image[i, :, reverse, :]

    # util.save_images(image2)

    return image2

def random_crop(image):
    n, c, h, w = image.shape
    image2 = torch.zeros(n, c, h, w).cuda()

    # crop_size = (9 * h // 10, 9 * w // 10)
    crop_size = (4 * h // 5, 4 * w // 5)

    for i in range(n):
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])

        bottom = top + crop_size[0]
        right = left + crop_size[1]

        x = image[i, :, top:bottom, left:right].clone()
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
        image2[i] = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

    # util.save_images(image)

    return image2

# 定义随机平移函数
def random_transfer(image):
    n, c, h, w = image.shape
    image2 = torch.zeros(n, c, h, w).cuda()

    offset = np.random.randint(-h//5, h//5 + 1, size=(n, 2))
    # offset1 = int(h * 0.2)
    # offset2 = int(h * 0.2)

    for i in range(n):
        offset1, offset2 = offset[i]

        left = max(0, offset1)
        top = max(0, offset2)
        right = min(w, w + offset1)
        bottom = min(h, h + offset2)
        image2[i, :, top - offset2:bottom - offset2, left - offset1:right - offset1] = image[i, :, top:bottom, left:right]

    # util.save_images(image2)

    return image2


def imresize(arr, size):
    img = Image.fromarray(arr)
    img = img.resize(size, Image.ANTIALIAS)
    return np.array(img)

def random_rotation(image):
    size, c, h, w = image.shape
    image2 = image.clone()
    image2 = np.array(image2.data.cpu())

    if c == 1:
        image2 = np.squeeze(image2, axis=1)
    else:
        image2 = image2.transpose((0, 2, 3, 1))

    for i in range(size):
        angle = np.random.randint(180)
        image_rotate = rotate(image2[i], angle, reshape=False)

        if c == 1:
            resized_image = imresize(image_rotate, (h, w))
            image2[i] = resized_image[:, :, np.newaxis]
        else:
            image2[i] = imresize(image_rotate, (h, w))

    if c == 1:
        image2 = image2[:, np.newaxis, :, :] / 255.0
    else:
        image2 = image2.transpose((0, 3, 1, 2)) / 255.0

    image2 = torch.from_numpy(image2).float()
    # util.save_images(image2)

    return image2

def mixup(image, label, num_classes, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs and soft labels'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = image.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    image2 = image[index, :]
    label2 = label[index]

    y_one_hot = torch.eye(num_classes, device=image.device)[label.long()]
    y2_one_hot = torch.eye(num_classes, device=image.device)[label2.long()]

    # Perform mixing
    mixed_image = lam * image + (1 - lam) * image2
    soft_label = lam * y_one_hot + (1 - lam) * y2_one_hot

    return mixed_image, soft_label


def cutout(image):
    image2 = image.clone()
    n, _, h, w = image2.shape
    mask_size = h // 2

    for i in range(n):
        mask_value = image2.mean()
        top = np.random.randint(0 - mask_size // 2, h - mask_size // 2)
        left = np.random.randint(0 - mask_size // 2, w - mask_size // 2)
        bottom = top + mask_size
        right = left + mask_size

        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > h:
            bottom = h
        if right > w:
            right = w

        image2[i][:, top:bottom, left:right] = mask_value

    # util.save_images(image2)

    return image2

def random_erasing(image, p=0.5, s=(0.05, 0.8), r=(0.3, 5), num_erasures=3):
    if np.random.rand() > p:
        return image

    image2 = image.clone()
    n, _, h, w = image2.shape

    for i in range(n):
        for _ in range(num_erasures):
            mask_value = np.random.rand()

            mask_area = np.random.uniform(s[0], s[1]) * h * w

            aspect_ratio = np.random.uniform(r[0], r[1])

            mask_height = int(np.round(np.sqrt(mask_area / aspect_ratio)))
            mask_width = int(np.round(np.sqrt(mask_area * aspect_ratio)))

            mask_height = min(mask_height, h)
            mask_width = min(mask_width, w)

            top = np.random.randint(0, h - mask_height + 1)
            left = np.random.randint(0, w - mask_width + 1)

            bottom = top + mask_height
            right = left + mask_width

            image2[i][:, top:bottom, left:right] = mask_value

    return image2

def ricap(image_batch, label_batch, num_classes):
    image_batch = np.array(image_batch.data.cpu())
    label_batch = np.array(label_batch.data.cpu())

    label_batch = np.identity(num_classes)[label_batch]

    alpha = 1.0
    use_same_random_value_on_batch = False

    batch_size = image_batch.shape[0]
    image_y = image_batch.shape[2]
    image_x = image_batch.shape[3]

    # crop_size w, h from beta distribution
    if use_same_random_value_on_batch:
        w_dash = np.random.beta(alpha, alpha) * np.ones(batch_size)
        h_dash = np.random.beta(alpha, alpha) * np.ones(batch_size)
    else:
        w_dash = np.random.beta(alpha, alpha, size=batch_size)
        h_dash = np.random.beta(alpha, alpha, size=batch_size)
    w = np.round(w_dash * image_x).astype(np.int32)
    h = np.round(h_dash * image_y).astype(np.int32)

    output_images = np.zeros(image_batch.shape)
    output_labels = np.zeros(label_batch.shape)

    def create_masks(start_xs, start_ys, end_xs, end_ys):
        mask_x = np.logical_and(np.arange(image_x).reshape(1, 1, 1, -1) >= start_xs.reshape(-1, 1, 1, 1),
                                np.arange(image_x).reshape(1, 1, 1, -1) < end_xs.reshape(-1, 1, 1, 1))
        mask_y = np.logical_and(np.arange(image_y).reshape(1, 1, -1, 1) >= start_ys.reshape(-1, 1, 1, 1),
                                np.arange(image_y).reshape(1, 1, -1, 1) < end_ys.reshape(-1, 1, 1, 1))
        mask = np.logical_and(mask_y, mask_x)
        mask = np.logical_and(mask, np.repeat(True, image_batch.shape[1]).reshape(1, -1, 1, 1))

        return mask

    def crop_concatenate(wk, hk, start_x, start_y, end_x, end_y):
        nonlocal output_images, output_labels
        xk = (np.random.rand(batch_size) * (image_x - wk)).astype(np.int32)
        yk = (np.random.rand(batch_size) * (image_y - hk)).astype(np.int32)
        target_indices = np.arange(batch_size)
        np.random.shuffle(target_indices)
        weights = wk * hk / image_x / image_y

        dest_mask = create_masks(start_x, start_y, end_x, end_y)
        target_mask = create_masks(xk, yk, xk + wk, yk + hk)

        output_images[dest_mask] = image_batch[target_indices][target_mask]
        output_labels += weights.reshape(-1, 1) * label_batch[target_indices]

    crop_concatenate(w, h,
                     np.repeat(0, batch_size), np.repeat(0, batch_size),
                     w, h)
    crop_concatenate(image_x - w, h,
                     w, np.repeat(0, batch_size),
                     np.repeat(image_x, batch_size), h)
    crop_concatenate(w, image_y - h,
                     np.repeat(0, batch_size), h,
                     w, np.repeat(image_y, batch_size))
    crop_concatenate(image_x - w, image_y - h,
                     w, h, np.repeat(image_x, batch_size),
                     np.repeat(image_y, batch_size))

    output_images = torch.from_numpy(output_images).float()
    output_labels = torch.from_numpy(output_labels).float()
    label_batch = torch.from_numpy(label_batch).float()

    # util.save_images(output_images)

    return output_images, output_labels
