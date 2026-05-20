import math
import torch as tc
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode as Interpolation


def IDBH(x, dataset):
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'svhn':
        augmentations = [
            T.RandomHorizontalFlip(),
            CropShift(0, 11),
            ColorShape('color'),
            T.RandomErasing(p=0.5)
        ]
    elif dataset == 'mnist' or dataset == 'fashionmnist':
        augmentations = [
            T.RandomHorizontalFlip(),
            CropShift(0, 9),
            ColorShape('shape'),
            T.RandomErasing(p=0.5, scale=(0.02, 0.4))
        ]

    # 对x进行增强操作
    for augment in augmentations:
        x = augment(x)

    return x


class ColorShape(tc.nn.Module):
    # 针对颜色偏向的增强变换列表
    ColorBiased = [
        (0.125, 'color', 0.1, 1.9),  # 以0.125的概率应用颜色调整，强度范围为0.1到1.9
        (0.125, 'brightness', 0.5, 1.9),  # 以0.125的概率应用亮度调整，强度范围为0.5到1.9
        (0.125, 'contrast', 0.5, 1.9),  # 以0.125的概率应用对比度调整，强度范围为0.5到1.9
        (0.125, 'sharpness', 0.1, 1.9),  # 以0.125的概率应用锐度调整，强度范围为0.1到1.9
        (0.125, 'autocontrast'),  # 以0.125的概率应用自动对比度调整
        (0.125, 'equalize'),  # 以0.125的概率应用直方图均衡化
        (0.125, 'shear', 0.05, 0.15),  # 以0.125的概率应用剪切变换，剪切强度范围为0.05到0.15
        (0.125, 'rotate', 1, 11)  # 以0.125的概率应用旋转，旋转角度范围为1到11度
    ]

    # 针对形状偏向的增强变换列表
    ShapeBiased = [
        (0.08, 'brightness', 0.5, 1.9),  # 以0.08的概率应用亮度调整，强度范围为0.5到1.9
        (0.04, 'contrast', 0.5, 1.9),  # 以0.04的概率应用对比度调整，强度范围为0.5到1.9
        (0.08, 'sharpness', 0.1, 1.9),  # 以0.08的概率应用锐度调整，强度范围为0.1到1.9
        (0.04, 'autocontrast'),  # 以0.04的概率应用自动对比度调整
        (0.08, 'equalize'),  # 以0.08的概率应用直方图均衡化
        (0.3, 'shear', 0.05, 0.35),  # 以0.3的概率应用剪切变换，剪切强度范围为0.05到0.35
        (0.3, 'rotate', 1, 31)  # 以0.3的概率应用旋转，旋转角度范围为1到31度
    ]

    def __init__(self, version='color'):
        super().__init__()

        assert version in ['color', 'shape']

        space = self.ColorBiased if version == 'color' else self.ShapeBiased

        self.space = {}
        p_accu = 0.0

        for trans in space:
            p = trans[0]
            self.space[(p_accu, p_accu + p)] = trans[1:]
            p_accu += p

    def transform(self, img, trans):

        if len(trans) == 1:
            trans = trans[0]
        else:
            lower, upper = trans[1:]
            trans = trans[0]
            if trans == 'rotate':
                strength = tc.randint(lower, upper, (1,)).item()
            else:
                strength = tc.rand(1) * (upper - lower) + lower
        if trans == 'color':
            img = F.adjust_saturation(img, strength)
        elif trans == 'brightness':
            img = F.adjust_brightness(img, strength)
        elif trans == 'contrast':
            img = F.adjust_contrast(img, strength)
        elif trans == 'sharpness':
            img = F.adjust_sharpness(img, strength)
        elif trans == 'shear':
            if tc.randint(2, (1,)):
                strength *= -1
            strength = math.degrees(strength)
            strength = [strength, 0.0] if tc.randint(2, (1,)) else [0.0, strength]
            img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=strength,
                           interpolation=Interpolation.NEAREST, fill=0)
        elif trans == 'rotate':
            if tc.randint(2, (1,)):
                strength *= -1
            img = F.rotate(img, angle=strength, interpolation=Interpolation.NEAREST, fill=0)
        elif trans == 'autocontrast':
            img = F.autocontrast(img)
        elif trans == 'equalize':
            img = (img * 255).to(tc.uint8)
            img = F.equalize(img)
            img = img.to(tc.float32) / 255

        return img

    def forward(self, img):

        roll = tc.rand(1)
        for (lower, upper), trans in self.space.items():
            if roll <= upper and roll >= lower:
                return self.transform(img, trans)

        return img


class CropShift(tc.nn.Module):
    def __init__(self, low, high=None):
        super().__init__()
        high = low if high is None else high
        self.low, self.high = int(low), int(high)

    def sample_top(self, x, y):
        x = tc.randint(0, x + 1, (1,)).item()
        y = tc.randint(0, y + 1, (1,)).item()
        return x, y

    def forward(self, img):
        if self.low == self.high:
            strength = self.low
        else:
            strength = tc.randint(self.low, self.high, (1,)).item()
        w, h = F.get_image_size(img)
        crop_x = tc.randint(0, strength + 1, (1,)).item()
        crop_y = strength - crop_x
        crop_w, crop_h = w - crop_x, h - crop_y
        top_x, top_y = self.sample_top(crop_x, crop_y)
        img = F.crop(img, top_y, top_x, crop_h, crop_w)
        img = F.pad(img, padding=[crop_x, crop_y], fill=0)

        top_x, top_y = self.sample_top(crop_x, crop_y)

        return F.crop(img, top_y, top_x, h, w)