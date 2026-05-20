#utf-8
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from src.utils.constants import DATA_SHAPE, NUM_CLASSES, INPUT_CHANNELS
from src.utils.tools import NestedNamespace


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: list[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def _get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.detach().clone())

        for module in target_modules:
            module.register_forward_hook(_get_feature_hook_fn)

    def check_and_preprocess(self, args: NestedNamespace):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]
        if args.common.buffers == "global":
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    buffers_list = list(module.named_buffers())
                    for name_buffer, buffer in buffers_list:
                        # transform buffer to parameter
                        # for showing out in model.parameters()
                        delattr(module, name_buffer)
                        module.register_parameter(
                            name_buffer,
                            torch.nn.Parameter(buffer.float(), requires_grad=False),
                        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))

    def get_last_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[list[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    feature_length = {
        "mnist": 1024,
        "medmnistS": 1024,
        "medmnistC": 1024,
        "medmnistA": 1024,
        "covid19": 196736,
        "fmnist": 1024,
        "emnist": 1024,
        "femnist": 1,
        "cifar10": 1600,
        "cinic10": 1600,
        "cifar100": 1600,
        "tiny_imagenet": 3200,
        "celeba": 133824,
        "svhn": 1600,
        "usps": 800,
    }

    def __init__(self, dataset: str):
        super(FedAvgCNN, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 512),
                activation3=nn.ReLU(),
            )
        )
        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])


class LeNet5(DecoupledModel):
    feature_length = {
        "mnist": 256,
        "medmnistS": 256,
        "medmnistC": 256,
        "medmnistA": 256,
        "covid19": 49184,
        "fmnist": 256,
        "emnist": 256,
        "femnist": 256,
        "cifar10": 400,
        "cinic10": 400,
        "svhn": 400,
        "cifar100": 400,
        "celeba": 33456,
        "usps": 200,
        "tiny_imagenet": 2704,
    }

    def __init__(self, dataset: str) -> None:
        super(LeNet5, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
                activation4=nn.ReLU(),
            )
        )

        self.classifier = nn.Linear(84, NUM_CLASSES[dataset])
class FashionmnistNet(DecoupledModel):
    feature_length = {
        'mnist':1152,
        "fmnist": 1152,        # 28x28 输入
        "cifar10": 2048,      # 32x32 输入
        "emnist":1152,
    }

    def __init__(self, dataset: str) -> None:
        super(FashionmnistNet, self).__init__()

        # 卷积层配置
        conv1_out_channels = 32
        conv2_out_channels = 64
        conv3_out_channels = 128

        self.base = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(INPUT_CHANNELS[dataset], conv1_out_channels, kernel_size=5, padding=2)),
                ('activation1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('conv2', nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=5, padding=2)),
                ('activation2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('conv3', nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size=3, padding=1)),
                ('activation3', nn.ReLU()),
                ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(self.feature_length[dataset], 2048)),  # 输入维度为 feature_length
                ('activation4', nn.ReLU()),
                ('fc2', nn.Linear(2048, 512)),
                ('activation5', nn.ReLU()),
            ])
        )

        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])
class FashionmnistNet_personal(DecoupledModel):
    feature_length = {
        'mnist': 1152,
        "fmnist": 1152,        # 28x28 输入
        "cifar10": 2048,       # 32x32 输入
        "emnist": 1152,
    }

    def __init__(self, dataset: str) -> None:
        super(FashionmnistNet_personal, self).__init__()

        # 卷积层配置
        conv1_out_channels = 32
        conv2_out_channels = 64
        conv3_out_channels = 128

        self.base = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(INPUT_CHANNELS[dataset], conv1_out_channels, kernel_size=5, padding=2)),
                ('activation1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('conv2', nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=5, padding=2)),
                ('activation2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('conv3', nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size=3, padding=1)),
                ('activation3', nn.ReLU()),
                ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('flatten', nn.Flatten()),  # 将特征图展平，送到分类器
            ])
        )

        # 定义分类器，将全连接层放入此处
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.feature_length[dataset], 2048)),  # 输入维度为 feature_length
                ('activation4', nn.ReLU()),
                ('fc2', nn.Linear(2048, 512)),
                ('activation5', nn.ReLU()),
                ('fc3', nn.Linear(512, NUM_CLASSES[dataset]))  # 输出类别数
            ])
        )

class TwoNN(DecoupledModel):
    feature_length = {
        "mnist": 784,
        "medmnistS": 784,
        "medmnistC": 784,
        "medmnistA": 784,
        "fmnist": 784,
        "emnist": 784,
        "femnist": 784,
        "cifar10": 3072,
        "cinic10": 3072,
        "svhn": 3072,
        "cifar100": 3072,
        "usps": 1536,
        "synthetic": DATA_SHAPE["synthetic"],
    }

    def __init__(self, dataset):
        super(TwoNN, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(self.feature_length[dataset], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        # self.base = nn.Linear(features_length[dataset], 200)
        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(self.base(x))
        return x

    def get_last_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        x = torch.flatten(x, start_dim=1)
        x = self.base(x)
        return func(x)

    def get_all_features(self, x):
        raise RuntimeError("2NN has 0 Conv layer, so is unable to get all features.")


class AlexNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        alexnet = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.base = alexnet
        self.classifier = nn.Linear(
            alexnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()

class SqueezeNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        archs = {
            "0": (models.squeezenet1_0, models.SqueezeNet1_0_Weights.DEFAULT),
            "1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
        }
        squeezenet: models.SqueezeNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = squeezenet.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(
                squeezenet.classifier[1].in_channels,
                NUM_CLASSES[dataset],
                kernel_size=1,
            ),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )


class DenseNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            "161": (models.densenet161, models.DenseNet161_Weights.DEFAULT),
            "169": (models.densenet169, models.DenseNet169_Weights.DEFAULT),
            "201": (models.densenet201, models.DenseNet201_Weights.DEFAULT),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        densenet: models.DenseNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = densenet
        self.classifier = nn.Linear(
            densenet.classifier.in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier = nn.Identity()


class ResNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
            "152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
        }

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        resnet: models.ResNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()


class MobileNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "3s": (
                models.mobilenet_v3_small,
                models.MobileNet_V3_Small_Weights.DEFAULT,
            ),
            "3l": (
                models.mobilenet_v3_large,
                models.MobileNet_V3_Large_Weights.DEFAULT,
            ),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        mobilenet = archs[version][0](weights=archs[version][1] if pretrained else None)
        self.base = mobilenet
        self.classifier = nn.Linear(
            mobilenet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class EfficientNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            "4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            "5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            "6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            "7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        efficientnet: models.EfficientNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = efficientnet
        self.classifier = nn.Linear(
            efficientnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class ShuffleNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "0_5": (
                models.shufflenet_v2_x0_5,
                models.ShuffleNet_V2_X0_5_Weights.DEFAULT,
            ),
            "1_0": (
                models.shufflenet_v2_x1_0,
                models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
            ),
            "1_5": (
                models.shufflenet_v2_x1_5,
                models.ShuffleNet_V2_X1_5_Weights.DEFAULT,
            ),
            "2_0": (
                models.shufflenet_v2_x2_0,
                models.ShuffleNet_V2_X2_0_Weights.DEFAULT,
            ),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        shufflenet: models.ShuffleNetV2 = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = shufflenet
        self.classifier = nn.Linear(shufflenet.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()


class VGG(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        archs = {
            "11": (models.vgg11, models.VGG11_Weights.DEFAULT),
            "13": (models.vgg13, models.VGG13_Weights.DEFAULT),
            "16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "19": (models.vgg19, models.VGG19_Weights.DEFAULT),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        vgg: models.VGG = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = vgg
        self.classifier = nn.Linear(
            vgg.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()



# NOTE: You can build your custom model here.
# What you only need to do is define the architecture in __init__().
# Don't need to consider anything else, which are handled by DecoupledModel well already.
# Run `python *.py -m custom` to use your custom model.
# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual  # 残差连接
        out = self.relu(out)
        return out

class ResNet9(DecoupledModel):
    def __init__(self, dataset: str) -> None:
        super(ResNet9, self).__init__()
        # 定义卷积层和第一个全连接层，使用OrderedDict来明确每一层的名称
        self.base = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
                ('bn2', nn.BatchNorm2d(128)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(2)),
                ('resblock1', ResidualBlock(128)),
                ('conv3', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
                ('bn3', nn.BatchNorm2d(256)),
                ('relu3', nn.ReLU(inplace=True)),
                ('conv4', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
                ('bn4', nn.BatchNorm2d(512)),
                ('relu4', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(2)),
                ('resblock2', ResidualBlock(512)),
                ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
                ('flatten', nn.Flatten())
            ])
        )
        # 只保留分类层
        self.classifier = nn.Linear(512, 10)  # FashionMNIST 数据集有 10 个类别

class FashionmnistAlex(DecoupledModel):
    def __init__(self, dataset: str) -> None:
        super(FashionmnistAlex, self).__init__()
        # 增大模型的规模
        self.base = nn.Sequential(
            OrderedDict([
                # 第一层卷积，增大输出通道数
                ('conv1', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)),  # 从32增至64
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),  # 输出尺寸：64 x 14 x 14
                # 第二层卷积
                ('conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1)),  # 从64增至128
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),  # 输出尺寸：128 x 7 x 7
                # 第三层卷积
                ('conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1)),  # 从128增至256
                ('relu3', nn.ReLU(inplace=True)),
                # 第四层卷积
                ('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),  # 保持256不变
                ('relu4', nn.ReLU(inplace=True)),
                # 第五层卷积
                ('conv5', nn.Conv2d(256, 128, kernel_size=3, padding=1)),  # 从256减至128
                ('relu5', nn.ReLU(inplace=True)),
                # 第三个池化层
                ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2)),  # 输出尺寸：128 x 3 x 3
                # 展平
                ('flatten', nn.Flatten()),
                # 全连接层，增大神经元数量
                ('dropout1', nn.Dropout()),
                ('fc1', nn.Linear(128 * 3 * 3, 1024)),  # 从512增至1024
                ('relu6', nn.ReLU(inplace=True)),
                ('dropout2', nn.Dropout()),
                ('fc2', nn.Linear(1024, 512)),  # 从256增至512
                ('relu7', nn.ReLU(inplace=True)),
            ])
        )
        # 分类层
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc3', nn.Linear(512, 10)),  # FashionMNIST 数据集有 10 个类别
            ])
        )



MODELS = {
    "FashionmnistNet": FashionmnistNet,
    "FashionmnistAlex":FashionmnistAlex,
    "FashionmnistNet_personal":FashionmnistNet_personal,
    "res9":ResNet9,
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "alex": AlexNet,
    "2nn": TwoNN,
    "squeeze0": partial(SqueezeNet, version="0"),
    "squeeze1": partial(SqueezeNet, version="1"),
    "res18": partial(ResNet, version="18"),
    "res34": partial(ResNet, version="34"),
    "res50": partial(ResNet, version="50"),
    "res101": partial(ResNet, version="101"),
    "res152": partial(ResNet, version="152"),
    "dense121": partial(DenseNet, version="121"),
    "dense161": partial(DenseNet, version="161"),
    "dense169": partial(DenseNet, version="169"),
    "dense201": partial(DenseNet, version="201"),
    "mobile2": partial(MobileNet, version="2"),
    "mobile3s": partial(MobileNet, version="3s"),
    "mobile3l": partial(MobileNet, version="3l"),
    "efficient0": partial(EfficientNet, version="0"),
    "efficient1": partial(EfficientNet, version="1"),
    "efficient2": partial(EfficientNet, version="2"),
    "efficient3": partial(EfficientNet, version="3"),
    "efficient4": partial(EfficientNet, version="4"),
    "efficient5": partial(EfficientNet, version="5"),
    "efficient6": partial(EfficientNet, version="6"),
    "efficient7": partial(EfficientNet, version="7"),
    "shuffle0_5": partial(ShuffleNet, version="0_5"),
    "shuffle1_0": partial(ShuffleNet, version="1_0"),
    "shuffle1_5": partial(ShuffleNet, version="1_5"),
    "shuffle2_0": partial(ShuffleNet, version="2_0"),
    "vgg11": partial(VGG, version="11"),
    "vgg13": partial(VGG, version="13"),
    "vgg16": partial(VGG, version="16"),
    "vgg19": partial(VGG, version="19"),
}
