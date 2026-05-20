from collections import OrderedDict
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig
from torch import Tensor

from src.utils.constants import DATA_SHAPE, INPUT_CHANNELS, NUM_CLASSES,DEFAULTS
import torch.nn.functional as F
import math
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

    def check_and_preprocess(self, args: DictConfig):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module for module in self.modules() if isinstance(module, nn.Dropout)
        ]
        if args.common.buffers == "global":
            for module in self.modules():
                if isinstance(
                    module,
                    (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
                ):
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
        try:
            out = self.base(x)
        except RuntimeError as err:
            if x.shape[1] == 1:
                x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
                try:
                    out = self.base(x)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"Seems {self.__class__.__name__} does not support this dataset. Data resizing may help."
                    ) from err
            else:
                raise RuntimeError(
                    f"Seems {self.__class__.__name__} does not support this dataset."
                ) from err
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
        try:
            _ = self.base(x)
        except RuntimeError as err:
            if x.shape[1] == 1:
                x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
                try:
                    _ = self.base(x)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"Seems {self.__class__.__name__} does not support this dataset. Data resizing may help."
                    ) from err
            else:
                raise RuntimeError(
                    f"Seems {self.__class__.__name__} does not support this dataset."
                ) from err
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

    def __init__(self, dataset: str, pretrained):
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

class ConvNet(DecoupledModel):
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

    def __init__(self, dataset: str,pretrained):
        super(ConvNet, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([
                # 第一个卷积层
                ('conv1', nn.Conv2d(INPUT_CHANNELS[dataset], 32, kernel_size=5)),
                ('bn1', nn.BatchNorm2d(32)),  # 在第一个卷积层后添加BN层
                ('activation1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=2)),
                # 第二个卷积层
                ('conv2', nn.Conv2d(32, 64, kernel_size=5)),
                ('bn2', nn.BatchNorm2d(64)),  # 在第二个卷积层后添加BN层
                ('activation2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=2)),
                # 展平层和全连接层
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(self.feature_length[dataset], 512)),
                ('activation3', nn.ReLU()),
            ])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc2',nn.Linear(512, NUM_CLASSES[dataset]))
            ])
        )
    def register_hooks(self):
        self.activations = {}
        def hook_fn(module, input, output):
            self.activations[module] = output
        # Register hooks for layers of interest
        self.base.conv1.register_forward_hook(hook_fn)
        self.base.pool1.register_forward_hook(hook_fn)
        self.base.conv2.register_forward_hook(hook_fn)
        self.base.pool2.register_forward_hook(hook_fn)

    def get_activations(self, x):
        self.activations = {}  # 清空之前保存的激活图
        self.register_hooks()  # 注册钩子
        # Perform a forward pass and capture activations
        _ = self.base(x)
        # Retrieve the activations
        activation_map1 = self.activations.get(self.base.conv1, None)
        activation_map2 = self.activations.get(self.base.pool1, None)
        activation_map3 = self.activations.get(self.base.conv2, None)
        activation_map4 = self.activations.get(self.base.pool2, None)
        # Return the activations
        return activation_map1, activation_map2, activation_map3, activation_map4

class PConvNet(DecoupledModel):
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

    def __init__(self, dataset: str, pretrained):
        super(PConvNet, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([
                # 第一个卷积层
                ('conv1', nn.Conv2d(INPUT_CHANNELS[dataset], 32, kernel_size=5)),
                ('bn1', nn.BatchNorm2d(32)),  # 在第一个卷积层后添加BN层
                ('activation1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=2)),
                # 第二个卷积层
                ('conv2', nn.Conv2d(32, 64, kernel_size=5)),
                ('bn2', nn.BatchNorm2d(64)),  # 在第二个卷积层后添加BN层
                ('activation2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=2)),
                # 展平层和全连接层
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(self.feature_length[dataset], 512)),
                ('activation3', nn.ReLU()),
            ])
        )

        self.project = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(512, 256)),
            # ('bn_1', nn.BatchNorm1d(256)),
            ('relu_1', nn.ReLU()),
            ('linear_2', nn.Linear(256, 512)),
            # ('bn_2', nn.BatchNorm1d(512)),
        ]))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc2', nn.Linear(512, NUM_CLASSES[dataset]))
            ])
        )
    def forward(self, x: Tensor,flag=False):
        base = self.base(x)
        project = self.project(base)
        x = self.classifier(project)
        if flag:
            return x, project
        else:
            return x


class CPNConvNet(DecoupledModel):
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

    def __init__(self, dataset: str, pretrained):
        super(CPNConvNet, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([
                # 第一个卷积层
                ('conv1', nn.Conv2d(INPUT_CHANNELS[dataset], 32, kernel_size=5)),
                ('bn1', nn.BatchNorm2d(32)),  # 在第一个卷积层后添加BN层
                ('activation1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=2)),
                # 第二个卷积层
                ('conv2', nn.Conv2d(32, 64, kernel_size=5)),
                ('bn2', nn.BatchNorm2d(64)),  # 在第二个卷积层后添加BN层
                ('activation2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=2)),
                # 展平层和全连接层
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(self.feature_length[dataset], 512)),
                ('activation3', nn.ReLU()),
            ])
        )

        self.cpn = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(512, 1024)),
            ('bn_1', nn.BatchNorm1d(1024)),
            ('relu_1', nn.ReLU()),
        ]))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc2', nn.Linear(512, NUM_CLASSES[dataset]))
            ])
        )
        self.clean_cpn_feature = None
        self.backdoor_cpn_feature = None

    def CPN(self, features):
        batch_size = features.shape[0]
        processed_features = self.cpn(features)  # shape: [batch_size, projection_path_dim * 2]
        view_as_candidates = processed_features.view(batch_size, 2, 512)
        gumbel_output = F.gumbel_softmax(view_as_candidates, tau=1, hard=False, dim=1)
        clean_features = gumbel_output[:, 0, :]  # shape: [batch_size, projection_path_dim]
        backdoor_features = gumbel_output[:, 1, :]  # shape: [batch_size, projection_path_dim]
        return clean_features, backdoor_features

    def forward(self, x, flag=False):
        x = self.base(x)
        clean_features, backdoor_features = self.CPN(x)
        x = self.classifier(clean_features)
        if flag==True:
            self.clean_cpn_feature = clean_features
            self.backdoor_cpn_feature = backdoor_features
        else:
            self.clean_cpn_feature = None
            self.backdoor_cpn_feature = None
        return x


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

    def __init__(self, dataset: str, pretrained):
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

    def __init__(self, dataset: str, pretrained):
        super(TwoNN, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(self.feature_length[dataset], 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
        )
        # self.base = nn.Linear(features_length[dataset], 200)
        self.classifier = nn.Linear(200, NUM_CLASSES[dataset])

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(self.base(x))
        return x

    def get_last_features(self, data, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        data = torch.flatten(data, start_dim=1)
        data = self.base(data)
        return func(data)

    def get_all_features(self, x):
        raise RuntimeError("2NN has 0 Conv layer, so is unable to get all features.")


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class resnet(DecoupledModel):
    def __init__(self, block, num_blocks, dataset:str,pretrained=False):
        super(resnet, self).__init__()
        self.in_planes = 64  # 初始化 in_planes 为 64
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 64, kernel_size=3, stride=1, padding=1, bias=False),  # 修改为64
                bn1=nn.BatchNorm2d(64),
                activation1=nn.ReLU(),
                layer1=self._make_layer(block, 64, num_blocks[0], stride=1),  # layer1的输入和输出通道为64
                layer2=self._make_layer(block, 128, num_blocks[1], stride=2),
                layer3=self._make_layer(block, 256, num_blocks[2], stride=2),
                layer4=self._make_layer(block, 512, num_blocks[3], stride=2),  # layer4输出通道512
                pool=nn.AvgPool2d(4),
                flatten = nn.Flatten(),
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                fc=nn.Linear(512 * block.expansion, NUM_CLASSES[dataset])  # 根据最后一层输出通道数更新
            )
        )
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def register_hooks(self):
        self.activations = {}
        # 定义钩子函数来存储每一层的激活图
        def hook_fn(module, input, output):
            self.activations[module] = output
        # 注册钩子，只针对layer1, layer2, layer3, layer4
        self.base.layer1.register_forward_hook(hook_fn)
        self.base.layer2.register_forward_hook(hook_fn)
        self.base.layer3.register_forward_hook(hook_fn)
        self.base.layer4.register_forward_hook(hook_fn)

    def get_activations(self, x):
        # 前向传播并返回激活图
        self.activations = {}  # 清空之前保存的激活图
        self.register_hooks()  # 注册钩子
        # 执行前向传播
        _ = self.base(x)
        # 将激活图赋值给指定的变量
        attention_map1 = self.activations.get(self.base.layer1, None)
        attention_map2 = self.activations.get(self.base.layer2, None)
        attention_map3 = self.activations.get(self.base.layer3, None)
        attention_map4 = self.activations.get(self.base.layer4, None)
        # 返回这些变量
        return attention_map1, attention_map2, attention_map3, attention_map4

class PResNet(DecoupledModel):
    def __init__(self, block, num_blocks, dataset:str,pretrained=False):
        super(PResNet, self).__init__()
        self.in_planes = 64  # 初始化 in_planes 为 64
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 64, kernel_size=3, stride=1, padding=1, bias=False),  # 修改为64
                bn1=nn.BatchNorm2d(64),
                activation1=nn.ReLU(),
                layer1=self._make_layer(block, 64, num_blocks[0], stride=1),  # layer1的输入和输出通道为64
                layer2=self._make_layer(block, 128, num_blocks[1], stride=2),
                layer3=self._make_layer(block, 256, num_blocks[2], stride=2),
                layer4=self._make_layer(block, 512, num_blocks[3], stride=2),  # layer4输出通道512
                pool=nn.AvgPool2d(4),
                flatten = nn.Flatten(),
            )
        )

        self.project = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(512, 256)),  # 输入维度为 512 * block.expansion
            ('relu_1', nn.ReLU()),
            ('linear_2', nn.Linear(256, 512)),  # 输出维度为 512
        ]))
        # self.project = nn.Sequential(OrderedDict([
        #     ('linear_1', nn.Linear(512, 512)),  # 输入维度为 512 * block.expansion
        # ]))

        self.classifier = nn.Sequential(
            OrderedDict(
                fc=nn.Linear(512, NUM_CLASSES[dataset])  # 根据最后一层输出通道数更新
            )
        )
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor,flag=False):
        base = self.base(x)
        project = self.project(base)
        x = self.classifier(project)
        if flag:
            return x, project
        else:
            return x

def ResNet10(dataset=None,pretrained=False):
    return resnet(BasicBlock, [1,1,1,1],dataset=dataset,pretrained=pretrained)
def ResNet18(dataset=None,pretrained=False):
    return resnet(BasicBlock, [2,2,2,2], dataset=dataset,pretrained=pretrained)
def ResNet34(dataset=None,pretrained=False):
    return resnet(BasicBlock, [3,4,6,3],dataset=dataset,pretrained=pretrained)
def ResNet50(dataset=None,pretrained=False):
    return resnet(Bottleneck, [3,4,6,3],dataset=dataset,pretrained=pretrained)
def ResNet101(dataset=None,pretrained=False):
    return resnet(Bottleneck, [3,4,23,3],dataset=dataset,pretrained=pretrained)
def ResNet152(dataset=None,pretrained=False):
    return resnet(Bottleneck, [3,8,36,3],dataset=dataset,pretrained=pretrained)

def PResNet18(dataset=None,pretrained=False):
    return PResNet(BasicBlock, [2,2,2,2], dataset=dataset,pretrained=pretrained)

# ---- 1. 配置表 -------------------------------------------------------------
cfgs = {
    # 说明：数字 = out_channels，'M' = MaxPool(2,2)
    "11": [64, 'M', 128, 'M',
           256, 256, 'M',
           512, 512, 'M',
           512, 512, 'M'],
    "13": [64, 64, 'M',
           128, 128, 'M',
           256, 256, 'M',
           512, 512, 'M',
           512, 512, 'M'],
    "16": [64, 64, 'M',
           128, 128, 'M',
           256, 256, 256, 'M',
           512, 512, 512, 'M',
           512, 512, 512, 'M'],
    "19": [64, 64, 'M',
           128, 128, 'M',
           256, 256, 256, 256, 'M',
           512, 512, 512, 512, 'M',
           512, 512, 512, 512, 'M'],
}

# ---- 2. 动态构建卷积模块 ---------------------------------------------------
# def make_layers(cfg):
#     layers, in_ch = [], 3
#     for v in cfg:
#         if v == 'M':
#             layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         else:
#             layers.extend([
#                 nn.Conv2d(in_ch, v, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(v),
#                 nn.ReLU(inplace=True),
#             ])
#             in_ch = v
#     return nn.Sequential(*layers)

def make_layers(cfg):
    layers, in_ch = [], 3
    first_dropout = True  # 用来标记是否是第一次加入Dropout
    for i, v in enumerate(cfg):
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            # 判断当前卷积层后面是否是MaxPool层
            if i+1 < len(cfg) and cfg[i+1] != 'M':  # 如果下一个不是'M'，则加入Dropout
                dropout_rate = 0.3 if first_dropout else 0.4  # 第一次加0.3，之后加0.4
                layers.extend([
                    nn.Conv2d(in_ch, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate)
                ])
                first_dropout = False  # 设置为False，表示后续的Dropout都是0.4
            else:  # 如果下一个是'M'，不加Dropout
                layers.extend([
                    nn.Conv2d(in_ch, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ])
            in_ch = v
    return nn.Sequential(*layers)

# ---- 3. 通用 VGG 类 --------------------------------------------------------
class VGG(DecoupledModel):                     # 若需要保持 DecoupledModel，请改成继承 DecoupledModel
    def __init__(self, version: str, dataset: str, pretrained: bool = False):
        super(VGG,self).__init__()
        if version not in cfgs:
            raise ValueError(f"Unsupported VGG version {version}. "
                             f"Choose from {list(cfgs.keys())}.")
        # 卷积骨干
        self.base = nn.Sequential(
            make_layers(cfgs[version]),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, NUM_CLASSES[dataset])
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)
        return self.classifier(x)


# NOTE: You can build your custom model here.
# What you only need to do is define the architecture in __init__().
# Don't need to consider anything else, which are handled by DecoupledModel well already.
# Run `python *.py -m custom` to use your custom model.
class CustomModel(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        # You need to define:
        # 1. self.base (the feature extractor part)
        # 2. self.classifier (normally the final fully connected layer)
        # The default forwarding process is: out = self.classifier(self.base(input))
        pass


MODELS = {
    "custom": CustomModel,
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "ConvNet":ConvNet,
    "PConvNet":PConvNet,
    "2nn": TwoNN,
    "resnet10":ResNet10,
    "resnet18":ResNet18,
    "resnet34":ResNet34,
    "resnet50":ResNet50,
    "resnet101":ResNet101,
    "resnet152":ResNet152,
    "PResNet18":PResNet18,
    "vgg11": partial(VGG, version="11"),
    "vgg13": partial(VGG, version="13"),
    "vgg16": partial(VGG, version="16"),
    "vgg19": partial(VGG, version="19"),

}