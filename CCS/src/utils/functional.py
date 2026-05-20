import os
import random
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np
import pynvml
import torch
from omegaconf import DictConfig
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.constants import DEFAULTS
from src.utils.metrics import Metrics
from src.attack.badnet import badnet
from src.attack.dba import dba
import torch.nn.functional as F


def fix_random_seed(seed: int, use_cuda=False) -> None:
    """Fix the random seed of FL training.

    Args:
        seed: Any number you like as the random seed.
        use_cuda: Flag indicates if using cuda.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available() and use_cuda:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimal_cuda_device(use_cuda: bool) -> torch.device:
    """Dynamically select CUDA device (has the most memory) for running FL
    experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
    if not torch.cuda.is_available() or not use_cuda:
        return torch.device("cpu")
    pynvml.nvmlInit()
    gpu_memory = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        gpu_ids = range(torch.cuda.device_count())

    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(memory_info.free)
    gpu_memory = np.array(gpu_memory)
    best_gpu_id = np.argmax(gpu_memory)
    return torch.device(f"cuda:{best_gpu_id}")


def vectorize(
    src: OrderedDict[str, torch.Tensor] | list[torch.Tensor] | torch.nn.Module,
    detach=True,
) -> torch.Tensor:
    """Vectorize(Flatten) and concatenate all tensors in `src`.

    Args:
        `src`: The source of tensors.
        `detach`: Set as `True` to return `tensor.detach().clone()`. Defaults to `True`.

    Returns:
        The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict) or isinstance(src, dict):
        return torch.cat([func(param).flatten() for param in src.values()])
    elif isinstance(src, torch.nn.Module):
        return torch.cat([func(param).flatten() for param in src.state_dict().values()])
    elif isinstance(src, Iterator):
        return torch.cat([func(param).flatten() for param in src])


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
    model_in_train_mode: bool = False,
) -> Metrics:
    """For evaluating the `model` over `dataloader` and return metrics.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").
        model_in_eval_mode (bool, optional): Set as `True` to switch model to eval mode. Defaults to `True`.

    Returns:
        Metrics: The metrics objective.
    """
    if model_in_train_mode:
        model.train()
    else:
        model.eval()
    model.to(device)
    metrics = Metrics()
    for x, y in dataloader:
        if len(x) <= 1:
            continue
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics

@torch.no_grad()
def evaluate_model_backdoor(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
    trigger=None,
    trigger_size=None,
    trigger_location=None,
    dataset=None,
    poison_method=None,
    generator = None,
    model_in_train_mode: bool = False,
) -> Metrics:
    if model_in_train_mode:
        model.train()
    else:
        model.eval()
    model.to(device)
    metrics = Metrics()
    generator.eval()
    for x, y in dataloader:
        if len(x) <= 1:
            continue
        if poison_method == 'badpfl':
            x, y = x.to(device), y.to(device)
            with torch.enable_grad():
                x, y = BadPFL(model, generator, x, y,dataset)
        elif poison_method == 'pfedba':
            x, y = x.to(device), y.to(device)
            x, y, poison_count = badnet(x, y, trigger=trigger, trigger_size=trigger_size, trigger_location=trigger_location,is_test=True, dataset=dataset)
        else:
            x, y = x.to(device), y.to(device)
            x, y, poison_count = dba(x, y, clientID=1, poison_client_ids=[1,15,23,37,46,55,69,75,97,84], is_test=True,dataset=dataset, attack='bapfl')

        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics

def pgd_attack(model, images, labels, epsilon=4. / 255., alpha=4. / 255., num_iter=1):
    adv_images = images.clone().detach() + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0, max=1)
    for _ in range(num_iter):
        torch.cuda.empty_cache()
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_images = adv_images + alpha * torch.sign(adv_images.grad)
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + eta, min=0, max=1)
        adv_images = adv_images.detach().clone()
        torch.cuda.empty_cache()
    return adv_images.detach().clone()


def BadPFL(model,trigger_gen, data,label, dataset,target_label=0, poison_ratio=1):
    # 根据给定的中毒比例（poison_ratio）生成中毒掩码
    poison_mask = torch.rand(label.size(0), device=label.device) <= poison_ratio
    # 如果没有数据需要中毒，返回原始数据
    if poison_mask.sum().item() == 0:
        return data, label
    else:
        # 克隆数据并准备设置中毒标签为目标标签
        poison_data, poison_label = data.clone(), torch.full([label.size(0), ], target_label,device=label.device)
    # 使用PGD攻击生成对抗样本并进行扰动
    poison_data = pgd_attack(model,poison_data,label)
    poison_data = poison_data.detach().clone()
    # 生成触发器
    gen_trigger = trigger_gen(data) / 255. * 4.
    if dataset == 'mnist' or dataset == 'fmnist':
        if gen_trigger.size(2) != data.size(2) or gen_trigger.size(3) != data.size(3):
            gen_trigger = F.interpolate(gen_trigger, size=(data.size(2), data.size(3)), mode='bilinear', align_corners=False)
    # 根据中毒掩码调整数据，只有中毒的数据会加上触发器
    poison_data = poison_mask.view(-1, 1, 1, 1).float() * (poison_data + gen_trigger) + (~poison_mask.view(-1, 1, 1, 1)).float() * data
    # 对应的标签设置为目标标签
    poison_label = poison_mask.float() * poison_label + (~poison_mask).float() * label
    # 返回中毒数据和标签
    return poison_data, poison_label.to(torch.long)



def parse_args(
    config: DictConfig,
    method_name: str,
    get_method_args_func: Callable[[Sequence[str] | None], Namespace] | None,
) -> DictConfig:
    """Purge arguments from default args dict, config file and CLI and produce
    the final arguments.

    Args:
        config: DictConfig set from .yaml config file.
        method_name: The FL method's name.
        get_method_args_func: The callable function of parsing FL method `method_name`'s spec arguments.
    Returns:
        DictConfig: The final argument namespace.
    """
    final_args = DictConfig(DEFAULTS)

    def _merge_configs(defaults: DictConfig, config: DictConfig) -> DictConfig:
        merged = DictConfig({})
        for key, default_value in defaults.items():
            if key in config:
                if isinstance(default_value, DictConfig) and isinstance(
                    config[key], DictConfig
                ):
                    merged[key] = _merge_configs(default_value, config[key])
                else:
                    merged[key] = config[key]
            else:
                merged[key] = default_value
        return merged

    final_args = _merge_configs(final_args, config)

    if hasattr(config, method_name):
        final_args[method_name] = config[method_name]

    if get_method_args_func is not None:
        default_method_args = DictConfig(get_method_args_func([]).__dict__)
        if hasattr(final_args, method_name):
            for key in default_method_args.keys():
                if key not in final_args[method_name].keys():
                    final_args[method_name][key] = default_method_args[key]
        else:
            final_args[method_name] = default_method_args

    assert final_args.mode in [
        "serial",
        "parallel",
    ], f"Unrecongnized mode: {final_args.mode}"
    if final_args.mode == "parallel":
        import ray

        num_available_gpus = final_args.parallel.num_gpus
        num_available_cpus = final_args.parallel.num_cpus
        if num_available_gpus is None:
            pynvml.nvmlInit()
            num_total_gpus = pynvml.nvmlDeviceGetCount()
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
                num_available_gpus = min(
                    len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), num_total_gpus
                )
            else:
                num_available_gpus = num_total_gpus
        if num_available_cpus is None:
            num_available_cpus = os.cpu_count()

        try:
            ray.init(
                address=config.parallel.ray_cluster_addr,
                namespace=method_name,
                num_cpus=num_available_cpus,
                num_gpus=num_available_gpus,
                ignore_reinit_error=True,
            )
        except ValueError:
            # have existing cluster
            # then ignore num_cpus and num_gpus
            ray.init(
                address=config.parallel.ray_cluster_addr,
                namespace=method_name,
                ignore_reinit_error=True,
            )

        cluster_resources = ray.cluster_resources()
        final_args.parallel.num_cpus = cluster_resources["CPU"]
        final_args.parallel.num_gpus = cluster_resources["GPU"]
        if final_args.parallel.num_workers < 2:
            print(
                f"num_workers is less than 2: {final_args.parallel.num_workers}, "
                "mode fallbacks to serial."
            )
            final_args.mode = "serial"
            del final_args.parallel

    return final_args


def initialize_data_loaders(
    dataset: Dataset,
    data_indices: List[Dict[str, List[int]]],
    batch_size: int = 32,
    **dataloader_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Subset, Subset, Subset]:
    """Initialize data loaders for training, validation, and testing.

    Args:
        dataset: The dataset to be used for creating subsets.
        data_indices: A list of dictionaries, where each dictionary contains
            the indices for 'train', 'val', and 'test' splits for a client.
        batch_size: The batch size for the data loaders. Defaults to 32.
        **dataloader_kwargs: Additional keyword arguments for the data loaders.

    Returns:
        A tuple containing:
        - trainloader: DataLoader for the training set.
        - testloader: DataLoader for the test set.
        - valloader: DataLoader for the validation set.
        - trainset: Subset of the dataset for training.
        - testset: Subset of the dataset for testing.
        - valset: Subset of the dataset for validation.
    """
    val_indices = np.concatenate(
        [client_i_indices["val"] for client_i_indices in data_indices]
    )
    test_indices = np.concatenate(
        [client_i_indices["test"] for client_i_indices in data_indices]
    )
    train_indices = np.concatenate(
        [client_i_indices["train"] for client_i_indices in data_indices]
    )

    valset = Subset(dataset, val_indices)
    testset = Subset(dataset, test_indices)
    trainset = Subset(dataset, train_indices)

    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, **dataloader_kwargs
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **dataloader_kwargs
    )

    return trainloader, testloader, valloader, trainset, testset, valset
