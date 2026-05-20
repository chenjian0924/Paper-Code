import json
import os
import random
from argparse import Namespace
from collections import OrderedDict
from typing import Callable, Iterator, Sequence, Union
from pathlib import Path

import torch
import pynvml
import numpy as np
from torch.utils.data import DataLoader
from rich.console import Console

from src.utils.metrics import Metrics
from src.utils.constants import DEFAULT_COMMON_ARGS, DEFAULT_PARALLEL_ARGS

from src.badnet.badnet_batch import MNISTPoison,CIFAR10Poison
from src.BapFL.BapFL_poison import get_poison_batch
from src.pFedBA.generate_trigger import get_blend_batch
# 固定随机种子
def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimal_cuda_device(use_cuda: bool) -> torch.device:
    """Dynamically select CUDA device (has the most memory) for running FL experiment.

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
        assert max(gpu_ids) < torch.cuda.device_count()
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
def evalutate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
) -> Metrics:
    model.eval()
    model.to(device)
    metrics = Metrics()
    for x, y in dataloader:

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics

@torch.no_grad()
def bapfl_evalutate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
    dataset=None,
    is_orginalpfedba = False
) -> Metrics:
    model.eval()
    model.to(device)
    metrics = Metrics()
    for x, y in dataloader:
        if dataset == 'mnist' or dataset == 'fmnist' or dataset == 'emnist':
            x, y, poison_count = get_poison_batch(x, y, is_test=True, is_cifar10=False,is_orginalpfedba = is_orginalpfedba)
        elif dataset == 'cifar10':
            x, y, poison_count = get_poison_batch(x, y, is_test=True, is_cifar10=True,is_orginalpfedba = is_orginalpfedba)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics

@torch.no_grad()
def pFedBA_evalutate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
    dataset=None,
    trigger = None,
    is_orginalpfedba = False
) -> Metrics:
    model.eval()
    model.to(device)
    metrics = Metrics()
    for x, y in dataloader:
        if dataset == 'mnist' or dataset == 'fmnist' or dataset == 'emnist':
            x, y, poison_count = get_blend_batch(x, y, trigger,is_test=True, is_cifar10=False,is_orginalpfedba = is_orginalpfedba)
        elif dataset == 'cifar10':
            x, y, poison_count = get_blend_batch(x, y, trigger,is_test=True, is_cifar10=True,is_orginalpfedba = is_orginalpfedba)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics

def parse_args(
    config_file_args: dict | None,
    method_name: str,
    get_method_args_func: Callable[[Sequence[str] | None], Namespace] | None,
    method_args_list: list[str],
) -> Namespace:
    """Purge arguments from default args dict, config file and CLI and produce the final arguments.

    Args:
        config_file_args (Union[dict, None]): Argument dictionary loaded from user-defined `.yml` file. `None` for unspecifying.
        method_name (str): The FL method's name.
        get_method_args_func (Union[ Callable[[Union[Sequence[str], None]], Namespace], None ]): The callable function of parsing FL method `method_name`'s spec arguments.
        method_args_list (list[str]): FL method `method_name`'s specified arguments set on CLI.

    Returns:
        NestedNamespace: The final argument namespace.
    """
    ARGS = dict(
        mode="serial", common=DEFAULT_COMMON_ARGS, parallel=DEFAULT_PARALLEL_ARGS
    )
    # 更新ARGS中的common和parallel参数，基于用户定义的配置文件
    if config_file_args is not None:
        if "common" in config_file_args.keys():
            ARGS["common"].update(config_file_args["common"])
        if "parallel" in config_file_args.keys():
            ARGS["parallel"].update(config_file_args["parallel"])
        if "mode" in config_file_args.keys():
            ARGS["mode"] = config_file_args["mode"]

    # 更新特定方法的参数，基于CLI输入和配置文件
    if get_method_args_func is not None:
        default_method_args = get_method_args_func([]).__dict__
        config_file_method_args = {}
        if config_file_args is not None:
            config_file_method_args = config_file_args.get(method_name, {})
        cli_method_args = get_method_args_func(method_args_list).__dict__

        # 提取在CLI中显式设置的参数
        for key in default_method_args.keys():
            if default_method_args[key] == cli_method_args[key]:
                cli_method_args.pop(key)

        # 对于相同参数，优先级为CLI > 配置文件 > 默认值
        method_args = default_method_args
        for key in default_method_args.keys():
            if key in cli_method_args.keys():
                method_args[key] = cli_method_args[key]
            elif key in config_file_method_args.keys():
                method_args[key] = config_file_method_args[key]

        ARGS[method_name] = method_args
    # 确保模式是 "serial" 或 "parallel"
    assert ARGS["mode"] in ["serial", "parallel"], f"Unrecongnized mode: {ARGS['mode']}"
    if ARGS["mode"] == "parallel":
        if ARGS["parallel"]["num_workers"] < 2:
            print(
                f"num_workers is less than 2: {ARGS['parallel']['num_workers']} and mode is fallback to serial."
            )
            ARGS["mode"] = "serial"
            del ARGS["parallel"]
    return NestedNamespace(ARGS)


class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """This class is for solving the incompatibility between the progress bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """
        self.stdout = stdout
        self.logfile_stream = None
        self.enable_log = enable_log

        # 如果启用了日志记录功能，则打开日志文件并创建日志记录器
        if self.enable_log:
            self.logfile_stream = open(logfile_path, "w")
            self.logger = Console(
                file=self.logfile_stream, record=True, log_path=False, log_time=False
            )

    def log(self, *args, **kwargs):
        # 打印日志信息到标准输出
        self.stdout.log(*args, **kwargs)
        # 如果启用了日志记录功能，则打印日志信息到日志文件
        if self.enable_log:
            self.logger.log(*args, **kwargs)

    def close(self):
        if self.logfile_stream:
            self.logfile_stream.close()


class NestedNamespace(Namespace):
    def __init__(self, args_dict: dict):
        # 递归地将字典转换为 NestedNamespace 对象
        super().__init__(
            **{
                key: self._nested_namespace(value) if isinstance(value, dict) else value
                for key, value in args_dict.items()
            }
        )

    def _nested_namespace(self, dictionary):
        return NestedNamespace(dictionary)

    def to_dict(self):
        # 递归地将 NestedNamespace 对象转换为字典
        return {
            key: (value.to_dict() if isinstance(value, NestedNamespace) else value)
            for key, value in self.__dict__.items()
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=False)
