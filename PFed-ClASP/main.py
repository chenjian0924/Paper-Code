import importlib
import os
import sys
import inspect
from pathlib import Path

import yaml
import pynvml

from src.server.fedavg import FedAvgServer

# 便于在代码中能够正确地引用和使用该路径下的模块或文件
FLBENCH_ROOT = Path(__file__).parent.absolute()
if FLBENCH_ROOT not in sys.path:
    sys.path.append(FLBENCH_ROOT.as_posix())


from src.utils.tools import parse_args

if __name__ == "__main__":
    # 检查命令行参数的数量
    if len(sys.argv) < 2:
        raise RuntimeError(
            "No method is specified. Run like `python main.py <method> [config_file_relative_path] [cli_method_args ...]`,",
            "e.g., python main.py fedavg config/template.yml`",
        )
    # 获取方法名
    method_name = sys.argv[1]

    # 初始化配置文件路径和命令行方法参数
    config_file_path = None
    cli_method_args = []

    # 检查命令行参数数量是否大于2
    if len(sys.argv) > 2:
        # 检查第二个参数是否为 .yaml 或 .yml 文件
        if ".yaml" in sys.argv[2] or ".yml" in sys.argv[2]:  # ***.yml or ***.yaml
            config_file_path = Path(sys.argv[2]).absolute()
            cli_method_args = sys.argv[3:]
        else:
            cli_method_args = sys.argv[2:]
    # 尝试导入指定的方法模块
    try:
        fl_method_server_module = importlib.import_module(f"src.server.{method_name}")
    except:
        raise ImportError(f"Can't import `src.server.{method_name}`.")

    # 获取模块中的所有属性
    module_attributes = inspect.getmembers(fl_method_server_module)
    # 查找与方法名对应的服务器类
    server_class = [
        attribute
        for attribute in module_attributes
        if attribute[0].lower() == method_name + "server"
    ][0][1]

    # 获取服务器类中的 get_hyperparams 方法（如果存在）
    get_method_hyperparams_func = getattr(server_class, f"get_hyperparams", None)

    # 初始化配置文件参数
    config_file_args = None

    # 如果配置文件路径存在且是文件，读取配置文件
    if config_file_path is not None and os.path.isfile(config_file_path):
        with open(config_file_path, "r") as f:
            try:
                config_file_args = yaml.safe_load(f)
            except:
                raise TypeError(
                    f"Config file's type should be yaml, now is {config_file_path}"
                )

    # 解析参数
    ARGS = parse_args(
        config_file_args, method_name, get_method_hyperparams_func, cli_method_args
    )

    # target method is not inherited from FedAvgServer
    # 检查 server_class 是否没有从 FedAvgServer 继承
    if server_class.__bases__[0] != FedAvgServer and server_class != FedAvgServer:
        # 获取 server_class 的第一个父类
        parent_server_class = server_class.__bases__[0]
        # 获取父类中的 get_hyperparams 方法，如果存在的话
        get_parent_method_hyperparams_func = getattr(
            parent_server_class, f"get_hyperparams", None
        )
        # class name: ***Server, only want ***
        # 获取父类类名，并将其转换为小写，并去掉末尾的 "Server" 部分
        parent_method_name = parent_server_class.__name__.lower()[:-6]
        # extract the hyperparams of parent method
        # 解析并提取父类方法的超参数
        PARENT_ARGS = parse_args(
            config_file_args,
            parent_method_name,
            get_parent_method_hyperparams_func,
            cli_method_args,
        )
        # 将提取到的父类方法参数设置到 ARGS 中
        setattr(ARGS, parent_method_name, getattr(PARENT_ARGS, parent_method_name))
    # 检查是否为并行模式
    if ARGS.mode == "parallel":
        import ray

        # 获取并行模式下可用的 GPU 和 CPU 数量
        num_available_gpus = ARGS.parallel.num_gpus
        num_available_cpus = ARGS.parallel.num_cpus
        # 如果未指定可用的 GPU 数量，则进行检测
        if num_available_gpus is None:
            pynvml.nvmlInit()
            num_total_gpus = pynvml.nvmlDeviceGetCount()# 获取系统中总的 GPU 数量
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
                num_available_gpus = min(
                    len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), num_total_gpus
                )
            else:
                num_available_gpus = num_total_gpus
        # 如果未指定可用的 CPU 数量，则获取系统的 CPU 数量
        if num_available_cpus is None:
            num_available_cpus = os.cpu_count()

        # 尝试初始化 Ray 集群
        try:
            ray.init(
                address=ARGS.parallel.ray_cluster_addr,# Ray 集群地址
                namespace=method_name,# 命名空间
                num_cpus=num_available_cpus,# 可用的 CPU 数量
                num_gpus=num_available_gpus,# 可用的 GPU 数量
                ignore_reinit_error=True,# 忽略重新初始化错误
            )
        except ValueError:
            # have existing cluster
            # then no pass num_cpus and num_gpus
            # 如果已经存在集群，则不传递 num_cpus 和 num_gpus 参数
            ray.init(
                address=ARGS.parallel.ray_cluster_addr,
                namespace=method_name,
                ignore_reinit_error=True,
            )
        # 获取集群资源
        cluster_resources = ray.cluster_resources()
        ARGS.parallel.num_cpus = cluster_resources["CPU"]
        ARGS.parallel.num_gpus = cluster_resources["GPU"]

    server = server_class(args=ARGS)
    server.run()
