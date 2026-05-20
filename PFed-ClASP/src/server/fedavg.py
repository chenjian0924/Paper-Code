import functools
import inspect
import pickle
import json
import os
import time
import random
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import ray
import torch
import numpy as np
from torchvision import transforms
from rich.console import Console
from rich.progress import track
from rich.json import JSON

from src.utils.models import MODELS, DecoupledModel
from src.utils.metrics import Metrics
from src.client.fedavg import FedAvgClient
from src.utils.constants import (
    FLBENCH_ROOT,
    LR_SCHEDULERS,
    OPTIMIZERS,
    OUT_DIR,
    DATA_MEAN,
    DATA_STD,
)
from src.utils.trainer import FLbenchTrainer
from data.utils.datasets import DATASETS, BaseDataset
from src.utils.tools import (
    Logger,
    NestedNamespace,
    fix_random_seed,
    get_optimal_cuda_device,
)

from src.Pruning.Pruning import Pruning
from torch.utils.data import Subset, DataLoader
from PIL import Image
class FedAvgServer:
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedAvg",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=False,
    ):
        self.args = args
        self.algo = algo
        self.unique_model = unique_model
        self.return_diff = return_diff
        self.trigger_optimizer = self.get_trigger()
        start_time = str(
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(round(time.time())))
        )
        self.output_dir = OUT_DIR / self.algo / start_time
        with open(
                FLBENCH_ROOT / "data" / self.args.common.dataset / "args.json", "r"
        ) as f:
            self.args.dataset = NestedNamespace(json.load(f))
        try:
            partition_path = (
                    FLBENCH_ROOT / "data" / self.args.common.dataset / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")
        self.train_clients: list[int] = partition["separation"]["train"]
        self.test_clients: list[int] = partition["separation"]["test"]
        self.val_clients: list[int] = partition["separation"]["val"]
        self.client_num: int = partition["separation"]["total"]
        self.device = get_optimal_cuda_device(self.args.common.use_cuda)
        self.model: DecoupledModel = MODELS[self.args.common.model](
            dataset=self.args.common.dataset
        )
        self.model.check_and_preprocess(self.args)
        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        self.public_model_params: OrderedDict[str, torch.Tensor] = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

        if self.args.common.external_model_params_file is not None:
            file_path = str(
                (FLBENCH_ROOT / self.args.common.external_model_params_file).absolute()
            )
            if os.path.isfile(file_path) and file_path.find(".pt") != -1:
                external_params = torch.load(file_path, map_location="cpu")
                self.public_model_params.update(external_params)
            elif not os.path.isfile(file_path):
                raise FileNotFoundError(f"{file_path} is not a valid file path.")
            elif file_path.find(".pt") == -1:
                raise TypeError(f"{file_path} is not a valid .pt file.")

        self.clients_personal_model_params = {i: {} for i in range(self.client_num)}

        if self.args.common.buffers == "local":
            _init_buffers = OrderedDict(self.model.named_buffers())
            for i in range(self.client_num):
                self.clients_personal_model_params[i] = deepcopy(_init_buffers)

        if self.unique_model:
            for params_dict in self.clients_personal_model_params.values():
                params_dict.update(deepcopy(self.model.state_dict()))
        self.clients_optimizer_state = {i: {} for i in range(self.client_num)}
        self.clients_lr_scheduler_state = {i: {} for i in range(self.client_num)}


        self.clients_local_epoch: list[int] = [
                                                  self.args.common.local_epoch
                                              ] * self.client_num

        if (
                self.args.common.straggler_ratio > 0
                and self.args.common.local_epoch
                > self.args.common.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.common.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.clients_local_epoch = [self.args.common.local_epoch] * (
                normal_num
            ) + random.choices(
                range(
                    self.args.common.straggler_min_local_epoch,
                    self.args.common.local_epoch,
                ),
                k=straggler_num,
            )
            random.shuffle(self.clients_local_epoch)

        if self.args.common.select_client == 1:

            fixed_clients = [1,15,21,35,42,56,63,76,88,92]
            self.client_sample_stream = []
            for _ in range(self.args.common.global_epoch):
                random_fixed_clients = random.sample(fixed_clients, min(len(fixed_clients), random.randint(2, 2)))
                random_non_fixed_clients = random.sample(
                    [client for client in self.train_clients if client not in fixed_clients],
                    max(1, int(self.client_num * self.args.common.join_ratio) - len(random_fixed_clients))
                )
                self.client_sample_stream.append(random_fixed_clients + random_non_fixed_clients)
        else:
            fixed_clients = [1,15]
            self.client_sample_stream = [
                fixed_clients + random.sample(
                    [client for client in self.train_clients if client not in fixed_clients],
                    max(1, int(self.client_num * self.args.common.join_ratio) - len(fixed_clients))
                )
                for _ in range(self.args.common.global_epoch)
            ]


        self.selected_clients: list[int] = []
        self.current_epoch = 0
        self.testing = False

        if not os.path.isdir(self.output_dir) and (
            self.args.common.save_log
            or self.args.common.save_fig
            or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        self.clients_metrics = {i: {} for i in self.train_clients}
        self.global_metrics = {
            "before": {"train": [], "val": [], "test": []},
            "after": {"train": [], "val": [], "test": []},
            "before_poison": {"train": [], "val": [], "test": []},
            "after_poison": {"train": [], "val": [], "test": []}
        }

        self.verbose = False
        stdout = Console(log_path=False, log_time=False, soft_wrap=True)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.common.save_log,
            logfile_path=OUT_DIR
            / self.algo
            / self.output_dir
            / f"{self.args.common.dataset}.log",
        )
        self.test_results: dict[int, dict[str, dict[str, Metrics]]] = {}
        self.train_progress_bar = track(
            range(self.args.common.global_epoch),
            "[bold green]Training...",
            console=stdout,
        )

        self.logger.log("=" * 20, self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:")
        self.logger.log(JSON(str(self.args)))

        if self.args.common.visible is not None:
            self.monitor_window_name_suffix = (
                self.args.dataset.monitor_window_name_suffix
            )

        if self.args.common.visible == "visdom":
            from visdom import Visdom

            self.viz = Visdom()
        elif self.args.common.visible == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=self.output_dir)
            self.tensorboard.add_text(
                f"ExperimentalArguments-{self.monitor_window_name_suffix}",
                f"<pre>{self.args}</pre>",
            )
        # init trainer
        self.trainer: FLbenchTrainer = None
        if use_fedavg_client_cls:
            self.init_trainer()

    def init_trainer(self, fl_client_cls=FedAvgClient, **extras):
        if self.args.mode == "serial" or self.args.parallel.num_workers < 2:
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=fl_client_cls,
                mode="serial",
                num_workers=0,
                init_args=dict(
                    model=deepcopy(self.model),
                    optimizer_cls=self.get_client_optimizer(),
                    lr_scheduler_cls=self.get_client_lr_scheduler(),
                    args=self.args,
                    dataset=self.get_client_dataset(),
                    data_indices=self.data_indices,
                    device=self.device,
                    return_diff=self.return_diff,
                    trainset_pFedBA=self.get_merged_dataset(self.args.common.poison_client_ids),
                    **extras,
                ),
            )

    def get_client_dataset(self) -> BaseDataset:
        try:
            partition_path = (
                FLBENCH_ROOT / "data" / self.args.common.dataset / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(
                f"Please partition {self.args.common.dataset} first."
            )
        self.data_indices: list[dict[str, list[int]]] = partition["data_indices"]
        dataset: BaseDataset = DATASETS[self.args.common.dataset](
            root=FLBENCH_ROOT / "data" / self.args.common.dataset,
            args=self.args.dataset,
            **self.get_dataset_transforms(),
        )

        return dataset

    def get_merged_dataset(self, client_ids):
        dataset = self.get_client_dataset()
        merged_indices = []
        for client_id in client_ids:
            merged_trainset = Subset(dataset, merged_indices)
        return merged_trainset
    def get_trigger(self):
        if self.args.common.dataset == 'mnist' or self.args.common.dataset == 'fmnist' or self.args.common.dataset == 'emnist':
            given_image_path = 'src/badnet/trigger_white.png'
            given_image = Image.open(given_image_path).convert('RGB')
            if self.args.common.is_orginalpfedba:
                given_image = given_image.resize((10,10))  # MNIST 的触发器大小为 10X10
            else:
                given_image = given_image.resize((5, 5))
        else:
            given_image_path = 'src/badnet/trigger_10.png'
            given_image = Image.open(given_image_path).convert('RGB')
            if self.args.common.is_orginalpfedba:
                given_image = given_image.resize((8, 8))  # CIFAR 的触发器大小为 8X8
            else:
                given_image = given_image.resize((5, 5))
        transform = transforms.ToTensor()
        given_image_tensor = transform(given_image)
        given_image_tensor = given_image_tensor.cuda()
        return given_image_tensor
    def get_dataset_transforms(self):
        test_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.common.dataset],
                    DATA_STD[self.args.common.dataset],
                )
            ]
            if self.args.common.dataset in DATA_MEAN
            and self.args.common.dataset in DATA_STD
            else []
        )
        test_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.common.dataset],
                    DATA_STD[self.args.common.dataset],
                )
            ]
            if self.args.common.dataset in DATA_MEAN
            and self.args.common.dataset in DATA_STD
            else []
        )
        train_target_transform = transforms.Compose([])
        return dict(
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
        )

    def get_client_optimizer(self):
        target_optimizer_cls: type[torch.optim.Optimizer] = OPTIMIZERS[
            self.args.common.optimizer.name
        ]
        _required_args = inspect.getfullargspec(target_optimizer_cls.__init__).args
        _opt_kwargs = {}
        for key, value in vars(self.args.common.optimizer).items():
            if key in _required_args:
                _opt_kwargs[key] = value
        optimizer = functools.partial(target_optimizer_cls, **_opt_kwargs)
        _opt_kwargs["name"] = self.args.common.optimizer.name
        self.args.common.optimizer = NestedNamespace(_opt_kwargs)
        return optimizer

    def get_client_lr_scheduler(self):
        try:
            lr_scheduler_args = getattr(self.args.common, "lr_scheduler")
            if lr_scheduler_args.name is not None:
                target_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] = (
                    LR_SCHEDULERS[lr_scheduler_args.name]
                )
                _required_args = inspect.getfullargspec(
                    target_scheduler_cls.__init__
                ).args

                _opt_kwargs = {}
                for key, value in vars(self.args.common.lr_scheduler).items():
                    if key in _required_args:
                        _opt_kwargs[key] = value

                lr_scheduler = functools.partial(target_scheduler_cls, **_opt_kwargs)
                _opt_kwargs["name"] = self.args.common.lr_scheduler.name
                self.args.common.lr_scheduler = NestedNamespace(_opt_kwargs)
                return lr_scheduler
        except:
            return None

    def train(self):
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E
            self.verbose = (self.current_epoch + 1) % self.args.common.verbose_gap == 0
            if self.verbose:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)
            if (E + 1) % self.args.common.test_interval == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            begin = time.time()
            self.train_one_round()
            end = time.time()
            self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )
        self.logger.log(
            f"{self.algo}'s average time taken by each global epoch: "
            f"{int(avg_round_time // 60)} min {(avg_round_time % 60):.2f} sec."
        )
    def train_one_round(self):
        clients_package = self.trainer.train()
        self.aggregate(clients_package)
    def package(self, client_id: int):
        return dict(
            client_id=client_id,
            local_epoch=self.clients_local_epoch[client_id],
            global_epoch=self.current_epoch,
            **self.get_client_model_params(client_id),
            optimizer_state=self.clients_optimizer_state[client_id],
            lr_scheduler_state=self.clients_lr_scheduler_state[client_id],
            return_diff=self.return_diff,
            trigger_optimizer = self.trigger_optimizer,
        )


    def test(self):
        self.testing = True
        clients = list(set(self.val_clients + self.test_clients))
        template = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "before_poison": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after_poison": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        if len(clients) > 0:
            if self.val_clients == self.train_clients == self.test_clients:
                results = {"all_clients": template}
                self.trainer.test(clients, results["all_clients"])
            else:
                results = {
                    "val_clients": deepcopy(template),
                    "test_clients": deepcopy(template),
                }
                if len(self.val_clients) > 0:
                    self.trainer.test(self.val_clients, results["val_clients"])
                #
                if len(self.test_clients) > 0:
                    self.trainer.test(self.test_clients, results["test_clients"])


            self.test_results[self.current_epoch + 1] = results

        self.testing = False


    def get_client_model_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:

        regular_params = deepcopy(self.public_model_params)
        # 获取指定客户端的个人模型参数
        personal_params = self.clients_personal_model_params[client_id]
        # 返回包含常规模型参数和个人模型参数的字典
        return dict(
            regular_model_params=regular_params, personal_model_params=personal_params
        )


    def show_convergence(self):

        colors = {
            "before": "blue",
            "after": "red",
            "train": "yellow",
            "val": "green",
            "test": "cyan",
            "before_poison": "purple",
            "after_poison": "brown",
        }
        self.logger.log("=" * 10, self.algo, "Convergence on train clients", "=" * 10)

        for stage in ["before", "after", "before_poison", "after_poison"]:
            for split in ["train", "val", "test"]:
                if len(self.global_metrics[stage][split]) > 0:
                    self.logger.log(
                        f"[{colors[split]}]{split}[/{colors[split]}] "
                        f"[{colors[stage]}]({stage} local training):"
                    )
                    acc_range = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
                    min_acc_idx = 10
                    max_acc = 0
                    accuracies = [
                        metrics.accuracy
                        for metrics in self.global_metrics[stage][split]
                    ]
                    for E, acc in enumerate(accuracies):
                        for i, target in enumerate(acc_range):
                            if acc >= target and acc > max_acc:
                                self.logger.log(f"{target}%({acc:.2f}%) at epoch: {E}")
                                max_acc = acc
                                min_acc_idx = i
                                break

                        acc_range = acc_range[:min_acc_idx]

    def log_info(self):
        for stage in ["before", "after","before_poison","after_poison"]:
            for split, flag in [
                ("train", self.args.common.eval_train),
                ("val", self.args.common.eval_val),
                ("test", self.args.common.eval_test),
            ]:
                if flag:
                    global_metrics = Metrics()
                    for i in self.selected_clients:
                        global_metrics.update(
                            self.clients_metrics[i][self.current_epoch][stage][split]
                        )
                    self.global_metrics[stage][split].append(global_metrics)
                    if self.args.common.visible == "visdom":
                        self.viz.line(
                            [global_metrics.accuracy],
                            [self.current_epoch],
                            win=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                            update="append",
                            name=self.algo,
                            opts=dict(
                                title=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                                xlabel="Communication Rounds",
                                ylabel="Accuracy",
                                legend=[self.algo],
                            ),
                        )
                    elif self.args.common.visible == "tensorboard":
                        self.tensorboard.add_scalar(
                            f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                            global_metrics.accuracy,
                            self.current_epoch,
                            new_style=True,
                        )

    def show_max_metrics(self):
        self.logger.log("=" * 20, self.algo, "Max Accuracy", "=" * 20)

        colors = {
            "before": "blue",
            "after": "red",
            "train": "yellow",
            "val": "green",
            "test": "cyan",
            "before_poison": "purple",
            "after_poison": "brown",
        }

        groups = ["val_clients", "test_clients"]
        if self.train_clients == self.val_clients == self.test_clients:
            groups = ["all_clients"]

        for group in groups:
            self.logger.log(f"{group}:")
            for stage in ["before", "after", "before_poison", "after_poison"]:
                for split, flag in [
                    ("train", self.args.common.eval_train),
                    ("val", self.args.common.eval_val),
                    ("test", self.args.common.eval_test),
                ]:
                    if flag:
                        metrics_list = list(
                            map(
                                lambda tup: (tup[0], tup[1][group][stage][split]),
                                self.test_results.items(),
                            )
                        )
                        if len(metrics_list) > 0:
                            epoch, max_acc = max(
                                [
                                    (epoch, metrics.accuracy)
                                    for epoch, metrics in metrics_list
                                ],
                                key=lambda tup: tup[1],
                            )
                            self.logger.log(
                                f"[{colors[split]}]({split})[/{colors[split]}] "
                                f"[{colors[stage]}]{stage}[/{colors[stage]}] "
                                f"fine - tuning: {max_acc:.2f}% at epoch {epoch}"
                            )

    def run(self):

        begin = time.time()
        self.train()
        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algo}'s total running time: "
            f"{int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log("=" * 20, self.algo, "Experiment Results:", "=" * 20)
        self.logger.log(
            "explain\n"
            "Format: [green](before local fine-tuning) -> [blue](after local fine-tuning)\n",
            "So if finetune_epoch = 0, x.xx% -> 0.00% is normal.",
        )
        all_test_results = {
            epoch: {
                group: {
                    split: {
                        "loss": f"{metrics['before'][split].loss:.4f} -> {metrics['after'][split].loss:.4f}",
                        "accuracy": f"{metrics['before'][split].accuracy:.2f}% -> {metrics['after'][split].accuracy:.2f}%",
                        "badnet_loss": f"{metrics['before_poison'][split].loss:.4f} -> {metrics['after_poison'][split].loss:.4f}",
                        "badnet_accuracy": f"{metrics['before_poison'][split].accuracy:.2f}% -> {metrics['after_poison'][split].accuracy:.2f}%",
                    }
                    for split, flag in [
                        ("train", self.args.common.eval_train),
                        ("val", self.args.common.eval_val),
                        ("test", self.args.common.eval_test),
                    ]
                    if flag
                }
                for group, metrics in results.items()
            }
            for epoch, results in self.test_results.items()
        }
        self.logger.log(all_test_results)
        if self.args.common.visible == "tensorboard":
            for epoch, results in all_test_results.items():
                self.tensorboard.add_text(
                    f"Results-{self.monitor_window_name_suffix}",
                    text_string=f"<pre>{results}</pre>",
                    global_step=epoch,
                )
        self.show_convergence()
        self.show_max_metrics()
        self.logger.close()
        if self.args.common.save_fig:
            import matplotlib
            from matplotlib import pyplot as plt

            matplotlib.use("Agg")
            linestyle = {
                "before": {"train": "dotted", "val": "dashed", "test": "solid"},
                "after": {"train": "dotted", "val": "dashed", "test": "solid"},
            }
            for stage in ["before", "after"]:
                for split in ["train", "val", "test"]:
                    if len(self.global_metrics[stage][split]) > 0:
                        plt.plot(
                            [metrics.accuracy for metrics in self.global_metrics[stage][split]],
                            label=f"{split}set ({stage}LocalTraining)",
                            ls=linestyle[stage][split],
                        )

            plt.title(f"{self.algo}_{self.args.common.dataset}")
            plt.ylim(0, 100)
            plt.xlabel("Communication Rounds")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(
                OUT_DIR / self.algo / self.output_dir / f"{self.args.common.dataset}.png",
                bbox_inches="tight",
            )
        if self.args.common.save_metrics:
            import pandas as pd
            df = pd.DataFrame()
            for stage in ["before", "after","before_poison","after_poison"]:
                for split in ["train", "val", "test"]:
                    if len(self.global_metrics[stage][split]) > 0:
                        for metric in ["accuracy"]:
                            stats = [getattr(metrics, metric) for metrics in self.global_metrics[stage][split]]
                            df.insert(
                                loc=df.shape[1],
                                column=f"{metric}_{split}_{stage}",
                                value=np.array(stats).T,
                            )
            df.to_csv(
                OUT_DIR / self.algo / self.output_dir / f"{self.args.common.dataset}_acc_metrics.csv",
                index=True,
                index_label="epoch",
            )
        if self.args.common.save_model:
            model_name = f"{self.args.common.dataset}_{self.args.common.global_epoch}_{self.args.common.model}.pt"
            if self.unique_model:
                torch.save(self.clients_personal_model_params, self.output_dir / model_name)
            else:
                torch.save(self.public_model_params, self.output_dir / model_name)
    @torch.no_grad()
    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):

        if self.args.common.use_multi_krum == 1:
            self._aggregate_multi_krum(clients_package)
        else:
            self._aggregate_average(clients_package)
        for client_id, package in clients_package.items():
            if client_id in self.args.common.poison_client_ids and "trigger_optimizer" in package:
                self.trigger_optimizer = package["trigger_optimizer"]

        if self.args.common.Pruning == 1 and self.current_epoch % self.args.common.pruning_epoch ==0:
            self.public_model_params = Pruning(self.public_model_params, self.args.common.u_conv, self.args.common.pruning_rate,self.args.common.bias)

    def _aggregate_average(self, clients_package: OrderedDict[int, dict[str, Any]]):

        clients_weight = [package["weight"] for package in clients_package.values()]
        weights = torch.tensor(clients_weight, dtype=torch.float32) / sum(clients_weight)  # 归一化权重

        if self.return_diff:
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in clients_package.values()
                    ],
                    dim=0,
                )  # [num_clients, *param_shape]
                aggregated = torch.sum(diffs * weights.view(-1, *([1] * (diffs.dim() - 1))), dim=0)
                self.public_model_params[name].data -= aggregated
        else:
            for name, global_param in self.public_model_params.items():
                client_params = torch.stack(
                    [
                        package["regular_model_params"][name]
                        for package in clients_package.values()
                    ],
                    dim=0,
                )  # [num_clients, *param_shape]
                aggregated = torch.sum(client_params * weights.view(-1, *([1] * (client_params.dim() - 1))), dim=0)
                global_param.data = aggregated