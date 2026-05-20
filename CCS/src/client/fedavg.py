from collections import OrderedDict
from copy import deepcopy
from typing import Any
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from data.utils.datasets import BaseDataset
from src.attack.BC import test_accuracy, FLS, BLS
from src.defence.AT.AUG import IDBH
from src.utils.functional import evaluate_model, get_optimal_cuda_device,evaluate_model_backdoor
from src.utils.metrics import Metrics
from src.utils.models import DecoupledModel,MODELS
from torch.autograd import Variable
#add
from src.attack.badnet import badnet
from src.attack.dba import dba
from src.defence.AT.AT import *
import torch.nn.functional as F
from src.defence.SARS.utils import Attention,PerturbedGradientDescent
from src.defence.PFedCLASP.util import flag_update,self_paced_augmentation,run_n_aug,collate_fn
from src.attack.BadPFL.generator import Autoencoder,Autoencoder_MNIST
from src.defence.WM.Loss import *
from src.defence.WM.WM import *

import numpy as np
class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler._LRScheduler],
        args: DictConfig,
        dataset: BaseDataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
        trigger, # 普通触发器
        noise,#PFedBA触发器
        poison_pattern,# PFedBA触发位置
    ):
        self.client_id: int = None
        self.args = args
        if device is None:
            self.device = get_optimal_cuda_device(use_cuda=self.args.common.use_cuda)
        else:
            self.device = device
        self.dataset = dataset
        self.model = model.to(self.device)
        self.regular_model_params: OrderedDict[str, torch.Tensor]
        self.personal_params_name: list[str] = []
        self.regular_params_name = list(key for key, _ in self.model.named_parameters())
        if self.args.common.buffers == "local":
            self.personal_params_name.extend(
                [name for name, _ in self.model.named_buffers()]
            )
        elif self.args.common.buffers == "drop":
            self.init_buffers = deepcopy(OrderedDict(self.model.named_buffers()))

        self.optimizer = optimizer_cls(params=self.model.parameters())
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.init_lr_scheduler_state: dict = None
        self.lr_scheduler_cls = None
        if lr_scheduler_cls is not None:
            self.lr_scheduler_cls = lr_scheduler_cls
            self.lr_scheduler = self.lr_scheduler_cls(optimizer=self.optimizer)
            self.init_lr_scheduler_state = deepcopy(self.lr_scheduler.state_dict())

        # [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices
        # Please don't bother with the [0], which is only for avoiding raising runtime error by setting Subset(indices=[]) with `DataLoader(shuffle=True)`
        self.trainset = Subset(self.dataset, indices=[0])
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        self.trainloader = DataLoader(self.trainset, batch_size=self.args.common.batch_size, shuffle=True)
        self.valloader = DataLoader(self.valset, batch_size=self.args.common.batch_size)
        self.testloader = DataLoader(self.testset, batch_size=self.args.common.batch_size)


        self.testing = False
        self.local_epoch = self.args.common.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.eval_results = {}
        self.return_diff = return_diff
        #trigger
        self.trigger = trigger
        #BadPFL
        if self.args.dataset.name =='mnist' or self.args.dataset.name == 'fmnist' or self.args.dataset.name == 'emnist':
            self.trigger_gen = Autoencoder_MNIST().to(self.device)
        else:
            self.trigger_gen = Autoencoder().to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.trigger_gen.parameters(), lr=0.01)
        #PFedBA
        self.PFedBA_trainset = Subset(self.dataset, indices=[])
        self.PFedBA_noise = noise
        self.PFedBA_trigger = self.trigger
        self.poison_pattern = poison_pattern
        # SARS
        if self.args.common.backdoor.defence_method == 'SARS':
            self.SARS_local_models = [MODELS[self.args.model.name](dataset=self.args.dataset.name,pretrained=self.args.model.use_torchvision_pretrained_weights) for _ in range(100)]
        #PFed-CLASP
        self.flag_spa = self.args.common.backdoor.flag_spa   #是否使用自步学习
        self.n_aug = self.args.common.backdoor.n_aug #数据增强的方式
        self.criterion_SPA = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.trainloader_SPA = DataLoader(self.trainset, batch_size=self.args.common.batch_size, shuffle=True, collate_fn=collate_fn)
        # WM
        self.wm_label = self.args.common.backdoor.wm_label
        self.Contrastive_Loss = Contrastive_Loss(temperature=0.07)
        # BC
        self.attack_layers = []
        # AT
        self.global_model = deepcopy(self.model)
        self.AT_alpha = self.args.common.backdoor.AT_alpha
        self.AT_beta = self.args.common.backdoor.AT_beta
    def get_PFedBA_trainset(self):
        merged_indices = []
        for client_id in self.args.common.backdoor.poison_client_ids:
            merged_indices.extend(self.data_indices[client_id]["train"])
        self.PFedBA_trainset.indices = merged_indices

    def load_data_indices(self):
        """This function is for loading data indices for No.`self.client_id`
        client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train_with_eval(self):
        eval_results = {
            "MTA_before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "ASR_before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "MTA_after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "ASR_after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        eval_results["MTA_before"] = self.MTA_evaluate()
        eval_results["ASR_before"] = self.ASR_evaluate()
        if self.local_epoch > 0:
            if self.client_id in self.args.common.backdoor.poison_client_ids:
                self.attack()
            else:
                self.fit()
            eval_results["MTA_after"] = self.MTA_evaluate()
            eval_results["ASR_after"] = self.ASR_evaluate()

        eval_msg = []
        for split, color, flag, subset in [
            ["train", "yellow", self.args.common.test.client.train, self.trainset],
            ["val", "green", self.args.common.test.client.val, self.valset],
            ["test", "cyan", self.args.common.test.client.test, self.testset],
        ]:
            if len(subset) > 0 and flag:
                eval_msg.append(
                    f"MTA_client [{self.client_id}]\t"
                    f"[{color}]({split}set)[/{color}]\t"
                    f"[red]loss: {eval_results['MTA_before'][split].loss:.4f} -> "
                    f"{eval_results['MTA_after'][split].loss:.4f}\t[/red]"
                    f"[blue]accuracy: {eval_results['MTA_before'][split].accuracy:.2f}% -> {eval_results['MTA_after'][split].accuracy:.2f}%[/blue]"
                )
                eval_msg.append(
                    f"ASR_client [{self.client_id}]\t"
                    f"[{color}]({split}set)[/{color}]\t"
                    f"[red]loss: {eval_results['ASR_before'][split].loss:.4f} -> "
                    f"{eval_results['ASR_after'][split].loss:.4f}\t[/red]"
                    f"[blue]accuracy: {eval_results['ASR_before'][split].accuracy:.2f}% -> {eval_results['ASR_after'][split].accuracy:.2f}%[/blue]"
                )

        eval_results["message"] = eval_msg
        self.eval_results = eval_results

    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()
        self.cu_epoch = package["current_epoch"]
        if self.args.common.backdoor.poison_method == 'pfedba':
            self.get_PFedBA_trainset()
        if (
            package["optimizer_state"]
            and not self.args.common.reset_optimizer_on_global_epoch
        ):
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.model.load_state_dict(package["regular_model_params"], strict=False)
        self.model.load_state_dict(package["personal_model_params"], strict=False)
        self.global_model.load_state_dict(package["regular_model_params"], strict=False)
        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        model_params = self.model.state_dict()
        self.client_gradient = OrderedDict(
            (key, model_params[key].clone().cpu())
            for key in self.regular_params_name
        )

        if self.return_diff:
            model_params = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )

    def train(self, server_package: dict[str, Any]) -> dict:
        self.set_parameters(server_package)
        self.train_with_eval()
        client_package = self.package()
        return client_package

    def package(self):
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )

        # client_package["client_gradient"] = {
        #     key: param_old - param_new
        #     for (key, param_new), param_old in zip(
        #         client_package["regular_model_params"].items(),
        #         self.client_gradient.values(),
        #     )
        # }
        client_package["client_gradient"] = {
            key: param_old - param_new
            for (key, param_new), param_old in zip(
                client_package["regular_model_params"].items(),
                self.client_gradient.values(),
            )
            if "base" in key  # 判断名字是否包含 "bias"
        }

        client_package["model_params_diff"] = {
            key: param_old - param_new
            for (key, param_new), param_old in zip(
                client_package["regular_model_params"].items(),
                self.client_gradient.values(),
            )
        }
        # if self.return_diff:
        #     client_package["model_params_diff"] = {
        #         key: param_old - param_new
        #         for (key, param_new), param_old in zip(
        #             client_package["regular_model_params"].items(),
        #             self.regular_model_params.values(),
        #         )
        #     }
        #     client_package.pop("regular_model_params")
        return client_package

    def fit(self):
        if self.args.common.backdoor.defence_method == 'AT':
            self.model.train()
            self.dataset.train()
            for _ in range(self.local_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    # x_aug = IDBH(x.clone(), self.args.dataset.name)
                    x_adv = perturb_input(self.model,  x.clone() , y.clone(), step_size=0.1, epsilon=0.5, perturb_steps=15, distance='l_inf_patch', device=self.device)
                    self.optimizer.zero_grad()
                    logits_adv, adv_pro_feature = self.model(x_adv, flag=True)
                    self.model.train()
                    logit, pro_feature = self.model(x , flag=True)
                    loss_ce = self.criterion(logit, y)
                    loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1),F.softmax(logit, dim=1),reduction='batchmean')
                    loss_mmd = MMD(pro_feature, adv_pro_feature,'rbf', self.device)
                    loss = loss_ce  + self.AT_alpha * loss_kl + self.AT_beta * loss_mmd
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
        else:
            self.model.train()
            self.dataset.train()
            for _ in range(self.local_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    loss.backward()
                    self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
    def attack(self):
        if self.args.common.backdoor.poison_method == 'badnet':
            self.model.train()
            self.dataset.train()
            for _ in range(self.local_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y, poison_count = dba(x, y,clientID = self.client_id,poison_client_ids = self.args.common.backdoor.poison_client_ids,is_test=False,dataset = self.args.dataset.name,attack = self.args.common.backdoor.poison_method)
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    loss.backward()
                    self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
        elif self.args.common.backdoor.poison_method == 'dba':
            self.model.train()
            self.dataset.train()
            for _ in range(self.local_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y, poison_count = dba(x, y,clientID = self.client_id,poison_client_ids = self.args.common.backdoor.poison_client_ids,is_test=False,dataset = self.args.dataset.name,attack = self.args.common.backdoor.poison_method)
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    loss.backward()
                    self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
        else:
            self.model.train()
            self.dataset.train()
            for _ in range(self.local_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    loss.backward()
                    self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()


    @torch.no_grad()
    def MTA_evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
        target_model = self.model if model is None else model
        if self.args.common.backdoor.defence_method == 'SARS' and self.client_id not in self.args.common.backdoor.poison_client_ids:
            target_model = self.SARS_local_models[self.client_id]
        target_model.eval()
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if (
            len(self.testset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.test
        ):
            test_metrics = evaluate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if (
            len(self.valset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.val
        ):
            val_metrics = evaluate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )

        if (
            len(self.trainset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.train
        ):
            train_metrics = evaluate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    @torch.no_grad()
    def ASR_evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
        target_model = self.model if model is None else model
        trigger_gen = self.trigger_gen
        self.ASR_trigger = self.trigger
        if self.args.common.backdoor.defence_method == 'SARS' and self.client_id not in self.args.common.backdoor.poison_client_ids:
            target_model = self.SARS_local_models[self.client_id]
        if self.args.common.backdoor.poison_method == 'pfedba':
            self.ASR_trigger = deepcopy(self.PFedBA_trigger)
        target_model.eval()
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        if (
            len(self.testset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.test
        ):
            test_metrics = evaluate_model_backdoor(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
                trigger=self.ASR_trigger,
                trigger_size=self.args.common.backdoor.trigger_size,
                trigger_location=self.args.common.backdoor.trigger_location,
                dataset=self.args.dataset.name,
                poison_method=self.args.common.backdoor.poison_method,
                generator=trigger_gen
            )

        if (
            len(self.valset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.val
        ):
            val_metrics = evaluate_model_backdoor(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )

        if (
            len(self.trainset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.train
        ):
            train_metrics = evaluate_model_backdoor(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    def test(self, server_package: dict[str, Any]) -> dict[str, dict[str, Metrics]]:
        self.testing = True
        self.set_parameters(server_package)
        results = {
            "MTA_before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "ASR_before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "MTA_after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "ASR_after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        results["MTA_before"] = self.MTA_evaluate()
        results["ASR_before"] = self.ASR_evaluate()
        if self.args.common.test.client.finetune_epoch > 0:
            frz_params_dict = deepcopy(self.model.state_dict())
            self.finetune()
            results["MTA_after"] = self.MTA_evaluate()
            results["ASR_after"] = self.ASR_evaluate()
            self.model.load_state_dict(frz_params_dict)
        self.testing = False
        return results

    def finetune(self):
        """Client model finetuning.

        This function will only be activated in `test()`
        """
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.test.client.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()





