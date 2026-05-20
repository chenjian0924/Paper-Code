from collections import OrderedDict
from copy import deepcopy
from typing import Any
import torch
from torch.utils.data import DataLoader, Subset
from src.utils.tools import NestedNamespace, get_optimal_cuda_device, evalutate_model,bapfl_evalutate_model,pFedBA_evalutate_model
from src.utils.metrics import Metrics
from src.utils.models import DecoupledModel
from data.utils.datasets import BaseDataset
# 导入函数
from src.SPA.util import *
from torch.distributions.uniform import Uniform #添加噪声

class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler._LRScheduler],
        args: NestedNamespace,
        dataset: BaseDataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
        trainset_pFedBA: Subset = None,
    ):
        self.client_id: int = None
        self.args = args
        if device is None:
            self.device = get_optimal_cuda_device(use_cuda=self.args.common.use_cuda)
        else:
            self.device = device
        self.dataset = dataset
        self.model = model.to(self.device)
        self.global_regular_model_params: OrderedDict[str, torch.Tensor]
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
            self.lr_scheduler = lr_scheduler_cls(optimizer=self.optimizer)
            self.init_lr_scheduler_state = deepcopy(self.lr_scheduler.state_dict())

        # [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices

        self.trainset = Subset(self.dataset, indices=[0])
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        self.trainset_pFedBA = trainset_pFedBA
        # 正常数据集
        self.trainloader = DataLoader(self.trainset, batch_size=self.args.common.batch_size, shuffle=True)
        self.valloader = DataLoader(self.valset, batch_size=self.args.common.batch_size)
        self.testloader = DataLoader(self.testset, batch_size=self.args.common.batch_size)

        self.testing = False
        self.local_epoch = self.args.common.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.eval_results = {}
        self.return_diff = return_diff

        #spa
        self.flag_spa = self.args.common.flag_spa
        self.n_aug = self.args.common.n_aug
        self.criterion_SPA = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.trainloader_SPA = DataLoader(
            self.trainset, batch_size=self.args.common.batch_size, shuffle=True, collate_fn=collate_fn
        )

    def load_data_indices(self):
        """This function is for loading data indices for No.`self.client_id` client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train_with_eval(self):
        eval_results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "before_poison": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after_poison": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }

        eval_results["before"] = self.evaluate()
        eval_results["before_poison"] = self.poison_evaluate()

        if self.local_epoch > 0:
            if self.client_id in self.args.common.poison_client_ids:
                self.poison_fit()
            else:
                self.fit()
            eval_results["after"] = self.evaluate()
            eval_results["after_poison"] = self.poison_evaluate()

        eval_msg = []

        for split, color, flag, subset in [
            ["train", "yellow", self.args.common.eval_train, self.trainset],
            ["val", "green", self.args.common.eval_val, self.valset],
            ["test", "cyan", self.args.common.eval_test, self.testset],
        ]:
            if len(subset) > 0 and flag:
                eval_msg.append(
                    "client [{}] [{}]({})  loss: {:.4f} -> {:.4f}   accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        color,
                        split,
                        eval_results["before"][split].loss,
                        eval_results["after"][split].loss,
                        eval_results["before"][split].accuracy,
                        eval_results["after"][split].accuracy,
                    )
                )

                eval_msg.append(
                    "client_poison [{}] [{}]({})  loss: {:.4f} -> {:.4f}   accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        color,
                        split,
                        eval_results["before_poison"][split].loss,
                        eval_results["after_poison"][split].loss,
                        eval_results["before_poison"][split].accuracy,
                        eval_results["after_poison"][split].accuracy,
                    )
                )

        eval_results["message"] = eval_msg
        self.eval_results = eval_results

    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.global_epoch = package.get("global_epoch", 0)
        self.trigger_optimizer = package.get("trigger_optimizer")
        self.load_data_indices()
        if package["optimizer_state"]:
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
        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        if self.return_diff:
            model_params = self.model.state_dict()
            self.global_regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )


    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()
        client_package = self.package()
        if self.client_id in self.args.common.poison_client_ids and hasattr(self, "trigger_optimizer"):
            client_package["trigger_optimizer"] = self.trigger_optimizer
        return client_package

    def package(self):
        model_params = self.model.state_dict()
        # print(model_params)
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
                {} if self.lr_scheduler is None else deepcopy(self.lr_scheduler.state_dict())
            ),
        )
        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip(
                    client_package["regular_model_params"].items(),
                    self.global_regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")
        if hasattr(self, "trigger_optimizer"):
            client_package["trigger_optimizer"] = self.trigger_optimizer.clone().cpu()
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        if self.args.common.SPA == 1:
            self.flag_noise = np.ones(len(self.trainset), dtype=int)
            for epoch in range(self.local_epoch):
                loss_each_all = np.zeros(len(self.trainset))
                for i, (x, y, index) in enumerate(self.trainloader_SPA):
                    if len(x) <= 1:
                        continue
                    if x.dim() == 3:
                        x = x.unsqueeze(1).to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.flag_spa == 1:
                        output = self.model(x)
                        y_spa = y.clone()
                        if y_spa.ndim == 1:
                            if self.args.common.dataset == 'emnist':
                                y_spa = torch.eye(62, device=self.device)[y_spa].clone()
                            else:
                                y_spa = torch.eye(10, device=self.device)[y_spa].clone()
                        self.loss_each = self.criterion_SPA(output, y_spa)
                        loss_each_all[index] = self.loss_each.detach().cpu().numpy()
                        self.judge_noise = self.args.common.judge_noise
                        flag_noise = flag_update(loss_each_all, self.judge_noise)
                    if self.args.common.dataset == 'emnist':
                        num_classes = 62
                    else:
                        num_classes = 10
                    if self.flag_spa == 1:
                        x, y = self_paced_augmentation(
                            images=x,
                            labels=y,
                            flag_noise=flag_noise,
                            index=index.cpu().numpy(),
                            n_aug=self.n_aug,
                            num_classes=num_classes
                        )
                    else:
                        x, y = run_n_aug(
                            x=x,
                            y=y,
                            n_aug=self.n_aug,
                            num_classes=num_classes
                        )
                    logit = self.model(x)
                    if y.ndim == 1:
                        if self.args.common.dataset == 'emnist':
                            y = torch.eye(62, device=self.device)[y].clone()
                        else:
                            y = torch.eye(10, device=self.device)[y].clone()
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
        else:
            for epoch in range(self.local_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
    def poison_fit(self):
        self.model.train()
        self.dataset.train()
        for epoch in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
        target_model = self.model if model is None else model
        target_model.eval()
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        if len(self.testset) > 0 and self.args.common.eval_test:
            test_metrics = evalutate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )
        if len(self.valset) > 0 and self.args.common.eval_val:
            val_metrics = evalutate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )
        if len(self.trainset) > 0 and self.args.common.eval_train:
            train_metrics = evalutate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    @torch.no_grad()
    def poison_evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
        target_model = self.model if model is None else model
        target_model.eval()
        self.dataset.eval()
        poison_train_metrics = Metrics()
        poison_val_metrics = Metrics()
        poison_test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        # 如果测试集非空并且参数设置中包含评估测试集，执行测试集评估
        if len(self.testset) > 0 and self.args.common.eval_test:
            if self.args.common.pFedBA == 1:
                poison_test_metrics = pFedBA_evalutate_model(
                    model=target_model,
                    dataloader=self.testloader,
                    criterion=criterion,
                    device=self.device,
                    dataset=self.args.common.dataset,
                    trigger = self.trigger_optimizer,
                    is_orginalpfedba = self.args.common.is_orginalpfedba
                )
            else:
                poison_test_metrics = bapfl_evalutate_model(
                    model=target_model,
                    dataloader=self.testloader,
                    criterion=criterion,
                    device=self.device,
                    dataset=self.args.common.dataset,
                    is_orginalpfedba = self.args.common.is_orginalpfedba
                )

        if len(self.valset) > 0 and self.args.common.eval_val:
            poison_val_metrics = bapfl_evalutate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )
        if len(self.trainset) > 0 and self.args.common.eval_train:
            poison_train_metrics = bapfl_evalutate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        return {"train": poison_train_metrics, "val": poison_val_metrics, "test": poison_test_metrics}


    def test(self, server_package: dict[str, Any]) -> dict[str, dict[str, Metrics]]:
        self.testing = True
        self.set_parameters(server_package)
        results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "before_poison": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after_poison": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }

        results["before"] = self.evaluate()
        results["before_poison"] = self.poison_evaluate()
        if self.args.common.finetune_epoch > 0:
            frz_params_dict = deepcopy(self.model.state_dict())
            self.finetune()
            results["after"] = self.evaluate()
            results["after_poison"] = self.poison_evaluate()
            self.model.load_state_dict(frz_params_dict)
        self.testing = False
        return results
    def finetune(self):
        """Client model finetuning. This function will only be activated in `test()`"""
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


