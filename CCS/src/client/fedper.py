from src.client.fedavg import FedAvgClient
import torch.nn.init as init
import math
class FedPerClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.personal_params_name.extend(
            [name for name in self.model.state_dict().keys() if "classifier" in name]
        )
        self.personal_params_name.extend(
            [name for name in self.model.state_dict().keys() if "project" in name]
        )

    def finetune(self):
        self.model.train()
        finetuning = 3
        if finetuning == 1:
            # fine-tune the full model
            super().finetune()
        elif finetuning == 2:
            # fine-tune the classifier only
            for _ in range(self.args.common.test.client.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # for name, param in self.model.named_parameters():
                    #     if name not in self.personal_params_name:
                    #         param.grad.zero_()
                    self.model.base.zero_grad()
                    self.optimizer.step()
        elif finetuning == 3:
            for param in self.model.classifier.parameters():
                if param.dim() > 1:  # 如果是权重矩阵
                    n = param.size(1) * param.size(2) * param.size(3) if param.dim() == 4 else param.size(0)
                    init.normal_(param, 0, math.sqrt(2. / n))
                else:  # 如果是偏置
                    init.zeros_(param)
            for _ in range(self.args.common.test.client.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.model.base.zero_grad()
                    self.optimizer.step()
