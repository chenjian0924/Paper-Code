from src.client.fedavg import FedAvgClient


class FedPerClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.personal_params_name.extend(
            [name for name in self.model.state_dict().keys() if "classifier" in name]
        )

    def finetune(self):
        self.model.train()
        full_model = False
        SIMPLE_TUNING= False
        if full_model:
            # fine-tune the full model
            super().finetune()
        elif full_model == False and SIMPLE_TUNING == False:
            # fine-tune the classifier only
            for _ in range(self.args.common.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        if name not in self.personal_params_name:
                            param.grad.zero_()
                    self.optimizer.step()
        elif SIMPLE_TUNING == True:
            self.model.classifier.reset_parameters()
            for param in self.model.base.parameters():
                param.requires_grad = False
            for epoch in range(200):
                self.model.train()
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            for param in self.model.base.parameters():
                param.requires_grad = True
