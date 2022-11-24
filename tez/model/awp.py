import torch


class AWP:
    def __init__(
            self,
            model,
            optimizer,
            adv_param="weight",
            adv_lr=1,
            adv_eps=0.2,
            adv_step=1,
            scaler=None,
            device="cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler
        self.device = device

    def attack_backward(self, data):
        self._save()
        for key, value in data.items():
            data[key] = value.to(self.device)
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                _, adv_loss, _ = self.model(**data)
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        # for name, param in self.model.named_parameters():
        for group in self.optimizer.param_groups:
            for i, param in enumerate(group["params"]):
                name = group["names"][i]
                if param.requires_grad and param.grad is not None and self.adv_param in name:
                    norm1 = torch.norm(param.grad)
                    norm2 = torch.norm(param.data.detach())
                    if norm1 != 0 and not torch.isnan(norm1):
                        lr = self.adv_lr * group["lr"]
                        r_at = lr * param.grad / (norm1 + e) * (norm2 + e)
                        param.data.add_(r_at)
                        param.data = torch.min(
                            torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                        )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}