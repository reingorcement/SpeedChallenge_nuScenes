"""Trainable parameter groups and a cosine factor shared across base learning rates."""
import math
import torch


class RelativeCosine(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, epochs):
        self.T_max=epochs
        super().__init__(optimizer)

    def get_lr(self):
        factor=.1+.9*(1+math.cos(math.pi*self.last_epoch/self.T_max))/2
        return [base*factor for base in self.base_lrs]


def make_optimizer(model, config):
    groups=[dict(name="flow_head",params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("rgb_encoder.")],
                 lr=config["learning_rate"])]
    rgb=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith("rgb_encoder.")]
    if rgb:groups.append(dict(name="rgb",params=rgb,lr=config["rgb_learning_rate"]))
    return torch.optim.AdamW(groups,lr=config["learning_rate"],weight_decay=config["weight_decay"])


def make_scheduler(optimizer, config):
    period = config.get("scheduler_epochs", config["epochs"])
    if not isinstance(period, int) or isinstance(period, bool) or period < config["epochs"]:
        raise ValueError("Scheduler period must be an integer >= stopping epoch budget")
    if config.get("scheduler_policy")=="cosine_each_group_to_0.1":return RelativeCosine(optimizer,period)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=period,eta_min=config["learning_rate"]*.1)
