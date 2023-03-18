from typing import List, Union
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch import LightningModule
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from utils import get_no_decay_groups



def count_params(opt: Optimizer, grad: bool = True):
    return sum(p.numel() for group in opt.param_groups for p in group["params"] if p.requires_grad == grad)


class GradualUnfreezingCallback(BaseFinetuning):
    def __init__(self, unfreeze_per_epoch: int, div_lr: float = 2.6, min_unfreeze_layer: int = 0):
        super().__init__()
        self.unfreeze_per_epoch = unfreeze_per_epoch
        self.last_unfrozen_layer = 0
        self.total_layers = 0
        self.div_lr = div_lr
        self.min_unfreeze_layer = min_unfreeze_layer


    def freeze_before_training(self, model):
        self.freeze(model)
        # Classifier unfreeze
        if hasattr(model.model, "classifier"):
            self.make_trainable(model.model.classifier)

        if hasattr(model.model, "roberta") and hasattr(
            model.model.roberta, "encoder"
        ):
            self.total_layers = len(model.model.roberta.encoder.layer)
            self.last_unfrozen_layer = self.total_layers

    def finetune_function(
        self, model: LightningModule, epoch: int, optimizer: Optimizer
    ):

        # Nothing to unfreeze
        if self.last_unfrozen_layer <= self.min_unfreeze_layer:
            return

        old_last_unfrozen_layer = self.last_unfrozen_layer
        self.last_unfrozen_layer = max(self.min_unfreeze_layer, self.last_unfrozen_layer - self.unfreeze_per_epoch)


        new_layers = model.model.roberta.encoder.layer[
            self.last_unfrozen_layer:old_last_unfrozen_layer
        ]
        print(f"Unfreezing {self.last_unfrozen_layer}-{old_last_unfrozen_layer} layers")
        self.make_trainable(new_layers)
        # Not first unfreeze than apply discriminative learning rate
        if old_last_unfrozen_layer == self.total_layers:
            new_lr = optimizer.param_groups[0]["lr"]
        else:
            new_lr = optimizer.param_groups[-1]["lr"] / self.div_lr



        decay = optimizer.param_groups[0]["weight_decay"]
        param_groups = get_no_decay_groups(new_layers, new_lr, decay, ["bias", "LayerNorm.weight"])
        optimizer.add_param_group(param_groups)
        print(f"Total trainable params: {count_params(optimizer)}")

            
