from dataclasses import dataclass
from typing import Callable
import torch
from lightning.pytorch import Trainer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class SchedulerConfig:
    scheduler: torch.optim.lr_scheduler._LRScheduler
    interval: str = "step"
    frequency: int = 1


CreateableScheduler = Callable[[torch.optim.Optimizer, Trainer], SchedulerConfig]

def get_linear_schedule_warmup(optimizer: torch.optim.Optimizer,
                            trainer,
                            num_warmup_steps: int, min_lr_ratio=0.0) -> SchedulerConfig:
    total_steps = trainer.estimated_stepping_batches
    if num_warmup_steps < 1:
        num_warmup_steps = int(total_steps * num_warmup_steps)
    
    return SchedulerConfig(
        scheduler=get_slated_lambda(optimizer, num_warmup_steps, total_steps, min_lr_ratio=min_lr_ratio),
        interval="step",
        frequency=1,
    )


def get_slated_lambda(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio:float, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return (1-min_lr_ratio) / num_warmup_steps * float(current_step) + min_lr_ratio
        return max(
            min_lr_ratio, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)) * (1-min_lr_ratio) + min_lr_ratio
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# As per smith with recommended epochs_per_cycle
def get_cycle_lr(optimizer: torch.optim.Optimizer,
                        trainer: Trainer,
                        base_lr: float = 0.0001,
                        max_lr: float = 0.001,
                        epochs_per_cycle: int = 4,
                        cycle_momentum: bool = False):
    steps_per_epoch = trainer.estimated_stepping_batches // trainer.max_epochs
    step_size_up = steps_per_epoch * epochs_per_cycle // 2


    return SchedulerConfig(scheduler=torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=cycle_momentum),
                                interval="step",
                                frequency=1)

def get_exponential_lr(optimizer: torch.optim.Optimizer,
                            trainer: Trainer,
                            gamma: float = 0.9) -> SchedulerConfig:
    return SchedulerConfig(
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma),
        interval="epoch",
        frequency=1,
    )

