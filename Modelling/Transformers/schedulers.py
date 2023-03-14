from dataclasses import dataclass
from typing import Callable
import torch
from lightning.pytorch import Trainer
from transformers.optimization import get_linear_schedule_with_warmup


@dataclass
class SchedulerConfig:
    scheduler: torch.optim.lr_scheduler._LRScheduler
    interval: str = "step"
    frequency: int = 1


CreateableScheduler = Callable[[torch.optim.Optimizer, Trainer], SchedulerConfig]

def get_linear_schedule_warmup(optimizer: torch.optim.Optimizer,
                           trainer,
                            num_warmup_steps: int) -> SchedulerConfig:
    total_steps = trainer.estimated_stepping_batches
    last_epoch = trainer.current_epoch
    if num_warmup_steps < 1:
        num_warmup_steps = int(total_steps * num_warmup_steps)
    
    return SchedulerConfig(
        scheduler=get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps, last_epoch=last_epoch),
        interval="step",
        frequency=1,
    )

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

