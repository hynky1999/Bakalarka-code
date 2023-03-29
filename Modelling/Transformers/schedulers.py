from dataclasses import dataclass
import math
from typing import Callable
import torch
from lightning.pytorch import Trainer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from dynam_lambda_lr import DynamicLambdaLR


@dataclass
class SchedulerConfig:
    scheduler: torch.optim.lr_scheduler.LRScheduler
    interval: str = "step"
    frequency: int = 1


CreateableScheduler = Callable[[torch.optim.Optimizer, Trainer], SchedulerConfig]


def get_slated_lambda(
    num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float, offset: int = 0
):
    def lr_lambda(current_step: int):
        current_step -= offset
        if current_step < num_warmup_steps:
            return max(
                min_lr_ratio,
                (1 - min_lr_ratio) / num_warmup_steps * float(current_step)
                + min_lr_ratio,
            )
        return max(
            min_lr_ratio,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps))
            * (1 - min_lr_ratio)
            + min_lr_ratio,
        )

    return lr_lambda


def get_linear_schedule_for_discriminiative_lr(
    optimizer: torch.optim.Optimizer,
    trainer: Trainer,
    total_unfreezes,
    groups_per_unfreeze: int,
    num_warmup_steps_classifier: float,
    min_lr_classifier: float,
    num_warmup_steps_backbone: float,
    min_lr_backbone: float,
):

    assert trainer.max_epochs != None and trainer.estimated_stepping_batches != None

    steps_per_epoch = math.ceil(
        trainer.num_training_batches / trainer.accumulate_grad_batches
    )
    training_steps = steps_per_epoch * trainer.max_epochs
    # Add classifier
    pg_lambdas = [
        get_slated_lambda(
            int(num_warmup_steps_classifier * training_steps),
            training_steps,
            min_lr_classifier,
            offset=0,
        )
        for _ in range(groups_per_unfreeze)
    ]
    for i in range(total_unfreezes):
        offset = i * steps_per_epoch
        remaining_steps = training_steps - offset
        pg_lambdas.extend(
            [
                get_slated_lambda(
                    int(num_warmup_steps_backbone * remaining_steps),
                    remaining_steps,
                    min_lr_backbone,
                    offset=offset,
                )
                for _ in range(groups_per_unfreeze)
            ]
        )

    return SchedulerConfig(
        scheduler=DynamicLambdaLR(
            optimizer,
            pg_lambdas,
        ),
        interval="step",
        frequency=1,
    )


def get_linear_schedule_warmup(
    optimizer: torch.optim.Optimizer, trainer, num_warmup_steps: int, min_lr_ratio=0.0
) -> SchedulerConfig:
    total_steps = trainer.estimated_stepping_batches
    if num_warmup_steps < 1:
        num_warmup_steps = int(total_steps * num_warmup_steps)

    return SchedulerConfig(
        scheduler=LambdaLR(
            optimizer=optimizer,
            lr_lambda=get_slated_lambda(
                num_warmup_steps, total_steps, min_lr_ratio=min_lr_ratio
            )),
            interval="step",
            frequency=1,
        )


# As per smith with recommended epochs_per_cycle
def get_cycle_lr(
    optimizer: torch.optim.Optimizer,
    trainer: Trainer,
    base_lr: float = 0.0001,
    max_lr: float = 0.001,
    epochs_per_cycle: int = 4,
    cycle_momentum: bool = False,
):
    steps_per_epoch = trainer.estimated_stepping_batches // trainer.max_epochs
    step_size_up = steps_per_epoch * epochs_per_cycle // 2

    return SchedulerConfig(
        scheduler=torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            cycle_momentum=cycle_momentum,
        ),
        interval="step",
        frequency=1,
    )


def get_exponential_lr(
    optimizer: torch.optim.Optimizer, trainer: Trainer, gamma: float = 0.9
) -> SchedulerConfig:
    return SchedulerConfig(
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma),
        interval="epoch",
        frequency=1,
    )
