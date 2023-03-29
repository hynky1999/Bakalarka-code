from argparse import ArgumentParser
from pathlib import Path
import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
import wandb
from lightning.pytorch.tuner.tuning import Tuner
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping, Timer, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


def fit(trainer: Trainer, model, datamodule, ckpt_path=None):
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


def validate(trainer: Trainer, model, datamodule, ckpt_path=None):
    trainer.validate(model, datamodule, ckpt_path=ckpt_path)


def test(trainer, model, datamodule, ckpt_path=None):
    trainer.test(model, datamodule, ckpt_path=ckpt_path)


def tune(trainer: Trainer, model, datamodule):
    tuner = Tuner(trainer)
    batch_size = tuner.scale_batch_size(
        model=model, datamodule=datamodule, mode="binsearch"
    )
    print(f"Found Batch size: {batch_size}")

    lr_finder = tuner.lr_find(model=model, datamodule=datamodule)
    print(f"Found lr: {lr_finder.suggestion()}")


def set_effective_batch_size(needed_size, devices, batch_size):
    acc_batches = needed_size // (devices * batch_size)
    if acc_batches * (devices * batch_size) != needed_size:
        raise ValueError(
            f"Effective batch size should be divisible by {devices * batch_size}"
        )
    return acc_batches

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    checkpoint_path = cfg.run.chckpt_path if "chckpt_path" in cfg.run else None
    if "logger" in cfg:
        project_name = f"{cfg.task.name.capitalize()}-Deep-Learning"
        if "debug" in cfg.run:
            project_name = f"Debug"
        logger = instantiate(cfg.logger, project=project_name)(config=vars(cfg))
    else:
        logger = TensorBoardLogger("tb_logs", name=cfg.task.name)

    if isinstance(logger, WandbLogger) and checkpoint_path in ["best_k", "last", "best"]:
        experiment = logger.experiment
        reference = f"{experiment.entity}/{experiment.project}/model-{experiment.id}:{checkpoint_path}"
        artifact = logger.use_artifact(reference)
        artifact_dir = artifact.download()
        checkpoint_path = Path(artifact_dir) / "model.ckpt"
    







    

    datamodule_kwargs = OmegaConf.to_container(cfg.task.settings) if "settings" in cfg.task else {}
    datamodule = instantiate(cfg.data, num_proc=cfg.num_proc, batch_size=cfg.batch_size, pin_memory=cfg.accelerator.pin_memory,effective_batch_size=cfg.effective_batch_size ,**datamodule_kwargs)
    optimizer = instantiate(cfg.optimizer, _partial_=True) if "optimizer" in cfg else None
    scheduler = instantiate(cfg.scheduler, _partial_=True) if "scheduler" in cfg else None
    
    model_kwargs = {
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    if hasattr(datamodule, "num_features"):
        model_kwargs["num_features"] = datamodule.num_features

    if hasattr(datamodule, "num_classes"):
        model_kwargs["num_classes"] = datamodule.num_classes
    

    model = instantiate(cfg.model.model, **model_kwargs)

<<<<<<< HEAD
    callbacks = [
        ModelCheckpoint(
            monitor=cfg.model.metrics.monitor,
            save_top_k=2,
            verbose=True,
            mode=cfg.model.metrics.mode,
        ),
        # For resuming
        ModelCheckpoint(
            monitor=None,
            save_top_k=1,
            verbose=True,
            save_on_train_epoch_end=True,
        ),
        EarlyStopping(
            monitor=cfg.model.metrics.monitor,
            patience=cfg.model.metrics.patience,
            verbose=True,
            mode=cfg.model.metrics.mode,
        ),
        LearningRateMonitor(logging_interval="step")
    ]
=======
    callbacks = []
    if cfg.run.mode == "fit":
        callbacks = [
            ModelCheckpoint(
                monitor=cfg.model.metrics.monitor,
                save_top_k=1,
                verbose=True,
                mode=cfg.model.metrics.mode,
            ),
            ModelCheckpoint(
                monitor=None,
                verbose=True,
                save_on_train_epoch_end=True,
            ),
            EarlyStopping(
                monitor=cfg.model.metrics.monitor,
                patience=cfg.model.metrics.patience,
                verbose=True,
                
                mode=cfg.model.metrics.mode,
            ),
            LearningRateMonitor(logging_interval="step")
        ]

    
>>>>>>> 778628c707c46aa4d6c3650c8f505f4e86ccdbae
    additional_callbacks = instantiate(cfg.callbacks)


    trainer: Trainer = Trainer(
        callbacks=callbacks + additional_callbacks,
        logger=logger,
        enable_model_summary=True,
        strategy=cfg.accelerator.strategy,
        devices=cfg.accelerator.devices,
        max_epochs=cfg.max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=cfg.limit_test_batches,
        fast_dev_run=cfg.fast_dev_run,
        accelerator=cfg.accelerator.accelerator,
        reload_dataloaders_every_n_epochs=1
    )

    if cfg.run.mode == "tune":
        tune(trainer, model, datamodule)

    if cfg.run.mode == "fit":
        fit(trainer, model, datamodule, ckpt_path=checkpoint_path)

    if cfg.run.mode == "validate":
        validate(trainer, model, datamodule, ckpt_path=checkpoint_path)

    if cfg.run.mode == "test":
        test(trainer, model, datamodule, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()