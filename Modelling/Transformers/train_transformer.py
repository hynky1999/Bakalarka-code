from argparse import ArgumentParser
import wandb
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner.tuning import Tuner

def fit(trainer, model, datamodule):
    trainer.fit(model, datamodule)

def validate(trainer, model, datamodule):
    trainer.validate(model, datamodule)

def test(trainer, model, datamodule):
    trainer.test(model, datamodule)

def tune(trainer, model, datamodule):
    tuner = Tuner(trainer)
    batch_size = tuner.scale_batch_size(model=model, datamodule=datamodule, mode="binsearch")
    print(f"Found Batch size: {batch_size}")

    lr_finder = tuner.lr_find(model=model, datamodule=datamodule)
    print(f"Found lr: {lr_finder.suggestion()}")
    wandb.log({"lr_graph": lr_finder.plot(suggest=True)})



class MyCli(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: ArgumentParser):
        parser.add_argument("--tune", action="store_true", default=False)
        parser.add_argument("--fit", action="store_true", default=False)
        parser.add_argument("--test", action="store_true", default=False)
        parser.add_argument("--validate", action="store_true", default=False)


if __name__ == "__main__":
    cli = MyCli(seed_everything_default=88, run=False, save_config_overwrite=True)
    print(cli.trainer.num_devices)

    if cli.config.tune == True:
        tune(cli.trainer, cli.model, cli.datamodule)
    
    if cli.config.fit == True:
        fit(cli.trainer, cli.model, cli.datamodule)

    if cli.config.validate == True:
        validate(cli.trainer, cli.model, cli.datamodule)

    if cli.config.test == True:
        test(cli.trainer, cli.model, cli.datamodule)



