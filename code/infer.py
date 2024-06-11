from code.classes.Model import NN
from code.classes.TwitterDataModule import TwitterDataModule

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3.2")
def infer(cfg: DictConfig):

    model = NN.load_from_checkpoint(
        "model.ckpt",
        input_size=cfg.params.input_size,
        num_classes=cfg.params.num_classes,
    )
    trainer = pl.Trainer()

    data_module = TwitterDataModule(batch_size=cfg.params.batch_size)
    data_module.setup()

    trainer.test(model, data_module.test_dataloader())
