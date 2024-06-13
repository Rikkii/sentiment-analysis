from code.classes.model import NN
from code.classes.twitter_data_module import TwitterDataModule

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3.2")
def train(cfg: DictConfig):

    data_module = TwitterDataModule(batch_size=cfg.params.batch_size)
    data_module.setup()

    loggers = [
        CSVLogger("./.logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri="file:./.logs/my-mlflow-logs",
        ),
    ]

    # Initialize network
    model = NN(input_size=cfg.params.input_size, num_classes=cfg.params.num_classes)
    trainer = pl.Trainer(max_epochs=cfg.params.num_epochs, logger=loggers)

    trainer.fit(model, data_module.train_dataloader())

    # Saving trained network
    trainer.save_checkpoint("model.ckpt")
