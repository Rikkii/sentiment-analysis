from code.classes.twitter_dataset import PreprocessAndVectorize, TwitterDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class TwitterDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TwitterDataset(
            mode="train", transform=PreprocessAndVectorize()
        )
        self.test_dataset = TwitterDataset(
            mode="test", transform=PreprocessAndVectorize()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
