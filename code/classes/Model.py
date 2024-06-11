import pytorch_lightning as pl
import torch
from torch import nn  # All neural network modules


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, num_classes),
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.layers(x)
        loss = self.loss_func(pred, y)
        self.log("train_loss", loss)
        return loss

    # Настраиваются параметры тестирования
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.layers(x)
        loss = self.loss_func(pred, y)
        pred = torch.argmax(pred, dim=1)
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", torch.tensor(accuracy), prog_bar=True)
        output = dict(
            {
                "test_loss": loss,
                "test_acc": torch.tensor(accuracy),
            }
        )
        return output

    # Конфигурируется оптимизатор
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
