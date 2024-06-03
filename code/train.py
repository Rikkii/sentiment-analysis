import torch
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader
from tqdm import tqdm  # For nice progress bar!
import hydra
from omegaconf import DictConfig

from code.classes.Model import NN
from code.classes.TwitterDataset import TwitterDataset, PreprocessAndVectorize

@hydra.main(config_path= 'config', config_name='config.yaml', version_base='1.3.2')
def train(cfg: DictConfig):

    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TwitterDataset(transform=PreprocessAndVectorize())
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.params.batch_size, shuffle=True)

    # # Initialize network
    model = NN(input_size=cfg.params.input_size, num_classes=cfg.params.num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.params.learning_rate)

    # Train Network
    for epoch in range(cfg.params.num_epochs):
        for src, target in tqdm(dataloader):

            scores = model(src)
            loss = criterion(scores, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent or adam step
            optimizer.step()

    # Saving trained network
    torch.save(model.state_dict(), 'model.pth')
