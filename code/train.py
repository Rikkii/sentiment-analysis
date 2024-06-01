# Imports
import torch
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader
from tqdm import tqdm  # For nice progress bar!

from code.classes.Model import NN
from code.classes.TwitterDataset import TwitterDataset, PreprocessAndVectorize


def train():
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 300
    num_classes = 4
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 3

    dataset = TwitterDataset(transform=PreprocessAndVectorize())
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # # Initialize network
    model = NN(input_size=input_size, num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
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
