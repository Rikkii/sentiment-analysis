# Check accuracy on training & test to see how good our model
import torch


def check_accuracy(loader, model):
    """
    Check accuracy of our trained model given a loader and a model

    Parameters:
        loader: torch.functions.data.DataLoader
            A loader for the dataset you want to check accuracy on
        model: nn.Module
            The model you want to check accuracy on

    Returns:
        acc: float
            The accuracy of the model on the dataset given by the loader
    """
    num_samples = len(loader.dataset)
    num_correct = 0

    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        for src, target in loader:
            scores = model(src)
            _, predictions = torch.max(scores, 1)
            num_correct += torch.sum(predictions == target).item()

    model.train()

    acc = num_correct / num_samples
    return acc