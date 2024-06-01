import torch
from torch.utils.data import DataLoader

from code.classes.Model import NN
from code.classes.TwitterDataset import TwitterDataset, PreprocessAndVectorize
from code.functions.check_accuracy import check_accuracy


def infer(input_size, num_classes, batch_size):
    model = NN(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load('model.pth'))


    test_dataset = TwitterDataset(mode='test', transform=PreprocessAndVectorize())
    testdataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


    print(f"Accuracy on test set: {check_accuracy(testdataloader, model) * 100:.2f}")