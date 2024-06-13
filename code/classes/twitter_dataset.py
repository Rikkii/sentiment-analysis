from code.functions.preprocess_and_vectorize import preprocess_and_vectorize

import pandas as pd
from torch.utils.data import Dataset


class TwitterDataset(Dataset):
    def __init__(self, mode="train", transform=None):
        if mode == "train":
            data = pd.read_csv("data/twitter_training_cleaned.csv")
        elif mode == "test":
            data = pd.read_csv("data/twitter_validation_cleaned.csv")
        else:
            raise Exception("enter correct mode parameter")

        self.x = data["text"]
        self.y = data["sentiment"]
        self.n_samples = data.shape[0]
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample


class PreprocessAndVectorize:
    def __call__(self, sample):
        x, y = sample
        return preprocess_and_vectorize(x, y)
