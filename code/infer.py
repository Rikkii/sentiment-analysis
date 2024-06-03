import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from code.classes.Model import NN
from code.classes.TwitterDataset import TwitterDataset, PreprocessAndVectorize
from code.functions.check_accuracy import check_accuracy

@hydra.main(config_path= 'config', config_name='config.yaml', version_base='1.3.2')
def infer(cfg: DictConfig):

    model = NN(input_size=cfg.params.input_size, num_classes=cfg.params.num_classes)
    model.load_state_dict(torch.load('model.pth'))


    test_dataset = TwitterDataset(mode='test', transform=PreprocessAndVectorize())
    testdataloader = DataLoader(dataset=test_dataset, batch_size=cfg.params.batch_size, shuffle=True)


    print(f"Accuracy on test set: {check_accuracy(testdataloader, model) * 100:.2f}")