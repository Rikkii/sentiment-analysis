from torch import nn  # All neural network modules

class NN(nn.Module):

    def __init__(self, input_size, num_classes):

        super(NN, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.model = nn.Sequential(
        nn.Linear(input_size, 10),
        nn.ReLU(),
        nn.Linear(10, num_classes)
        )

    def forward(self, x):

        x = self.model(x)
        return x





