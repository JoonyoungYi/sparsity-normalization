import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_input_unit, n_hidden_unit, n_output_unit):
        super(MLP, self).__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_input_unit, n_hidden_unit)
        self.fc2 = nn.Linear(n_hidden_unit, n_hidden_unit)
        self.fc3 = nn.Linear(n_hidden_unit, n_output_unit)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
