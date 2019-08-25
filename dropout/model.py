import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_unit, layer_num, dropout_rate,
                 dropout_option):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_unit = hidden_unit
        self.layer_num = layer_num
        self.dropout_rate = dropout_rate
        self.dropout_option = dropout_option

        self.linears = [nn.Linear(self.input_dim, self.hidden_unit)]
        for i in range(self.layer_num - 1):
            self.linears.append(nn.Linear(self.hidden_unit, self.hidden_unit))
        self.linears = nn.ModuleList(self.linears)

        self.out = nn.Linear(self.hidden_unit, 1)

    def forward(self, x, dropout_new=None):
        for linear in self.linears:
            x = linear(x)

            # control sparsity for ablation study
            if dropout_new is not None:
                x = F.dropout(x, p=dropout_new, training=self.training)
                if self.training:
                    x = x * (1 - dropout_new)

            # normal forward pass
            else:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                if self.training:
                    x = x * (1 - self.dropout_rate)

            if self.training:
                if self.dropout_option == 'sn':
                    l0 = (torch.abs(x).detach() > 1e-20).sum(
                        dim=1).type(torch.FloatTensor)
                    l0 = torch.max(l0, torch.ones_like(l0))
                    multiplier = (self.hidden_unit / l0).unsqueeze(1)
                    x = x * multiplier.cuda()
                elif self.dropout_option == 'default_train':
                    x = x * 1 / (1 - self.dropout_rate)
                elif self.dropout_option == 'default_test':
                    pass
                elif self.dropout_option == 'none':
                    pass
                else:
                    raise NotImplementedError()
            else:
                if self.dropout_option == 'default_test':
                    x = x * (1 - self.dropout_rate)

            x = F.relu(x)

        y_pred = self.out(x)

        return y_pred
