import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_input_unit, n_hidden_unit, n_hidden_layer,
                 dropout_rate, dropout_mode):
        super().__init__()

        linears = [nn.Linear(n_input_unit, n_hidden_unit)]
        for _ in range(n_hidden_layer):
            linears.append(nn.Linear(n_hidden_unit, n_hidden_unit))
        linears.append(nn.Linear(n_hidden_unit, 1))

        self.n_input_unit = n_input_unit
        self.n_hidden_unit = n_hidden_unit
        self.n_hidden_layer = n_hidden_layer
        self.dropout_rate = dropout_rate
        self.dropout_mode = dropout_mode
        self.activation_function = F.relu
        self.linears = nn.ModuleList(linears)

    def forward(self, x, test_dropout_rate=0.0):
        linears = self.linears
        activation_function = self.activation_function
        training = self.training
        dropout_rate = self.dropout_rate
        multiplier = 1.0
        dropout_mode = self.dropout_mode
        if test_dropout_rate > 1e-20:
            assert not training
            training = True
            dropout_rate = test_dropout_rate

        if activation_function not in [F.relu, F.leaky_relu]:
            print("relu와 leaky relu가 아닐 경우 dropout 메커니즘을 바꿔야 합니다.")
            raise NotImplementedError()

        for linear in linears[:-1]:
            x = linear(x)
            x = F.dropout(x, p=dropout_rate, training=training)

            if dropout_mode == 'default':
                if test_dropout_rate > 1e-20:
                    multiplier = (1 - test_dropout_rate) / (
                        1 - self.dropout_rate)  # 예외적 self 사용
            elif dropout_mode == 'sparsity-normalization':
                n_unit = x.shape[1]
                l0 = (torch.abs(x).detach() > 1e-20).sum(
                    dim=1).type(torch.FloatTensor)
                l0 = torch.max(l0, torch.ones_like(l0))
                multiplier = (n_unit / l0).unsqueeze(1).cuda()
            else:
                raise NotImplementedError()
            x = x * multiplier

            x = activation_function(x)

        y_pred = linears[-1](x)
        return y_pred
