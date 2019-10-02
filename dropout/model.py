import torch
import torch.nn as nn
import torch.nn.functional as F


def _dropout(x,
             training,
             dropout_rate,
             dropout_mode,
             n_hidden_unit,
             original_dropout_rate=None):
    if training:
        x = F.dropout(x, p=dropout_rate)

        if dropout_mode == 'default':
            if original_dropout_rate is not None:
                # 예외적 self 사용
                multiplier = (1 - dropout_rate) / (1 - original_dropout_rate)
            else:
                multiplier = 1.0
        elif dropout_mode == 'sparsity-normalization':
            multiplier = 1 - dropout_rate
            l0 = (torch.abs(x).detach() > 1e-20).sum(
                dim=1).type(torch.FloatTensor)
            l0 = torch.max(l0, torch.ones_like(l0))
            multiplier *= (n_hidden_unit / l0).unsqueeze(1)
            multiplier = multiplier.cuda()
        else:
            raise NotImplementedError()
        x = x * multiplier

    return x, (torch.abs(x).detach() > 1e-20).sum(
        dim=1).type(torch.FloatTensor)


class MLP(nn.Module):
    def __init__(self, n_input_unit, n_output_unit, n_hidden_unit, n_hidden_layer,
                 dropout_rate, dropout_mode, activation_fn):
        super().__init__()

        linears = [nn.Linear(n_input_unit, n_hidden_unit)]
        for _ in range(n_hidden_layer - 1):
            linears.append(nn.Linear(n_hidden_unit, n_hidden_unit))
        linears.append(nn.Linear(n_hidden_unit, n_output_unit))

        # self.n_input_unit = n_input_unit
        self.n_hidden_unit = n_hidden_unit
        self.n_hidden_layer = n_hidden_layer
        self.dropout_rate = dropout_rate
        self.dropout_mode = dropout_mode
        self.activation_fn = activation_fn
        self.linears = nn.ModuleList(linears)

    def forward(self, x, test_dropout_rate=0.0, return_mask=False):
        linears = self.linears
        training = self.training
        n_hidden_unit = self.n_hidden_unit
        dropout_rate = self.dropout_rate
        dropout_mode = self.dropout_mode
        activation_fn = self.activation_fn
        if test_dropout_rate > 1e-20:
            assert not training
            training = True
            dropout_rate = test_dropout_rate

        ms = []
        for linear in linears[:-1]:
            x = linear(x)

            if activation_fn == 'relu':
                x, m = _dropout(
                    x,
                    training,
                    dropout_rate,
                    dropout_mode,
                    n_hidden_unit,
                    original_dropout_rate=self.dropout_rate
                    if test_dropout_rate > 1e-20 else None)
                x = F.relu(x)

            elif activation_fn in ['leaky_relu', 'elu', 'sigmoid', 'tanh']:
                if activation_fn == 'leaky_relu':
                    x = F.leaky_relu(x)
                elif activation_fn == 'elu':
                    x = nn.ELU()(x)
                elif activation_fn == 'sigmoid':
                    x = torch.sigmoid(x)
                elif activation_fn == 'tanh':
                    x = torch.tanh(x)
                else:
                    raise NotImplementedError()

                x, m = _dropout(
                    x,
                    training,
                    dropout_rate,
                    dropout_mode,
                    n_hidden_unit,
                    original_dropout_rate=self.dropout_rate
                    if test_dropout_rate > 1e-20 else None)
            else:
                raise NotImplementedError()

            ms.append(m)

        y_pred = linears[-1](x)

        if return_mask:
            return y_pred, ms
        else:
            return y_pred
