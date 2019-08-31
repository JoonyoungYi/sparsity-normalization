import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, batch_norm=False,
                 dropout=False, dropout_rate=0.5, dropout_mode='default'):
        super(LeNet, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.dropout_mode = dropout_mode

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1   = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.bias2 = nn.Parameter(torch.zeros(16))

        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.leaky_relu(out)
        out = F.max_pool2d(out, 2)

        if self.dropout:
            out = F.dropout(out, self.dropout_rate, training=self.training)

            # compute multiplier
            multiplier = self.compute_multiplier_2d(out.detach(), self.dropout_rate, self.dropout_mode,
                                              6, 16, 5,
                                              training=self.training)

            # forward pass without bias
            out = self.conv2(out)
            # calibration
            out *= multiplier
            # add bias
            out += self.bias2.reshape(-1, 1, 1)
        else:
            out = self.conv2(out)
            out += self.bias2.reshape(-1, 1, 1)

        if self.batch_norm:
            out = self.bn2(out)
        out = F.leaky_relu(out)
        out = F.max_pool2d(out, 2)

        if self.dropout:
            out = F.dropout(out, self.dropout_rate, training=self.training)

            out = out.view(out.size(0), -1)

            # compute multiplier
            multiplier = self.compute_multiplier_1d(out.detach(), self.dropout_rate, self.dropout_mode,
                                  training=self.training)
            # calibration
            out *= multiplier
        else:
            out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = F.leaky_relu(out)
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)

        return out

    def compute_multiplier_1d(self, x, dropout_rate, dropout_mode,
                              training=True):

        multiplier = 1.0

        if training:
            if dropout_mode == 'default':
                multiplier = 1.0
            
            elif dropout_mode == 'sparsity-normalization':
                # original dropout automatically multiply 1 / (1 - dropout_rate)
                # restore this
                multiplier = 1 - dropout_rate

                # compute l0 norm
                l0 = (torch.abs(x) > 1e-20).sum(
                    dim=1).type(torch.cuda.FloatTensor)
                # zero to one to prevent nan
                l0 = torch.max(l0, torch.ones_like(l0))

                # calculate multiplier
                multiplier *= (x.shape[1] / l0).unsqueeze(1)
            
            else:
                raise NotImplementedError()

        return multiplier

    def compute_multiplier_2d(self, x, dropout_rate, dropout_mode,
                      in_channels, out_channels, kernel_size, stride=1, padding=0,
                      training=True):

        multiplier = 1.0

        if training:
            if dropout_mode == 'default':
                multiplier = 1.0

            elif dropout_mode == 'sparsity-normalization':
                # original dropout automatically multiply 1 / (1 - dropout_rate)
                # restore this
                multiplier = 1 - dropout_rate

                # binarize x and get x_ones
                x_ones = torch.ones_like(x)
                xb = (torch.abs(x) > 1e-20).type(torch.cuda.FloatTensor)
                # W_ones for conv2d filter filled with 1
                W_ones = torch.ones(out_channels, in_channels, kernel_size, kernel_size).cuda()

                # calculate number of contributed neurons
                l0_i = F.conv2d(x_ones, W_ones, stride=stride, padding=padding)
                l0 = F.conv2d(xb, W_ones, stride=stride, padding=padding)
                # zero to one to prevent nan
                l0 = torch.max(l0, torch.ones_like(l0))

                # calculate multiplier
                multiplier *= (l0_i / l0)

            else:
                raise NotImplementedError()
            
        return multiplier

def init_bn(m):
    if type(m) == nn.BatchNorm2d:
        # set bn in training mode to update post-statistics
        m.training = True
        # reset all values for post-statistics
        m.reset_running_stats()
        # use cumulative moving average (exact average)
        m.momentum = None

def add_momentum(m):
    if type(m) == nn.BatchNorm2d:
        m.momentum = 0.1
