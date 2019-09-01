from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _forward_conv2d(x,
                    conv,
                    batch_norm,
                    dropout_rate=0.0,
                    activation_fn=F.leaky_relu,
                    training=True,
                    use_batch_normalization=False):
    x = conv(x)
    if use_batch_normalization:
        x = batch_norm(x)
    x = activation_fn(x)
    x = F.max_pool2d(x, 2)
    if dropout_rate > 1e-20:
        x = F.dropout(x, dropout_rate, training=training)
    return x


def _forward_mlp(x, linears, activation_fn):
    for linear in linears[:-1]:
        x = linear(x)
        x = activation_fn(x)
    return linears[-1](x)


class LeNet(nn.Module):
    def __init__(self,
                 dropout_rate=0.0,
                 dropout_mode='default',
                 use_batch_normalization=False):
        super(LeNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.dropout_mode = dropout_mode
        self.use_batch_normalization = use_batch_normalization
        self.activation_fn = F.leaky_relu

        self.conv1, self.bn1 = nn.Conv2d(3, 6, 5), nn.BatchNorm2d(6)
        self.conv2, self.bn2 = nn.Conv2d(6, 16, 5), nn.BatchNorm2d(16)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

        self.convs = [self.conv1, self.conv2]
        self.batch_norms = [self.bn1, self.bn2]
        self.linears = [self.linear1, self.linear2, self.linear3]

    def forward(self, x):
        activation_fn = self.activation_fn
        dropout_rate = self.dropout_rate
        convs = self.convs
        batch_norms = self.batch_norms
        linears = self.linears
        use_batch_normalization = self.use_batch_normalization

        # 2 convolution layers
        for conv, batch_norm in zip(convs, batch_norms):
            x = _forward_conv2d(
                x,
                conv,
                batch_norm,
                activation_fn=activation_fn,
                dropout_rate=dropout_rate,
                training=self.training,
                use_batch_normalization=use_batch_normalization)
        x = x.view(x.size(0), -1)

        # full connected layers
        x = _forward_mlp(x, linears=linears, activation_fn=activation_fn)
        return x

    def start_batch_norm_post_statistics(self, train_loader=None):
        assert not self.training
        assert train_loader
        batch_norms = self.batch_norms

        # 일단 batch norm state_dict 저장
        self.batch_norm_state_dicts = [
            deepcopy(batch_norm.state_dict()) for batch_norm in batch_norms
        ]

        # exact average 취할 수 있도록 설정 변경
        for batch_norm in batch_norms:
            batch_norm.training = True
            # reset all values for post-statistics
            batch_norm.reset_running_stats()
            # use cumulative moving average (exact average)
            batch_norm.momentum = None

        # print([batch_norm.state_dict() for batch_norm in batch_norms])

        # train_loader를 이용해서 batch norm statistics 재계산
        with torch.no_grad():
            for x, y in tqdm(train_loader):
                self(x.cuda())

        # print([batch_norm.state_dict() for batch_norm in batch_norms])

    def finish_batch_norm_post_statistics(self):
        assert not self.training
        batch_norms = self.batch_norms
        state_dicts = self.batch_norm_state_dicts

        for batch_norm, state_dict in zip(batch_norms, state_dicts):
            batch_norm.load_state_dict(state_dict)

        # 다시 moving average로 돌아가도록 설정 변경
        for batch_norm in batch_norms:
            batch_norm.momentum = 0.1
            batch_norm.training = False

    # def compute_multiplier_1d(self,
    #                           x,
    #                           dropout_rate,
    #                           dropout_mode,
    #                           training=True):
    #
    #     multiplier = 1.0
    #
    #     if training:
    #         if dropout_mode == 'default':
    #             multiplier = 1.0
    #
    #         elif dropout_mode == 'sparsity-normalization':
    #             # original dropout automatically multiply 1 / (1 - dropout_rate)
    #             # restore this
    #             multiplier = 1 - dropout_rate
    #
    #             # compute l0 norm
    #             l0 = (torch.abs(x) > 1e-20).sum(
    #                 dim=1).type(torch.cuda.FloatTensor)
    #             # zero to one to prevent nan
    #             l0 = torch.max(l0, torch.ones_like(l0))
    #
    #             # calculate multiplier
    #             multiplier *= (x.shape[1] / l0).unsqueeze(1)
    #
    #         else:
    #             raise NotImplementedError()
    #
    #     return multiplier
    #
    # def compute_multiplier_2d(self,
    #                           x,
    #                           dropout_rate,
    #                           dropout_mode,
    #                           in_channels,
    #                           out_channels,
    #                           kernel_size,
    #                           stride=1,
    #                           padding=0,
    #                           training=True):
    #     multiplier = 1.0
    #     if training:
    #         if dropout_mode == 'default':
    #             multiplier = 1.0
    #
    #         elif dropout_mode == 'sparsity-normalization':
    #             # original dropout automatically multiply 1 / (1 - dropout_rate)
    #             # restore this
    #             multiplier = 1 - dropout_rate
    #
    #             # binarize x and get x_ones
    #             x_ones = torch.ones_like(x)
    #             xb = (torch.abs(x) > 1e-20).type(torch.cuda.FloatTensor)
    #             # W_ones for conv2d filter filled with 1
    #             W_ones = torch.ones(out_channels, in_channels, kernel_size,
    #                                 kernel_size).cuda()
    #
    #             # calculate number of contributed neurons
    #             l0_i = F.conv2d(x_ones, W_ones, stride=stride, padding=padding)
    #             l0 = F.conv2d(xb, W_ones, stride=stride, padding=padding)
    #             # zero to one to prevent nan
    #             l0 = torch.max(l0, torch.ones_like(l0))
    #
    #             # calculate multiplier
    #             multiplier *= (l0_i / l0)
    #
    #         else:
    #             raise NotImplementedError()
    #
    #     return multiplier

    # def _________init_bn(m):
    #     if type(m) == nn.BatchNorm2d:
    #         # set bn in training mode to update post-statistics
    #         m.training = True
    #         # reset all values for post-statistics
    #         m.reset_running_stats()
    #         # use cumulative moving average (exact average)
    #         m.momentum = None
    #
    #
    # def _________add_momentum(m):
    #     if type(m) == nn.BatchNorm2d:
    #         m.momentum = 0.1
