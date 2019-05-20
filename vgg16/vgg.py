import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

cfg = {
    'vanilla': ['64_X_X', '64_X_X', 'M', '128_X_X', '128_X_X', 'M', '256_X_X', '256_X_X', '256_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M'],
    'normal-0': ['64_X_I', '64_X_X', 'M', '128_X_X', '128_X_X', 'M', '256_X_X', '256_X_X', '256_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M'],
    'dropout': ['64_X_X', '64_X_X', 'M', '128_X_X', '128_X_X', 'M', '256_X_X', '256_X_X', '256_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M'],
    'dropout_normal-0': ['64_X_I', '64_X_X', 'M', '128_X_X', '128_X_X', 'M', '256_X_X', '256_X_X', '256_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M'],
    'normal-07': ['64_X_I', '64_X_X', 'M', '128_X_X', '128_X_X', 'M', '256_X_X', '256_X_X', '256_X_N', 'M', '512_X_X', '512_X_X', '512_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M'],
    'normal-049': ['64_X_I', '64_X_X', 'M', '128_X_X', '128_X_N', 'M', '256_X_X', '256_X_X', '256_X_X', 'M', '512_X_X', '512_X_N', '512_X_X', 'M', '512_X_X', '512_X_X', '512_X_X', 'M'],
    'normal-04712': ['64_X_I', '64_X_X', 'M', '128_X_X', '128_X_N', 'M', '256_X_X', '256_X_X', '256_X_N', 'M', '512_X_X', '512_X_X', '512_X_X', 'M', '512_X_X', '512_X_N', '512_X_X', 'M'],
    'normal-all': ['64_X_I', '64_X_N', 'M', '128_X_X', '128_X_N', 'M', '256_X_X', '256_X_N', '256_X_N', 'M', '512_X_X', '512_X_N', '512_X_N', 'M', '512_X_X', '512_X_N', '512_X_N', 'M'],
    'bn_vanilla': ['64_B_X', '64_B_X', 'M', '128_B_X', '128_B_X', 'M', '256_B_X', '256_B_X', '256_B_X', 'M', '512_B_X', '512_B_X', '512_B_X', 'M', '512_B_X', '512_B_X', '512_B_X', 'M'],
    'bn_normal-0': ['64_B_I', '64_B_X', 'M', '128_B_X', '128_B_X', 'M', '256_B_X', '256_B_X', '256_B_X', 'M', '512_B_X', '512_B_X', '512_B_X', 'M', '512_B_X', '512_B_X', '512_B_X', 'M'],
    'bn_normal-07': ['64_B_I', '64_B_X', 'M', '128_B_X', '128_B_X', 'M', '256_B_X', '256_B_X', '256_B_N', 'M', '512_B_X', '512_B_X', '512_B_X', 'M', '512_B_X', '512_B_X', '512_B_X', 'M'],
    'bn_normal-049': ['64_B_I', '64_B_X', 'M', '128_B_X', '128_B_N', 'M', '256_B_X', '256_B_X', '256_B_X', 'M', '512_B_X', '512_B_N', '512_B_X', 'M', '512_B_X', '512_B_X', '512_B_X', 'M'],
    'bn_normal-04712': ['64_B_I', '64_B_X', 'M', '128_B_X', '128_B_N', 'M', '256_B_X', '256_B_X', '256_B_N', 'M', '512_B_X', '512_B_X', '512_B_X', 'M', '512_B_X', '512_B_N', '512_B_X', 'M'],
    'bn_normal-all': ['64_B_I', '64_B_N', 'M', '128_B_X', '128_B_N', 'M', '256_B_X', '256_B_N', '256_B_N', 'M', '512_B_X', '512_B_N', '512_B_N', 'M', '512_B_X', '512_B_N', '512_B_N', 'M']
}

class Conv2d_input_normal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(Conv2d_input_normal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=False)

        self.conv_C = nn.Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=False)

        self.conv_C.weight.requires_grad = False
        nn.init.constant_(self.conv_C.weight, 1.0)

        self.bias = nn.Parameter(torch.zeros(self.out_channels))

        self.fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
        bound = 1 / sqrt(self.fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        binarized_x = (x > 1e-5).type(torch.cuda.FloatTensor)
        binarized_x.requires_grad = False

        x = self.conv(x)
        divider = self.conv_C(binarized_x)
        divider = torch.max(divider, torch.Tensor([1.0]).to('cuda'))

        x = torch.div(x*self.fan_in, divider)

        x = x + self.bias.unsqueeze(1).unsqueeze(1)
        return x 

class Conv2d_hidden_normal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(Conv2d_hidden_normal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=False)
        
        self.conv_C = nn.Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=False)
        
        self.conv_C.weight.requires_grad = False
        nn.init.constant_(self.conv_C.weight, 1.0)

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        
        self.fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
        bound = 1 / sqrt(self.fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        sigmoid_x = torch.sigmoid(x)

        x = F.relu(x)
        x = self.conv(x)

        divider = self.conv_C(sigmoid_x)

        x = torch.div(x*self.fan_in, divider)

        x = x + self.bias.unsqueeze(1).unsqueeze(1)
        return x

class VGG(nn.Module):
    def __init__(self, vgg_name, dropout):
        super(VGG, self).__init__()
        self.dropout = dropout
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if self.dropout:
                    layers += [nn.Dropout(p=0.25)]
            else:
                x = x.split('_')
                out_channels, batch_norm, input_norm, hidden_norm = int(x[0]), x[1] == 'B', x[2] == 'I', x[2] == 'N'

                if hidden_norm:
                    layers.pop()
                    layers += [Conv2d_hidden_normal(in_channels, out_channels, kernel_size=3, padding=1)]
                elif input_norm:
                    layers += [Conv2d_input_normal(in_channels, out_channels, kernel_size=3, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]

                if batch_norm:
                    layers += [nn.BatchNorm2d(out_channels)]

                layers += [nn.ReLU(inplace=True)]
                           
                in_channels = out_channels
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
