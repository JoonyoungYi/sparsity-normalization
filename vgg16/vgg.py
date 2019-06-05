import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

cfg = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

class Conv2d_sparsity_normal(nn.Module):
    """Convolutional layer for sparsity normalization. (See Algorithm 2 in the paper)"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, fan_in_K_prime):
        super(Conv2d_sparsity_normal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.fan_in_K_prime = fan_in_K_prime

        # Remove bias from a convolution layer. (See Algorithm 2 in the paper)
        self.conv = nn.Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=False)

        # Make a copy of self.conv which is not trainable
        # and set its weights to one. (See Algorithm 2 in the paper)
        self.conv_C = nn.Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=False)

        self.conv_C.weight.requires_grad = False
        nn.init.constant_(self.conv_C.weight, 1.0)

        # Define bias. (See Algorithm 2 in the paper)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

        self.fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
        bound = 1 / sqrt(self.fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # These procedure follows Algorithm 2 of the paper.
        binarized_x = (x > 1e-5).type(torch.cuda.FloatTensor)
        binarized_x.requires_grad = False

        x = self.conv(x)
        divider = self.conv_C(binarized_x)
        divider = torch.max(divider, torch.Tensor([1.0]).to('cuda'))

        if self.fan_in_K_prime:
            x *= self.fan_in
        x = torch.div(x, divider)

        x = x + self.bias.unsqueeze(1).unsqueeze(1)
        return x 

class Conv2d_hidden_normal(nn.Module):
    """Convolutional layer for hidden layer normalization."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, fan_in_K_prime):
        super(Conv2d_hidden_normal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.fan_in_K_prime = fan_in_K_prime

        # Remove bias from a convolution layer. (See Algorithm 2 in the paper)
        self.conv = nn.Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=False)
        
        # Make a copy of self.conv which is not trainable
        # and set its weights to one. (See Algorithm 2 in the paper)
        self.conv_C = nn.Conv2d(self.in_channels,
                         self.out_channels,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=False)
        
        self.conv_C.weight.requires_grad = False
        nn.init.constant_(self.conv_C.weight, 1.0)

        # Define bias. (See Algorithm 2 in the paper)
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        
        self.fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
        bound = 1 / sqrt(self.fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        sigmoid_x = torch.sigmoid(x)

        x = F.relu(x)
        x = self.conv(x)

        divider = self.conv_C(sigmoid_x)

        if self.fan_in_K_prime:
            x *= self.fan_in
        x = torch.div(x, divider)

        x = x + self.bias.unsqueeze(1).unsqueeze(1)
        return x

class VGG(nn.Module):
    """VGG architecture"""
    def __init__(self, vgg_name, sparsity_normalization, normalize_layer_indexes, batch_norm, dropout, fan_in_K_prime):
        super(VGG, self).__init__()
        self.sparsity_normalization = sparsity_normalization
        self.normalize_layer_indexes = normalize_layer_indexes
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.fan_in_K_prime = fan_in_K_prime

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
        layer_idx = 0

        for x in cfg:
            # Add max pooling layers
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if self.dropout:
                    layers += [nn.Dropout(p=0.25)]

            # Add convolutional layers
            else:
                out_channels = x

                if layer_idx in self.normalize_layer_indexes:
                    layers.pop() # There is a relu activation inside the Conv2d_hidden_normal
                    layers += [Conv2d_hidden_normal(in_channels, out_channels, kernel_size=3, padding=1, self.fan_in_K_prime)]
                elif layer_idx == 0 and self.sparsity_normalization:
                    layers += [Conv2d_sparsity_normal(in_channels, out_channels, kernel_size=3, padding=1, self.fan_in_K_prime)]
                else:
                    layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]

                if batch_norm:
                    layers += [nn.BatchNorm2d(out_channels)]

                layers += [nn.ReLU(inplace=True)]
                layer_idx += 1
                           
                in_channels = out_channels
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
