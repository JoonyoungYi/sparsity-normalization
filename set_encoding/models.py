import torch
import torch.nn as nn

from .modules import SAB, PMA


class DeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=100),
            nn.Tanh(),
            # nn.Linear(in_features=300, out_features=100),
            # nn.Tanh(),
            nn.Linear(in_features=100, out_features=30),
            nn.Tanh(), )
        self.dec = nn.Sequential(
            nn.Linear(in_features=30, out_features=1), )
        self.pool = pool

    def forward(self, x):
        # print(x.shape)
        x = self.enc(x)
        # print(x.shape)
        if self.pool == "max":
            x = x.max(dim=1)
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        elif self.pool == "min":
            x = x.min(dim=1)
        else:
            raise NotImplementedError()
        # print(x.shape)
        x = self.dec(x)
        # print(x.shape)
        # input('')
        return x


class SetTransformer(nn.Module):
    def __init__(
            self, ):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4), )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=1), )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)
