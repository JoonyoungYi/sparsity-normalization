import torch
from torch import nn


class Dropout(nn.Module):
    def __init__(self, drop_rate=0.5, drop_mode="standard"):
        super(Dropout, self).__init__()
        assert drop_mode in ["standard", "sparsity-normalization"]
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(drop_rate))

        self.drop_rate = drop_rate
        self.drop_mode = drop_mode

    def forward(self, X):
        # 귀찮아서 cuda 쓴다고 가정했음.
        drop_mode = self.drop_mode

        if self.training:
            keep_prob = 1 - self.drop_rate
            binomial = torch.distributions.binomial.Binomial(probs=keep_prob)
            m = binomial.sample(X.size()).cuda()
            y = X * m
            if drop_mode == "standard":
                return y * (1.0 / keep_prob)
            elif drop_mode == "sparsity-normalization":
                l0 = (m.detach() > 1e-20).sum(
                    dim=-1, keepdims=True).type(torch.FloatTensor).cuda()
                return y * (m.shape[-1] / l0)
            else:
                raise NotImplementedError()
        return X
