import numpy as np


class MultiLayerPerceptron:
    def __init__(self, n_units):
        assert n_units and len(n_units) >= 2
        assert n_units[-1] == 1
        weights = [
            np.ones([n_in, n_out])
            for n_in, n_out in zip(n_units[:-1], n_units[1:])
        ]
        x = np.ones(n_units[0])

        self.weights = weights
        self.x = x


class DropOut(MultiLayerPerceptron):
    def __init__(self,
                 n_units=[100, 100, 100, 1],
                 dropout_rate=0.5,
                 mode='default'):
        MultiLayerPerceptron.__init__(self, n_units)
        assert mode in ['default', 'exact', 'expectation']

        self.mode = mode
        self.dropout_rate = dropout_rate

    def __call__(self):
        x = self.x
        weights = self.weights
        dropout_rate = self.dropout_rate
        mode = self.mode

        h = x
        for w in weights[:-1]:
            h = np.matmul(h, w)

            m = np.ones(h.shape[0])
            m[np.random.rand(h.shape[0]) < dropout_rate] = 0
            if mode == 'default':
                h = np.multiply(h, m)
            elif mode == 'exact':
                h = np.multiply(h, m) * h.shape[0] / np.sum(m)
            elif mode == 'expectation':
                h = np.multiply(h, m) * 2
            else:
                raise NotImplementedError()

        return np.matmul(h, weights[-1])
