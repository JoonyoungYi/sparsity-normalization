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

    def __call__(self):
        x = self.x
        weights = self.weights

        h = x
        for w in weights[:-1]:
            h = np.matmul(h, w)
        return np.matmul(h, weights[-1])


class DropOut(MultiLayerPerceptron):
    def __init__(self, n_units, drop_rate=0.5, mode='default'):
        MultiLayerPerceptron.__init__(self, n_units)
        assert mode in ['default', 'exact', 'expectation']

        self.mode = mode
        self.drop_rate = drop_rate

    def __call__(self):
        x = self.x
        weights = self.weights
        drop_rate = self.drop_rate
        mode = self.mode

        h = x
        for w in weights[:-1]:
            h = np.matmul(h, w)

            m = np.ones(h.shape[0])
            m[np.random.rand(h.shape[0]) < drop_rate] = 0
            if mode == 'default':
                h = np.multiply(h, m)
            elif mode == 'exact':
                h = np.multiply(h, m) * h.shape[0] / np.sum(m)
            elif mode == 'expectation':
                h = np.multiply(h, m) / (1 - drop_rate)
            else:
                raise NotImplementedError()

        return np.matmul(h, weights[-1])


class DropConnect(MultiLayerPerceptron):
    def __init__(self, n_units, drop_rate=0.5, mode='default'):
        MultiLayerPerceptron.__init__(self, n_units)
        assert mode in ['default', 'exact', 'expectation']

        self.mode = mode
        self.drop_rate = drop_rate

    def __call__(self):
        x = self.x
        weights = self.weights
        drop_rate = self.drop_rate
        mode = self.mode

        h = x
        for w in weights[:-1]:
            m = np.ones(w.shape)
            m[np.random.rand(w.shape[0], w.shape[1]) < drop_rate] = 0
            # print(w)
            w = np.multiply(w, m)
            # print(w)
            h = np.matmul(h, w)

            if mode == 'default':
                pass  # do nothing
            elif mode == 'exact':
                h = h * h.shape[0] / np.sum(m, axis=0)
            elif mode == 'expectation':
                h = h / (1 - drop_rate)
            else:
                raise NotImplementedError()
            # print(h)
            # input('')

            # elif mode == 'exact':
            #     h = np.multiply(h, m) * h.shape[0] / np.sum(m)
            # elif mode == 'expectation':
            #     h = np.multiply(h, m) * 2

        return np.matmul(h, weights[-1])
