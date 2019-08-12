import numpy as np
from tqdm import tqdm

from .models import MultiLayerPerceptron
from .models import DropOut
from .models import DropConnect


def _simulate(model, n_trial=1000):
    ys = []
    for _ in tqdm(range(n_trial)):
        y = model()
        ys.append(y[0])
    ys = np.array(ys)
    mean = np.mean(ys)
    std = np.std(ys)
    return mean, std


def main():
    n_units = [100, 100, 100, 1]
    # n_units = [8, 8, 8, 1]

    # model = MultiLayerPerceptron(n_units=n_units)

    model = DropOut(mode='default', n_units=n_units)
    # model = DropOut(mode='exact', n_units=n_units)
    # model = DropOut(mode='expectation', n_units=n_units)

    # model = DropConnect(mode='default', n_units=n_units)
    # model = DropConnect(mode='exact', n_units=n_units)
    # model = DropConnect(mode='expectation', n_units=n_units)

    mean, std = _simulate(model)
    print('Mean:', '%10.2f' % mean)
    print('Std :', '%10.2f' % std)


if __name__ == '__main__':
    main()
