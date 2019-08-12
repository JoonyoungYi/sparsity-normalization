import numpy as np
from tqdm import tqdm

from .models import DropOut


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
    model = DropOut(mode='default')
    # model = DropOut(mode='exact')
    # model = DropOut(mode='expectation')

    mean, std = _simulate(model)
    print('Mean:', '%10.2f' % mean)
    print('Std :', '%10.2f' % std)


if __name__ == '__main__':
    main()
