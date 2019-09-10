import math

import torch
import numpy as np
from tqdm import tqdm

from .dataset import Dataset
from .models import MLP


def main():
    dataset = Dataset()
    # print(dataset.train_x)
    # print(dataset.train_y)
    train_x = dataset.train_x
    train_y = dataset.train_y
    test_x = dataset.test_x
    test_y = dataset.test_y

    model = MLP(
        n_input_unit=train_x.shape[1],
        n_output_unit=train_y.shape[1],
        n_hidden_unit=50)
    model = model.cuda()

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, eps=1e-8)

    min_test_rmse = float('inf')

    for epoch_idx in tqdm(range(100000)):
        train_y_hat = model(train_x)
        train_loss = criterion(train_y_hat, train_y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_rmse = math.sqrt(train_loss.item())

        test_y_hat = model(test_x)
        test_loss = criterion(test_y_hat, test_y)
        test_rmse = np.sqrt(test_loss.item())

        if min_test_rmse > test_rmse:
            min_test_rmse = test_rmse
            # print(test_y_hat[0], test_y[0])

        if epoch_idx % 100 == 0:
            print(
                epoch_idx,
                '%.4f' % train_rmse,
                '%.4f' % test_rmse,
                '%.4f' % min_test_rmse, )


if __name__ == '__main__':
    main()
