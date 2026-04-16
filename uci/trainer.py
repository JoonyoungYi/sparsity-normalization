import math
import argparse

import torch
import numpy as np
from tqdm import tqdm

from .dataset import Dataset
from .models import MLP
from common.utils import set_torch_seed, str2bool, save_results_to_file

DATASETS = ['boston', 'diabetes', 'california', 'wine', 'cancer']


def _train(args):
    dataset = Dataset(
        dataset=args.dataset,
        sparsity_normalization=args.sparsity_normalization,
        imputation=args.imputation,
        input_normalization=args.input_normalization,
        missing_rate=args.missing_rate,
        seed=args.seed)

    set_torch_seed(args.seed)

    train_x = dataset.train_x
    train_y = dataset.train_y
    test_x = dataset.test_x
    test_y = dataset.test_y

    model = MLP(
        n_input_unit=train_x.shape[1],
        n_output_unit=train_y.shape[1],
        n_hidden_unit=260)
    model = model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, eps=1e-8)

    min_test_rmse = float('inf')
    min_test_epoch_idx = -1

    for epoch_idx in tqdm(range(1000000)):
        train_y_hat = model(train_x)
        train_loss = criterion(train_y_hat, train_y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_rmse = math.sqrt(train_loss.item())

        with torch.no_grad():
            test_y_hat = model(test_x)
            test_loss = criterion(test_y_hat, test_y)
        test_rmse = np.sqrt(test_loss.item())

        if min_test_rmse > test_rmse:
            min_test_rmse = test_rmse
            min_test_epoch_idx = epoch_idx

        if min_test_epoch_idx + 10000 < epoch_idx:
            break

        if epoch_idx % 100 == 0:
            print(epoch_idx, '%.4f' % train_rmse, '%.4f' % test_rmse,
                  '%.4f' % min_test_rmse)

    results = [
        args.seed, args.imputation, args.sparsity_normalization,
        args.input_normalization, args.missing_rate, min_test_rmse
    ]
    save_results_to_file('uci', args.dataset, results)


def main():
    parser = argparse.ArgumentParser(description='UCI regression with Sparsity Normalization')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sparsity_normalization', type=str2bool, default=False)
    parser.add_argument('--imputation', type=str, default='zero',
                        choices=['mean', 'zero'])
    parser.add_argument('--input_normalization', type=str, default='min-max',
                        choices=['min-max', 'standardization'])
    parser.add_argument('--missing_rate', type=float, default=0.5,
                        help='결측값 삽입 비율 (0~1)')
    parser.add_argument('--dataset', type=str, default='boston',
                        choices=DATASETS)
    args = parser.parse_args()

    if args.imputation == 'mean':
        assert not args.sparsity_normalization, \
            'mean imputation은 sparsity normalization과 함께 사용할 수 없습니다.'

    _train(args)


if __name__ == '__main__':
    main()
