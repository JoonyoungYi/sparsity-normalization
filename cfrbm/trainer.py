import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from common.datasets.movielens import Dataset
from common.utils import set_torch_seed
from common.utils import str2bool
from .models import RBM


def __expand(x, n_class=5):
    x = x.astype(np.int64)
    new_x = np.eye(n_class + 1)[x][:, :, 1:]
    new_x = new_x.reshape([-1, n_class * x.shape[1]])
    return new_x.astype(np.float32)


# def __get_rmse()

def _train(args):
    dataset = Dataset(args.dataset, args.dataset_split, seed=args.dataset_seed)
    dataset.init_full_matrices()

    n_user, n_item = dataset.get_n_user_and_n_item()
    n_class = dataset.n_class

    train_x = dataset.train_matrix
    valid_x = dataset.train_matrix
    valid_y = dataset.valid_matrix
    test_x = (dataset.train_matrix + dataset.valid_matrix)
    # test_x = dataset.train_matrix
    test_y = dataset.test_matrix

    if args.dataset == Dataset.KIND_MOVIELENS_10M:
        train_x = train_x * 2
        valid_x = valid_x * 2
        valid_y = valid_y * 2
        test_x = test_x * 2
        test_y = test_y * 2

    if args.use_user_vector:
        n_row = n_user
        n_col = n_item
    else:
        train_x = train_x.T
        valid_x = valid_x.T
        valid_y = valid_y.T
        test_x = test_x.T
        test_y = test_y.T

        n_row = n_item
        n_col = n_user

    train_x = __expand(train_x, n_class=n_class)
    valid_x = __expand(valid_x, n_class=n_class)
    valid_y = __expand(valid_y, n_class=n_class)
    test_x = __expand(test_x, n_class=n_class)
    test_y = __expand(test_y, n_class=n_class)

    set_torch_seed(args.seed)

    rbm = RBM(n_vis=train_x.shape[1], k=1).cuda()

    train_op = optim.SGD(rbm.parameters(), 0.0001)

    for epoch in range(10):
        data = torch.from_numpy(train_x).cuda()
        sample_data = data.bernoulli()

        v, v1, vp = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        print(loss)
        print(vp.shape)
        train_op.zero_grad()
        loss.backward()
        train_op.step()
        # break

    # print(np.mean(loss_))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CFRBM-tensorflow')
    parser.add_argument(
        '--dataset',
        type=str,
        default=Dataset.KIND_MOVIELENS_100k,
        # default=Dataset.KIND_MOVIELENS_1M,
        choices=[
            Dataset.KIND_MOVIELENS_100k,
            Dataset.KIND_MOVIELENS_1M,
            Dataset.KIND_MOVIELENS_10M,
        ],
        help='Specify dataset.')
    parser.add_argument(
        '--dataset_split',
        type=str,
        # default='default-10-percent',
        default='5-fold',
        choices=[
            'default-10-percent',
            '5-fold',
        ],
        help='Specify dataset split.')
    parser.add_argument(
        '--n_batch', type=int, default=512, help='the batch size')
    parser.add_argument('--seed', type=int, default=1, help='the seed')
    parser.add_argument(
        '--use_user_vector',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether use user vector')
    parser.add_argument(
        '--dataset_seed',
        type=int,
        # default=None,
        default=1,
        help='the seed for dataset. None 일 때는 seed 값을 따라갑니다.')
    args = parser.parse_args()

    _train(args)


if __name__ == '__main__':
    main()
