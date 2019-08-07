import numpy as np

from .few_shots import omniglot
from ..utils import set_seed


class Dataset:
    def __init__(self, args):
        set_seed(args.dataset_seed)
        images = omniglot.load_data()

        np.random.shuffle(images)

        train_images = images[:args.n_train_class]
        valid_images = images[args.n_train_class:args.n_train_class + args.n_valid_class]
        test_images = images[args.n_train_class + args.n_valid_class:]

        print(train_images.shape)
        print(valid_images.shape)
        print(test_images.shape)

        np.random.choice(args.n_train_class, args.n_way, replace=False)


        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images


if __name__ == '__main__':
    from collections import namedtuple
    Arguments = namedtuple(
        'Arguments',
        'dataset_seed n_train_class n_valid_class n_way k_shot n_batch_task n_train_task n_valid_task n_test_task'
    )
    args = Arguments(
        dataset_seed=1,
        n_train_class=1100,
        n_valid_class=100,
        n_way=5,
        k_shot=1,
        n_batch_task=32,
        n_train_task=200000,
        n_valid_task=600,
        n_test_task=600)

    dataset = Dataset(args)
