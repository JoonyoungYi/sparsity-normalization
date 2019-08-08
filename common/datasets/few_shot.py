import numpy as np

from .few_shots import omniglot
from ..utils import set_seed


class Dataset:
    def __init__(self, args):
        set_seed(args.dataset_seed)
        images = omniglot.load_data()

        np.random.shuffle(images)

        train_images = images[:args.n_train_class]
        valid_images = images[args.n_train_class:
                              args.n_train_class + args.n_valid_class]
        test_images = images[args.n_train_class + args.n_valid_class:]

        # print(train_images.shape)
        # print(valid_images.shape)
        # print(test_images.shape)

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images

        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_batch_task = args.n_batch_task
        self.n_train_class = args.n_train_class

    def _get_batch(self, images, n_class, pixel=28, n_sample_per_class=80):
        n_way = self.n_way
        k_shot = self.k_shot
        n_batch_task = self.n_batch_task

        # n_batch_task = args.n_batch_task
        alpha_xs = np.zeros(n_batch_task, n_way * k_shot, pixel, pixel)
        beta_xs = np.zeros(n_batch_task, n_way * k_shot, pixel, pixel)
        for task_idx in range(n_batch_task):
            class_idxs = np.random.choice(n_class, n_way, replace=False)
            _images = images[class_idxs]

            train_shot_idxs = np.random.randint(
                n_sample_per_class, size=[n_way, k_shot])
            valid_shot_idxs = np.random.randint(
                n_sample_per_class, size=[n_way, k_shot])
            _train_images = _images[[[_way] for _way in range(n_way)],
                                    train_shot_idxs]
            _valid_images = _images[[[_way] for _way in range(n_way)],
                                    valid_shot_idxs]
            _train_images = _train_images.reshape(
                [n_way * k_shot, pixel, pixel])
            _valid_images = _train_images.reshape(
                [n_way * k_shot, pixel, pixel])

            alpha_xs[task_idx] = _train_images
            beta_xs[task_idx] = _valid_images

        return alpha_xs, beta_xs

    def get_train_batch(self):
        return self._get_batch(self.train_images, self.n_train_class)


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
        k_shot=2,
        n_batch_task=32,
        n_train_task=200000,
        n_valid_task=600,
        n_test_task=600)

    dataset = Dataset(args)
