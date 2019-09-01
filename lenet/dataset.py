import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from common.utils import set_torch_seed


def _data_transforms():
    """get transform of dataset"""
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                        0.2010)),
    ])

    train_transforms = data_transforms
    valid_transforms = data_transforms
    test_transforms = data_transforms

    return train_transforms, valid_transforms, test_transforms


def _dataset(train_transforms, valid_transforms, test_transforms):
    """get dataset for classification"""
    train_set = torchvision.datasets.CIFAR10(
        root='.datasets/cifar-10-torch',
        train=True,
        download=True,
        transform=train_transforms)
    remain_set = torchvision.datasets.CIFAR10(
        root='.datasets/cifar-10-torch',
        train=False,
        download=True,
        transform=test_transforms)

    # Split validation set from remain set
    num_remain = len(remain_set)
    indices = list(range(num_remain))
    split = int(np.floor(0.5 * num_remain))

    np.random.shuffle(indices)

    test_idx, valid_idx = indices[split:], indices[:split]

    valid_set = torch.utils.data.Subset(remain_set, valid_idx)
    test_set = torch.utils.data.Subset(remain_set, test_idx)

    return train_set, valid_set, test_set


def _data_loader(train_set, valid_set, test_set, batch_size):
    """get data loader"""
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def load_data(batch_size, dataset_seed=1):
    # set dataset seed and get data
    set_torch_seed(dataset_seed)
    train_transforms, valid_transforms, test_transforms = _data_transforms()
    train_set, valid_set, test_set = _dataset(train_transforms, valid_transforms,
                                            test_transforms)
    train_loader, valid_loader, test_loader = _data_loader(
        train_set, valid_set, test_set, batch_size)
    return train_loader, valid_loader, test_loader
