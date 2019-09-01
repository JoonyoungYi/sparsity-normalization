import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def data_transforms():
    """get transform of dataset"""
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                        0.2010)),
    ])

    train_transforms = data_transforms
    val_transforms = data_transforms
    test_transforms = data_transforms

    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transforms)
    remain_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transforms)

    # Split validation set from remain set
    num_remain = len(remain_set)
    indices = list(range(num_remain))
    split = int(np.floor(0.5 * num_remain))

    np.random.shuffle(indices)

    test_idx, val_idx = indices[split:], indices[:split]

    val_set = torch.utils.data.Subset(remain_set, val_idx)
    test_set = torch.utils.data.Subset(remain_set, test_idx)

    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set, batch_size):
    """get data loader"""
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
