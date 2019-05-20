'''https://github.com/kuangliu/pytorch-cifar'''

from __future__ import print_function

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
import argparse
from common.utils import str2bool

from .vgg import *
from .utils import progress_bar

parser = argparse.ArgumentParser(description='VGG16')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--dropout', type=str2bool, default=False)
parser.add_argument('--epochs', default=3000, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_epoch = 0
evaluated_acc = 0
test_best_acc = 0
test_best_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='vgg16/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

validset = torchvision.datasets.CIFAR10(root='vgg16/data', train=False, download=True, transform=transform_test)
testset = torchvision.datasets.CIFAR10(root='vgg16/data', train=False, download=True, transform=transform_test)

num_test = len(testset)
indices = list(range(num_test))
split = int(np.floor(0.5 * num_test))

np.random.shuffle(indices)

test_idx, valid_idx = indices[split:], indices[:split]
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, sampler=test_sampler,
    num_workers=2,
)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=100, sampler=valid_sampler,
    num_workers=2,
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG(args.model, args.dropout)
net = net.to(device)
print(net)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('vgg16/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('vgg16/checkpoint/ckpt.{}_{}'.format(args.model, args.seed))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def valid(epoch):
    global best_acc
    global best_epoch
    global evaluated_acc
    global test_best_acc
    global test_best_epoch
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*test_correct/test_total, test_correct, test_total))

    test_acc = 100.*test_correct/test_total

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('vgg16/checkpoint'):
            os.mkdir('vgg16/checkpoint')
        torch.save(state, 'vgg16/checkpoint/ckpt.{}_{}'.format(args.model, args.seed))
        best_acc = acc
        best_epoch = epoch
        evaluated_acc = test_acc

    if test_acc > test_best_acc:
        test_best_acc = test_acc
        test_best_epoch = epoch

    print("Best")
    print("valid acc: {}".format(best_acc))
    print("valid epoch: {}".format(best_epoch))
    print("evaluated acc: {}".format(evaluated_acc))
    print("test acc: {}".format(test_best_acc))
    print("test epoch: {}".format(test_best_epoch))

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    valid(epoch)
