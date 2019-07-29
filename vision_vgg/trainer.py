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

import os
import argparse
from common.utils import str2bool, integer_list

from .vgg import *
from .utils import progress_bar, print2file

parser = argparse.ArgumentParser(description='VGG16')
parser.add_argument('--lr',
                    default=1e-3,
                    type=float,
                    help='learning rate')
parser.add_argument('--resume', '-r',
                    action='store_true',
                    help='resume from last')
parser.add_argument('--resume_best',
                    action='store_true',
                    help='resume from best model')
parser.add_argument('--seed',
                    default=1,
                    type=int)
parser.add_argument('--model',
                    type=str,
                    help='model name for checkpoint')
parser.add_argument('--sparsity_normalization',
                    default=False,
                    type=str2bool,
                    help='whether to apply sparsity normalization')
parser.add_argument('--normalize_layer_indexes',
                    default=[],
                    type=integer_list,
                    help='list of layers to which normalization is applied')
parser.add_argument('--batch_norm',
                    default=False,
                    type=str2bool)
parser.add_argument('--dropout',
                    default=False,
                    type=str2bool)
parser.add_argument('--fan_in_K_prime',
                    default=False,
                    type=str2bool,
                    help='Set K_prime to fan_in (See Algorithm 1 in the paper)')
parser.add_argument('--K_prime',
                    default=1.0,
                    type=float,
                    help='Set K_prime (See Algorithm 1 in the paper)')
parser.add_argument('--epochs',
                    default=3000,
                    type=int)
parser.add_argument('--batch_size',
                    default=128,
                    type=int)
args = parser.parse_args()

forbidden_layer_index = [0, 3, 6, 10, 14]
for layer_index in args.normalize_layer_indexes:
    assert layer_index not in forbidden_layer_index

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # start from 0 or last best validation accuracy
best_epoch = 0
evaluated_acc = 0
test_best_acc = 0
test_best_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Setting seed for reproducing
# fix seed for split dataset
random.seed(2019)
os.environ['PYTHONHASHSEED'] = str(2019)
np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data preparation
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='vision_vgg/data', train=True, download=True, transform=transform_train)
# Load trainloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)

remain_set = torchvision.datasets.CIFAR10(root='vision_vgg/data', train=False, download=True, transform=transform_test)

# Split validation set from remain set
num_remain = len(remain_set)
indices = list(range(num_remain))
split = int(np.floor(0.5 * num_remain))

np.random.shuffle(indices)

test_idx, valid_idx = indices[split:], indices[:split]

validset = torch.utils.data.Subset(remain_set, valid_idx)
testset = torch.utils.data.Subset(remain_set, test_idx)

# Load testloader and validloader
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False,
    num_workers=2,
)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=100, shuffle=False,
    num_workers=2,
)

# change seed for training
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model construction
print('==> Building model..')
net = VGG('vgg16', args.sparsity_normalization, args.normalize_layer_indexes, args.batch_norm, args.dropout, args.fan_in_K_prime, args.K_prime)
net = net.to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.resume_best:
    # Load checkpoint.
    print('==> Resuming from best checkpoint..')
    assert os.path.isdir('vision_vgg/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('vision_vgg/checkpoint/ckpt.{}_{}'.format(args.model, args.seed))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optim'])
    best_acc = checkpoint['best_acc']
    best_epoch = checkpoint['best_epoch']
    evaluated_acc = checkpoint['eval_acc']
    test_best_acc = checkpoint['test_best_acc']
    test_best_epoch = checkpoint['test_best_epoch']
    start_epoch = checkpoint['epoch']

elif args.resume:
    # Load checkpoint.
    print('==> Resuming from last checkpoint..')
    assert os.path.isdir('vision_vgg/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('vision_vgg/checkpoint/ckpt.last_{}_{}'.format(args.model, args.seed))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optim'])
    best_acc = checkpoint['best_acc']
    best_epoch = checkpoint['best_epoch']
    evaluated_acc = checkpoint['eval_acc']
    test_best_acc = checkpoint['test_best_acc']
    test_best_epoch = checkpoint['test_best_epoch']
    start_epoch = checkpoint['epoch']+1

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

        print2file('Epoch: {}'.format(epoch), args.model)
        print2file('Train Accuracy: {:.4f}'.format(correct/total), args.model)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Validation
def valid(epoch):
    # define as global to use again
    global best_acc
    global best_epoch
    global evaluated_acc
    global test_best_acc
    global test_best_epoch
    net.eval()

    # calculate validation accuracy
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

    acc = correct/total
    print2file('Valid Accuracy: {:.4f}'.format(acc), args.model)

    # calculate test accuracy 
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

    test_acc = test_correct/test_total
    print2file('Test  Accuracy: {:.4f}\n'.format(test_acc), args.model)

    # Tracking best test accuracy
    if test_acc > (test_best_acc + 1e-4):
        test_best_acc = test_acc
        test_best_epoch = epoch

    # Save checkpoint.
    if acc > (best_acc + 1e-4):
        best_acc = acc
        best_epoch = epoch
        # Tracking test accuracy of the best model
        evaluated_acc = test_acc

        print('Saving..')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'eval_acc': evaluated_acc,
            'test_best_acc': test_best_acc,
            'test_best_epoch': test_best_epoch,
            'epoch': epoch
        }
        if not os.path.isdir('vision_vgg/checkpoint'):
            os.mkdir('vision_vgg/checkpoint')
        torch.save(state, 'vision_vgg/checkpoint/ckpt.{}_{}'.format(args.model, args.seed))

    print("Best")
    print("valid acc: {}".format(best_acc))
    print("valid epoch: {}".format(best_epoch))
    print("evaluated acc: {}".format(evaluated_acc))
    print("test acc: {}".format(test_best_acc))
    print("test epoch: {}".format(test_best_epoch))

    state = {
        'net': net.state_dict(),
        'optim': optimizer.state_dict(),
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'eval_acc': evaluated_acc,
        'test_best_acc': test_best_acc,
        'test_best_epoch': test_best_epoch,
        'epoch': epoch
    }
    if not os.path.isdir('vision_vgg/checkpoint'):
        os.mkdir('vision_vgg/checkpoint')
    torch.save(state, 'vision_vgg/checkpoint/ckpt.last_{}_{}'.format(args.model, args.seed))

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    valid(epoch)
