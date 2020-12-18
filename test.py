'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from DDN import DDN

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='VGG16', type=str, help='name of the model')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/fs/vulcan-datasets/CIFAR/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/fs/vulcan-datasets/CIFAR/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

resnet_dict = {'18':ResNet18, '34':ResNet34, '50':ResNet50, '101':ResNet101, '152':ResNet152}

def get_net(model):
    if model.startswith('VGG'):
        return VGG(model)
    elif model.startswith('resnet'):
        n_layer = model.split('_')[-1]
        if 'no_pooling' in model:
            return resnet_dict[n_layer](False)
        else:
            return resnet_dict[n_layer](True)


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()


# Model
print('==> Building model..')
net = get_net(args.model)
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    net = net.cuda()
    cudnn.benchmark = True

attacker = DDN(steps=100, device=torch.device('cuda'))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/%s.pth'%args.model)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']


adv_norm = 0.
correct = 0
adv_correct = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    adv = attacker.attack(net, inputs.to(device), labels=targets.to(device), targeted=False)
    outputs = net(inputs)
    _, predicted = outputs.max(1)
    adv_outputs = net(adv)
    _, adv_predicted = adv_outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    adv_correct += adv_predicted.eq(targets).sum().item()
    adv_norm += l2_norm(adv - inputs.to(device)).sum().item()
    progress_bar(batch_idx, len(testloader), 'Adv Acc: %.3f | Acc: %.3f%% (%d/%d)'
                 % (100.*adv_correct/total, 100.*correct/total, correct, total))


print('Raw done in error: {:.2f}%.'.format(100.*correct/total))
print('DDN done in Success: {:.2f}%, Mean L2: {:.4f}.'.format(
    100.*adv_correct/total,
    adv_norm/total
))





# pred_orig = net(inputs.to(device)).argmax(dim=1).cpu()
# pred_ddn = net(adv).argmax(dim=1).cpu()
# print('Raw done in error: {:.2f}%.'.format(
#     (pred_orig != targets).float().mean().item() * 100,
# ))
# print('DDN done in Success: {:.2f}%, Mean L2: {:.4f}.'.format(
#     (pred_ddn != targets).float().mean().item() * 100,
#     l2_norm(adv - inputs.to(device)).mean().item()
# ))